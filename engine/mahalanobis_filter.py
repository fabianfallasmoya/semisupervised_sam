import os
import json
import torch
import numpy as np
from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground, Transform_To_Models
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class MahalanobisFilter:

    def __init__(self,
                timm_model=None,
                timm_pretrained=True,
                num_classes=1,
                sam_model=None,
                use_sam_embeddings=False,
                is_single_class=True,
                device="cpu"):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        self.mean = None
        self.inv_cov = None
        self.device = device
        self.num_samples = None
        self.num_classes = num_classes
        self.timm_model = timm_model
        self.sam_model = sam_model
        self.is_single_class = is_single_class

        if not use_sam_embeddings:
            # create a model for feature extraction
            feature_extractor = MyFeatureExtractor(
                timm_model, timm_pretrained, num_classes
            ).to(self.device)
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = sam_model
        self.sam_model = sam_model

        # create the default transformation
        if use_sam_embeddings:
            trans_norm = Transform_To_Models()
        else:
            if feature_extractor.is_transformer:
                trans_norm = Transform_To_Models(
                        size=feature_extractor.input_size,
                        force_resize=True, keep_aspect_ratio=False
                    )
            else:
                trans_norm = Transform_To_Models(
                        size=33, force_resize=False, keep_aspect_ratio=True
                    )
        self.trans_norm = trans_norm
        self.use_sam_embeddings = use_sam_embeddings

    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        # Covariance matrix
        cov_matrix = torch.cov(embeddings.T)
        # Pseudo inverse of the covariance matrix
        inv_cov = torch.pinverse(cov_matrix)
        self.inv_cov = inv_cov

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        #if mean.dim() == 1:  # Distribution mean.
        #    mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        # x_mu shape (samples x embedding size), inv_covariance (embedding size x embedding size), dist shape ()
        #dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        dist = torch.sqrt(torch.diagonal(torch.mm(torch.mm(x_mu, inv_covariance), x_mu.T)))

        #print("x_mu:", x_mu)
        #print("covariance: ", inv_covariance)
        return dist #.sqrt()
    
    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        mean_value = torch.mean(distances).item()
        std_deviation = torch.std(distances).item()
        print(distances)
        print("Mean predicted:", mean_value)
        print("Standard Deviation predicted:", std_deviation)
        return distances
    
    def run_filter(self,
        labeled_loader,
        unlabeled_loader,
        dir_filtered_root = None, get_background_samples=False):

        # 1. Get feature maps from the labeled set
        labeled_imgs = []
        labeled_labels = []
        
        for batch in labeled_loader:
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from bounding boxes)
                imgs_f, labels_f = get_foreground(
                    batch, idx, self.trans_norm,
                    self.use_sam_embeddings
                )
                labeled_imgs += imgs_f
                labeled_labels += labels_f
        #print("Len labeled imgs: ", len(labeled_imgs))
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]

        # get all features maps using: the extractor + the imgs
        feature_maps_list = self.get_all_features(labeled_imgs)
        #print("feature_maps_list imgs: ", feature_maps_list)
        #----------------------------------------------------------------
        if self.is_single_class:
            labels = np.zeros(len(feature_maps_list))
        else:
            labels = np.array(labels)

        imgs_1, imgs_2, _, _ = train_test_split(
            feature_maps_list, labels,
            train_size = 0.9,
            shuffle=True # shuffle the data before splitting
        )

        # 2. Calculating the mean prototype for the labeled data
        #----------------------------------------------------------------
        #support_features_imgs_1 = torch.stack(imgs_1)
        #support_features_imgs_2 = torch.stack(imgs_2)
        support_features = torch.stack(feature_maps_list)
        #print("Support features mean: ", support_features.mean())
        #print("Features size: ", support_features.shape)

        # 3. Calculating the sigma (covariance matrix), the distances 
        # with respect of the support features and get the threshold
        #----------------------------------------------------------------
        if self.is_single_class:
            #self.fit(support_features_imgs_1)
            self.fit(support_features)
            #distances = self.predict(support_features_imgs_2)
            distances = self.predict(support_features)
            mean_value = torch.mean(distances).item()
            std_deviation = torch.std(distances).item()
            #print("distances: ", distances)
            #print("Mean calculate:", mean_value)
            #print("Standard Deviation calculate:", std_deviation)
            self.threshold = max(distances).item()
            #print("threshold:", self.threshold)

        # go through each batch unlabeled
        distances_all = 0

        # keep track of the img id for every sample created by sam
        imgs_ids = []
        imgs_box_coords = []
        imgs_scores = []

        # 3. Get batch of unlabeled // Evaluating the likelihood of unlabeled data
        for (batch_num, batch) in tqdm(
            enumerate(unlabeled_loader), total= len(unlabeled_loader)
        ):
            unlabeled_imgs = []
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from sam)
                imgs_s, box_coords, scores = self.sam_model.get_unlabeled_samples(
                    batch, idx, self.trans_norm, self.use_sam_embeddings
                )
                unlabeled_imgs += imgs_s

                # accumulate SAM info (inferences)
                imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
                imgs_box_coords += box_coords
                imgs_scores += scores

            # get all features maps using: the extractor + the imgs
            featuremaps_list = self.get_all_features(unlabeled_imgs)
            featuremaps = torch.stack(featuremaps_list) # e.g. [387 x 512]

            # init buffer with distances
            support_set_distances = []
            distances = self.predict(featuremaps)
            support_set_distances = distances

            # accumulate
            if (batch_num == 0):
                distances_all = support_set_distances
            else:
                distances_all = torch.cat((distances_all, support_set_distances), 0)

        # transform data 
        scores = []
        for j in range(0, distances_all.shape[0]):
            scores += [distances_all[j].item()]
        scores = np.array(scores).reshape((len(scores),1))

        limit = self.threshold 
        # accumulate results
        results = []
        for index, score in enumerate(scores):
            if(score.item() <= limit):
                image_result = {
                    'image_id': imgs_ids[index],
                    'category_id': 1, # fix this
                    'score': imgs_scores[index],
                    'bbox': imgs_box_coords[index],
                }
                results.append(image_result)

        if len(results) > 0:
            # write output
            results_file = f"{dir_filtered_root}/bbox_results.json"
            if os.path.isfile(results_file):
                os.remove(results_file)
            json.dump(results, open(results_file, 'w'), indent=4)

    def get_all_features(self, images):
        """
        Extract feature vectors from the images.
        
        Params
        :images (List<tensor>) images to be used to extract features
        """
        features = []
        # get feature maps from the images
        if self.use_sam_embeddings:
            with torch.no_grad():
                for img in images:
                    t_temp = self.feature_extractor.get_embeddings(img)
                    features.append(t_temp.squeeze().cpu())
        else:
            with torch.no_grad():
                for img in images:
                    t_temp = self.feature_extractor(img.unsqueeze(dim=0).to(self.device))
                    features.append(t_temp.squeeze().cpu())
        return features