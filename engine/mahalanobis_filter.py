import os
import json
import torch
import numpy as np
from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground, Transform_To_Models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import linalg as la
from sklearn.covariance import LedoitWolf

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
                timm_model, timm_pretrained, num_classes #256, use_fc=True
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

    def is_positive_semidefinite(self, covariance_matrix):
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        positive_semidefinite = all(eigenvalues >= 0)
        return positive_semidefinite

    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3
        
    def isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False
        
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        # Covariance matrix
        #covariance_matrix = ShrunkCovariance().fit(embeddings).covariance_
        #covariance_matrix = OAS().fit(embeddings).covariance_
        #self.inv_cov = torch.tensor(np.linalg.pinv(covariance_matrix), dtype=torch.float) #torch.tensor(covariance_matrix, dtype=torch.float, device=self.device) #torch.tensor(np.linalg.pinv(covariance_matrix), dtype=torch.float)
        #covariance_matrix = np.cov(embeddings, rowvar=False)
        #covariance_matrix = torch.matmul(embeddings.t(), embeddings) / (embeddings.size(0) - 1)
        self.ledoitwolf = LedoitWolf()
        self.ledoitwolf.fit(embeddings.detach().numpy())
        covariance_matrix = self.ledoitwolf.precision_
        print("covariance_matrix: ", covariance_matrix.shape)

        #covariance_matrix = torch.Tensor(embeddings.cpu()).t().mm(torch.Tensor(embeddings.cpu())).cov()
        #print(covariance_matrix)
        #if not np.all(np.linalg.eigvals(covariance_matrix) > 0):
        #    covariance_matrix = self.higham_regularization(covariance_matrix)
            #inv_cov = self.higham_regularization(inv_cov)
            ##inv_cov = self.logarithmic_transform(inv_cov)
            #inv_cov = self.diagonal_loading(inv_cov)
            ##inv_cov = self.cholesky_decomposition(inv_cov)
        #    print(covariance_matrix)
        #    print("Positive defined ")
        #if self.is_pos_semidef(covariance_matrix.detach().numpy()):
        #    print("Pos semidef")
        #covariance_matrix = self.get_near_psd(covariance_matrix.detach().numpy())
        #covariance_matrix = np.array(self.nearPSD(covariance_matrix.detach().numpy()))

        #if not self.is_positive_semidefinite(covariance_matrix.detach().numpy()):
        #    print("Covariance Matrix is Positive Semidefinite: ", self.is_positive_semidefinite(covariance_matrix.detach().numpy()))
        #    are_equal = np.array_equal(self.nearestPD(covariance_matrix.detach().numpy()), covariance_matrix.detach().numpy())
        #    print("Are equal nearest Positive definite: ", are_equal)
        #
        #    covariance_matrix = self.nearestPD(covariance_matrix.detach().numpy())

        # Calculate determinant
        #det = np.linalg.det(covariance_matrix)
        # Check if the determinant is non-zero
        #if det != 0:
        #    print("The matrix is non-singular (invertible).")
        #    covariance_matrix = torch.Tensor(covariance_matrix)
        #    self.inv_cov = covariance_matrix
        #else:
        #    print("The matrix is singular.")
        #    covariance_matrix = torch.Tensor(covariance_matrix)
        #    self.inv_cov =  torch.pinverse(covariance_matrix)
        #    det = np.linalg.det(covariance_matrix)
        #    print("Determinant: ", det)
            
        
        #self.inv_cov  = np.linalg.pinv(covariance_matrix)
        #self.inv_cov = torch.pinverse(covariance_matrix)
        self.inv_cov = torch.tensor(covariance_matrix, dtype=torch.float)
        

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
        #dist = torch.sqrt(torch.diagonal(torch.mm(torch.mm(x_mu, inv_covariance), x_mu.T)))
        dist = torch.diagonal(torch.mm(torch.mm(x_mu, inv_covariance), x_mu.T))
        #print("x_mu:", x_mu)
        #print("covariance: ", inv_covariance)
        return dist.sqrt()
        #return self.ledoitwolf.

    def mahalanobis_distance_v2(self, X):
        return torch.tensor(self.ledoitwolf.mahalanobis(X))
    
    def predict(self, embeddings):
        #distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        distances = self.mahalanobis_distance_v2(embeddings)
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
            #mean_value = torch.mean(distances).item()
            #std_deviation = torch.std(distances).item()
            #print("distances: ", distances)
            #print("Mean calculate:", mean_value)
            #print("Standard Deviation calculate:", std_deviation)
            self.threshold = max(distances).item()
            print("Threshold: ", self.threshold)
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
        print("Scores: ", len(scores))
        count = 0
        for index, score in enumerate(scores):
            if(score.item() <= limit):
                image_result = {
                    'image_id': imgs_ids[index],
                    'category_id': 1, # fix this
                    'score': imgs_scores[index],
                    'bbox': imgs_box_coords[index],
                }
                results.append(image_result)
                count=count+1
                print("Score.item(): ", score.item())
        print("Count: ", count)

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