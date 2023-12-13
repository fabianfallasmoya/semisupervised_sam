import os
import json
import torch
import numpy as np
from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground, Transform_To_Models, get_background
from tqdm import tqdm
from numpy import linalg as la
from sklearn.covariance import LedoitWolf, MinCovDet
import cv2

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
        #self.power_transf = PowerTransformer(method='yeo-johnson', standardize=True)

        if not use_sam_embeddings:
            # create a model for feature extraction
            feature_extractor = MyFeatureExtractor(
                timm_model, timm_pretrained, num_classes #128, use_fc=True
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
    
    def is_positive_definite(self, matrix):
        try:
            np.linalg.cholesky(matrix)
            return True  # Cholesky decomposition succeeded, matrix is positive definite
        except np.linalg.LinAlgError:
            return False  # Cholesky decomposition failed, matrix is not positive definite

    def calculate_svd(self, covariance_matrix, k=512):
        U, Sigma, Vt = np.linalg.svd(covariance_matrix)
        U_k = U[:, :k]
        Sigma_k = np.diag(Sigma)
        Vt_k = Vt[:k, :]
        self.Vt_k = Vt_k
        # Compute the new matrix in the principal component space
        new_cov = U_k.T @ Sigma_k @ Vt_k.T
        self.new_cov = new_cov
        return new_cov

    def inference_points(self, data_point):
        print(self.Vt_k.T.shape)
        print(data_point.shape)

        return data_point @ self.Vt_k.T
    
    def fit_svd(self, embeddings, k=512):
        covariance_matrix = np.cov(embeddings.detach().numpy(), rowvar=False)
        new_covariance = self.calculate_svd(covariance_matrix, k=k)
        embeddings_new = self.inference_points(embeddings.detach().numpy())
        self.mean = torch.mean(torch.Tensor(embeddings_new), axis=0)
        self.inv_cov = torch.pinverse(torch.Tensor(new_covariance))

    def fit_mcd(self, embeddings):
        #embeddings = torch.Tensor(torch.sigmoid(embeddings))
        self.mean = torch.mean(embeddings, axis=0)
        print(self.mean.shape)

        self.mincovdet = MinCovDet()
        self.mincovdet.fit(embeddings.detach().numpy())
        #self.ledoitwolf.fit(embeddings)
        covariance_matrix = self.mincovdet.precision_

        #self.inv_cov = torch.pinverse(torch.Tensor(covariance_matrix))
        self.inv_cov =torch.Tensor(covariance_matrix)
        #covariance_matrix = covariance_matrix.detach().numpy()
        if self.is_positive_definite(self.inv_cov):
            print("The matrix is positive definite.")
        elif self.is_positive_semidefinite(self.inv_cov):
            print("The matrix is positive semi-definite.")
        else:
            print("The matrix is neither positive definite nor positive semi-definite.")

    def estimate_covariance(self, examples, rowvar=False, inplace=False):
        """
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()
    
    def fit_regularization(self, examples, context_features=None):
        self.mean = torch.mean(examples, axis=0)
        
        covariance_matrix = self.estimate_covariance_ledoitwolf(examples) #self.estimate_covariance(examples)
        if context_features != None:
            context_covariance_matrix = self.estimate_covariance_ledoitwolf(context_features) #self.estimate_covariance(context_features)

        lambda_k_tau = (examples.size(0) / (examples.size(0) + 1))

        print("Covariance matrix shape: ", covariance_matrix.shape)
        print("torch.eye(examples.size(1), examples.size(1)) shape: ", torch.eye(examples.size(1), examples.size(1)).shape)

        self.inv_cov = torch.inverse((lambda_k_tau * covariance_matrix) + ((1 - lambda_k_tau) * context_covariance_matrix) \
                    + torch.eye(examples.size(1), examples.size(1)))
        
        print("Self.mean shape: ", self.mean.shape)

    def estimate_covariance_ledoitwolf(self, embeddings):
        self.ledoitwolf = LedoitWolf()
        self.ledoitwolf.fit(embeddings.detach().numpy())
        covariance_matrix = self.ledoitwolf.covariance_
        return torch.Tensor(covariance_matrix)

    def fit(self, embeddings):
        #embeddings = torch.Tensor(torch.sigmoid(embeddings))
        self.mean = torch.mean(embeddings, axis=0)
        print(self.mean.shape)
        # Covariance matrix
        #covariance_matrix = ShrunkCovariance().fit(embeddings).covariance_
        #covariance_matrix = OAS().fit(embeddings).covariance_
        #self.inv_cov = torch.tensor(np.linalg.pinv(covariance_matrix), dtype=torch.float) #torch.tensor(covariance_matrix, dtype=torch.float, device=self.device) #torch.tensor(np.linalg.pinv(covariance_matrix), dtype=torch.float)
        #covariance_matrix = np.cov(embeddings, rowvar=False)
        
        #covariance_matrix = torch.matmul(embeddings.t(), embeddings) / (embeddings.size(0) - 1)
        
        #embeddings = self.power_transf.fit_transform(embeddings.detach().numpy())

        self.ledoitwolf = LedoitWolf()
        self.ledoitwolf.fit(embeddings.detach().numpy())
        #self.ledoitwolf.fit(embeddings)
        covariance_matrix = self.ledoitwolf.precision_
        #print("covariance_matrix: ", covariance_matrix.shape)

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
            #covariance_matrix = self.nearestPD(covariance_matrix)
        #covariance_matrix = torch.cov(embeddings.T)
        #print(covariance_matrix.shape)
        self.inv_cov = torch.Tensor(covariance_matrix)
        #covariance_matrix = covariance_matrix.detach().numpy()
        if self.is_positive_definite(self.inv_cov):
            print("The matrix is positive definite.")
        elif self.is_positive_semidefinite(self.inv_cov):
            print("The matrix is positive semi-definite.")
        else:
            print("The matrix is neither positive definite nor positive semi-definite.")
        #    print("Covariance Matrix is Positive Semidefinite: ", self.is_positive_semidefinite(covariance_matrix.detach().numpy()))
        #    are_equal = np.array_equal(self.nearestPD(covariance_matrix.detach().numpy()), covariance_matrix.detach().numpy())
        #    print("Are equal nearest Positive definite: ", are_equal)
        #
        #    covariance_matrix = self.nearestPD(covariance_matrix)

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
        
        #self.inv_cov = torch.tensor(covariance_matrix, dtype=torch.float)
        #nearest_cov = covariance_matrix
        #if self.is_positive_semidefinite(covariance_matrix):
            ## added 
        #    print("Not positive semidefite")
            #nearest_cov = self.nearestPD(covariance_matrix)
            #print(nearest_cov)

        #self.inv_cov = torch.tensor(covariance_matrix, dtype=torch.float)
        #print("Near Covariance Positive semi definite")

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
        x_mu = mean - values  # batch x features
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
        
    def predict_svd(self, embeddings):

        embeddings = embeddings.detach().numpy()
        #new_embeddings = self.inference_points(embeddings)
        distances = self.mahalanobis_distance(torch.Tensor(embeddings), self.mean, self.inv_cov)

        mean_value = torch.mean(distances).item()
        std_deviation = torch.std(distances).item()
        print(distances)
        print("Mean predicted:", mean_value)
        print("Standard Deviation predicted:", std_deviation)
        return distances

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        #distances = self.mahalanobis_distance_v2(embeddings)
        mean_value = torch.mean(distances).item()
        std_deviation = torch.std(distances).item()
        print(distances)
        print("Mean predicted:", mean_value)
        print("Standard Deviation predicted:", std_deviation)
        return distances

    def run_filter(self,
        labeled_loader,
        unlabeled_loader,
        dir_filtered_root = None, get_background_samples=True,
        num_classes:float=0):

        # 1. Get feature maps from the labeled set
        labeled_imgs = []
        labeled_labels = []

        #all_imgs_context = []
        #all_label_context = []
        
        back_imgs_context = []
        back_label_context = []

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

                #all_imgs_context += imgs_f
                #all_label_context += labels_f

                if get_background_samples:
                    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

                    imgs_b, labels_b = get_background(
                        batch, idx, self.trans_norm, ss, 
                        num_classes, self.use_sam_embeddings)
                    back_imgs_context += imgs_b
                    back_label_context += labels_b

        all_imgs_context = labeled_imgs + back_imgs_context[:len(labeled_imgs)]

        #print("Len labeled imgs: ", len(labeled_imgs))
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]

        # get all features maps using: the extractor + the imgs
        feature_maps_list = self.get_all_features(labeled_imgs)
        all_feature_maps_list = self.get_all_features(all_imgs_context)

        #print("Feature map list: ", feature_maps_list)
        #print("feature_maps_list imgs: ", feature_maps_list)
        #----------------------------------------------------------------
        if self.is_single_class:
            labels = np.zeros(len(feature_maps_list))
        else:
            labels = np.array(labels)

        # 2. Calculating the mean prototype for the labeled data
        #----------------------------------------------------------------
        #support_features_imgs_1 = torch.stack(imgs_1)
        #support_features_imgs_2 = torch.stack(imgs_2)
        support_features = torch.stack(feature_maps_list)
        all_support_features = torch.stack(all_feature_maps_list)
        print("Support features shape: ", support_features.shape)
        print("all_support_features shape: ", all_support_features.shape)

        #support_features = torch.sigmoid(support_features)
        #print("Support features mean: ", support_features.mean())
        #print("Features size: ", support_features.shape)

        # 3. Calculating the sigma (covariance matrix), the distances 
        # with respect of the support features and get the threshold
        #----------------------------------------------------------------
        if self.is_single_class:
            #self.fit(support_features_imgs_1)
            #support_features = self.power_transf.fit_transform(support_features.detach().numpy())
            #support_features = torch.Tensor(support_features)
            
            #self.fit(support_features)
            self.fit_regularization(support_features, all_support_features)
            #self.fit_mcd(support_features)
            #self.fit_svd(support_features)
            #distances = self.predict(support_features_imgs_2)
            distances = self.predict(support_features)
            #support_features_new = torch.Tensor(self.inference_points(support_features.detach().numpy()))
            #distances = self.predict_svd(support_features_new)
            
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
            
            #featuremaps_new = torch.Tensor(self.inference_points(featuremaps.detach().numpy()))

            #featuremaps = self.power_transf.transform(featuremaps.detach().numpy())
            #featuremaps = torch.Tensor(featuremaps)
            #featuremaps = torch.sigmoid(featuremaps)
            # init buffer with distances
            support_set_distances = []
            distances = self.predict(featuremaps)
            #distances = self.predict_svd(featuremaps_new)

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