from typing import List
import torch
import torch.nn as nn

import numpy as np
from torch import Tensor
from .fewshot_model import FewShot
from .fewshot_utils import compute_prototypes, compute_prototypes_singleclass
# import statistics
import scipy
from sklearn.model_selection import train_test_split

class BDCSPN(FewShot):

    """
    Jinlu Liu, Liang Song, Yongqiang Qin
    "Prototype Rectification for Few-Shot Learning" (ECCV 2020)
    https://arxiv.org/abs/1911.10713

    Rectify prototypes with label propagation and feature shifting.
    Classify queries based on their cosine distance to prototypes.
    This is a transductive method.
    """
    def __init__(
        self,
        *args,
        is_single_class, use_sam_embeddings,
        **kwargs,
    ):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: dimension of the feature vectors extracted by the backbone.
            support_encoder: module encoding support features. If none is specific, we use
                the default encoder from the original paper.
            query_encoder: module encoding query features. If none is specific, we use
                the default encoder from the original paper.
        """
        super().__init__(*args, **kwargs)

        # DANNY ADDED
        self.mean = None
        self.std = None
        self.num_samples = None
        self.is_single_class = is_single_class
        self.use_sam_embeddings = use_sam_embeddings
        # DANNY ADDED

    # DANNY ADDED
    def get_embeddings_timm(self, img):
        """
        Returns the embeddings from the backbone which is a timm model.
        """
        with torch.no_grad():
            x = self.backbone.forward(img.unsqueeze(dim=0))#.to('cuda'))
        return x

    def get_embeddings_sam(self, img):
        """
        Returns the embeddings from the backbone which SAM.
        """
        with torch.no_grad():
            x = self.backbone.get_embeddings(img)
        return x
    # DANNY ADDED
    def process_support_set(
        self,
        #support_images: Tensor,
        #support_labels: Tensor,
        support_images: List,
        support_labels: List = None,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract features from the support set with full context embedding.
        Store contextualized feature vectors, as well as support labels in the one hot format.

        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        #support_features = self.compute_features(support_images)
        support_features = []
        self.num_samples = len(support_images)

        #---------------------------------------
        # split the ids
        if support_labels is None:
            y_labels = np.zeros(len(support_images))
        else:
            y_labels = np.zeros(len(support_images))
        imgs_1, imgs_2, lbl_1, lbl_2 = train_test_split(
            support_images, y_labels,
            train_size = 0.6,
            shuffle=True # shuffle the data before splitting
        )
        #---------------------------------------

        # DANNY ADDED
        support_labels = torch.Tensor(lbl_1)
        # DANNY ADDED


        # get feature maps from the images
        for img in imgs_1:
            if self.use_sam_embeddings:
                t_temp = self.get_embeddings_sam(img)
            else:
                t_temp = self.get_embeddings_timm(img)
            support_features.append(t_temp.squeeze().cpu())

        # get prototypes and save them into cuda memory
        self.support_features = torch.stack(support_features)
        if self.is_single_class:
            prototypes = compute_prototypes_singleclass(self.support_features)
            prototypes = prototypes.unsqueeze(dim=0) # 2D tensor
        else:
            support_labels = torch.Tensor(lbl_1)
            prototypes = compute_prototypes(self.support_features, support_labels)
        self.prototypes = prototypes#.to('cuda')
        self.support_labels = support_labels
        #---------------------------------------
        if self.is_single_class:
            self._calculate_statistics(imgs_2)
        #---------------------------------------

    def _calculate_statistics(
        self,
        imgs: List,
    ) -> Tensor:
        """
        Get metrics from the embeddings.

        Params
        :imgs (tensor) -> embedding to calculate metrics.
        """
        assert self.is_single_class, "This method can be used just in single class"
        scores = []
        for img in imgs:
            score = self.forward(img)
            scores.append(score.cpu().item())
        self.mean = scipy.mean(scores)
        self.std = scipy.std(scores)

    def rectify_prototypes(self, query_features: Tensor):
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels.long(), n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift



        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()

        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )

        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [n_query, n_classes]

        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Update prototypes using query images, then classify query images based
        on their cosine distance to updated prototypes.
        """
        #query_features = self.compute_features(query_images)

        # DANNY CODE
        if self.use_sam_embeddings:
            z_query = self.get_embeddings_sam(query_images)
        else:
            z_query = self.get_embeddings_timm(query_images)
        # DANNY CODE
        self.rectify_prototypes(
            query_features=z_query,
        )
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(z_query)
        )

    @staticmethod
    def is_transductive() -> bool:
        return True