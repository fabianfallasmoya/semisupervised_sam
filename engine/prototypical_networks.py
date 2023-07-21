"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""
from typing import List
import torch
from torch import Tensor
from .fewshot_model import FewShot
from .fewshot_utils import compute_prototypes, compute_prototypes_singleclass
import statistics


class PrototypicalNetworks(FewShot):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    # def __init__(self, *args, **kwargs):
    #     """
    #     Raises:
    #         ValueError: if the backbone is not a feature extractor,
    #         i.e. if its output for a given image is not a 1-dim tensor.
    #     """
    #     super().__init__(*args, **kwargs)

    #     if len(self.backbone_output_shape) != 1:
    #         raise ValueError(
    #             "Illegal backbone for Prototypical Networks. "
    #             "Expected output for an image is a 1-dim tensor."
    #         )

    def process_support_set(
        self,
        support_images: List,
        support_labels: List,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        # support_features = self.backbone.forward(support_images)
        support_features = []
        self.backbone.eval()
        # get feature maps from the images
        with torch.no_grad():
            for img in support_images:
                t_temp = self.backbone.forward(img.unsqueeze(dim=0).to('cuda'))
                support_features.append(t_temp.squeeze().cpu())
        self.backbone.train()
        
        # get prototypes and save them into cuda memory
        support_features = torch.stack(support_features)
        support_labels = torch.Tensor(support_labels)
        prototypes = compute_prototypes(support_features, support_labels)
        self.prototypes = prototypes.to('cuda')

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of support and query images
        z_query = self.backbone.forward(query_images)

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, self.prototypes)

        # Use it to compute classification scores
        # FEW SHOT ALWAYS LOOK FOR THE MAX, AND SINCE DISTANCES ARE ALWAYS 
        # POSITIVE, WE NEGATE THE RESULTS IN ORDER TO LOOK FOR THE MAX
        scores = -dists 

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False

class PrototypicalNetworks_SingleClass(FewShot):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def __init__(self, *args, **kwargs):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)
        self.mean = None
        self.std = None

    #     if len(self.backbone_output_shape) != 1:
    #         raise ValueError(
    #             "Illegal backbone for Prototypical Networks. "
    #             "Expected output for an image is a 1-dim tensor."
    #         )

    def process_support_set(
        self,
        support_images: List
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        support_features = []
        self.backbone.eval()
        # get feature maps from the images
        with torch.no_grad():
            for img in support_images:
                t_temp = self.backbone.forward(img.unsqueeze(dim=0).to('cuda'))
                support_features.append(t_temp.squeeze().cpu())
        self.backbone.train()
        
        # get prototypes and save them into cuda memory
        support_features = torch.stack(support_features)
        prototypes = compute_prototypes_singleclass(support_features)
        self.prototypes = prototypes.unsqueeze(dim=0).to('cuda')

    def forward_groundtruth(
        self,
        query_images: Tensor,
    ) -> Tensor:
        scores = []
        with torch.no_grad():
            for img in query_images:
                score = self.forward(img.unsqueeze(dim=0).to('cuda'))
                scores.append(score.cpu().item())

        self.mean = statistics.mean(scores)
        self.std = statistics.stdev(scores)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of support and query images
        z_query = self.backbone.forward(query_images)

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, self.prototypes)

        # Use it to compute classification scores
        # FEW SHOT ALWAYS LOOK FOR THE MAX, AND SINCE DISTANCES ARE ALWAYS 
        # POSITIVE, WE NEGATE THE RESULTS IN ORDER TO LOOK FOR THE MAX
        scores = -dists 

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
