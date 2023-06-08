import copy
from typing import Tuple
import torch
from torch import nn, Tensor

# def compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
#     """
#     Compute the dimension of the feature space defined by a feature extractor.
#     Args:
#         backbone: feature extractor

#     Returns:
#         shape of the feature vector computed by the feature extractor for an instance

#     """
#     input_images = torch.ones((4, 3, 32, 32))
#     # Use a copy of the backbone on CPU, to avoid device conflict
#     output = copy.deepcopy(backbone).cpu()(input_images)
#     return tuple(output.shape[1:])


def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """
    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )