from torch import nn 
import torch
import timm
import copy
import torch.nn as nn

class Timm_head_names:
    RESNET18 = 'resnet18'
    RESNETV2_50 = 'resnetv2_50'
    SWINV2_BASE_WINDOW8_256 = 'swinv2_base_window8_256.ms_in1k'
    RESNETRS_420 = "resnetrs420"
    RESTNET10 = "resnet10t.c3_in1k"
    ViT = "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"


class Identity(nn.Module):
    """ Identity to remove one layer """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class MyBackbone(nn.Module):

    def __init__(self, model_name, pretrained, num_c, use_fc=False, freeze_all=True):
        super(MyBackbone,self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        temp_input_size = 32

        # different types of head
        if model_name == Timm_head_names.RESNET18 or \
            model_name == Timm_head_names.RESNETRS_420 or \
            model_name == Timm_head_names.RESTNET10:
            self.backbone.fc = Identity()
        elif model_name == Timm_head_names.RESNETV2_50:
            self.backbone.head.fc = Identity()
        elif model_name == Timm_head_names.SWINV2_BASE_WINDOW8_256:
            #(norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            #(head): Linear(in_features=1024, out_features=1000, bias=True)
            self.backbone.head.fc = Identity()
            self.backbone.head.flatten = Identity()
            model_name=model_name.split('.')[0]
            temp_input_size = int(model_name.split('_')[-1])
        elif model_name == Timm_head_names.ViT:
            self.backbone.head = Identity()
            model_name=model_name.split('.')[0]
            temp_input_size = int(model_name.split('_')[-1])

        # if freeze_all:
        #     for param in self.backbone.parameters():
        #         param.requires_grad=False 

        # get the proper size of feature maps
        features_size = self.compute_backbone_output_shape(temp_input_size)
        if len(features_size) != 1:
            raise ValueError(
                "Illegal backbone for Prototypical Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

        # final fc layer
        self.fc_final = nn.Linear(in_features=features_size[0], out_features=num_c)
        self.use_fc=use_fc

    def compute_backbone_output_shape(self, input_size=32):
        """ Compute the dimension of the feature space defined by a feature extractor.
        Params
        :backbone (nn.Module) -> feature extractor
        Returns
        :shape (int) -> shape of the feature vector computed by the feature extractor for an instance
        """
        input_images = torch.ones((4, 3, input_size, input_size))
        # Use a copy of the backbone on CPU, to avoid device conflict
        output = copy.deepcopy(self.backbone).cpu()(input_images)

        return tuple(output.shape[1:])

    def forward(self, x):
        """ Conditional forwarding (useful when we like the final feature maps) """
        x = self.backbone(x)
        if self.use_fc:
            x = self.fc_final(x)
        return x

    #----------------------
    @property
    def use_fc(self):
        return self._use_fc 

    @use_fc.setter
    def use_fc(self,value):
        self._use_fc=value
    #----------------------

# class MyBackbone_transformer(nn.Module):

#     def __init__(self, model_name, pretrained, num_c, use_fc=False, freeze_all=True):
#         super(MyBackbone_resnet,self).__init__()
#         self.backbone = timm.create_model(model_name, pretrained=pretrained)

#         # different types of head
#         if model_name == Timm_head_names.RESNET18:
#             self.backbone.fc = Identity()
#         elif model_name == Timm_head_names.RESNETV2_50:
#             self.backbone.head.fc = Identity()

#         if freeze_all:
#             for param in self.backbone.parameters():
#                 param.requires_grad=False 

#         # get the proper size of feature maps
#         features_size = self.compute_backbone_output_shape(self.backbone)

#         # final fc layer
#         self.fc_final = nn.Linear(in_features=features_size[0], out_features=num_c)
#         self.use_fc=use_fc

#     def compute_backbone_output_shape(self, backbone):
#         """ Compute the dimension of the feature space defined by a feature extractor.
#         Params
#         :backbone (nn.Module) -> feature extractor
#         Returns
#         :shape (int) -> shape of the feature vector computed by the feature extractor for an instance
#         """
#         input_images = torch.ones((4, 3, 32, 32))
#         # Use a copy of the backbone on CPU, to avoid device conflict
#         output = copy.deepcopy(backbone).cpu()(input_images)

#         return tuple(output.shape[1:])

#     def forward(self, x):
#         """ Conditional forwarding (useful when we like the final feature maps) """
#         x = self.backbone(x)
#         if self.use_fc:
#             x = self.fc_final(x)
#         return x

#     #----------------------
#     @property
#     def use_fc(self):
#         return self._use_fc 

#     @use_fc.setter
#     def use_fc(self,value):
#         self._use_fc=value
#     #----------------------