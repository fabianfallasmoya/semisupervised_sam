from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
import numpy as np
import torch
import torchvision
from PIL import Image


class SAM:

    def __init__(self, args) -> None:
        if args.sam_model == 'b':
            self.checkpoint = "weights/sam_vit_b_01ec64.pth"
            self.model_type = "vit_b"
        elif args.sam_model == 'h':
            self.checkpoint = "weights/sam_vit_h_4b8939.pth"
            self.model_type = "vit_h"
        else:
            RuntimeError("No sam config found")
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to('cuda')
        self.mask_generator = None

    def load_simple_mask(self):
        #There are several tunable parameters in automatic mask generation that control 
        # how densely points are sampled and what the thresholds are for removing low 
        # quality or duplicate masks. Additionally, generation can be automatically 
        # run on crops of the image to get improved performance on smaller objects, 
        # and post-processing can remove stray pixels and holes. 
        # Here is an example configuration that samples more masks:
        #https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35    

        #Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
        # and 0.92 and 0.96 for score_thresh

        mask_generator_ = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            # pred_iou_thresh=0.9,
            # stability_score_thresh=0.96,
            # crop_n_layers=1, default:0
            # crop_n_points_downscale_factor=1,default:1
            min_mask_region_area=100,  # Requires open-cv to run post-processing
            output_mode="coco_rle",
        )
        self.mask_generator = mask_generator_

    def get_unlabeled_sam(self, batch, idx, transform):
        """ From a batch and its index get samples 
        Params
        :batch (<tensor, >)
        """
        imgs = []
        box_coords = []
        scores = []

        # batch[0] has the images    
        img = batch[0][idx].cpu().numpy().transpose(1,2,0)
        img_pil = Image.fromarray(img)

        # run sam to create proposals
        masks = self.mask_generator.generate(img)

        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), in_fmt='xywh', out_fmt='xyxy'
            )
            # get img
            crop = img_pil.crop(np.array(xyxy))  
            sample = transform.preprocess(crop)

            # accumulate
            imgs.append(sample)
            box_coords.append(xywh)
            scores.append(float(ann['predicted_iou']))
        return imgs, box_coords, scores


        