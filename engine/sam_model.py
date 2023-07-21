from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry

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

    def get_simple_mask(self):
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
        return mask_generator_


        