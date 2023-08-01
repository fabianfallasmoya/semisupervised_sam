import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision
from data import transforms_toNumpy
from pycocotools.cocoeval import COCOeval

from data.transforms import Transform_Normalization

def save_inferences_simple(
    sam_model, unlabeled_loader, filepath
    ):
    results = []
    trans_norm = Transform_Normalization(
        size=33, force_resize=False, keep_aspect_ratio=True
    )
    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    for (_,batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get foreground samples (from sam predictions)
            imgs_s, box_coords, scores = sam_model.get_unlabeled_sam(
                batch, idx, trans_norm
            )
            # accumulate SAM info (inferences)
            # no need to store the imgs, just the rest of the information
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    for idx_,_ in enumerate(imgs_ids):
        image_result = {
            'image_id': imgs_ids[idx_],
            'category_id': 1,
            'score': imgs_scores[idx_],
            'bbox': imgs_box_coords[idx_],
        }
        results.append(image_result)
        
    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

def save_inferences_singleclass(
        fs_model, unlabeled_loader,
        sam_model, filepath,
        trans_norm
    ):
    results = []
    fs_model.backbone.use_fc = False
    lower = fs_model.mean - (1*fs_model.std)
    upper = fs_model.mean + (1*fs_model.std)

    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    unlabeled_imgs = []
    for (_, batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get foreground samples (from sam predictions)
            imgs_s, box_coords, scores = sam_model.get_unlabeled_sam(
                batch, idx, trans_norm
            )
            # accumulate SAM info (inferences)
            unlabeled_imgs += imgs_s
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    for idx_,sample in enumerate(unlabeled_imgs):
        with torch.no_grad():
            sample = sample.unsqueeze(dim=0).to('cuda')
            res = fs_model(sample).cpu().item()

        if  res < upper and res > lower:
            image_result = {
                'image_id': imgs_ids[idx_],
                'category_id': 1,
                'score': imgs_scores[idx_],
                'bbox': imgs_box_coords[idx_],
            }
            results.append(image_result)
        
    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

def save_inferences_twoclasses(
    fs_model, unlabeled_loader, sam_model, filepath,
    trans_norm
    ):
    results = []
    fs_model.backbone.use_fc = False

    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    unlabeled_imgs = []
    for (_,batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get foreground samples (from sam predictions)
            imgs_s, box_coords, scores = sam_model.get_unlabeled_sam(
                batch, idx, trans_norm
            )
            # accumulate SAM info (inferences)
            unlabeled_imgs += imgs_s
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    pineapples = 0
    for idx_,sample in enumerate(unlabeled_imgs):
        with torch.no_grad():
            sample = sample.unsqueeze(dim=0).to('cuda')
            res = fs_model(sample)
            label = torch.max(res.detach().data, 1)[1].item() #keep index

        # if class is zero! Currently, this does not work with multi-class
        if label == 0: # not background
            pineapples += 1
            image_result = {
                'image_id': imgs_ids[idx_],
                'category_id': 1,
                'score': imgs_scores[idx_],
                'bbox': imgs_box_coords[idx_],
            }
            results.append(image_result)

    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)


def eval_sam(coco_gt, image_ids, pred_json_path, output_root, method="xyz"):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # write results into a file
    with open(f"{output_root}/mAP_{method}.txt", 'w') as file:
        for i in coco_eval.stats:
            file.write(f"{str(i)}\n")
    