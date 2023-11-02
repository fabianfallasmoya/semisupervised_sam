import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision
from data import transforms_toNumpy
from pycocotools.cocoeval import COCOeval

from data.transforms import Transform_To_Models

def save_inferences_simple(
    sam_model, unlabeled_loader, 
    filepath, use_sam_embeddings
    ):
    results = []
    trans_norm = Transform_To_Models(
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
            imgs_s, box_coords, scores = sam_model.get_unlabeled_samples(
                batch, idx, trans_norm, use_sam_embeddings
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
        trans_norm,
        use_sam_embeddings
    ):
    fs_model.backbone.use_fc = False
    
    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    unlabeled_imgs = []

    # collect all inferences from SAM
    for (_, batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get foreground samples (from sam predictions)
            imgs_s, box_coords, scores = sam_model.get_unlabeled_samples(
                batch, idx, trans_norm, use_sam_embeddings
            )
            # accumulate SAM info (inferences)
            unlabeled_imgs += imgs_s
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    # store std for 1 and for 2 and 3
    # for idx_1 in range(1,4):
    idx_1 = 2
    idx_float = float(idx_1)
    lower = fs_model.mean - (idx_float*fs_model.std)
    upper = fs_model.mean + (idx_float*fs_model.std)

    results = []
    for idx_,sample in enumerate(unlabeled_imgs):
        distance = fs_model(sample).cpu().item()

        # distance is inside the first std from the mean
        if  distance <= upper and distance >= lower:
            image_result = {
                'image_id': imgs_ids[idx_],
                'category_id': 1,
                'score': imgs_scores[idx_],
                'bbox': imgs_box_coords[idx_],
            }
            results.append(image_result)
        
    if len(results) > 0:
        # write output
        f_ = f"{filepath}/bbox_results_std{idx_1}.json"
        if os.path.exists(f_):
            os.remove(f_)
        json.dump(results, open(f_, 'w'), indent=4)

def save_inferences_twoclasses(
    fs_model, unlabeled_loader, sam_model, 
    filepath, trans_norm, use_sam_embeddings
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
            imgs_s, box_coords, scores = sam_model.get_unlabeled_samples(
                batch, idx, trans_norm, use_sam_embeddings
            )
            # accumulate SAM info (inferences)
            unlabeled_imgs += imgs_s
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    pineapples = 0
    for idx_,sample in enumerate(unlabeled_imgs):
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


def eval_sam(coco_gt, image_ids, pred_json_path, output_root, method="xyz", number=None):
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
    if number is None:
        file_name = f"{output_root}/mAP_{method}.txt"
    else:
        file_name = f"{output_root}/mAP_{method}_std{number}.txt"
    with open(file_name, 'w') as file:
        for i in coco_eval.stats:
            file.write(f"{str(i)}\n")
    