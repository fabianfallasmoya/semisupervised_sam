import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision
from data import Transforms_fabian
from pycocotools.cocoeval import COCOeval

def save_inferences(
    fs_model, mask_gen, image_ids, coco_gt, img_path, 
    sample_resolution, num_classes, filepath
    ):
    results = []
    trans = Transforms_fabian(sample_resolution)
    fs_model.backbone.use_fc = False

    temp = 1
    for image_id in tqdm(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = f"{img_path}/{image_info['file_name']}"
        img_pil = Image.open(image_path)
        img_draw = ImageDraw.Draw(img_pil)
        img = np.array(img_pil)

        # run sam to create proposals
        masks = mask_gen.generate(img)

        # predict the class using the few-shot model
        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), 'xywh', 'xyxy'
            )

            # get img
            crop = img_pil.crop(np.array(xyxy))  
            sample = trans.preprocess(crop)
            sample = sample.unsqueeze(dim=0).to('cuda')

            res = fs_model(sample)
            # torch.max (x, 1) -> (a,b) where a:score, b:index
            label = torch.max(res.detach().data, 1)[1] #keep index
            if label <= (num_classes - 1):
                image_result = {
                    'image_id': image_id,
                    'category_id': label.item() + 1,
                    'score': float(ann['predicted_iou']),
                    'bbox': xywh,
                }
                results.append(image_result)
                #---------- see image
                                
                x1,y1,x2,y2 = xyxy
                shape = [(x1, y1), (x2, y2)]
                img_draw.rectangle(shape, outline="red")
        img_pil.save(f"w{temp}.png")
        temp += 1
        
    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

def save_inferences_singleclass(
    fs_model, mask_gen, image_ids, coco_gt, img_path, 
    sample_resolution, num_classes, filepath
    ):
    results = []
    trans = Transforms_fabian(sample_resolution)
    fs_model.backbone.use_fc = False
    lower = fs_model.mean - (1*fs_model.std)
    upper = fs_model.mean + (1*fs_model.std)

    temp = 1
    for image_id in tqdm(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = f"{img_path}/{image_info['file_name']}"
        img_pil = Image.open(image_path)
        img_draw = ImageDraw.Draw(img_pil)
        img = np.array(img_pil)

        # run sam to create proposals
        masks = mask_gen.generate(img)

        # predict the class using the few-shot model
        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), 'xywh', 'xyxy'
            )

            # get img
            crop = img_pil.crop(np.array(xyxy))  
            sample = trans.preprocess(crop)
            sample = sample.unsqueeze(dim=0).to('cuda')

            with torch.no_grad():
                res = fs_model(sample)

            if  res < upper and res > lower:
                image_result = {
                    'image_id': image_id,
                    'category_id': 1,
                    'score': float(ann['predicted_iou']),
                    'bbox': xywh,
                }
                results.append(image_result)
                #---------- see image
                                
                x1,y1,x2,y2 = xyxy
                shape = [(x1, y1), (x2, y2)]
                img_draw.rectangle(shape, outline="red")
        img_pil.save(f"w{temp}.png")
        temp += 1
        
    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

def save_inferences_simple(
    mask_gen, image_ids, coco_gt, img_path, filepath
    ):
    results = []

    temp = 1
    for image_id in tqdm(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = f"{img_path}/{image_info['file_name']}"
        img_pil = Image.open(image_path)
        img = np.array(img_pil)
        img_draw = ImageDraw.Draw(img_pil)

        # run sam to create proposals
        masks = mask_gen.generate(img)

        # every segment is an instance
        for ann in masks:
            xywh = ann['bbox']
            
            image_result = {
                'image_id': image_id,
                'category_id': 1, # assuming that there is only a single class
                'score': float(ann['predicted_iou']),
                'bbox': xywh,
            }
            results.append(image_result)

            #---------- see image
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), 'xywh', 'xyxy'
            )
            
            x1,y1,x2,y2 = xyxy
            shape = [(x1, y1), (x2, y2)]
            img_draw.rectangle(shape, outline="red")
        img_pil.save(f"z{temp}.png")
        temp += 1

    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

def eval_sam(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print()