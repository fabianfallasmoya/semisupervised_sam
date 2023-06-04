from random import sample
import torch
from PIL import Image
from .transforms import Transforms_fabian
import numpy as np
import cv2
from metrics import intersection_over_union, intersection
import random

def get_batch_prototypes(
    dataloader_fewshot, sample_resolution:int=None, num_classes:float=None, img_resolution:int=None):
    """ Get the images that will be used to calculate the prototypes.

    Params:
    :None -> but it uses the dataloader to load labeled images.
    Return
    :imgs (tensor) -> all images from the bounding boxes.
    :labels (tensor) -> labels of the bounding boxes.
    """
    trans = Transforms_fabian(sample_resolution)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    imgs = []
    labels = []
    
    # ITERATE: BATCH
    count_imgs = 0
    for batch in dataloader_fewshot:
        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get background samples
            imgs_b, labels_b = get_background(batch, idx, trans, ss, num_classes, img_resolution)
            # get foreground samples
            imgs_f, labels_f = get_foreground(batch, idx, trans)

            # accumulate
            imgs += imgs_b + imgs_f
            labels += labels_b + labels_f
            count_imgs += 1
    return imgs, labels

def get_foreground(batch, idx, transform):
    """ From a batch and its index get samples """
    imgs = []
    labels = []
    # batch[0] has the images    
    image = batch[0][idx].cpu().numpy().transpose(1,2,0)
    img_pil = Image.fromarray(image)

    # batch[1] has the metadata and bboxes
    # tensor dim where the boxes are, is: [100x4]
    # where no box is present, the row is [-1,-1,-1,-1]. So, get all boxes and classes as:
    bbox_indx = (batch[1]['bbox'][idx].sum(axis=1)>0).nonzero(as_tuple=True)[0].cpu()
    boxes = batch[1]['bbox'][idx][bbox_indx].cpu()
    classes = batch[1]['cls'][idx][bbox_indx].cpu() 

    # ITERATE: BBOX
    for idx_bbox, bbox in enumerate(boxes):
        #FOR SOME REASON, THE COORDINATES COMES:
        # [y1,x1,y2,x2] and need to be translated to: [x1,y1,x2,y2]
        bbox = bbox[[1,0,3,2]]
        bbox = bbox.numpy()

        # get img
        crop = img_pil.crop(bbox)  
        new_sample = transform.preprocess(crop)
        labels.append(classes[idx_bbox].item())
        imgs.append(new_sample)
    return imgs, labels

def get_background(batch, idx, transform, selective_search, num_classes, img_resolution):
    """ From a batch and its index get samples """
    imgs = []
    labels = []
    ious = torch.Tensor([])
    areas = torch.Tensor([])
    
    # 1. GET bbox proposals from selective search
    image = batch[0][idx].cpu().numpy().transpose(1,2,0)
    img_pil = Image.fromarray(image)
    orig_width, orig_height = batch[1]['img_size'][idx].tolist()

    # the image comes with padding, this code removes the padding
    # using the original resolution
    if orig_width > orig_height:
        new_width = img_resolution
        new_height = (orig_height * img_resolution) / orig_width
    else:
        new_height = img_resolution
        new_width = (orig_width * img_resolution) / orig_height
    img_pil = img_pil.crop([0,0,new_width,new_height])  
    open_cv_image = np.array(img_pil)

    # Convert RGB to BGR and apply selective search
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    selective_search.setBaseImage(open_cv_image)
    selective_search.switchToSelectiveSearchFast()
    proposals = selective_search.process()

    # 2. CALCULATE ious (permutation) ground truth vs proposals
    #  from numpy to list
    #proposals = [list(i) for i in list(proposals)]
    # from [x,y,w,h] to [x1,y1,x2,y2]
    proposals[:,[2]] = proposals[:,[0]] + proposals[:,[2]]
    proposals[:,[3]] = proposals[:,[1]] + proposals[:,[3]]
    proposals = torch.from_numpy(proposals)

    #  batch[1] has the metadata and bboxes
    #  tensor dim where the boxes are, is: [100x4]
    #  where no box is present, the row is [-1,-1,-1,-1]. So, get all boxes and classes as:
    bbox_indx = (batch[1]['bbox'][idx].sum(axis=1)>0).nonzero(as_tuple=True)[0].cpu()
    ground_truth = batch[1]['bbox'][idx][bbox_indx].cpu()
    classes = batch[1]['cls'][idx][bbox_indx].cpu()

    # iterate: bbox
    for gt in ground_truth:
        # FOR SOME REASON, THE COORDINATES COMES:
        # [y1,x1,y2,x2] and need to be translated to: [x1,y1,x2,y2]
        gt_new = gt[[1,0,3,2]]
        gt_repeated = gt_new.repeat(len(proposals),1)

        # get just the intersection w.r.t. ground truth
        # also, get the are of the proposed box
        # (a, b) = intersection(proposals[830], gt_new)
        (res, areas) = intersection(proposals, gt_repeated)
        if ious.numel() > 0:
            ious = torch.cat((ious,res), dim=1)
        else:
            ious = res

    # 3. CHOOSE the winner proposals
    for idx_ground in list(range(ious.shape[1])):
        coords = []
        proposal_areas = []
        labels_temp = []

        # get indices that have small intersection (less than 30%) with respect current gt
        main_indices = (ious[:,idx_ground] < 0.3) # & (ious[:,idx] > 0.05)
        main_indices = main_indices.nonzero().squeeze().tolist()

        # get submatrix excluding current column
        rest_cols = list(range(ious.shape[1]))
        del rest_cols[idx_ground]

        # idea: get proposals with low iou with respect other gt
        if len(rest_cols) > 0:
            # keep only the proposals that also have small iou with respect others
            small_set = ious[main_indices,:]
            small_set = small_set[:, rest_cols]
            
            # get 
            finalist = ((small_set < 0.005).sum(dim=1) > 1).nonzero()
            final_indxs = [main_indices[i] for i in finalist.squeeze().tolist()]
        else:
            final_indxs = main_indices#[i.item() for i in main_indices]

        # get areas, coordinatates and labels
        proposal_areas += [areas[i].item() for i in final_indxs]
        coords += [proposals[i].tolist() for i in final_indxs]
        labels_temp += [classes[idx_ground].item() + num_classes] * len(final_indxs)
        if len(coords) < 1:
            continue

        # keep proposals with high areas: sort them, then random sample of n
        # ration of 1:n (one foreground and n background)
        results = list(zip(proposal_areas, coords, labels_temp))
        results.sort(key = lambda x: x[0], reverse=True)
        proposal_areas, coords, labels_temp = zip(*results)
        proposal_areas = proposal_areas[0:int(len(proposal_areas)*.2)]
        coords = coords[0:int(len(coords)*.2)] #
        labels_temp = labels_temp[0:int(len(labels_temp)*.2)]
        if len(labels_temp) > 1:
            indices = random.sample(range(0, len(labels_temp)), k=1)
            coords = [coords[i] for i in indices]
            labels_temp = [labels_temp[i] for i in indices]

        # 4. GET images as tensors
        labels += labels_temp
        image = batch[0][idx].cpu().numpy().transpose(1,2,0)
        img_pil = Image.fromarray(image)
        for (x1,y1,x2,y2) in coords:
            crop = img_pil.crop([x1,y1,x2,y2])  
            new_sample = transform.preprocess(crop)
            imgs.append(new_sample)
    return imgs, labels

