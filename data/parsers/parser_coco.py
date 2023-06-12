""" COCO dataset parser
Copyright 2020 Ross Wightman
"""
import numpy as np
#from pycocotools.coco import COCO
from .fabian_COCO import COCO
from .parser import Parser
from .parser_config import CocoParserCfg
from PIL import Image


class CocoParser(Parser):

    def __init__(self, cfg: CocoParserCfg):
        super().__init__(
            bbox_yxyx=cfg.bbox_yxyx,
            has_labels=cfg.has_labels,
            include_masks=cfg.include_masks,
            include_bboxes_ignore=cfg.include_bboxes_ignore,
            ignore_empty_gt=cfg.has_labels and cfg.ignore_empty_gt,
            min_img_size=cfg.min_img_size
        )
        self.cat_ids_as_labels = True  # this is the default for original TF EfficientDet models
        self.coco = None
       
        self._load_annotations (cfg.ann_filename, cfg.json_dict)

    def get_ann_info(self, idx, verbose=False):        
        img_id = self.img_ids[idx]
        if self.include_masks:
            return self._parse_img_ann_segm(img_id, verbose=verbose)
        else:
            return self._parse_img_ann(img_id, verbose=verbose)

    def _load_annotations(self, ann_file, json_):
        assert self.coco is None
        self.coco = COCO(annotation_file=ann_file, json_dict=json_)
        self.cat_ids = self.coco.getCatIds()
        self.cat_names = [c['name'] for c in self.coco.loadCats(ids=self.cat_ids)]
        if not self.cat_ids_as_labels:
            self.cat_id_to_label = {cat_id: i + self.label_offset for i, cat_id in enumerate(self.cat_ids)}
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.loadImgs([img_id])[0]
            if (min(info['width'], info['height']) < self.min_img_size or
                    (self.ignore_empty_gt and img_id not in img_ids_with_ann)):
                self.img_ids_invalid.append(img_id)
                continue
            self.img_ids.append(img_id)
            self.img_infos.append(info)

    def _parse_img_ann_segm(self, img_id, verbose=False):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        masks = []
        masks_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            mask_file_path = ann['mask_file_path']
            mask = np.array(Image.open(mask_file_path))
            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    masks_ignore.append(mask)
            else:
                masks.append(mask)
        if masks:
            masks = np.array(masks, ndmin=2, dtype=np.float32)
        else:
            masks = np.zeros((0, 4), dtype=np.float32)  # TODO: fixthis

        if self.include_bboxes_ignore:
            if masks_ignore:
                masks_ignore = np.array(masks_ignore, ndmin=2, dtype=np.float32)
            else:
                masks_ignore = np.zeros((0, 4), dtype=np.float32)  # TODO: fixthis
        
        ann = dict(mask=masks)

        if self.include_bboxes_ignore:
            ann['mask_ignore'] = masks_ignore

        return ann
                
                
    def _parse_img_ann(self, img_id, verbose=False):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        full_json_annotation = []
        cls = []
        
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            if self.yxyx:
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_id_to_label[ann['category_id']] if self.cat_id_to_label else ann['category_id'])
            full_json_annotation.append(ann)
        if bboxes:
            bboxes = np.array(bboxes, ndmin=2, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if verbose:
            ann = dict(bbox=bboxes, cls=cls, full_ann_json=full_json_annotation)
            # print("\n******* COCO PARSER **********")
            # print("\tTarget Boxes: ", bboxes)
            # print("\tFull ann json: ", ann_info)
            # print("*****************\n")
        else:
            ann = dict(bbox=bboxes, cls=cls)

        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann
