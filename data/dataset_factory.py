""" Dataset factory
Copyright 2020 Ross Wightman
"""
import os
import json
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

from .dataset_config import *
from .parsers import *
from .dataset import DetectionDatset
from .parsers import create_parser

from sklearn.model_selection import train_test_split


def create_dataset(name, root, splits=('train', 'val'), use_semi_split=False, 
                   seed=42, semi_percentage=1.0, verbose=False):
    
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    dataset_cls = DetectionDatset
    datasets = OrderedDict()
    if name.startswith('coco'):
        if 'coco2014' in name:
            dataset_cfg = Coco2014Cfg()
        elif 'cocobear' in name:
            dataset_cfg = CocoBearCfg()
        else:
            dataset_cfg = Coco2017Cfg()

        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']

            if use_semi_split and s.startswith('train'):
                #--------------------------------------------
                # open file if semi to split the labeled dict
                with open(ann_file, 'r') as f:
                    new_labeled = json.load(f)
                with open(ann_file, 'r') as f2:
                    new_unlabeled = json.load(f2)
                # get all image ids 
                ids = [item['id'] for item in new_labeled['images']]

                # split the ids
                y_dumpy = np.zeros(len(ids))
                labeled_idx, _, _, _ = train_test_split(
                                        ids, y_dumpy, 
                                        train_size = semi_percentage / 100.0, 
                                        shuffle = True, 
                                        random_state = seed
                                        )
                # keep just the necessary ids
                new_labeled['images'] = [i for i in new_labeled['images'] if i['id'] in labeled_idx]
                new_labeled['annotations'] = [i for i in new_labeled['annotations'] if i['image_id'] in labeled_idx]
                
                new_unlabeled['images'] = [i for i in new_unlabeled['images'] if i['id'] not in labeled_idx]
                new_unlabeled['annotations'] = [i for i in new_unlabeled['annotations'] if i['image_id'] not in labeled_idx]
                #--------------------------------------------

                # labeled dataset
                parser_cfg_labeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_labeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_labeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_labeled),
                    verbose=verbose
                )

                # unlabeled dataset
                parser_cfg_unlabeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_unlabeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_unlabeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_unlabeled),
                    verbose=verbose
                )
                
            else:
                parser_cfg = CocoParserCfg(
                    ann_filename=ann_file,
                    has_labels=split_cfg['has_labels']
                )
                
                datasets[s] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
                    verbose=verbose
                )
    elif name.startswith('voc'):
        if 'voc0712' in name:
            dataset_cfg = Voc0712Cfg()
        elif 'voc2007' in name:
            dataset_cfg = Voc2007Cfg()
        else:
            dataset_cfg = Voc2012Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            if isinstance(split_cfg['split_filename'], (tuple, list)):
                assert len(split_cfg['split_filename']) == len(split_cfg['ann_filename'])
                parser = None
                for sf, af, id in zip(
                        split_cfg['split_filename'], split_cfg['ann_filename'], split_cfg['img_dir']):
                    parser_cfg = VocParserCfg(
                        split_filename=root / sf,
                        ann_filename=os.path.join(root, af),
                        img_filename=os.path.join(id, dataset_cfg.img_filename))
                    if parser is None:
                        parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                    else:
                        other_parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                        parser.merge(other=other_parser)
            else:
                parser_cfg = VocParserCfg(
                    split_filename=root / split_cfg['split_filename'],
                    ann_filename=os.path.join(root, split_cfg['ann_filename']),
                    img_filename=os.path.join(split_cfg['img_dir'], dataset_cfg.img_filename),
                )
                parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
            datasets[s] = dataset_cls(data_dir=root, parser=parser)
    elif name.startswith('openimages'):
        if 'challenge2019' in name:
            dataset_cfg = OpenImagesObjChallenge2019Cfg()
        else:
            dataset_cfg = OpenImagesObjV5Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            parser_cfg = OpenImagesParserCfg(
                categories_filename=root / dataset_cfg.categories_map,
                img_info_filename=root / split_cfg['img_info'],
                bbox_filename=root / split_cfg['ann_bbox'],
                img_label_filename=root / split_cfg['ann_img_label'],
                img_filename=dataset_cfg.img_filename,
                prefix_levels=split_cfg['prefix_levels'],
                has_labels=split_cfg['has_labels'],
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg)
            )
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
