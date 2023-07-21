#!/usr/bin/env python
import warnings
import torch
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
from data import create_loader, create_dataset
from utils import get_parameters, seed_everything, throttle_cpu
from engine import SAM
from pycocotools.coco import COCO
from eval import *
#------------------------------------------------------------------------------------------------


def create_datasets_and_loaders(args):
    """ Setup datasets, transforms, loaders, evaluator.
    Params
    :args: Model specific configuration dict / struct

    Returns
    :Train loader, validation loader
    """
    datasets = create_dataset(
        args.dataset, args.root, use_semi_split=False,
        semi_percentage=args.semi_percentage
    )
    dataset_train = datasets[0]
    dataset_eval = datasets[1]

    loader_eval = create_loader(
        dataset_eval,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_val,
        is_training=False,
    )
    return loader_eval

def run_experiment(args):
    """ Run simple experiment """
    path_output = f"./output/{args.output_folder}/{args.semi_percentage}/{args.run_name}"

    # STEP 1: create data loaders
    loader_eval = create_datasets_and_loaders(args)
    
    # STEP 2: run inferences using SAM
    sam = SAM(args)
    mask_gen = sam.get_simple_mask()

    # STEP 3: get segments as the inferences
    filepath = f"weights/bbox_results.json"
    MAX_IMAGES = 10000
    VAL_IMGS = str(loader_eval.dataset.data_dir)
    coco_gt = COCO(f"{str(loader_eval.dataset.data_dir.parent)}/annotations/instances_val2017.json")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    save_inferences_simple(
        mask_gen, image_ids, coco_gt, VAL_IMGS, filepath
    )
    eval_sam(coco_gt, image_ids, filepath)

if __name__ == '__main__':
    args = get_parameters()
    throttle_cpu(args.numa)
    if not args.seed == None:
        seed_everything(args.seed)
    run_experiment(args)