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
from data import create_loader, create_dataset, create_dataset_ood
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
    datasets = create_dataset_ood(
        args.dataset, args.root, 
        labeled_samples=1,
        unlabeled_samples=100
    )
    dataset_train_labeled = datasets[0]
    dataset_train_unlabeled = datasets[1]

    # create data loaders
    loader_labeled = create_loader(
        dataset_train_labeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_labeled,
        is_training=False
    )

    loader_unlabeled = create_loader(
        dataset_train_unlabeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_unlabeled,
        is_training=False,
    )
    return loader_labeled, loader_unlabeled

def run_experiment(args):
    """ Run simple experiment """
    path_output = f"./output/{args.output_folder}/{args.semi_percentage}/{args.run_name}"

    # STEP 1: create data loaders
    loader_eval = create_datasets_and_loaders(args)

    # STEP 2: run ood filter
    
    
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