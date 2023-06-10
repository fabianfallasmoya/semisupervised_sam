#!/usr/bin/env python
import warnings
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from engine.my_backbones import MyBackbone
from engine.prototypical_networks import PrototypicalNetworks
from utils import Constants_AugMethod, add_bool_arg
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
import timm.utils as utils
from timm.models import resume_checkpoint, load_checkpoint
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
try:
    # new timm import
    from timm.layers import set_layer_config, convert_sync_batchnorm
except ImportError:
    # support several versions back in timm
    from timm.models.layers import set_layer_config
    try:
        # timm has its own convert_sync_bn to support BatchNormAct layers in updated models
        from timm.models.layers import convert_sync_batchnorm
    except ImportError:
        warnings.warn(
            "Falling back to torch.nn.SyncBatchNorm, does not properly work with timm models in new versions.")
        convert_sync_batchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm
from timm.data.transforms_factory import create_transform 
from data import create_loader, create_dataset, create_parser, \
    DetectionDatset, SkipSubset, resolve_input_config
from utils import add_bool_arg, get_parameters, seed_everything, throttle_cpu
from data.fewshot_data import get_batch_prototypes
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
        args.dataset, args.root, use_semi_split=args.use_semi_split,
        semi_percentage=args.semi_percentage
    )
    dataset_train_labeled = datasets[0]
    dataset_train_unlabeled = datasets[1]
    dataset_eval = datasets[2]

    # setup labeler in loader/collate_fn if not enabled in the model bench
    # labeler = None
    # if not args.bench_labeler:
    #     labeler = AnchorLabeler(
    #         Anchors.from_config(args), args.num_classes, match_threshold=0.5)
    loader_labeled = create_loader(
        dataset_train_labeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=args.reprob,
    )

    # if args.val_skip > 1:
    #     dataset_eval = SkipSubset(dataset_eval, args.val_skip)
    loader_eval = create_loader(
        dataset_eval,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_val,
        is_training=False,
    )
    return loader_labeled, loader_eval

def run_experiment(args):
    """ Run simple experiment """
    path_output = f"./output/{args.output_folder}/{args.semi_percentage}/{args.run_name}"

    # STEP 1: create data loaders
    loader_labeled, loader_eval = create_datasets_and_loaders(args)
    
    # STEP 2: get the raw support set
    imgs, labels = get_batch_prototypes(
        loader_labeled, args.new_sample_size, args.num_classes, args.img_resolution
    )
    # labels start from index 1 to n, translate to start from 0 to n.
    labels = [i-1 for i in labels]
    # to tensors
    imgs = torch.stack(imgs).to('cuda')
    labels = torch.Tensor(labels).to('cuda')
    # one additional class for every class
    num_classes = args.num_classes * 2

    # STEP 3: create few-shot model 
    backbone = MyBackbone(
        args.timm_model, args.load_pretrained, args.num_classes
    )
    fs_model = PrototypicalNetworks(backbone, use_softmax=False).to('cuda')
    #  create the prototypes
    fs_model.process_support_set(imgs, labels)

    # STEP 4: run inferences using SAM
    sam = SAM(args)
    mask_gen = sam.get_simple_mask()

    # STEP 5: classify these inferences using the few-shot model
    filepath = f"weights/bbox_results.json"
    MAX_IMAGES = 10000
    VAL_IMGS = str(loader_eval.dataset.data_dir)
    coco_gt = COCO(f"{str(loader_eval.dataset.data_dir.parent)}/annotations/instances_val2017.json")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    save_inferences(
        fs_model, mask_gen, image_ids, coco_gt, VAL_IMGS, 
        args.new_sample_size, args.num_classes, filepath
    )
    eval_sam(coco_gt, image_ids, filepath)

if __name__ == '__main__':
    args = get_parameters()
    throttle_cpu(args.numa)
    if not args.seed == None:
        seed_everything(args.seed)
    run_experiment(args)