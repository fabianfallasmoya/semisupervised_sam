#!/usr/bin/env python
import warnings
from segment_anything.modeling import sam
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from engine.feature_extractor import MyFeatureExtractor
from engine.prototypical_networks import PrototypicalNetworks, PrototypicalNetworks_SingleClass
from utils import Constants_AugMethod, Constants_MainMethod
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
from data import create_dataset_ood
from utils import get_parameters, seed_everything, throttle_cpu, save_gt
from data.fewshot_data import get_batch_prototypes
from engine import SAM
from pycocotools.coco import COCO
from eval import *
from data import transforms_toNumpy, create_loader
from ood_filter import OOD_filter_neg_likelihood
#------------------------------------------------------------------------------------------------


def create_datasets_and_loaders(args):
    """ Setup datasets, transforms, loaders, evaluator.
    Params
    :args -> Model specific configuration dict / struct

    Returns
    :loaders
    """
    datasets = create_dataset_ood(
        args.dataset, args.root, 
        labeled_samples=args.ood_labeled_samples,
        unlabeled_samples=args.ood_unlabeled_samples
    )
    dataset_train_labeled = datasets[0]
    dataset_train_unlabeled = datasets[1]

    # create data loaders
    trans_numpy = transforms_toNumpy()
    normalize_imgs = False
    loader_labeled = create_loader(
        dataset_train_labeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_labeled,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    loader_unlabeled = create_loader(
        dataset_train_unlabeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_unlabeled,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    return loader_labeled, loader_unlabeled

def sam_simple(args, output_root):
    """ Run sam and store all masks as inferences """

    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    _, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root, method=args.method)
    
    # STEP 2: create an SAM instance
    sam = SAM(args)
    sam.load_simple_mask()

    # STEP 3: classify these inferences using the few-shot model
    # and SAM predictions.
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/gt.json"
    coco_gt = COCO(f"{gt_eval_path}")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results.json"

    save_inferences_simple(
        sam, unlabeled_loader, res_data
    )
    eval_sam(
        coco_gt, image_ids, res_data, 
        output_root, method=args.method
    )

def few_shot_single_class(args, output_root):
    """ Run few shot with a single class """

    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    labeled_loader, unlabeled_loader = create_datasets_and_loaders(args)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root, method=args.method)

    # STEP 2: create few-shot model 
    feature_extractor = MyFeatureExtractor(
        args.timm_model, args.load_pretrained, args.num_classes
    )
    fs_model = PrototypicalNetworks_SingleClass(
        feature_extractor, use_softmax=False
    ).to('cuda')

    # STEP 3: get the raw support set
    trans_norm = None
    if feature_extractor.is_transformer:
        trans_norm = Transform_Normalization(
                size=feature_extractor.input_size, 
                force_resize=True, keep_aspect_ratio=False
            )
    else:
        trans_norm = Transform_Normalization(
                size=33, force_resize=False, keep_aspect_ratio=True
            )
    imgs, _ = get_batch_prototypes( 
        labeled_loader, args.num_classes,
        get_background_samples=False,
        trans_norm=trans_norm
    )
    
    #  create the prototypes
    fs_model.process_support_set(imgs)
    fs_model.forward_groundtruth(imgs)

    # STEP 4: create an SAM instance
    sam = SAM(args)
    sam.load_simple_mask()

    # STEP 5: classify these inferences using the few-shot model
    # and SAM predictions.
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/gt.json"
    coco_gt = COCO(f"{gt_eval_path}")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results.json"

    save_inferences_singleclass(
        fs_model, unlabeled_loader, sam, 
        res_data, trans_norm
    )

    # STEP 6: evaluate model
    eval_sam(
        coco_gt, image_ids, res_data, 
        output_root, method=args.method
    )

def few_shot_two_classes(args, output_root):
    """ Run few shot with two classes: object and no-object """

    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    labeled_loader, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    save_gt(unlabeled_loader, output_root, method=args.method)
    
    # STEP 2: create few-shot model 
    num_classes = args.num_classes * 2 # one additional class for every class
    feature_extractor = MyFeatureExtractor(
        args.timm_model, args.load_pretrained, num_classes
    )
    fs_model = PrototypicalNetworks(
        feature_extractor, use_softmax=False
    ).to('cuda')

    # STEP 3: get the raw support set
    trans_norm = None
    if feature_extractor.is_transformer:
        trans_norm = Transform_Normalization(
                size=feature_extractor.input_size, 
                force_resize=True, keep_aspect_ratio=False
            )
    else:
        trans_norm = Transform_Normalization(
                size=33, force_resize=False, keep_aspect_ratio=True
            )
    imgs, labels = get_batch_prototypes(
        labeled_loader, args.num_classes, 
        get_background_samples=True,
        trans_norm=trans_norm
    )
    # labels start from index 1 to n, translate to start from 0 to n.
    labels = [i-1 for i in labels]    
    #  create the prototypes
    fs_model.process_support_set(imgs, labels)

    # STEP 4: run inferences using SAM
    sam = SAM(args)
    sam.load_simple_mask()

    # STEP 5: classify these inferences using the few-shot model
    # and SAM predictions.
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/gt.json"
    coco_gt = COCO(f"{gt_eval_path}")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results.json"

    save_inferences_twoclasses(
        fs_model, unlabeled_loader, sam, 
        res_data, trans_norm
    )

    # STEP 6: evaluate model
    eval_sam(
        coco_gt, image_ids, res_data, 
        output_root, method=args.method
    )

def ood_filter(args, output_root):
    """ Run ood filter """

    # STEP 1: create data loaders
    labeled_loader, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root, args.method)

    # STEP 2: run the ood filter using inferences from SAM
    # sam instance - default values of the model
    sam = SAM(args)
    sam.load_simple_mask()

    # instance the main class and instance the timm model
    ood_filter_neg_likelihood = OOD_filter_neg_likelihood(
        timm_model=args.timm_model, 
        timm_pretrained=args.load_pretrained,
        sam_model=sam
    )

    # run filter using the backbone, sam, and ood
    ood_filter_neg_likelihood.run_filter(
        labeled_loader, unlabeled_loader, 
        dir_filtered_root=output_root,
        ood_thresh=args.ood_thresh,
        ood_hist_bins=args.ood_histogram_bins
    )

    # STEP 3: evaluate results
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/gt.json"
    coco_gt = COCO(gt_eval_path)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results.json"

    eval_sam(
        coco_gt, image_ids, res_data, 
        output_root, method=args.method
    )


if __name__ == '__main__':
    args = get_parameters()

    if not args.numa == -1:
        throttle_cpu(args.numa)
    if not args.seed == None:
        seed_everything(args.seed)

    output_root = f"./output/{args.output_folder}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}_seed{args.seed}_{args.timm_model}/{args.method}"

    if args.method == Constants_MainMethod.SAM_FEWSHOT_SINGLE_CLASS:
        few_shot_single_class(args, output_root)
    elif args.method == Constants_MainMethod.SAM_ALONE:
        sam_simple(args, output_root)
    elif args.method == Constants_MainMethod.SAM_FEWSHOT_TWO_CLASSES:
        few_shot_two_classes(args, output_root)
    elif args.method == Constants_MainMethod.SAM_OOD:
        ood_filter(args, output_root)