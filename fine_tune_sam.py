# import warnings
import torch
# from torch.nn.parallel import DistributedDataParallel as NativeDDP
# from utils import Constants_AugMethod, add_bool_arg
# try:
#     from apex import amp
#     from apex.parallel import DistributedDataParallel as ApexDDP
#     from apex.parallel import convert_syncbn_model
#     has_apex = True
# except ImportError:
from data import create_loader, create_dataset  # ,
#  create_parser, DetectionDatset, SkipSubset, resolve_input_config
from utils import get_parameters, seed_everything  # , throttle_cpu, add_bool_arg
from engine import SAM
from pycocotools.coco import COCO
from eval import save_inferences, eval_sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
# ------------------------------------------------------------------------------------------------


def create_datasets_and_loaders(args, verbose=False, include_masks=False):
    """ Setup datasets, transforms, loaders, evaluator.
    Params
    :args: Model specific configuration dict / struct

    Returns
    :Train loader, validation loader
    """
    datasets = create_dataset(
        args.dataset, args.root, use_semi_split=args.use_semi_split,
        semi_percentage=args.semi_percentage,
        include_masks=include_masks,
        verbose=verbose
    )
    dataset_train_labeled = datasets[0]
    dataset_eval = None
    dataset_train_unlabeled = None
    if len(datasets) > 2:
        dataset_train_unlabeled = datasets[1]
        dataset_eval = datasets[2]
    else:
        dataset_eval = datasets[1]

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
    if dataset_eval is not None:
        # if args.val_skip > 1:
        #     dataset_eval = SkipSubset(dataset_eval, args.val_skip)
        loader_eval = create_loader(
            dataset_eval,
            img_resolution=args.img_resolution,
            batch_size=args.batch_size_val,
            is_training=False,
        )
        return loader_labeled, loader_eval


def run_finetuning(args):
    """ Run simple experiment """
    path_output = f"./output/{args.output_folder}/{args.semi_percentage}/{args.run_name}"

    # STEP 1: create data loaders
    loader_labeled, loader_eval = create_datasets_and_loaders(args, include_masks=True)

    # STEP 2: create a SAM model
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # STEP 3: fine-tune SAM
    # Resources:
    # official Github repo: https://github.com/facebookresearch/segment-anything
    # An example of SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
    # An example of how to fine-tune SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/


if __name__ == '__main__':
    args = get_parameters()
    # throttle_cpu(args.numa)
    if args.seed is not None:
        seed_everything(args.seed)
    run_finetuning(args)
