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
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
# ------------------------------------------------------------------------------------------------


def _plot_image(img, name):
    dpi = 100
    # Create figure and axes
    fig, ax = plt.subplots(1,
                           figsize=(img.shape[0] / dpi, img.shape[1] / dpi),
                           dpi=dpi)

    # Display the image
    ax.imshow(img)

    plt.axis('off')  # Hide axes
    plt.tight_layout()  # Remove padding
    plt.savefig(f'images/{name}.png')
    plt.close(fig)


def plot_image(img, name, is_mask=False):
    if not is_mask:
        _plot_image(img, name)
    else:
        for i in range(img.shape[-1]):
            _plot_image(img[..., i], f'{name}_{i}')
    

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


def get_image(img):
    _img = img.squeeze(0).permute(1, 2, 0).numpy()
    # Make sure that img is in the range [0, 1]
    assert _img.min() >= 0 and _img.max() <= 255, f"img min: {_img.min()}, img max: {_img.max()}"
    return _img.astype(np.uint8)


def run_finetuning(args):
    """ Run simple experiment """
    path_output = f"./output/{args.output_folder}/{args.semi_percentage}/{args.run_name}"

    # STEP 1: create data loaders
    loader_labeled, loader_eval = create_datasets_and_loaders(args, include_masks=True)

    for ind, batch in tqdm(enumerate(loader_labeled), total=len(loader_labeled)):
        # every batch is a tuple: (torch.imgs , metadata_and_bboxes, full_json_annotation)
        # metadata_and_bboxes keys:
        #   dict_keys(['img_idx', 'img_size', 'mask', 'img_scale'])
        img = batch[0]
        mask = np.flip(batch[1]['mask'].numpy(), axis=-1)
        tqdm.write(f"\tImg shape: {img.shape}\n\tmetadata keys: {batch[1].keys()}\n\tmask shape: {mask.shape}")
        
        # Plotting for debugging
        # img_for_plot = get_image(img)        
        # mask_for_plot = mask.squeeze(0).transpose(1, 2, 0)
        # plot_image(img=img_for_plot, name=f"finetune_image_{ind}")
        # plot_image(img=mask_for_plot, name=f"finetune_mask_{ind}", is_mask=True)


    # STEP 2: create a SAM model
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

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
