#!/usr/bin/env python
import warnings
import torch
import cv2
from torch.nn.parallel import DistributedDataParallel as NativeDDP
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
try:
    from timm.layers import set_layer_config, convert_sync_batchnorm
except ImportError:
    from timm.models.layers import set_layer_config
    try:
        # timm has its own convert_sync_bn to support BatchNormAct layers in updated models
        from timm.models.layers import convert_sync_batchnorm
    except ImportError:
        warnings.warn(
            "Falling back to torch.nn.SyncBatchNorm, does not properly work with timm models in new versions.")
        convert_sync_batchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm
from utils import *
from data.fewshot_data import get_batch_prototypes
from data.transforms import Transform_To_Models
from pycocotools.coco import COCO
from engine import SAM
from engine.feature_extractor import MyFeatureExtractor
from engine.prototypical_networks import PrototypicalNetworks
from engine.ood_filter_neg_likelihood import OOD_filter_neg_likelihood
#------------------------------------------------------------------------------------------------


def sam_simple(args, output_root):
    """ Run sam and store all masks as inferences 
    Params
    :args -> parameters from bash.
    :output_root (str) -> output folder location.
    """
    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    _, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root)
    
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
        sam, unlabeled_loader, res_data,
        args.use_sam_embeddings
    )
    eval_sam(
        coco_gt, image_ids, res_data, 
        output_root, method=args.method
    )

def few_shot(args, is_single_class=None, output_root=None):
    """ Use sam and fewshot to classify the masks.
    Params
    :args -> parameters from bash.
    :output_root (str) -> output folder location.
    """
    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    labeled_loader, unlabeled_loader = create_datasets_and_loaders(args)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root)

    # STEP 2: create an SAM instance
    sam = SAM(args)
    sam.load_simple_mask()

    # STEP 3: create few-shot model
    if args.use_sam_embeddings:
        feature_extractor = sam
    elif is_single_class:
        feature_extractor = MyFeatureExtractor(
            args.timm_model, args.load_pretrained, args.num_classes
        )
    else:
        feature_extractor = MyFeatureExtractor(
            args.timm_model, 
            args.load_pretrained, 
            args.num_classes * 2 # one additional class for background
        )

    fs_model = PrototypicalNetworks(
        is_single_class,
        args.use_sam_embeddings,
        feature_extractor, 
        use_softmax=False,
    ).to('cuda')

    # STEP 4: get the raw support set
    trans_norm = None
    if args.use_sam_embeddings:
        trans_norm = Transform_To_Models()
    elif feature_extractor.is_transformer:
        trans_norm = Transform_To_Models(
                size=feature_extractor.input_size, 
                force_resize=True, keep_aspect_ratio=False
            )
    else:
        trans_norm = Transform_To_Models(
                size=33, force_resize=False, keep_aspect_ratio=True
            )

    if is_single_class:
        # single class does not require background class
        imgs, _ = get_batch_prototypes( 
            labeled_loader, args.num_classes,
            get_background_samples=False, # single class
            trans_norm=trans_norm,
            use_sam_embeddings=args.use_sam_embeddings
        )
        #  create the prototypes
        fs_model.process_support_set(imgs) # just one class
    else:
        # REQUIRES background class
        imgs, labels = get_batch_prototypes(
            labeled_loader, args.num_classes, 
            get_background_samples=True, # two classes
            trans_norm=trans_norm,
            use_sam_embeddings=args.use_sam_embeddings
        )
        # create prototypes
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [i-1 for i in labels]    
        fs_model.process_support_set(imgs, labels)
    
    # STEP 5: classify these inferences using the few-shot model
    # and SAM predictions.
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/gt.json"
    coco_gt = COCO(f"{gt_eval_path}")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results.json"

    if is_single_class:
        save_inferences_singleclass(
            fs_model, unlabeled_loader, sam, 
            output_root, trans_norm,
            args.use_sam_embeddings
        )
    else:
        save_inferences_twoclasses(
            fs_model, unlabeled_loader, sam, 
            res_data, trans_norm,
            args.use_sam_embeddings
        )

    # STEP 6: evaluate model
    if is_single_class:
        for idx_ in range(1,4):
            MAX_IMAGES = 100000
            gt_eval_path = f"{output_root}/gt.json"
            coco_gt = COCO(gt_eval_path)
            image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
            res_data = f"{output_root}/bbox_results_std{idx_}.json"

            eval_sam(
                coco_gt, image_ids, res_data, 
                output_root, method=args.method,
                number=idx_
            )
    else:
        eval_sam(
            coco_gt, image_ids, res_data, 
            output_root, method=args.method
        )

def ood_filter(args, output_root):
    """ Use sam and fewshot (maximum likelihood) to classify masks.
    Params
    :args -> parameters from bash.
    :output_root (str) -> output folder location.
    """
    # STEP 1: create data loaders
    labeled_loader, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root)

    # STEP 2: run the ood filter using inferences from SAM
    # sam instance - default values of the model
    sam = SAM(args)
    sam.load_simple_mask()

    # instance the main class and instance the timm model
    ood_filter_neg_likelihood = OOD_filter_neg_likelihood(
        timm_model=args.timm_model, 
        timm_pretrained=args.load_pretrained,
        sam_model=sam,
        use_sam_embeddings=args.use_sam_embeddings
    )

    # #------------------------------------------------
    # ood_filter_neg_likelihood.sanity_check_bootstrapping(
    #     labeled_loader,  
    #     dir_filtered_root=output_root,
    #     ood_hist_bins=args.ood_histogram_bins
    # )
    # #------------------------------------------------

    # run filter using the backbone, sam, and ood
    ood_filter_neg_likelihood.run_filter(
        labeled_loader, unlabeled_loader, 
        dir_filtered_root=output_root,
        ood_thresh=args.ood_thresh,
        ood_hist_bins=args.ood_histogram_bins
    )

    # STEP 3: evaluate results
    # run for std 1 and 2
    for idx_ in range(1,4):
        MAX_IMAGES = 100000
        gt_eval_path = f"{output_root}/gt.json"
        coco_gt = COCO(gt_eval_path)
        image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
        res_data = f"{output_root}/bbox_results_std{idx_}.json"

        eval_sam(
            coco_gt, image_ids, res_data, 
            output_root, method=args.method,
            number=idx_
        )

def selective_search(args, output_root):
    """ Run selective search (ss) and stores the results.
    Params
    :args -> parameters from bash.
    :output_root (str) -> output folder location.
    """
    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    _, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root)

    # STEP 2: run inferences using selective search
    results = []
    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    max = 100
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for (_,batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # run ss over the current img
            # a. get image
            image = batch[0][idx].cpu().numpy().transpose(1,2,0)
            # b. to pil
            img_pil = Image.fromarray(image)
            # c. convert RGB to BGR and apply selective search (ss)
            open_cv_image = np.array(img_pil)[:, :, ::-1].copy() 
            # d. run ss
            ss.setBaseImage(open_cv_image)
            ss.switchToSelectiveSearchFast()
            proposals = ss.process()

            # e. store metadata
            for num,proposal in enumerate(proposals):
                imgs_box_coords += [proposal.tolist()]
                imgs_scores += [1]

                if (num + 1) >= max:
                    break
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * max
    
    for idx_,_ in enumerate(imgs_ids):
        image_result = {
            'image_id': imgs_ids[idx_],
            'category_id': 1,
            'score': imgs_scores[idx_],
            'bbox': imgs_box_coords[idx_],
        }
        results.append(image_result)
    
    filepath = f"{output_root}/bbox_results.json"
    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

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

    if args.use_sam_embeddings:
        output_root = f"./output/{args.output_folder}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}_seed{args.seed}/{args.method}@samEmbed"
    else:
        output_root = f"./output/{args.output_folder}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}_seed{args.seed}/{args.method}@{args.timm_model}"

    if args.method == Constants_MainMethod.SELECTIVE_SEARCH:
        output_root = f"./output/{args.output_folder}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}_seed{args.seed}/{args.method}"
        selective_search(args, output_root)
    if args.method == Constants_MainMethod.ALONE:
        output_root = f"./output/{args.output_folder}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}_seed{args.seed}/{args.method}"
        sam_simple(args, output_root)
    elif args.method == Constants_MainMethod.FEWSHOT_1_CLASS:
        few_shot(args, is_single_class=True, output_root=output_root)
    elif args.method == Constants_MainMethod.FEWSHOT_2_CLASSES:
        few_shot(args, is_single_class=False, output_root=output_root)
    elif args.method == Constants_MainMethod.FEWSHOT_OOD:
        ood_filter(args, output_root)