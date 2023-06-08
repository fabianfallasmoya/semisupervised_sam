import argparse
from utils import seed_everything, get_parameters
from run import create_datasets_and_loaders
from engine import SAM
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm
import numpy as np
import cv2
import os
import json
from PIL import Image
import torch
# from pprint import pprint

PLOT_EVERY = 25


def plot_image_and_bboxes(img, bboxes, masks_per_bbox, ind):
    # Create figure and axes
    dpi = 100
    fig, ax = plt.subplots(1,
                           figsize=(img.shape[0] / dpi, img.shape[1] / dpi),
                           dpi=dpi)

    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch for each bounding box and plot the segmentation masks
    cmap = plt.get_cmap('tab20')
    for i, (box, mask) in enumerate(zip(bboxes, masks_per_bbox)):
        if mask is None:
            continue
        # Create a Rectangle patch for each bounding box
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0],
                                  linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # If mask's first dimension is one, squeeze it
        if mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)

        # Convert the boolean mask to integer format
        mask_uint8 = mask.astype(np.uint8)
        
        # Create mask_color
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 4))

        # Apply color to the places where mask is 1
        color = cmap(i % cmap.N)
        mask_color[mask == 1, :] = color  # Assign color to the mask

        # Overlay the colored mask on the image
        ax.imshow(mask_color, alpha=0.5, interpolation='none')
        
        # Create a border for the mask
        border_uint8 = cv2.dilate(mask_uint8, np.ones((5, 5), np.uint8), iterations=1) - mask_uint8

        # Create a color border
        border_color = np.zeros((*border_uint8.shape, 4))
        border_color[border_uint8 == 1] = color
    
        # Overlay the colored border on the image
        ax.imshow(border_color, alpha=0.7, interpolation='none')  # Adjust alpha for transparency

    plt.axis('off')  # Hide axes
    plt.tight_layout()  # Remove padding
    plt.savefig(f'images/image_{ind}.png')
    plt.close(fig)
    

def get_image(batch):
    img = batch[0].squeeze(0).permute(1, 2, 0).numpy()
    # Make sure that img is in the range [0, 1]
    assert img.min() >= 0 and img.max() <= 255, f"img min: {img.min()}, img max: {img.max()}"
    return img.astype(np.uint8)


def tensor_to_item(d):
    if isinstance(d, list):
        return [tensor_to_item(v) for v in d]
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                d[key] = value.item()
            elif isinstance(value, list):
                d[key] = [v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for v in value]
            elif isinstance(value, dict):
                tensor_to_item(value)
        return d
    else:
        raise ValueError(f"Unsupported type: {type(d)}")


def match_bboxes_to_annot(bboxes, scale, annot):
    matched_pairs = []
    for bbox in bboxes:
        # Scale the bbox
        scaled_bbox = bbox * scale

        # Attempt to match to an annot
        matched_annot = None
        for a in annot:
            annot_bbox = torch.cat(a['bbox'])
            if torch.allclose(scaled_bbox[0], annot_bbox[1], atol=1e-8):
                matched_annot = a
                break

        # If a match was found, append it to the results
        if matched_annot is not None:
            matched_pairs.append((bbox, matched_annot))
        else:
            matched_pairs.append((bbox, None))
            
    return matched_pairs


def convert_labels_for_single_loader(args: argparse.Namespace,
                                     loader,
                                     instances_json_path,
                                     masks_save_path,
                                     json_save_path):
    # Load the SAM model
    tqdm.write("loading SAM model")
    sam = SAM(args)
    
    # Load the existing annotations
    with open(os.path.join(args.root, instances_json_path), 'r') as f:
        all_annotations_dict = json.load(f)

    # Use the bounding box labels as target boxes for SAM
    mask_paths = []  # Initialize an empty list to hold the paths to the mask files
    new_annotations = []
    tqdm.write("Iterating over the data")
    for ind, batch in tqdm(enumerate(loader), total=len(loader)):
        # every batch is a tuple: (torch.imgs , metadata_and_bboxes, full_json_annotation)
        # metadata_and_bboxes keys:
        #   dict_keys(['img_idx', 'img_size', 'bbox', 'cls', 'img_scale'])

        if ind == 0:
            tqdm.write(f"Imgs shape: {batch[0].shape}, metadata keys: {batch[1].keys()}, bbox shape: {batch[1]['bbox'].shape}")
        
        if len(batch) == 3:
            full_json_annotation = batch[2]
            # print(f"full_json_annotation: len ({len(full_json_annotation)}):")
            # pprint(full_json_annotation)
        else:
            raise Exception("No full json annotation found in batch")
        
        # print(f"Target img_idx: {batch[1]['img_idx'].item()}")
        # print(f"Target img_size: {batch[1]['img_size']}")
        # print(f"Target img_scale: {batch[1]['img_scale']}")
        # Get the image id from the batch metadata
        # image_id = batch[1]['img_idx'].item()
        # print(f"Metadata: {batch[1]}")
        # Get the bounding boxes
        bboxes_orig = batch[1]['bbox']
        # Squeeze the batch dimension and transpose the tensor to a PIL-friendly format
        img = get_image(batch)

        # Run SAM on each image+box and predict the masks
        num_valid_boxes = 0
        masks_per_bbox = []
        bboxes = []
        # print("Target Raw boxes:")
        for bbox in bboxes_orig:
            for box in bbox:
                all_minus_one = torch.all(box == -1.).item()
                if all_minus_one:
                    masks_per_bbox.append(None)
                    continue
                else:
                    num_valid_boxes += 1
                    # print("\t", box)
                masks = sam.get_mask_for_bbox(img, box.squeeze(0).numpy()[[1, 0, 3, 2]])
                masks_per_bbox.append(masks)
                bboxes.append(box)
        # print("num_valid_boxes:", num_valid_boxes)
        matched_pairs = match_bboxes_to_annot(bboxes, batch[1]['img_scale'], full_json_annotation)
        # print("Matched pairs:")
        # for pair in matched_pairs:
        #     print("\t", pair)
        #     print("--------------------")
        # Save the masks as images
        box_ind = 0
        for (box, json_annot), mask in zip(matched_pairs, masks_per_bbox):
            if mask is None:
                continue
            # Ensure mask is a numpy array
            mask_np = np.array(mask).squeeze().astype(np.uint8)
            
            # Ensure it's a 2D array
            assert len(mask_np.shape) == 2, \
                f"Expected a 2D mask but got a {len(mask_np.shape)}D mask (shape: {mask_np.shape})"

            # Save mask as image
            # Multiply by 255 to convert binary mask to an image
            mask_img = Image.fromarray(mask_np * 255)
            mask_path = os.path.join(args.root,
                                     f'{masks_save_path}/mask_{ind}_{box_ind}.png')
            mask_img.save(mask_path)
            mask_paths.append(mask_path)

            # Append the matched annotation along with the mask path
            json_annot['mask_file_path'] = mask_path
            new_annotations.append(json_annot)
            box_ind += 1
        
        # Plot the image and the bounding boxes
        if ind % PLOT_EVERY == 0:
            plot_image_and_bboxes(img, bboxes, masks_per_bbox, ind)

    # Save the updated annotations
    with open(os.path.join(args.root, json_save_path), 'w') as f:
        # Update the original annotations dictionary with the new annotations and save it as a new JSON file
        all_annotations_dict['annotations'] = tensor_to_item(new_annotations)
        try:
            json.dump(all_annotations_dict, f)
        except TypeError:
            print("Error while dumping json")
            print("all_annotations_dict:", all_annotations_dict)
            raise


def convert_labels(args: argparse.Namespace):
    # Load the data
    print("Loading dataloaders")
    loader_labeled, loader_eval = create_datasets_and_loaders(args, verbose=True)

    print("Converting labels for the trained data")
    convert_labels_for_single_loader(args,
                                     loader_labeled,
                                     instances_json_path='annotations/instances_train.json',
                                     masks_save_path='annotations/masks_train',
                                     json_save_path='annotations/segmentation_masks_train.json')

    print("Converting labels for the eval data")
    convert_labels_for_single_loader(args,
                                     loader_eval,
                                     instances_json_path='annotations/instances_val.json',
                                     masks_save_path='annotations/masks_val',
                                     json_save_path='annotations/segmentation_masks_val.json')
    

if __name__ == '__main__':
    args = get_parameters()
    # throttle_cpu(args.numa)
    if args.seed is not None:
        seed_everything(args.seed)
    convert_labels(args)
    print("Finished")
