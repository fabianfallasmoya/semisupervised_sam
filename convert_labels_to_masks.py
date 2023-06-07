import argparse
from utils import seed_everything, get_parameters
from run import create_datasets_and_loaders
from engine import SAM
import matplotlib.pyplot as plt
from matplotlib import patches, colors as mcolors
from tqdm import tqdm
import numpy as np
import cv2

PLOT_EVERY = 40


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
    for i, (bbox, masks_for_this_bbox) in enumerate(zip(bboxes, masks_per_bbox)):
        for j, (box, mask) in enumerate(zip(bbox, masks_for_this_bbox)):
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
            color = cmap(j % cmap.N)
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


def convert_labels(args: argparse.Namespace):
    # Load the data
    tqdm.write("loading data")
    loader_labeled, loader_eval = create_datasets_and_loaders(args)

    # Load the SAM model
    tqdm.write("loading SAM model")
    sam = SAM(args)
    
    # Use the bounding box labels as target boxes for SAM
    tqdm.write("Iterating over the data")
    for ind, batch in tqdm(enumerate(loader_labeled)):
        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        if ind == 0:
            tqdm.write(f"Imgs shape: {batch[0].shape}, metadata keys: {batch[1].keys()}, bbox shape: {batch[1]['bbox'].shape}")
        if ind < 2:
            continue
        # Get the bounding boxes
        bboxes = batch[1]['bbox']
        # Squeeze the batch dimension and transpose the tensor to a PIL-friendly format
        img = get_image(batch)

        # Iterate over the bounding boxes
        masks_per_bbox = []
        for bbox in bboxes:
            masks_for_this_bbox = []
            for box in bbox:
                masks = sam.get_mask_for_bbox(img, box.squeeze(0).numpy()[[1, 0, 3, 2]])
                masks_for_this_bbox.append(masks)
            masks_per_bbox.append(masks_for_this_bbox)
                
        # Plot the image and the bounding boxes
        if ind % PLOT_EVERY == 0:
            plot_image_and_bboxes(img, bboxes, masks_per_bbox, ind)

    # sam.get_mask_for_bbox

    # Run SAM and predict the masks

    # Save the masks


if __name__ == '__main__':
    args = get_parameters()
    # throttle_cpu(args.numa)
    if args.seed is not None:
        seed_everything(args.seed)
    convert_labels(args)
    print("Finished")
