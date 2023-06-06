import argparse
from utils import seed_everything, get_parameters
from run import create_datasets_and_loaders
from engine import SAM


def convert_labels(args: argparse.Namespace):
    # Load the data
    print("loading data")
    loader_labeled, loader_eval = create_datasets_and_loaders(args)

    # Load the SAM model
    print("loading SAM model")
    sam = SAM(args)
    
    # Use the bounding box labels as target boxes for SAM

    # Run SAM and predict the masks

    # Save the masks


if __name__ == '__main__':
    args = get_parameters()
    # throttle_cpu(args.numa)
    if args.seed is not None:
        seed_everything(args.seed)
    convert_labels(args)
    print("Finished")
