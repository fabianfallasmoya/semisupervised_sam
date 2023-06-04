import argparse
import psutil

class Constants_AugMethod:
    NO_AUGMENTATION = 'no_augmentation'
    RAND_AUGMENT = 'rand_augmentation'

def add_bool_arg(parser, name, default=False, help=''):
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})

def seed_everything(seed: int = 42, rank: int = 0):
    """ Try to seed everything to reproduce results """
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed + rank)
    os.environ['PYTHONHASHSEED'] = str(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameters():
    # parameters
    parser = argparse.ArgumentParser(description='TIMM + new data aug')
    # Model 
    parser.add_argument('--root',type=str, default='.')
    parser.add_argument('--num-classes', type=int, default=1) 
    add_bool_arg(parser, 'load-pretrained', default=False) 
    parser.add_argument('--timm-model', type=str, default="")  
    parser.add_argument('--loss', type=str, default="mse")  
    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--val-freq', type=int, default=1)

    # semi
    add_bool_arg(parser, 'use-semi-split', default=False) 
    parser.add_argument('--semi-percentage', type=float, default=10.)

    # training
    parser.add_argument('--epochs', type=int, default=1)

    # dataset
    parser.add_argument('--dataset', default='coco17', type=str, metavar='DATASET')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--batch-size-val', type=int, default=64)
    parser.add_argument('--reprob', type=float, default=0.)
    parser.add_argument('--aug-method', type=str, default=Constants_AugMethod.NO_AUGMENTATION)
    parser.add_argument('--img-resolution', type=int, default=512)
    parser.add_argument('--new-sample-size', type=int, default=224)
    # add_bool_arg(parser, 'store-val', default=False) 

    # general
    parser.add_argument('--numa', type=int, default=None)
    parser.add_argument('--output-folder', type=str, default=None)  
    parser.add_argument('--run-name', type=str, default=None)  
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sam-model', type=str, default=None)

    return parser.parse_args()

def get_cpu_list(val: int = None):
    """
    Accodring lscpu the NUMA nodes are:
    
    NUMA node0 CPU(s):     0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38
    NUMA node1 CPU(s):     1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39

    so, accordint to this distribution every gpu will be assign 10 cpus in the same NUMA node.
    cpu list goes from 0 to 39.
    """
    # numa 0
    if val == 0:
        return list(range(0,20,2))
    if val == 1:
        return list(range(20,40,2))

    # numa 1
    if val == 2:
        return list(range(1,20,2))
    if val == 3:
        return list(range(21,40,2))

# LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(numa: int = None):
    cpu_list = get_cpu_list(numa)
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])