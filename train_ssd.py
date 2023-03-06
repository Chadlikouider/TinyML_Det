"""
    This file is for training the ssd_models including(Squezzenet-SSD, EtinyNet-SSD , and mobileNetV2-SSD)

    @author: CHADLI KOUIDER
"""

import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from dataset.pascal_voc import PascalVOCDataset
from models.ssd.ssd import MatchPrior
from models.ssd.etinynet_ssd import create_etinynet_ssd_lite
from models.ssd.Squeezenet_ssd import create_squeezenet_ssd_lite
from config import squeezenet_ssd_config
from config import etinynet_ssd_config



parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')

parser.add_argument('--net', default="etinynet-ssd",
                    help="The network architecture, it can be squeezenet-ssd, etinynet-ssd, or mb2-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


if __name__ == '__main__':
    timer = Timer()

    if args.net == 'etinynet-ssd':
        create_net = create_etinynet_ssd_lite
        config = etinynet_ssd_config
    elif args.net == 'squeezenet-ssd':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    if config.type == 'float64':
        dtype = torch.float64
    elif config.type == 'float32':
        dtype = torch.float32
    elif config.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8
    
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, config.ssd_iou_thresh)
    logging.info("Prepare training datasets.")
    datasets = []
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean = config.MEAN, 
                                                                std = config.STD)])
    
    train_dataset = PascalVOCDataset(root_dir = config.train_path, transform = train_transforms)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=False, collate_fn=train_dataset.collate)


    logging.info("Prepare Validation datasets.")
    val_dataset = PascalVOCDataset(root_dir = config.val_path, transform = train_transforms)
    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle = False, collate_fn=val_dataset.collate)
    

    
