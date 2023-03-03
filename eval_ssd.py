import torch
import argparse
import pathlib
import numpy as np
import logging
import sys


from utils import box_utils, common_tools
from utils.misc import str2bool, Timer

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="etinynet-ssd",
                    help="The network architecture, it should be of squeezenet-ssd,etinynet-ssd, or mobilenet-ssd..")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support only voc.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, 
                    help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str,
                    help="The directory to store evaluation results.")
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")



def main():
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    
if __name__ == '__main__':
    main()