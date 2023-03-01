
"""
Configuration of training

@author: CHADLI KOUIDER
"""

import yaml
from utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
# determine the current device and based on that set the pin memory
# flag
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False
# specify ImageNet mean and standard deviation
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
# INIT_LR = 1e-4
# NUM_EPOCHS = 20
# specify the loss weights
# LABELS = 1.0
# BBOX = 1.0


# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# parameters for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# General settings
num_classes = config['num_classes']
image_size = config['image_size']
batch_size = config['batch_size']
num_epochs = config['num_epochs']



# Model settings
backbone = config['backbone']
num_anchors = config['num_anchors']
center_variance = config['center_variance']
size_variance = config['size_variance']


# Loss function settings
loc_loss_type = config['loc_loss_type']
loc_weight = config['loc_weight']
conf_loss_type = config['conf_loss_type']
conf_weight = config['conf_weight']
gamma = config['gamma']

# Optimizer settings
optimizer = config['optimizer']
lr = config['lr']
momentum = config['momentum']
weight_decay = config['weight_decay']

# OD parameters
iou_threshold = config['iou_threshold']
nms_threshold = config['nms_threshold']
# Dataset settings
train_dataset = config['train_dataset']
val_dataset = config['val_dataset']
train_data_dir = config['train_data_dir']
val_data_dir = config['val_data_dir']

specs = [
    SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
    SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
]


priors = generate_ssd_priors(specs, image_size)