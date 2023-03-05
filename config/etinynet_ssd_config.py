
"""
Configuration of training using EtinyNet-SSD

@author: CHADLI KOUIDER
"""
import yaml
import os
import torch
from utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
# determine the current device and based on that set the pin memory
# flag
# PIN_MEMORY = True if DEVICE == "cuda" else False


# Load the configuration file
# Get the absolute path of the config.yaml file
config_path = os.path.join('config', 'Config_etinynet_ssd.yaml')
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# parameters for image normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Dataset configuration
train_path = config['dataset']['train']
val_path = config['dataset']['val']
num_classes = config['dataset']['num_classes']
class_names = config['dataset']['class_names']
input_size = config['dataset']['input_size']

# Model configuration
width_multiplier = config['model']['width_multiplier']
pretrained = config['model']['pretrained']
type = config['model']['type']
ssd_feature_maps = config['model']['ssd']['feature_maps']
ssd_shrinkage = config['model']['ssd']['shrinkage']
ssd_min_sizes = config['model']['ssd']['min_sizes']
ssd_max_sizes = config['model']['ssd']['max_sizes']
ssd_aspect_ratios = config['model']['ssd']['aspect_ratios']
center_variance = config['model']['ssd']['center_variance']
size_variance = config['model']['ssd']['size_variance']
ssd_iou_thresh = config['model']['ssd']['iou_thresh']
ssd_nms_thresh = config['model']['ssd']['nms_thresh']

# Training configuration
batch_size = config['train']['batch_size']
optimizer_type = config['train']['optimizer']['type']
learning_rate = config['train']['optimizer']['lr']
momentum = config['train']['optimizer']['momentum']
weight_decay = config['train']['optimizer']['weight_decay']
scheduler_type = config['train']['scheduler']['type']
milestones = config['train']['scheduler']['milestones']
gamma = config['train']['scheduler']['gamma']
num_epochs = config['train']['num_epochs']

# Testing configuration
test_batch_size = config['test']['batch_size']

# Miscellaneous configuration

if config['misc']['device'] != 'cpu':
    device = config['misc']['device'] if torch.cuda.is_available() else 'cpu' # Load device from config file
else:
    device = config['misc']['device']

checkpoint_dir = config['misc']['checkpoint_dir']
log_dir = config['misc']['log_dir']
log_interval = config['misc']['log_interval']
save_interval = config['misc']['save_interval']
output_dir = config['output_dir']


#   1-Generate small-sized square box: This section appends 1 element to priors.
#   2-Generate big-sized square box: This section appends 1 element to priors.
#   3-Change h/w ratio of the small sized box: 
#     This section appends 2 * len(ssd_aspect_ratios) elements to priors. 
#     For each aspect ratio, we append 2 priors: one with width = w * sqrt(ratio) and height = h / sqrt(ratio), 
#     and one with width = w / sqrt(ratio) and height = h * sqrt(ratio).
num_priors = []
for i ,aspect_ratio in enumerate(ssd_aspect_ratios):
    num_priors.append( 1 + 1 + 2 * len(aspect_ratio))

# Define specs list
specs = []

for i, fmap in enumerate(ssd_feature_maps):
    specs.append(SSDSpec(fmap, ssd_shrinkage[i], 
                         SSDBoxSizes(ssd_min_sizes[i], ssd_max_sizes[i]),
                         ssd_aspect_ratios[i]))

priors = generate_ssd_priors(specs, input_size)