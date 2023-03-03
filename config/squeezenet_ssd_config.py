import yaml
import os
import torch
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
# Get the absolute path of the config.yaml file
config_path = os.path.join('config', 'Config_squeezenet_ssd.yaml')
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
backbone = config['model']['backbone']
num_classes_model = config['model']['num_classes']
pretrained = config['model']['pretrained']
ssd_feature_maps = config['model']['ssd']['feature_maps']
ssd_shrinkage = config['model']['ssd']['shrinkage']
ssd_min_sizes = config['model']['ssd']['min_sizes']
ssd_max_sizes = config['model']['ssd']['max_sizes']
ssd_aspect_ratios = config['model']['ssd']['aspect_ratios']
ssd_variance = config['model']['ssd']['variance']
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

print(ssd_min_sizes)
print(ssd_max_sizes)
print(ssd_aspect_ratios)
print(test_batch_size)
print(checkpoint_dir)
print(output_dir)

# Define specs list
specs = []
for i, fmap in enumerate(ssd_feature_maps):
    specs.append(SSDSpec(fmap, ssd_shrinkage[i], 
                         SSDBoxSizes(ssd_min_sizes[i], ssd_max_sizes[i]),
                         ssd_aspect_ratios[i]))

priors = generate_ssd_priors(specs, input_size)