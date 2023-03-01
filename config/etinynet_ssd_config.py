
"""
Configuration of training

@author: CHADLI KOUIDER
"""
import yaml
from utils.box_utils import generate_ssd_priors, SSDSpec, SSDBoxSizes
# determine the current device and based on that set the pin memory
# flag
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False
# specify ImageNet mean and standard deviation
# specify the loss weights
# LABELS = 1.0
# BBOX = 1.0

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


image_size = config["image_size"]
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
iou_threshold = config["IoU_threshold"]
center_variance = config["center_variance"]
size_variance = config["size_variance"]
num_class = config["num_classes"]

#============== Generate anchor boxes ==================
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)
#=======================================================

# Access the hyperparameters
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]


# Directories
train_dir = config["train_dir"]
val_dir = config["val_dir"]
test_dir = config["test_dir"]