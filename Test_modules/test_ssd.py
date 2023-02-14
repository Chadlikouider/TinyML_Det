
"""
A file to test ssd module

@author: Chadli kouider
"""
import torch
import torch.nn as nn
from models.ssd.ssd import SSD


num_classes = 3
# Define the base network as a list of layers
base_net = nn.ModuleList([nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3), nn.ReLU(),
                          nn.Conv2d(32, 64, 3), nn.ReLU()])
    
# Define the source layer indexes
source_layer_indexes = [2]
    
# Define the extra layers
extras = nn.ModuleList([nn.Conv2d(64, 18, 1)])
    
    # Define the classification headers
classification_headers = nn.ModuleList([nn.Conv2d(16, 6* num_classes, 1),
                                        nn.Conv2d(18, 6* num_classes, 1)])
    
# Define the regression headers
regression_headers = nn.ModuleList([nn.Conv2d(16, 6* 4, 1),
                                    nn.Conv2d(18, 6* 4, 1)])
    
# Create an instance of the SSD class
model = SSD(num_classes, base_net, source_layer_indexes, extras, classification_headers, regression_headers)
    
# Define an input tensor
x = torch.randn(10, 3, 128, 128)
    
# Pass the input tensor through the model
confidences, locations = model(x)
    
    
print("Test Passed!")