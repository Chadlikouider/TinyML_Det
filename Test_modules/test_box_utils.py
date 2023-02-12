# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:39:33 2023

@author: wakcomputer
"""

import collections
import torch
import itertools
from typing import List
import math
from utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

################################## generate_ssd_priors #################################
# Define named tuples for the SSD specification
SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

# Define the SSD specifications
specs = [
    SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
    SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
]

# Define the image size
image_size = 300

# Generate the SSD priors
priors = generate_ssd_priors(specs, image_size)

# Print the resulting priors
print(priors)


################################# convert_locations_to_boxes ########################


################################# convert_boxes_to_locations ########################


################################  center_form_to_corner_form ########################


################################  corner_form_to_center_form ########################


###############################  area_of and iou_of and assign_priors ###############



################################# NMS(Hard and soft) ################################
from utils.box_utils import nms
#                        x_min,y_min,x_max,y_max,propability
box_scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.9],
                           [0.2, 0.3, 0.4, 0.5, 0.8],
                           [0.3, 0.4, 0.5, 0.6, 0.7],
                           [0.4, 0.5, 0.6, 0.7, 0.6],
                           [0.5, 0.6, 0.7, 0.8, 0.5],
                           [0.6, 0.7, 0.8, 0.9, 0.4],
                           [0.7, 0.8, 0.9, 1.0, 0.3],
                           [0.8, 0.9, 1.0, 1.1, 0.2],
                           [0.9, 1.0, 1.1, 1.2, 0.1],
                           [1.0, 1.1, 1.2, 1.3, 0.0]])
# Test hard NMS
result = nms(box_scores, nms_method="hard", iou_threshold=0.5, top_k=3, candidate_size=6)
print(result)

# Test soft NMS
result = nms(box_scores, nms_method="soft", score_threshold=0.4, sigma=0.3, top_k=3)
print(result)


################################# hard_negative_mining ################################

from utils.box_utils import hard_negative_mining
loss = torch.tensor([[0.3, 0.2, 0.1, 0.4, 0.5],
                     [0.5, 0.4, 0.3, 0.2, 0.1]])
labels = torch.tensor([[1, 0, 1, 0, 0],
                       [0, 1, 1, 0, 1]])
neg_pos_ratio = 3

# Call the hard_negative_mining function
mask = hard_negative_mining(loss, labels, neg_pos_ratio)

# Print the output mask
print(mask)
# output should be :tensor([[1, 1, 1, 0, 0],[0, 1, 1, 0, 1]])