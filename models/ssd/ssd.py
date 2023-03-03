"""
    Implementation of SSD module Adapted from:
    https://github.com/tranleanh/mobilenets-ssd-pytorch/blob/master/vision/ssd/ssd.py
    @author: CHADLI KOUIDER
"""

import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from utils import box_utils


#from collections import namedtuple
#GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])

class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        
        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes 
                                                   if isinstance(t, tuple)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            # self.priors = config.priors.to(self.device)
            self.priors = config.priors.to("cpu")
        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # forward function returns
        #the confidence scores and the predicted locations of the bounding boxes
        confidences = []
        locations = []
        # Index to keep track of the starting layer in the base network
        start_layer_index = 0
        # Index to keep track of the header index in the list of headers
        header_index = 0

        # Loop through the list of end layer indexes for the source layers
        for end_layer_index in self.source_layer_indexes:
            # Check if the current end layer index is a tuple (layer index, added layer)
            if isinstance(end_layer_index, tuple):
                # If it is, the added layer is extracted and stored in the added_layer variable
                added_layer = end_layer_index[1]
                # The end layer index is also extracted
                end_layer_index = end_layer_index[0]
            else:
                # If it's not a tuple, the added layer is set to None
                added_layer = None
    
            # Loop through the base network layers from the starting layer to the end layer index
            for layer in self.base_net[start_layer_index: end_layer_index]:
                # Pass the input through the current layer
                x = layer(x)
    
            # Check if there is an added layer
            if added_layer:
                # If there is, pass the output from the base network to the added layer
                y = added_layer(x)
            else:
                # If there isn't, set y to be the output from the base network
                y = x
    
            # Update the starting layer index to the end layer index
            start_layer_index = end_layer_index
    
            # Compute the header for the current layer
            confidence, location = self.compute_header(header_index, y)
    
            # Increment the header index
            header_index += 1
    
            # Append the confidence and location values to their respective lists
            confidences.append(confidence)
            locations.append(location)

        # Loop through the remaining layers in the base network
        for layer in self.base_net[end_layer_index:]:
            # Pass the input through the current layer
            x = layer(x)

        # Loop through the extra layers
        for layer in self.extras:
            # Pass the input through the current layer
            x = layer(x)
    
            # Compute the header for the current layer
            confidence, location = self.compute_header(header_index, x)
    
            # Increment the header index
            header_index += 1
    
            # Append the confidence and location values to their respective lists
            confidences.append(confidence)
            locations.append(location)

        # Concatenate the lists of confidence scores and locations
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        # Check if the current mode is test or not
        if self.is_test:
            # Calculate the softmax of confidences along the 2nd dimension
            confidences = F.softmax(confidences, dim=2)
            # Convert the locations to boxes based on the priors, center variance, and size variance
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance)
            # Convert the boxes from center form to corner form
            boxes = box_utils.center_form_to_corner_form(boxes)
            # Return the confidences and boxes
            return confidences, boxes
        # If the mode is not test, simply return the confidences and locations
        else:
            return confidences, locations
        
    # This function takes as input the index of a header and a tensor x, 
    # and returns the confidence scores and the predicted locations of the 
    # bounding boxes for the header. 
    def compute_header(self, i, x):
        # Pass the input tensor x through the i-th classification
        # header and store the output in the "confidence" variable
        confidence = self.classification_headers[i](x)
        # Permute the dimensions of the "confidence" tensor to
        # (batch size, width, height, number of classes) and make sure it is contiguous in memory.
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        
        #print("confidence shape = ", confidence.shape)
        # Reshape the "confidence" tensor to (batch size, width * height, number of classes) 
        # so that it can be easily processed in the next step.
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        #print("reshape = ", confidence.shape)
        # Pass the input tensor x through the i-th regression header 
        # and store the output in the "location" variable
        location = self.regression_headers[i](x)

        # Permute the dimensions of the "location" tensor to 
        # (batch size, width, height, 4) and make sure it is contiguous in memory.
        location = location.permute(0, 2, 3, 1).contiguous()

        # Reshape the "location" tensor to (batch size, width * height, 4) 
        # so that it can be easily processed in the next step.
        location = location.view(location.size(0), -1, 4)

        # Return the "confidence" and "location" tensors as a tuple
        return confidence, location

    # This function takes as input a trained base network and initializes 
    # the weights of the added layers and the headers with the Xavier initialization
    def init_from_base_net(self, model):
        # Load the state dict of the base network from the specified model file
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        # Initialize the source layer add-ons using xavier initialization
        self.source_layer_add_ons.apply(_xavier_init_)
        # Initialize the extras using xavier initialization
        self.extras.apply(_xavier_init_)
        # Initialize the classification headers using xavier initialization
        self.classification_headers.apply(_xavier_init_)
        # Initialize the regression headers using xavier initialization
        self.regression_headers.apply(_xavier_init_)

    # This function takes as input a pretrained SSD model and initializes 
    # the weights of the added layers and the headers with the Xavier 
    # initialization, while keeping the weights of the base network unchanged.
    def init_from_pretrained_ssd(self, model):
        # Load the state dict of the pretrained SSD model from the specified model file
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        # Filter out the state dict entries that correspond to the classification and regression headers
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        # Get the state dict of the current model
        model_dict = self.state_dict()
        # Update the current model's state dict with the state dict of the pretrained SSD model
        model_dict.update(state_dict)
        # Load the updated state dict into the current model
        self.load_state_dict(model_dict)
        # Initialize the classification headers using xavier initialization
        self.classification_headers.apply(_xavier_init_)
        # Initialize the regression headers using xavier initialization
        self.regression_headers.apply(_xavier_init_)


    def init(self):
        # Initialize the parameters of the base network using the xavier initialization method.
        self.base_net.apply(_xavier_init_)
        # Initialize the parameters of the source layer add-ons using the xavier initialization method.
        self.source_layer_add_ons.apply(_xavier_init_)

        # Initialize the parameters of the extras using the xavier initialization method.
        self.extras.apply(_xavier_init_)

        # Initialize the parameters of the classification headers using the xavier initialization method.
        self.classification_headers.apply(_xavier_init_)

        # Initialize the parameters of the regression headers using the xavier initialization method.
        self.regression_headers.apply(_xavier_init_)


    def load(self, model):
        # Load the model state from a file and store it in the network.
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        # Save the current state of the model to a file.
        torch.save(self.state_dict(), model_path)


# Class definition for MatchPrior
class MatchPrior(object):

    # Initialize the class with center_form_priors, center_variance, size_variance and iou_threshold
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        # Assign center_form_priors to class object attribute
        self.center_form_priors = center_form_priors
        # Convert center_form_priors to corner_form_priors using box_utils.center_form_to_corner_form
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        # Assign center_variance to class object attribute
        self.center_variance = center_variance
        # Assign size_variance to class object attribute
        self.size_variance = size_variance
        # Assign iou_threshold to class object attribute
        self.iou_threshold = iou_threshold

    # Define the callable method that takes gt_boxes and gt_labels as inputs
    def __call__(self, gt_boxes, gt_labels):
        # Convert numpy ndarray gt_boxes to torch tensor if gt_boxes is of type numpy ndarray
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        # Convert numpy ndarray gt_labels to torch tensor if gt_labels is of type numpy ndarray
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        # Assign prior boxes to gt boxes using box_utils.assign_priors with self.corner_form_priors and self.iou_threshold
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold)
        # Convert assigned boxes back to center form using box_utils.corner_form_to_center_form
        boxes = box_utils.corner_form_to_center_form(boxes)
        # Convert boxes to locations using box_utils.convert_boxes_to_locations with center_form_priors, center_variance and size_variance
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        # Return locations and labels
        return locations, labels




def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)