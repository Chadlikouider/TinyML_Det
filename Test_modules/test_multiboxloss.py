# -*- coding: utf-8 -*-
"""
Test the Multibox_loss module with fake data

@author: chadli kouider
"""
import torch

import numpy as np

from models.nn.multibox_loss import MultiboxLoss
# Create fake data for testing
num_classes = 2
num_priors = 5
batch_size = 3
confidence = torch.tensor(np.random.rand(batch_size, num_priors, num_classes), dtype=torch.float32)
predicted_locations = torch.tensor(np.random.rand(batch_size, num_priors, 4), dtype=torch.float32)
labels = torch.tensor(np.random.randint(0, 2, (batch_size, num_priors)), dtype=torch.long)
gt_locations = torch.tensor(np.random.rand(batch_size, num_priors, 4), dtype=torch.float32)
priors = torch.tensor(np.random.rand(num_priors, 4), dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the MultiboxLoss class
multibox_loss = MultiboxLoss(priors, 0.5, 3, 0.1, 0.2, device)

# Call the forward function
smooth_l1_loss, classification_loss = multibox_loss(confidence, predicted_locations, labels, gt_locations)

# Print the results
print("Smooth L1 loss: ", smooth_l1_loss.item())
print("Classification loss: ", classification_loss.item())
