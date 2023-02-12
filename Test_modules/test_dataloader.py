# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:27:44 2023

@author: Chadli Kouider
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PascalVOCDataset(Dataset):
    def __init__(self, data_dir, split, transforms=None):
        # Store the data directory, dataset split, and any image transforms
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        # Define the directories for images and annotations
        self.images_dir = os.path.join(data_dir, self.split,'JPGImages')
        self.annotations_dir = os.path.join(data_dir, self.split, 'labels')
        # Get a list of all the image filenames
        self.image_list = os.listdir(self.images_dir)

    def __getitem__(self, idx):
        # Get the image file name
        image_name = self.image_list[idx]
        # Construct the full image path
        image_path = os.path.join(self.images_dir, image_name)
        # Open and convert the image to RGB
        image = Image.open(image_path).convert("RGB")
        # Apply image transforms, if any
        if self.transforms:
            image = self.transforms(image)

        # Get the annotation file name
        annotation_name = image_name.split('.')[0] + '.xml'
        # Construct the full annotation path
        annotation_path = os.path.join(self.annotations_dir, annotation_name)
        # Parse the annotation XML file
        annotation_tree = ET.parse(annotation_path)
        annotation_root = annotation_tree.getroot()
        # Get object information from the annotation
        objects = []
        for obj in annotation_root.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['bbox'] = [int(obj.find('bndbox').find('xmin').text),
                                 int(obj.find('bndbox').find('ymin').text),
                                 int(obj.find('bndbox').find('xmax').text),
                                 int(obj.find('bndbox').find('ymax').text)]
            objects.append(obj_struct)

        return image, objects

    def __len__(self):
        return len(self.image_list)

# Usage
data_dir = 'D:\\master_project_codes\\heridal'
transforms = transforms.Compose([transforms.ToTensor()])

# train_dataset = PascalVOCDataset(data_dir, 'train', transforms)
# val_dataset = PascalVOCDataset(data_dir, 'val', transforms)
test_dataset = PascalVOCDataset(data_dir, 'testImages',transforms)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=True)



for i, (image, objects) in enumerate(test_loader):
    print(f'Image tensor: {image.shape}')
    print(f'Object information: {objects}')


import matplotlib.pyplot as plt
import numpy as np

def show_image_with_objects(image, objects):
    # Convert the image tensor to a numpy array
    image = np.transpose(image.numpy(), (1, 2, 0))
    # Plot the image
    plt.imshow(image)
    # Draw rectangles around the objects
    xmin, ymin, xmax, ymax = [x.item() for x in objects['bbox']]
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red'))
    # Show the plot
    plt.show()

for i, (image, objects) in enumerate(test_loader):
    #[x.item() for x in objects[0]['bbox']]
    for j in range(image.size()[0]):
        show_image_with_objects(image[j], objects[j])
    break


import matplotlib.pyplot as plt
import numpy as np

# Function to display the image with its annotations
def show_img_with_boxes(img, annotations):
    # Convert the image from tensor to numpy array
    img = img.permute(1, 2, 0).numpy()

    # Plot the image
    plt.imshow(img)

    # Plot the bounding boxes on the image
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red'))
    plt.show()

# Iterate over the test dataset and display the images with their annotations
for img, annotations in test_loader:
    for i in range(img.shape[0]):
        show_img_with_boxes(img[i], annotations[i])


import logging

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler to log the output to a file
file_handler = logging.FileHandler('detection.log')
file_handler.setLevel(logging.INFO)

# create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(file_handler)

# example usage
logger.info("Starting detection...")
# detect objects in the image

# log the loss
loss = 0.12
logger.info("Loss: {:.4f}".format(loss))

# log the accuracy
accuracy = 0.98
logger.info("Accuracy: {:.4f}".format(accuracy))

# log the precision
precision = 0.96
logger.info("Precision: {:.4f}".format(precision))

# log the recall
recall = 0.97
logger.info("Recall: {:.4f}".format(recall))

# log the F1 score
f1_score = 0.97
logger.info("F1 Score: {:.4f}".format(f1_score))

logger.info("Detection complete.")


import logging

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set up the logging with basicConfig()
logging.basicConfig(filename='detection.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# example usage
logger.info("Starting detection...")
# detect objects in the image

# log the loss
loss = 0.12
logger.info("Loss: {:.4f}".format(loss))

# log the accuracy
accuracy = 0.98
logger.info("Accuracy: {:.4f}".format(accuracy))

# log the precision
precision = 0.96
logger.info("Precision: {:.4f}".format(precision))

# log the recall
recall = 0.97
logger.info("Recall: {:.4f}".format(recall))

# log the F1 score
f1_score = 0.97
logger.info("F1 Score: {:.4f}".format(f1_score))
logger.info("Accuracy: {:.4f} Recall: {:.4f}".format(accuracy, recall))

logger.info("Detection complete.")


# Loop over the train data loader
for i, (images, annotations, labels) in enumerate(test_loader):
    # Access the images and annotations in each mini-batch
    print("Mini-batch {0}".format(i))
    print("Images:", images.shape)
    print("Annotations:", annotations)
    print("Annotations:", labels)
    break # break the loop after accessing one mini-batch