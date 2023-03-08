"""
Pipeline for dataloader for datasets with Pascal VOC format annotation.
The structure of the dataset should be as follow:
    --Dataset_folder_name
            |__ annotation
                    |__ image_1.xml
                    |__ image_2.xml
                    |__ ........
            |__ image_1.JPG
            |__ image_2.JPG
            |__ .......
@author: CHADLI KOUIDER
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2

# Custom dataset class to load images with annotations in Pascal VOC format
class PascalVOCDataset(Dataset):
    # Initialize the dataset with the root directory and optional transform
    def __init__(self, root_dir, size, class_names,transform=None):
        self.root_dir = root_dir # root directory of the dataset
        self.transform = transform # optional transform to be applied to the image
        self.size = size # target image size
        self.img_list = [file for file in os.listdir(self.root_dir) if file.endswith('.JPG')] # list of all jpg files in the root directory

        class_names.insert(0,'BACKGROUND')
        self.class_names = tuple(class_names)


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
    
    def get_image(self, idx):
        img_path = os.path.join(self.root_dir, self.img_list[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.size)
        if self.transform:
            img, _ = self.transform(img)
        return img
    
    def get_annotations(self, idx):
        img_path = os.path.join(self.root_dir, self.img_list[idx])
        annotations, labels = self.parse_annotations(img_path,self.size)
        return annotations, labels

    # Returns the number of images in the dataset
    def __len__(self):
        return len(self.img_list)

    # Returns the image and its annotations for a given index
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_list[idx]) # full path of the image
        img = cv2.imread(img_path) # read the image
        img = cv2.resize(img,self.size) #resize the image
        annotations, labels = self.parse_annotations(img_path,self.size) # get the annotations and label for the image

        # Apply the transform to the image if it is specified
        if self.transform:
            img = self.transform(img)
            
        # Return in tuple form
        return img, annotations, labels

    # Parse the annotations for the image from its XML file
    def parse_annotations(self, img_path, size = None):
        annotations = [] # list to store the annotations
        labels = [] # list to store label
        xml_path = img_path.replace('.JPG', '.xml')
        xml_path = os.path.join(os.path.dirname(img_path), 'annotations', os.path.basename(xml_path)) # path of the XML file
        tree = ET.parse(xml_path) # parse the XML file
        root = tree.getroot() # get the root element of the XML file

        # get the original image size
        img = cv2.imread(img_path)
        orig_h, orig_w, _ = img.shape

        for obj in root.iter('object'): # iterate over the objects in the XML file
            bbox = obj.find('bndbox') # find the bounding box element
            xmin = int(bbox.find('xmin').text) # get the xmin coordinate of the bounding box
            ymin = int(bbox.find('ymin').text) # get the ymin coordinate of the bounding box
            xmax = int(bbox.find('xmax').text) # get the xmax coordinate of the bounding box
            ymax = int(bbox.find('ymax').text) # get the ymax coordinate of the bounding box

            # if the size is specified, rescale the coordinates accordingly
            if size is not None:
                scale_x = size[0] / orig_w
                scale_y = size[1] / orig_h
                xmin = int(scale_x * xmin)
                ymin = int(scale_y * ymin)
                xmax = int(scale_x * xmax)
                ymax = int(scale_y * ymax)
            annotations.append([xmin, ymin, xmax, ymax]) # add the bounding box coordinates to the annotations list
            # Get labels then do integer encoding
            label = obj.find('name').text.lower().strip()
            if label in self.class_dict: # check if label exists in the class dictionary
                labels.append(self.class_dict[label])
            else:
                print(f"Warning: Label {label} not found in class dictionary. Skipping.")
        return annotations, labels # return the annotations and labels
    """note to self: output is in form of : Annotations:[[[list xy_person1_img1],[xy_person2_img1],..],[xy_person1_img2],[xy_person2_img2],..]
                                            labels(integer encoding):  [[list labels of image 1],[list labels of image 2]]"""
    
    # Method to collate the data into mini-batches
    def collate(self, batch):
        
        if not batch:
            raise ValueError("Batch is empty.")

        item = batch[0]
        if len(item) != 3:
            raise ValueError(f"Expected 3 elements in item but got {len(item)}")
        
        images = [item[0] for item in batch] # get the images from the batch
        annotations = [item[1] for item in batch] # get the annotations from the batch
        labels = [item[2] for item in batch] # get the labels from the batch
        
        return torch.stack(images), annotations, labels # return the mini-batch as tensor of images and list of annotations



def plot_image(img, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation
        bbox = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red')
        ax.add_patch(bbox)
    plt.show()


if __name__ == "__main__": 
    # Create instances of the custom dataset class for train, validation and test datasets
    labels=['human']
    test_dataset = PascalVOCDataset(root_dir='D:\\master_project_codes\\heridal\\testImages',
                                    size= (300,300),class_names = labels, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=test_dataset.collate)
    
    # Loop through the mini-batches and plot the images with their bounding boxes
    #for i, (images, annotations, labels) in enumerate(test_loader):
        # Loop through the images in the mini-batch
    #    for j in range(images.shape[0]):
    #        img = images[j].permute(1, 2, 0).numpy() # Convert the image from Tensor to NumPy array and permute the dimensions
    #        annotations_j = annotations[j] # Get the annotations for the image
    #        plot_image(img, annotations_j) # Plot the image with its bounding boxes
    #    break # break the loop after accessing one mini-batch
    for i, (images, annotations, labels) in enumerate(test_loader):
        # Access the images and annotations in each mini-batch
        print("Mini-batch {0}".format(i))
        print("Images:", images.shape)
        print("Annotations:", annotations)
        print("labels:", labels)
        
        break # break the loop after accessing one mini-batch


