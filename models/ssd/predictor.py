# import required libraries
import torch
from torchvision import transforms
from utils import box_utils
from utils.misc import Timer

# create Predictor class
class Predictor:
    
    # constructor to initialize Predictor class
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        # set the network model
        self.net = net
        # set image transform
        self.transform = transforms.Compose([transforms.Resize(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)
                                             ])
        # set IoU threshold
        self.iou_threshold = iou_threshold
        # set filter threshold
        self.filter_threshold = filter_threshold
        # set candidate size
        self.candidate_size = candidate_size
        # set NMS method
        self.nms_method = nms_method
        # set sigma
        self.sigma = sigma
        # check if device is given, otherwise set to CUDA if available, else CPU
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # move the model to the device
        self.net.to(self.device)
        # set the model to evaluation mode
        self.net.eval()
        # create timer object
        self.timer = Timer()

    # method to predict object detection from input image
    def predict(self, image, top_k=-1, prob_threshold=None):
        # set the device to CPU
        cpu_device = torch.device("cpu")
        # get the height, width and channels of the image
        height, width, _ = image.shape
        # apply the image transformation
        image = self.transform(image)
        # convert the image to tensor and move to the device
        images = image.unsqueeze(0)
        images = images.to(self.device)
        # turn off the gradient calculation
        with torch.no_grad():
            # start the timer
            self.timer.start()
            # forward pass the image to the network model
            scores, boxes = self.net.forward(images)
            # print the inference time
            print("Inference time: ", self.timer.end())
        # select the first image in the batch
        boxes = boxes[0]
        scores = scores[0]
        # set the filter threshold to default value if not given
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # move the boxes and scores to CPU
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        # create empty lists to store the predicted boxes and labels
        picked_box_probs = []
        picked_labels = []
        # loop through each class, starting from the second class
        for class_index in range(1, scores.size(1)):
            # get the scores of the current class
            probs = scores[:, class_index]
            # create a boolean mask based on the probability threshold
            mask = probs > prob_threshold
            # apply the mask to the probabilities
            probs = probs[mask]
            # if there are no boxes, skip to the next class
            if probs.size(0) == 0:
                continue
            # subset the boxes based on the mask
            subset_boxes = boxes[mask, :]
            # concatenate the boxes and probabilities along the 2nd axis
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            # apply NMS on the box_probs
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        # concatenate the box coordinates and probabilities for all classes
        picked_box_probs = torch.cat(picked_box_probs)
        # scale the coordinates back to the original image size
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        # return the final results, including the box coordinates, predicted labels, and probabilities
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]