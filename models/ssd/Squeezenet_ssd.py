"""
    Implementation of Squeezenet-SSD, this file include the convolutional layers
    of classifiaction headers and regression headers as well as extra layers

@author: Chadli kouider
"""
import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from models.nn.squeezenet import squeezenet1_0, squeezenet1_1

from models.ssd.ssd import SSD
from config import squeezenet_ssd_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_squeezenet_ssd_lite(num_classes ,is_test=False):

    if config.version == '1_0':
        base_net = squeezenet1_0(config.pretrained).features  
    elif config.version == '1_1':
        base_net = squeezenet1_1(config.pretrained).features  
    else:
        raise ValueError('Invalid squeezenet version')

    source_layer_indexes = [
        7,
        12
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),#8
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),#4
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),#2
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)#1
        )
    ])
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[0] * 4, kernel_size=3, padding=1), # 1st feature map
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[1] * 4, kernel_size=3, padding=1), # 2nd feature map
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[2] * 4, kernel_size=3, padding=1), # 3rd feature map
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[3] * 4, kernel_size=3, padding=1), # 4th feature map
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[4] * 4, kernel_size=3, padding=1), # 5th feature map
        Conv2d(in_channels=256, out_channels=config.num_priors[5] * 4, kernel_size=1),                     # 6th feature map 
    ])
    classification_headers = ModuleList([
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[0] * num_classes, kernel_size=3, padding=1), # 1st feature map
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[1] * num_classes, kernel_size=3, padding=1), # 2nd feature map
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[2] * num_classes, kernel_size=3, padding=1), # 3rd feature map
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[3] * num_classes, kernel_size=3, padding=1), # 4th feature map
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[4] * num_classes, kernel_size=3, padding=1), # 5th feature map
        Conv2d(in_channels=256, out_channels=config.num_priors[5] * num_classes, kernel_size=1),                     # 6th feature map
    ])

    #def count_parameters(model):
    #    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #model = SSD(num_classes, base_net, source_layer_indexes, extras, classification_headers, regression_headers, is_test=is_test, config=config)  
    #print("Number of parameters in the model:", count_parameters(model))
    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)
