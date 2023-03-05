"""
    Implementation of EtinyNet-SSD, this file include the convolutional layers
    of classifiaction headers and regression headers as well as extra layers

@author: CHADLI KOUIDER
"""
import torch
import torch.nn as nn
from models.ssd.ssd import SSD
from models.nn.etinynet import etinynet_100, etinynet_075, etinynet_050, etinynet_035

from config import etinynet_ssd_config as config

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                    padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def create_etinynet_ssd_lite(num_classes, is_test=False):
    
    # Set an EtinyNet backbone with specific width multiplier (1.0, 0.75, 0.5, 0.35)
    if config.width_multiplier == 1:
        base_net = etinynet_100(pretrained = config.pretrained).features
    elif config.width_multiplier == 0.75:
        base_net = etinynet_075(pretrained = config.pretrained).features
    elif config.width_multiplier == 0.5:
        base_net = etinynet_050(pretrained = config.pretrained).features
    elif config.width_multiplier == 0.35:
        base_net = etinynet_035(pretrained = config.pretrained).features
    else:
        raise ValueError('Invalid etinynet version')
    


    source_layer_indexes = [
        17
    ]
    extras = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2),
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        )
    ])

    num_classes = 2
    is_test = False
    regression_headers = nn.ModuleList([
        SeperableConv2d(in_channels=round(512 * config.width_multiplier), 
                        out_channels=config.num_priors[0] * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[1] * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[2] * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[3] * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[4] * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=config.num_priors[5] * 4, kernel_size=1),
    ])

    classification_headers = nn.ModuleList([
        SeperableConv2d(in_channels=round(512 * config.width_multiplier), 
                        out_channels=config.num_priors[0] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[1] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=config.num_priors[2] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[3] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=config.num_priors[4] * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=config.num_priors[5] * num_classes, kernel_size=1),
    ])

    #def count_parameters(model):
    #    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #model = SSD(num_classes, base_net, source_layer_indexes, extras, classification_headers, regression_headers, is_test=is_test, config=config)

    #print("Number of parameters in the model:", count_parameters(model))
    
    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)