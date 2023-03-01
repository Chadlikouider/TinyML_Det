"""
    Implementation of EtinyNet-SSD, this file include the convolutional layers
    of classifiaction headers and regression headers as well as extra layers

@author: CHADLI KOUIDER
"""
import torch
import torch.nn as nn
from models.ssd.ssd import SSD
from models.nn.etinynet import EtinyNet


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

def create_etinynet_ssd_lite(num_classes, width_mult=1.0, is_test=False):
    base_net = EtinyNet(multiplier=width_mult).features
    
    source_layer_indexes = []
    extras = nn.ModuleList([])

    regression_headers = nn.ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = nn.ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)