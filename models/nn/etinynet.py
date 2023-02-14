"""
This architecture is taken from: EtinyNet: Extremely Tiny Network for TinyML 
<https://doi.org/10.1609/aaai.v36i4.20387 > paper

@author: CHADLI KOUIDER
"""
import torch
import torch.nn as nn
import numpy as np

__all__ = ['EtinyNet']


###############################################################################
# Functions (Private)
###############################################################################

def conv_3x3_bn(inp, oup, stride, pad=0, active = True, batch_norm=True):
    """This function takes an input and apply 2D convolution with kernal (3x3)
    and a stride 2 then do batch normalization and finally pass it through a 
    ReLU function
        Args:
            inp: number of input channels
            out: number of output channels
            stride: stride number
        Returns:
            """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding=pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    """This function takes an input and apply 2D convolution with kernal (1x1)
    and a stride 1 then do batch normalization and finally pass it through a 
    ReLU function
        Args:
            inp: number of input channels
            out: number of output channels
        Returns:
            new convolutional layer containing:
                -convolution kernel=1,stride=1,padding=1
                -BatchNormalization
                -ReLU activation function
            """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def add_conv(out, channels=1, in_channels=0, kernel=1, stride=1, pad=0, num_group=1,
             active=True, norm=True):
    """
    add_conv: A helper function to add a convolutional layer to a given neural network.
            follwing a certain pattern depending on the args and flags

    args:
        out (list): A list that represents the layers in the neural network(nn.Sequential).
        channels (int): The number of channels in the output data.
        in_channels(int): The number of channels in the output data.
        kernel (int): The size of the kernel for the convolutional layer.
        stride (int): The stride of the convolutional layer.
        pad (int): The padding of the convolutional layer.
        num_group (int): The number of groups for the convolutional layer(for depthwise
                                                                          convolution).
        active (bool): A flag to indicate whether to include the ReLU activation function.
        norm (bool): A flag to indicate whether to include normalization.

    return:
        None. The out list is modified in-place to include the new convolutional layer.
    """
    
    out.append(nn.Conv2d(in_channels = in_channels, out_channels = channels, 
                         kernel_size = kernel, stride = stride, padding=pad, 
                         groups=num_group, bias=False))
    
    if active:
        out.append(nn.ReLU(inplace=True))
        
    if norm:
        out.append(nn.BatchNorm2d(channels))

        
###############################################################################
# Internal module (Private)
###############################################################################
class LinearBottleneck(nn.Module):
    r"""A modified version of LinearBottleneck used in MobileNetV2 model
    from the`"Inverted Residuals and Linear Bottlenecks:
     Mobile Networks for Classification, Detection and Segmentation"
   <https://arxiv.org/abs/1801.04381>`_ paper.
   Parameters
   ----------
   in_channels : int
       Number of input channels.
   channels : int
       Number of output channels.
   stride : int
       stride
   norm_layer : object
       Normalization layer used (default: :class:`torch.nn.BatchNorm2d`)
   """
    def __init__(self, channels, stride, shortcut):
        super(LinearBottleneck,self).__init__()
        
        self.use_shortcut1 = (stride == 1 and channels[0] == channels[1] and shortcut)
        self.use_shortcut2 = shortcut
        
        
        # out1 = dconv+BN => 1x1 conv
        self.out1 = nn.Sequential() # 1x1
        add_conv(self.out1, in_channels=channels[0], channels = channels[0],
                 kernel=3, stride=stride, pad=1, num_group=channels[0],
                 active = False)
        add_conv(self.out1, in_channels=channels[0], channels=channels[1],
                 active = True)
        
        
        # out2 = dconv+BN+ReLU 
        self.out2 = nn.Sequential()#1x1
        add_conv(self.out2,
                 in_channels=channels[1], channels=channels[1], kernel=3,
                 stride=1, pad=1, num_group=channels[1], active=True)

    def forward(self,x):
        out = self.out1(x)
        if self.use_shortcut1:
            out = out + x
        x1 = out
        out = self.out2(out)
        if self.use_shortcut2:
            if self.use_shortcut1:
                out = out + x + x1
            else:
                out = out + x1
        return out


###############################################################################
# Backbone network
###############################################################################
class EtinyNet(nn.Module):
    def __init__(self, multiplier=1., n_class=1000):
        super(EtinyNet, self).__init__()
        
        self.features = []
        add_conv(self.features, int(32 * multiplier), in_channels=3, kernel=3,
                 stride = (2,2), pad=1)
        self.features.append(nn.MaxPool2d((2,2)))
        
        # make the linear depthwise block (LB)
        channels_group = [[32, 32],  [32, 32], [32, 32], [32, 32],
                                  [32, 128], [128, 128], [128, 128], [128, 128]]
        strides = [1, 1, 1, 1] + [2, 1, 1, 1]
        shortcuts = [0, 0, 0, 0] + [0, 0, 0, 0]
        for cg, s, sc in zip(channels_group, strides, shortcuts):
            self.features.append(LinearBottleneck(channels=np.int32(np.array(cg)*multiplier),
                                                  stride=s, shortcut=sc))
        
        # Make the first the dense linear depthwise block (DLB)
        
        
        channels_group = [[128, 192], [192, 192], [192, 192]]
        strides = [2, 1, 1]
        shortcuts = [1, 1, 1]
        for cg, s, sc in zip(channels_group, strides, shortcuts):
            self.features.append(LinearBottleneck(channels=np.int32(np.array(cg)*multiplier),
                                                  stride=s, shortcut=sc))
        
        # Make the second the dense linear depthwise block (DLB)
        
        
        channels_group = [[192, 256], [256, 256], [256, 512]]
        strides = [2, 1, 1]
        shortcuts = [1,1,1]
        for cg, s, sc in zip(channels_group, strides, shortcuts):
            self.features.append(LinearBottleneck(channels=np.int32(np.array(cg)*multiplier),
                                                  stride=s, shortcut=sc))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features) 
    
    def forward(self, x):
        
        x = self.features(x)
        
        return x


if __name__ == "__main__": 
    model = EtinyNet()
    output = model(torch.zeros((1, 3, 224, 224)))
    print(output)
    