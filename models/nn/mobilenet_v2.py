import torch
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
import math
# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(
                c * width_mult, 4 if width_mult == 0.1 else 8
            )
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, s if i == 0 else 1, t)
                )
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        #output_channel = (
        #    _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8)
        #    if width_mult > 1.0
        #    else 1280
        #)
        #self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



UID = "Lyken17"
model_urls = {
    "mbv2_100": f"https://github.com/{UID}/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_1.0-0c6065bc.pth",
    "mbv2_075": f"https://github.com/{UID}/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.75-dace9791.pth",
    "mbv2_050": f"https://github.com/{UID}/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth",
    "mbv2_035": f"https://github.com/{UID}/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.35-b2e15951.pth",
    "mbv2_025": f"https://github.com/{UID}/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.25-b61d2159.pth",
    "mbv2_010": f"https://github.com/{UID}/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.1-7d1d638a.pth",
}


def mbv2_100(pretrained = False, progress = True):

    model = MobileNetV2(width_mult = 1)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["etn_100"],
                                              progress=progress, map_location=torch.device('cpu'))
        # Remove classifier parameters from the state dictionary
        state_dict.pop('classifier.1.weight')
        state_dict.pop('classifier.1.bias')
        model.load_state_dict(state_dict, strict=False)
    return model

def mbv2_075(pretrained = False, progress = True):

    model = MobileNetV2(width_mult = 0.75)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["etn_075"],
                                              progress=progress, map_location=torch.device('cpu'))
        # Remove classifier parameters from the state dictionary
        state_dict.pop('classifier.1.weight')
        state_dict.pop('classifier.1.bias')
        model.load_state_dict(state_dict, strict=False)
    return model

def mbv2_050(pretrained = False, progress = True):

    model = MobileNetV2(width_mult = 0.5)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["etn_050"],
                                              progress=progress, map_location=torch.device('cpu'))
        # Remove classifier parameters from the state dictionary
        state_dict.pop('classifier.1.weight')
        state_dict.pop('classifier.1.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


def mbv2_035(pretrained = False, progress = True):

    model = MobileNetV2(width_mult = 0.35)
    if pretrained:
        state_dict = torch.load('D:\\master_project_codes\\mobilenetv2_0.35-b2e15951.pth', map_location=torch.device('cpu'))
        state_dict = torch.hub.load_state_dict_from_url(model_urls["mbv2_035"],
                                                        progress=progress, map_location=torch.device('cpu'))
        print(state_dict.keys())
        # Remove classifier parameters from the state dictionary
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        state_dict.pop('conv.1.num_batches_tracked')
        state_dict.pop('conv.1.running_var')
        state_dict.pop('conv.1.running_mean')
        state_dict.pop('conv.1.bias')
        state_dict.pop('conv.1.weight')
        state_dict.pop('conv.0.bias')
        state_dict.pop('conv.0.weight')
        model.load_state_dict(state_dict, strict=False)
    return model
if __name__ == "__main__": 
    model = MobileNetV2()
    output=model(torch.zeros((1, 3, 224, 224)))


    import subprocess
    from torch.utils.tensorboard import SummaryWriter
    logdir = r'C:\Users\wakcomputer\Downloads\logg'
    command = f"tensorboard --logdir={logdir}"

    subprocess.Popen(command.split())