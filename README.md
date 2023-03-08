# TinyML_Det
This repository primarily incorporates [EtinyNet: Extremely Tiny Network for TinyML ](https://ojs.aaai.org/index.php/AAAI/article/view/20387) as its backbone and adds [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). to achieve high accuracy object detection. The implementation is influenced by [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).
The goal of this implementation is to create a lightweight and accurate object detection model suitable for embedded systems. <br>
Additionally, the repository also includes SSD/SSD-Lite implementations based on MobileNetV2, and Squeezenet.

## Codebase Structure 
(The codebase structure after completing the codebase construction process, which is shown below in [`Detailed Instruction`](#detailed-instruction))

    .
    ├── ...
    ├── config/                                # configuration file folder.
    │   ├── etinynet_ssd_config.py             # The config file to read configuration of network from .yaml.
    │   └── Config_etinynet_ssd.yaml           # The configuration parameters of the etinynet network
    |   └── squeezenet_ssd_config.py           # The config file to read configuration of network from .yaml.
    |   └── Config_squeezenet_ssd.yaml         # The configuration parameters of the squeezenet network
    |
    ├── dataset/                               # dataset file folder.
    |   ├── pascal_voc.py                      # dataloader file for images with annotation in pascal voc format
    |
    ├── models/                                # Models file folder.
    |   ├──nn/                                 # Backbone file folder + boxes loss function
    |       ├── etinynet.py                    # The feature extractor based on EtinyNet architecture 
    |       └── squeezenet.py                  # The feature extractor based on SqueezeNet architecture 
    |       └── multibox_loss.py               # Object file to compute classification and regression loss
    |   ├──ssd/                                # Backbone + SSD file folder
    |       ├── ssd.py                         # The SSD object file 
    |       └── Squeezenet_ssd.py              # The file to connect SqueezeNet feature extractor with extra-layers and headers,return SSD object
    |       └── etinynet_ssd.py                # The file to connect etinynet feature extractor with extra-layers and headers,return SSD object
    ├── utils/                                 # Utilities file folder.
    │   ├── box_utils.py                       # Functions and utilities related to localization boxes.
    │   ├── common_tools.py                    # Common tools for evaluation such as mAP.
    │   └── misc.py                            # file containig the function to save checkpoints and perform TF 
    └── tain_ssd.py                            # file to perform the training
    └── eval_ssd.py                            # file to perform evaluation on test data
    
    
    
## Requirement
1. Python 3.6+
2. OpenCV
3. PyTorch 1.4.0+
4. Tensorflow lite


## base network
![image](https://user-images.githubusercontent.com/110521665/218944268-bfb1682f-2327-485c-8d44-006441233249.png)<center>**<u>Figure 1</u>**:   *Backbone architecture of Etinynet1.0.*</center><p>&nbsp;</p>

### Model List

- We provide information on the number of parameters in each model, as well as the size of each parameter in three different floating-point formats: 32-bit, 16-bit, and 8-bit.
- In this project, we are exclusively working with images of one class, which is Human. The images are in RGB format and have a resolution of 256x256 pixels.

The model list:

| net_id                    | MACs      | #Parameters | Params size (fp32)  | Params size (fp16)  | Params size (fp8)  |
| ------------------------- | --------- | ----------- | ------------------- | ------------------- | ------------------ |
| *# baseline models*       |           |             |                     |                     |                    |
| mbv2-w0.35                | 72.537M   |   0.39M     |       1.48MB        |       0.74MB        |       0.37MB       |
| SqueezeNet1.1             | 346.897M  |   0.74M     |       2.76MB        |       1.39MB        |       0.70MB       |
| *# SSD baseline models*   |           |             |                     |                     |                    |
| mbv2-w0.35-SSD            | -----     |   -----     |       -----         |       ------        |       ------       |
| SqueezeNet1.1-SSD         | 407.178M  |   1.32M     |       5.01MB        |       2.50MB        |       1.25MB       |
| *# backbone models*       |           |             |                     |                     |                    |
| etn-w1.0                  | 182.673M  |   0.66M     |       2.51MB        |       1.26MB        |       0.63MB       |
| etn-w0.75                 | 112.626M  |   0.38M     |       1.45MB        |       0.73MB        |       0.36MB       |
| etn-w0.50                 | 58.831M   |   0.18M     |       0.68MB        |       0.34MB        |       0.17MB       |
| etn-w0.35                 | 33.669M   |   91.637K   |       0.35MB        |       0.18MB        |         90kB       |
| *# SSDLite models*        |           |             |                     |                     |                    |
| etn-w1.0-SSDLite          | 199.145M  |   0.97M     |       3.71MB        |       1.86MB        |       0.93MB       |
| etn-w0.75-SSDLite         | 125.394M  |   0.69M     |       2.64MB        |       1.32MB        |       0.66MB       |
| etn-w0.5-SSDLite          | 67.896M   |   0.49M     |       1.89MB        |       0.95MB        |       0.48MB       |
| etn-w0.35-SSDLite         | 40.470M   |   0.40M     |       1.53MB        |       0.77MB        |       0.39MB       |

### EtinyNet-SSDLite
Using the SSD-Lite and EtinyNet as a starting point, EtinyNet-SSDLite is an attempt to get a real time object detection algorithm on non-GPU computers and edge device such as STEM32 or ESP32. Since STEM32  by itself does not have enought computing capabilites, it requires more powerful base station or cloud to process the image/video information captured and detect objects in real-time. When EtinyNet-SSDLite is used, it eliminates the requirement of the base station and cloud process for real-time object detection.
![image](https://github.com/Chadlikouider/TinyML_Det/blob/main/assets/overview%20architecture%20of%20EtinyNetw1.0-SSDLite.png?raw=true) <center>**<u>Figure 2</u>**:   *Overview architecture of EtinyNetw1.0-SSDLite.*</center><p>&nbsp;</p>
