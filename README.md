# TinyML_Det
This repository primarily incorporates [EtinyNet: Extremely Tiny Network for TinyML ](https://ojs.aaai.org/index.php/AAAI/article/view/20387) as its backbone and adds [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). to achieve high accuracy object detection. The implementation is influenced by [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).
The goal of this implementation is to create a lightweight and accurate object detection model suitable for embedded systems. <br>
Additionally, the repository also includes SSD/SSD-Lite implementations based on MobileNetV2, Squeezenet, and Squeeznext.

## Codebase Structure 
(The codebase structure after completing the codebase construction process, which is shown below in [`Detailed Instruction`](#detailed-instruction))

    .
    ├── ...
    ├── config/                                # configuration file folder.
    │   ├── config.py                          # The config file to set configuration of network.
    │   └── ...                                
    ├── dataset/                               # dataset file folder.
    |   ├── pascal_voc.py                      # dataloader file for images with annotation in pascal voc format
    ├── models/                                # Models file folder.
    |   ├──nn/                                 # backbone file folder
    |   ├──ssd/                                # Backbone + SSD file folder
    ├── utils/                                 # utilities file folder.
    │   ├── main.cpp                           # Main source file.
    │   ├── TinyEngine                         # TinyEngine folder.
    │   └── ...                                
    └── ...
    
    
    
## Requirement
1. Python 3.6+
2. OpenCV
3. PyTorch 1.4.0+
4. Tensorflow lite


## base network
![image](https://user-images.githubusercontent.com/110521665/218944268-bfb1682f-2327-485c-8d44-006441233249.png)<center>**<u>Figure 1</u>**:   *Backbone architecture of Etinynet1.0.*</center><p>&nbsp;</p>

### Model List

- We provide information on the number of parameters in each model, as well as the size of each parameter in three different floating-point formats: 32-bit, 16-bit, and 8-bit.
- Throughout this project we're only using 1 class (Human)

The model list:

| net_id                    | MACs   | #Parameters | Params size (fp32)  | Params size (fp16)  | Params size (fp8)  |
| ------------------------- | ------ | ----------- | ------------------- | ------------------- | ------------------ |
| *# baseline models*       |        |             |                     |                     |                    |
| mbv2-w0.35                | ----M  |   0.39M     |       1.48MB        |       0.74MB        |       0.37MB       |
| SqueezeNet1.1             | ----M  |   0.74M     |       2.76MB        |       1.39MB        |       0.70MB       |
| *# SSD baseline models*   |        |             |                     |                     |                    |
| mbv2-w0.35-SSD            | ----M  |   ----M     |       -----         |       ----MB        |       ----MB       |
| SqueezeNet1.1-SSD         | ----M  |   1.32M     |       5.01MB        |       2.50MB        |       1.25MB       |
| *# backbone models*       |        |             |                     |                     |                    |
| etn-w1.0                  | ---M   |   0.66M     |       2.51MB        |       1.26MB        |       0.63MB       |
| etn-w0.75                 | ----M  |   0.38M     |       1.45MB        |       0.73MB        |       0.36MB       |
| etn-w0.50                 | ----M  |   0.18M     |       0.68MB        |       0.34MB        |       0.17MB       |
| etn-w0.35                 | ----M  |   0.09M     |       0.35MB        |       0.18MB        |         90kB       |
| *# SSDLite models*        |        |             |                     |                     |                    |
| etn-w1.0-SSDLite          | ---M   |   0.97M     |       3.71MB        |       1.86MB        |       0.93MB       |
| etn-w0.75-SSDLite         | ----M  |   0.69M     |       2.64MB        |       1.32MB        |       0.66MB       |
| etn-w0.5-SSDLite          | ----M  |   0.49M     |       1.89MB        |       0.95MB        |       0.48MB       |
| etn-w0.35-SSDLite         | ----M  |   0.40M     |       1.53MB        |       0.77MB        |       0.39MB       |

