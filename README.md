# TinyML_Det
This repository primarily incorporates [EtinyNet: Extremely Tiny Network for TinyML ](https://ojs.aaai.org/index.php/AAAI/article/view/20387) as its backbone and adds [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). to achieve high accuracy object detection. The implementation is influenced by [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).
The goal of this implementation is to create a lightweight and accurate object detection model suitable for embedded systems. <br>
Additionally, the repository also includes SSD/SSD-Lite implementations based on MobileNetV2, Squeezenet, and Squeeznext.


## Requirement
1. Python 3.6+
2. OpenCV
3. PyTorch 1.4.0+
4. Tensorflow lite


## base network

![backboneEtinynet1 0](https://user-images.githubusercontent.com/110521665/218684229-e5bc6e2b-49fb-4a56-bec4-f2768fd7c43c.png)
![backboneEtinynet0 75](https://user-images.githubusercontent.com/110521665/218686290-2bb646f9-4cf6-46fe-a5a7-08c8e140c645.png)<center>**<u>Figure</u>**:   *(left)Backbone architecture of Etinynet1.0.(right)Backbone architecture of Etinynet0.75.*</center>
