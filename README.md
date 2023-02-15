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
![image](https://user-images.githubusercontent.com/110521665/218944268-bfb1682f-2327-485c-8d44-006441233249.png)<center>**<u>Figure 1</u>**:   *Backbone architecture of Etinynet1.0.*</center><p>&nbsp;</p>
![image](https://user-images.githubusercontent.com/110521665/218944368-12731790-d95e-4aa2-be53-9b62216829f4.png)<center>**<u>Figure 2</u>**:   *Backbone architecture of Etinynet0.75.*</center>
