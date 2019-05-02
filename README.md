# deep-sensor

A [PyTorch](http://pytorch.org/) framework for sensory signals in machinery fault diagnostics.

[![](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/redone17/conv-rotor) [![](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)  [![](https://img.shields.io/badge/license-BSD3-ff69b4.svg)](https://github.com/redone17/conv-rotor/blob/master/LICENSE)

## requirements: 
* numpy (>= 1.13.1)
* pytorch (>= 0.1.12.post2)

## usage
1. data: 
    * .txt file for data: \# signal_samples x \# signal_length
    * .txt file for label: \# signal_samples x 1
    * An example of fake dataset with 100 samples and labels are in [toydata](./toydata). 
2. ``` $ python run.py ```

## models
### 1D
* [WDCNN](https://github.com/redone17/conv-rotor/blob/master/models/wdcnn.py) from paper: [A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals](http://dx.doi.org/10.3390/s17020425)
* [Ince's](https://github.com/redone17/deep-sensor/blob/master/models/ince.py) from paper: [Real-Time Motor Fault Detection by 1-D Convolutional Neural Networks](https://doi.org/10.1109/TIE.2016.2582729)
* [1DCNN](https://github.com/redone17/deep-sensor/blob/master/models/dcnn.py) from paper: [Convolutional Neural Networks for Fault Diagnosis Using Rotating Speed Normalized Vibration](https://doi.org/10.1007/978-3-030-11220-2_8) (cite as)
~~~~
@InProceedings{10.1007/978-3-030-11220-2_8,
author="Wei, Dongdong
and Wang, KeSheng
and Heyns, Stephan
and Zuo, Ming J.",
editor="Fernandez Del Rincon, Alfonso
and Viadero Rueda, Fernando
and Chaari, Fakher
and Zimroz, Radoslaw
and Haddar, Mohamed",
title="Convolutional Neural Networks for Fault Diagnosis Using Rotating Speed Normalized Vibration",
booktitle="Advances in Condition Monitoring of Machinery in Non-Stationary Operations",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="67--76",
abstract="Fault diagnosis is vital for the health management of rotating machinery. The non-stationary working conditions is one of the major challenges in this field. The key is to extract working-condition-invariant but fault-discriminative features. Traditional methods use expert knowledge on the machines and signal processing to extract fault features from vibration signals manually. This paper regards this issue as a domain adaption problem and utilizes deep learning technique to learn fault discriminative features automatically. We teach deep Convolutional Neural Networks to pronounce diagnostic results from raw vibration data and propose a Rotating Speed Normalization method to improve the domain adaption ability of the neural network models. A case study of rotor crack diagnosis under non-stationary and ever-changing rotating speeds is presented. Using 95600 signal segments, we compare the diagnostic performance of ours and reported Convolutional Neural Network models. The results show that our model gives solid diagnostic accuracy from non-stationary vibration signals, and the proposed Rotating Speed Normalization method can successfully boost the performance of all investigated CNN models.",
isbn="978-3-030-11220-2"
}
~~~~

### 2D
* [LeNet5](https://github.com/redone17/deep-sensor/blob/master/models/lenet.py) for spectrogram of signals
* [VGG](https://github.com/redone17/deep-sensor/blob/master/models/vgg.py) forked from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

