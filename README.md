# conv-rotor

A deep learning framework for fault diagnositc with [PyTorch](http://pytorch.org/).
[![](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/redone17/conv-rotor) [![](https://img.shields.io/badge/python-2.7.13-blue.svg)](https://www.python.org/)  [![](https://img.shields.io/badge/license-BSD3-ff69b4.svg)](https://github.com/redone17/conv-rotor/blob/master/LICENSE)

## requirements: 
* numpy (>= 1.13.1)
* pytorch (>= 0.1.12.post2)

## usage
1. data: 
    * text file for data: \# signal_samples x \# signal_length
    * text file for label: \# signal_samples x 1
2. ``` $ python run.py ```

## models
* [WDCNN](https://github.com/redone17/conv-rotor/blob/master/models/wdcnn.py) from paper: [A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals](http://dx.doi.org/10.3390/s17020425)
