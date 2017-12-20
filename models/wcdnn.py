'''
WDCNN model with pytorch

Reference:
Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen and Zhujun Zhang
A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals
Sensors, MDPI
doi:10.3390/s17020425
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super(BasicBlock, self).__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm1d(out_channels)
    self.pool = nn.MaxPool1d(2,stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = F.relu(x)

class WDCNN(nn.Module):
    def __init__(self, in_channels, n_class, use_feature=False):
        super(WDCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels)
            BasicBlock(in_channels, 16, 64, 16, 24)
            BasicBlock(16, 32)
            BasicBlock(32, 64)
            BasicBlock(64, 64)
            BasicBlock(64, 64, padding=0)
        )
        self.n_features = 64*3
        self.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        features = self.conv(x)
        activations = self.fc(features.view(-1, n_features))
        if use_feature:
            out = (activations, features)
        else:
            out = (activations, None)
        return out

