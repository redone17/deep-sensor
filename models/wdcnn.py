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
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_channels, n_class, use_feature=False):
        super(Net, self).__init__()
        self.name = 'WDCNN'
        self.use_feature = use_feature
        self.b0 = nn.BatchNorm1d(in_channels)
        self.b1 = BasicBlock(in_channels, 16, kernel_size=64, stride=16, padding=24)
        self.b2 = BasicBlock(16, 32)
        self.b3 = BasicBlock(32, 64)
        self.b4 = BasicBlock(64, 64)
        self.b5 = BasicBlock(64, 64, padding=0)
        self.n_features = 64*3
        self.fc = nn.Linear(self.n_features, n_class)

    def forward(self, x):
        f0 = self.b0(x)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        f5 = self.b5(f4)
        features = (f0,f1,f2,f3,f4,f5)
        activations = self.fc(features[-1].view(-1, self.n_features))
        if self.use_feature:
            out = (activations, features)
        else:
            out = activations
        return out
'''
net = Net(1, 4)
x = torch.randn(10,1,2048)
y = net(Variable(x))
print(y.size)
'''

