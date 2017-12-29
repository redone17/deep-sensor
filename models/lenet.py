'''
classical LeNet5 for 32x32 images with pytorch 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2,stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_channels, n_class, use_feature=False):
        super(Net, self).__init__()
        self.name = 'LeNet5'
        self.use_feature = use_feature
        self.b0 = nn.BatchNorm2d(in_channels)
        self.b1 = BasicBlock(in_channels, 6)
        self.b2 = BasicBlock(6, 16)
        self.n_features = 16*5*5
        self.fc = nn.Sequential(
                      nn.Linear(self.n_features, 120),
                      nn.ReLU(),
                      nn.Linear(120, 84),
                      nn.ReLU(),
                      nn.Linear( 84, n_class)
                  )

    def forward(self, x):
        f0 = self.b0(x)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        features = (f0,f1,f2)
        activations = self.fc(features[-1].view(-1, self.n_features))
        if self.use_feature:
            out = (activations, features)
        else:
            out = activations
        return out

'''
net = Net(1, 4)
x = torch.randn(10,1,32,32)
y = net(Variable(x))
print(y.data.size())
'''

