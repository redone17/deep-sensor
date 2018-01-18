'''
mport torch
import torch.nn as nn
from torch.autograd import Variable
DCNN for one-dimensional signals in Pytorch.
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

cfg = {
    'DCNN08': [8, 'M', 16, 'M', 32, 'M', 64, 'M', 128, 'M'],
    'DCNN11': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'DCNN13': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'DCNN16': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M'],
    'DCNN19': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M'],
}

class Net(nn.Module):
    def __init__(self, vgg_name, in_channels=1, n_class=5, use_feature=False):
        super(Net, self).__init__()
        self.name = vgg_name
        self.in_channels = in_channels
        self.use_feature = use_feature
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(128, n_class)

    def forward(self, x):
        features = self.features(x)
        activations = self.classifier(features.view(features.size(0), -1))
        if self.use_feature:
            out = (activations, features)
        else:
            out = activations
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool1d(kernel_size=64, stride=1)] # AvePool for bigger input sizes
        return nn.Sequential(*layers)

'''
net = Net('DCNN08',1,5)
x = torch.randn(10,1,2048)
print(net(Variable(x)).size())
'''

