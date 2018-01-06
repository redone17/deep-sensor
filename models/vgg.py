'''
VGG11/13/16/19 in Pytorch.
forked from kuangliu/pytorch-cifar
https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'cVGG19': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M'],
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
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=1)] # AvePool for bigger input sizes
        return nn.Sequential(*layers)

'''
net = Net('cVGG19',3,4)
x = torch.randn(10,3,32,32)
print(net(Variable(x)).size())
'''

