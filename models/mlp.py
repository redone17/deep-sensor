'''
Multi-Layer Perceptron
a baseline
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Mudule):
    def __init__(self, in_features, n_class, n_hiddens=[500, 100], use_batchnorm=False, use_dropout=False):
        super(Net, self).__init__()
        self.name = vgg_name
        self.in_features = in_features
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.n_layers = len(n_hiddens) + 1
        self.n_hiddens = n_hiddens
        self.mlp = self._make_layers(in_features, n_hiddens, n_class)

    def _make_layers(in_features, n_hiddens, n_class):
        TODO

