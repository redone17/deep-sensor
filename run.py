# -*- coding:utf-8 -*-
# python2

'''
mian .py for conv-rotor project
0. initialize
1. get data;
2. make models;
3. train;
4. test;
5. visualize;
'''

# initialize
from __future__ import print_function, division 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import time
import copy
import os

import data_loader
import iter_utils
import torch.utils.data as data_utils
from models import wcdnn

# load data
data_arr = data_loader.load_data('dis_data.txt','rpm_data.txt')
label_vec = data_loader.load_label('label_vec.txt')
train_dict, test_dict = data_loader.split_arr(data_arr, label_vec)
trainset = data_loader.arr_to_dataset(train_dict['data'], train_dict['label'])
testset = data_loader.arr_to_dataset(test_dict['data'], test_dict['label'])

train_loader = data_utils.DataLoader(
    dataset = trainset,
    batch_size = 200,
    shuffle = True,
    num_workers = 2,
)
print(len(train_loader.dataset))

test_loader = data_utils.DataLoader(
    dataset = testset,
    batch_size = 200,
    shuffle = True,
    num_workers = 2,
)
print(len(test_loader.dataset))

# make models
model = wcdnn.WDCNN(4, )


