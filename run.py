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
import torch.optim as optim
import torch.nn as nn
import data_loader
import iter_utils
import torch.utils.data as data_utils
from models import *

# load data
data_arr = data_loader.load_data('dis_data.txt')
# data_arr = data_arr[:,:,:240] # add for Ince's model
# amp, ang = data_loader.fft_arr(data_arr) # add for fft wdcnn
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
print('Number of training samples: {}'.format(len(train_loader.dataset)))

test_loader = data_utils.DataLoader(
    dataset = testset,
    batch_size = 200,
    shuffle = True,
    num_workers = 2,
)
print('Number of testing samples: {}'.format(len(test_loader.dataset)))

# make models
model = wdcnn.Net(1, 4)

# train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001 )
best_model, loss_curve = iter_utils.train(model, train_loader, criterion, optimizer,
    init_lr=0.001, decay_epoch=10, n_epoch=2)

# test
test_accuracy = iter_utils.test(best_model, test_loader)
print('Test accuracy: {:.4f}%'.format(100*test_accuracy))

# visualization
# TODO


