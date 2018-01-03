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

## initialize
from __future__ import print_function, division 
import torch.optim as optim
import torch.nn as nn
import data_loader
import iter_utils
import torch.utils.data as data_utils
from models import *

## load data
data_arr_01 = data_loader.load_data('fluc03/vib_data.txt')
# data_arr_03 = data_loader.load_data('fluc03/vib_data.txt')
# data_arr = data_arr[:,:,:240] # add for Ince's model
# amp, ang = data_loader.fft_arr(data_arr) # add for fft wdcnn
spectrogram = data_loader.stft_arr(data_arr_01) # add for stft-LeNet
label_vec = data_loader.load_label('label_vec.txt')
train_dict_01, test_dict_01 = data_loader.split_arr(spectrogram, label_vec)
# train_dict_03, test_dict_03 = data_loader.split_arr(data_arr_03, label_vec)
trainset_01 = data_loader.arr_to_dataset(train_dict_01['data'], train_dict_01['label'])
testset_01 = data_loader.arr_to_dataset(test_dict_01['data'], test_dict_01['label'])
# trainset_03 = data_loader.arr_to_dataset(train_dict_03['data'], train_dict_03['label'])
# testset_03 = data_loader.arr_to_dataset(test_dict_03['data'], test_dict_03['label'])

train_loader = data_utils.DataLoader(dataset = trainset_01, batch_size = 200, shuffle = True, num_workers = 2)
test_loader = data_utils.DataLoader(dataset = testset_01, batch_size = 200, shuffle = True, num_workers = 2)
print('Number of training samples: {}'.format(len(train_loader.dataset)))
print('Number of testing samples: {}'.format(len(test_loader.dataset)))
print( )

## make models
model = lenet.Net(1, 4)

## train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.000)
best_model, loss_curve = iter_utils.train(model, train_loader, criterion, optimizer,
    init_lr=0.0001, decay_epoch=5, n_epoch=30)

# test
test_accuracy = iter_utils.test(best_model, test_loader)
print('Test accuracy: {:.4f}%'.format(100*test_accuracy))

## visualization
# TODO

