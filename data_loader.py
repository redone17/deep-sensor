# -*- coding:utf-8 -*-
# python2

'''
1. load data from txt files
2. reshape and preprocess the data
3. split train and test
4. encapsule data as datasets 
'''

import numpy as np
import torch
import torch.utils.data as data_utils

def load_data(*data_file):
    '''
    data_file is a tuple with 1 or 2 elements;
    first is vibration matrix,
    second can be rotating speed matrix.

    input two files only for rsn,
    load_data twice if you have two vibration signal files.

    rsn is a preprocessing for signal, 
    it requires rotating speed matrix.

    see paper "Convolutional Neural Networks for Fault Diagnosis
    Using Rotating Speed Normalized Vibration".
    '''
    data_arr = np.loadtxt(data_file[0])
    if len(data_file)>1:
        print('rsn used (see the paper).')
        rpm_arr = np.loadtxt(data_file[1])
        mean_rpm = np.mean(rpm_arr)
        data_arr = np.power(mean_rpm,2)*data_arr / (rpm_arr*rpm_arr)
    return np.expand_dims(data_arr, axis=1)

def load_label(label_file):
    '''
    load labels corrsponding to data
    '''
    return np.loadtxt(label_file, ndmin=1)

def split_arr(data, label, p=0.8):
    '''
    split data and label array into train and test partitions
    '''
    train = {'data':None, 'label':None}
    test = {'data':None, 'label':None}
    n_total = np.shape(data)[0]
    n_train = int(n_total*p)
    n_test = n_total - n_train
    idx = np.random.permutation(n_total)
    train_mask = idx[:n_train]
    test_mask = idx[n_total-n_test:]
    train['data'] = data[train_mask]
    train['label'] = label[train_mask]
    test['data'] = data[test_mask]
    test['label'] = label[test_mask]
    return train, test

def arr_to_dataset(data_arr, label_vec):
    '''
    convert numpy array into tensor dataset
    dataset = (X,y)
    '''
    data_ten = torch.from_numpy(data_arr).float()
    label_ten = torch.from_numpy(label_vec).long()
    dataset = data_utils.TensorDataset(data_ten, label_ten)
    return dataset

def fft_arr(arr):
    '''
    Fourier transform for signals in a Numpy array
    '''
    (n, _, l) = arr.shape
    amp = np.zeros((n,1,l/2))
    ang = np.zeros((n,1,l/2))
    for idx in range(n):
        ft = np.fft.fft(arr[idx,:,:])[:,:l/2]
        amp[idx] = np.absolute(ft)
        ang[idx] = np.angle(ft)
    return amp, ang

