# -*- coding:utf-8 -*-
# python3

'''
1. load data from txt files
2. reshape and preprocess the data
3. encapsule data as datasets 
'''

import numpy as np
def load_data(*data_file):
	'''
	data_file is a tuple with 1 or 2 elements;
	first is vibration matrix,
	second can be rotating speed matrix.
	
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
	return data_arr
	
def load_label(label_file):
	'''
	load labels corrsponding to data
	'''
	return np.loadtxt(label_file)
