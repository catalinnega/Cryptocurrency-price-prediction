#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 13:21:04 2019

@author: catalin
"""

import numpy as np
import matplotlib.pyplot as plt

import mylib_dataset as md
import mylib_normalize as mn

#from matplotlib.finance import candlestick
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import talib as ta
import math
import time
        

start_time = time.time()
directory = '/home/catalin/databases/klines_2014-2018_15min/'
hard_coded_file_number = 0

data = md.get_dataset_with_descriptors(concatenate_datasets_preproc_flag = True, 
                                       preproc_constant = 0.99, 
                                       normalization_method = "rescale",
                                       dataset_directory = directory,
                                       hard_coded_file_number = hard_coded_file_number)
X = data['preprocessed_data'] ## this will be used for training
X_unprocessed = data['data']

open_prices = X_unprocessed[:,0]
close_prices = X_unprocessed[:,1]
high_prices = X_unprocessed[:,2]
low_prices = X_unprocessed[:,3]
volume = X_unprocessed[:,4]

#def DCT_transform_II(vector):
#	result = []
#	factor = math.pi / len(vector)
#	for i in range(len(vector)):
#		summ = 0.0
#		for (j, val) in enumerate(vector):
#			summ += val * math.cos((j + 0.5) * i * factor)
#		result.append(summ)
#	return np.array(result)

def DCT_transform_II(vector):
	if vector.ndim != 1:
		raise ValueError()
	n = vector.size
	if n == 1:
		return vector.copy()
	elif n == 0 or n % 2 != 0:
		raise ValueError()
	else:
		half = n // 2
		gamma = vector[ : half]
		delta = vector[n - 1 : half - 1 : -1]
		alpha = DCT_transform_II(gamma + delta)
		beta  = DCT_transform_II((gamma - delta) / (np.cos(np.arange(0.5, half + 0.5) * (np.pi / n)) * 2.0))
		result = np.zeros_like(vector)
		result[0 : : 2] = alpha
		result[1 : : 2] = beta
		result[1 : n - 1 : 2] += beta[1 : ]
		return result

def RLS_indicator(close, time_period = 128, lam = 0.9, delta = 1, smoothing = 0.9, dct_transform = False):
    sample_circular_buffer = np.zeros(time_period)
    dct_sample_circular_buffer = np.zeros(time_period)
    rls_filter = np.zeros(time_period)
    rls_indicator_error = []
    rls_indicator_output = []
    rls_smoothed_indicator = []
    P = (delta**(-1)) * np.identity(time_period);
    for i in range(len(close)):
        if(dct_transform == "type II"):
            dct_sample_circular_buffer = np.hstack((close[i], dct_sample_circular_buffer[:-1]))
            sample_circular_buffer = DCT_transform_II(dct_sample_circular_buffer)
        else:
            sample_circular_buffer = np.hstack((close[i], sample_circular_buffer[:-1]))
        z = np.matmul(sample_circular_buffer,P)
        k = np.matmul(P, sample_circular_buffer.T) / (lam + np.matmul(z, sample_circular_buffer.T))
        y = np.matmul(rls_filter, sample_circular_buffer.T)
        error = close[i] - y
        rls_filter = rls_filter + np.multiply(k, np.conj(error))
        P = (1 / lam) * (P - np.multiply(np.matmul(z, k), P))
        rls_indicator_error.append(error)
        rls_indicator_output.append(y)
        if(i==0):
             rls_smoothed_indicator.append(rls_indicator_error[-1])
        else:
             rls_smoothed_indicator.append(rls_smoothed_indicator[-1] * smoothing + \
                                           (1 - smoothing) * rls_indicator_error[-1])
    return np.array(rls_indicator_error), np.array(rls_indicator_output)

rls_indicator_error, rls_indicator_output = RLS_indicator(close_prices,
                                                          time_period = 256,
                                                          lam = 0.9, 
                                                          delta = 1/(np.var(close_prices)*100),
                                                          dct_transform = "type II")

plt.figure()
plt.title('RLS indicator. filter length = 128, lam = 0.9, delta = 1')
plt.plot(rls_indicator_error,label =  'RLS error signal')
plt.plot(rls_indicator_output,label =  'RLS filter output')
plt.plot(close_prices, label = 'close prices')
plt.legend(loc='best')
plt.show()

end_time = time.time()
print('process time: '+ str(end_time-start_time))