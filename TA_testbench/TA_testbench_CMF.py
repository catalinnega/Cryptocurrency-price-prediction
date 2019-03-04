#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:18:50 2018

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
        


directory = '/home/catalin/databases/klines_2014-2018_15min/'
hard_coded_file_number = 0

data = md.get_dataset_with_descriptors(concatenate_datasets_preproc_flag = True, 
                                       preproc_constant = 0.99, 
                                       normalization_method = "rescale",
                                       dataset_directory = directory,
                                       hard_coded_file_number = hard_coded_file_number)
X = data['preprocessed_data'] ## this will be used for training
X_unprocessed = data['data']

close = X_unprocessed[:,1]
high = X_unprocessed[:,2]
low = X_unprocessed[:,3]
volume = X_unprocessed[:,4]



def Chaikin_money_flow(close, high, low, volume, time_period = 21):
    MF_volume_circular_buffer = np.zeros(time_period)
    volume_circular_buffer = np.zeros(time_period)
    CMF = []
    for i in range(len(close)):
        MF_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        MF_volume = MF_multiplier * volume[i]
        MF_volume_circular_buffer = np.hstack((MF_volume, MF_volume_circular_buffer[:-1]))
        volume_circular_buffer = np.hstack((volume[i], volume_circular_buffer[:-1]))
        CMF.append(np.sum(MF_volume_circular_buffer) / np.sum(volume_circular_buffer))
    return CMF

plt.figure()
plt.plot(CMF)
plt.plot(close)
