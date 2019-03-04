#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:53:12 2018

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

open_prices = X_unprocessed[:,0]
close_prices = X_unprocessed[:,1]
high_prices = X_unprocessed[:,2]
low_prices = X_unprocessed[:,3]
volume = X_unprocessed[:,4]

time_period = 20
def bollinger_bands(close_prices, time_period):
    sample_circular_buffer = np.zeros(time_period)
    SMA_values = []
    upperline_values = []
    lowerline_values = []
    squeeze_values = []
    SMA_crossings = [0]
    for i in range(len(close_prices)):
        sample_circular_buffer = np.hstack((close_prices[i], sample_circular_buffer[:-1]))
        mean = (np.sum(sample_circular_buffer) ) / time_period
        SMA_values.append(mean)
        
        std = np.sqrt(np.sum( (np.array(sample_circular_buffer) - mean) ** 2 ) / time_period)
        
        upperline_values.append(mean + 2 * std)
        lowerline_values.append(mean - 2 * std)
        squeeze_values.append(upperline_values[-1] - lowerline_values[-1])
        if(i>0):
            if((close_prices[i] < SMA_values[-1]) and (close_prices[i - 1] > SMA_values[-2])):
                SMA_crossings.append(-1) ## bearish reversal
            elif ((close_prices[i] > SMA_values[-1]) and (close_prices[i - 1] < SMA_values[-2])):
                SMA_crossings.append(1) ## bullish reversal
            else:
                if(SMA_crossings[-1] == 1):
                    SMA_crossings.append(1)
                elif(SMA_crossings[-1] == -1):
                    SMA_crossings.append(-1)
                else:
                    SMA_crossings.append(0)
    return SMA_values, upperline_values, lowerline_values, squeeze_values, SMA_crossings   

SMA_values, upperline_values, lowerline_values, squeeze_values, SMA_crossings = bollinger_bands(close_prices, time_period)  
plt.close('all')
plt.plot(close_prices)
plt.plot(SMA_values)
plt.plot(upperline_values)
plt.plot(lowerline_values)
plt.plot(SMA_crossings)
    


