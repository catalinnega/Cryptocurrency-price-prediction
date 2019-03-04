#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 23:12:56 2018

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

#EMA_time_period = 14
#EMA = (close_prices[i] - EMA) * EMA_time_period + EMA
def ATR(close, high, low, time_frame = 14):
    ATR_EMA = []
    ATR_EMA_Wilder = []
    lam_EMA = 2 / (time_frame + 1)
    lam_Wilder = 1 / time_frame
    for i in range(len(close)):
            a1 = high[i] - low[i]
            a2 = abs(high[i] - close[i])
            a3 = abs(low[i] - close[i])
            TR = max(a1,a2,a3)
            if(i == 0):
                ATR_EMA.append(TR)
                ATR_EMA_Wilder.append(TR)
            else:
                ATR_EMA.append(lam_EMA * TR + (1 - lam_EMA) * ATR_EMA[-1])
                ATR_EMA_Wilder.append(lam_Wilder * TR + (1 - lam_Wilder) * ATR_EMA[-1])
    return ATR_EMA, ATR_EMA_Wilder

ATR_EMA, ATR_EMA_Wilder =  ATR(close, high, low, time_frame = 14)
plt.figure()
plt.plot(ATR_EMA)
plt.plot(ATR_EMA_Wilder)
plt.plot(close)