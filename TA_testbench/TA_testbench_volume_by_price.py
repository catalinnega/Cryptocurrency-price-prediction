#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:13:11 2018

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

close_prices = X_unprocessed[:,1]
volume = X_unprocessed[:,4]


def VBP(close_prices, volume, bins = 12, lam = 0.99):
    max_price = np.max(close_prices)
    bin_values = [i*max_price/bins for i in range(bins + 1)]
    
    VBP = []
    bin_volume = np.zeros(len(bin_values))
    slope_VBP = []
    slope_VPB_smooth = []
    for i in range(len(close_prices)):
        for j in range(1,len(bin_values)):
            if close_prices[i] <= bin_values[j]:
                bin_volume[j-1] += volume[i]
                if(VBP != []):
                    VBP.append(bin_volume[j-1]/np.max(VBP))
                else:
                    VBP.append(bin_volume[j-1])
                if(i == 0):
                    slope_VBP.append(0)
                    slope_VPB_smooth.append(0)
                else:
                    max1 = np.max(slope_VBP)
                    max2 = np.max(slope_VPB_smooth)
                    if(max1 > 0):
                        slope_VBP.append((VBP[-1] - VBP[-2])/max1)
                    else:
                        slope_VBP.append((VBP[-1] - VBP[-2]))
                    if(max2):
                        slope_VPB_smooth.append((slope_VBP[-2] * lam + slope_VBP[-1] * (1-lam))/max2)
                    else:
                        slope_VPB_smooth.append((slope_VBP[-2] * lam + slope_VBP[-1] * (1-lam)))
                break
    
    return np.array(VBP), np.array(slope_VBP), np.array(slope_VPB_smooth)

VBP, slope_VBP = VBP(close_prices,volume, bins = 12)
plt.figure()
plt.plot(close_prices)     
plt.plot(VBP)
plt.plot(slope_VBP)
lam = 0.99
for i in range(len(slope_VBP)):
    slope_VBP[i] = lam * slope_VBP[i-1] + (1-lam) * slope_VBP[i]
plt.plot(slope_VBP)
    
    
plot_levels = True
if(plot_levels):
    plt.figure()
    plt.plot(close_prices)
    for i in range(len(bin_values)):
        plt.axhline(bin_values[i])
#        plt.plot(bin_volume[i])

scale = len(close_prices) / np.max(bin_volume)
ceva = plt.hist2d(bin_volume*scale, bin_values, len(bin_volume))
for i in ceva[2]:
    plt.axhline(i, color = 'r')