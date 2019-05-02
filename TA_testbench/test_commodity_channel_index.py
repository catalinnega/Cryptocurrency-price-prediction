#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:55:13 2019

@author: catalin
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
#import mylib_TA2 as mta2
import mylib_dataset2 as md2
import mpl_finance


    
def commodity_channel_index(close_prices, high_prices, low_prices, param_dict):
    if('skip' in param_dict):
        if(param_dict['skip']):
           return {}
    
    dict_return = {}
    constant = param_dict['constant'] 
    for time_period in param_dict['periods']:
        typical_price_circular_buffer = np.zeros(time_period)
        CCI_values = []
#        CCI_threshold_values = []
        #constant = 0.015
        for i in range(len(close_prices)):
            typical_price = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
            typical_price_circular_buffer = np.hstack((typical_price, typical_price_circular_buffer[:-1]))
            TP_SMA = np.sum(typical_price_circular_buffer) / time_period
            mean_deviation = np.sum(abs(typical_price_circular_buffer - TP_SMA)) / time_period
            if(mean_deviation < 0.000001):
                CCI_values.append(300)
            else:
                CCI_values.append((typical_price - TP_SMA) / (constant * mean_deviation))
#            if(CCI_values[-1] > 190):
#                CCI_threshold_values.append(1)
#            elif(CCI_values[-1] < -190):
#                CCI_threshold_values.append(-1)
#            else:
#                 CCI_threshold_values.append(0)
        dict_return.update({
                           'CCI_values_' + str(time_period): np.array(CCI_values),
#                           'CCI_threshold_values' + str(time_period): np.array(CCI_threshold_values),
                           })
    return dict_return


with open('/home/catalin/git_workspace/disertatie/unprocessed_btc_data.pkl', 'rb') as f:
    data = pickle.load(f)

test_window_len = 1000
zeros_len = 700
open_princes = data['open'][-test_window_len:-zeros_len]
close_prices = data['close'][-test_window_len:-zeros_len]
high_prices = data['high'][-test_window_len:-zeros_len]
low_prices = data['low'][-test_window_len:-zeros_len]
volume = data['volume'][-test_window_len:-zeros_len]
utc_time = data['UTC'][-test_window_len:-zeros_len]
   

dict_data = md2.get_database_data(dataset_directory = '', 
                                  normalization_method = 'rescale',
                                  datetime_interval = {},
                                  preproc_constant = 0, 
                                  lookback = 0,
                                  input_noise_debug = False,
                                  dataset_type = 'dataset2')
open_prices = dict_data['X'][:,0]
close_prices = dict_data['X'][:,1]
high_prices = dict_data['X'][:,2]
low_prices = dict_data['X'][:,3]
volume = dict_data['X'][:,4]


period = 14   
cci_dict = commodity_channel_index(close_prices, high_prices, low_prices, {'periods' : [period],
                                                                           'constant': 0.015})
cci = cci_dict['CCI_values_' + str(period)]
#
#fig = plt.figure()
#ax1 = plt.subplot2grid((6,4), (1,0), rowspan=3, colspan=4)
#ax1.set_ylim([min(low_prices),max(high_prices)])
#mpl_finance.candlestick2_ochl(ax1, 
#                              open_princes[100:],
#                              close_prices[100:],
#                              high_prices[100:], 
#                              low_prices[100:], 
#                              colorup = 'green', 
#                              colordown = 'red', 
#                              width = 0.6,
#                              alpha = 0.9)
#plt.text(155, 6375, "CCI window: "+ str(period) + ' candles')
#
#plt.ylabel('Bitcoin price in USD')
#plt.xlabel('15 minute candles')
#plt.title('Bitcoin-USD historical data')
#plt.legend(loc = 'best')
#plt.show()
##
#ax0 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4)
#ax0.plot(cci[100:], color = 'blue', linewidth=1.5, label = 'squeeze')
#ax0.axhline(-150, color='black')
#ax0.axhline(150, color='black')
##ax0.fill_between(mfi2, len(mfi2), where=(mfi2>=80), facecolor='black', edgecolor='black', alpha=0.5)
##ax0.fill_between(mfi2, 20, where=(mfi2<=20), facecolor='black', edgecolor='black', alpha=0.5)
#ax0.set_yticks([-300,300])
#ax0.yaxis.label.set_color("black")
#ax0.spines['bottom'].set_color("black")
#ax0.spines['top'].set_color("black")
#ax0.spines['left'].set_color("black")
#ax0.spines['right'].set_color("black")
#ax0.tick_params(axis='y', colors='black')
#ax0.tick_params(axis='x', colors='black')
#plt.ylabel('CCI')
#plt.xlabel('15 minute candles')
#plt.title('Commodity Channel Index')
#plt.legend(loc = 'best')
#plt.show()