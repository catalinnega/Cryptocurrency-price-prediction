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
import mpl_finance


    
def bollinger_bands(close_prices, param_dict):
    if('skip' in param_dict):
        if(param_dict['skip']):
           return {}
    
    dict_return = {}
    for time_period in param_dict['periods']:
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
    
        dict_return.update({
                            'SMA_values': np.array(SMA_values),
                            'upperline_values': np.array(upperline_values),
                            'lowerline_values': np.array(lowerline_values),
                            'squeeze_values': np.array(squeeze_values),
                            'SMA_crossings': np.array(SMA_crossings)
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
      
bb_dict = bollinger_bands(close_prices, {'periods' : [14]})
SMA_values = bb_dict['SMA_values']
upperline_values = bb_dict['upperline_values']
lowerline_values = bb_dict['lowerline_values']
squeeze_values = bb_dict['squeeze_values']
SMA_crossings = bb_dict['SMA_crossings']


a = np.array([utc_time, open_princes,close_prices, high_prices, low_prices])

fig = plt.figure()
ax1 = plt.subplot2grid((6,4), (1,0), rowspan=3, colspan=4)
ax1.set_ylim([min(low_prices),max(high_prices)])
mpl_finance.candlestick2_ochl(ax1, 
                              open_princes[100:],
                              close_prices[100:],
                              high_prices[100:], 
                              low_prices[100:], 
                              colorup = 'green', 
                              colordown = 'red', 
                              width = 0.6,
                              alpha = 0.9)
ax1.plot(SMA_values[100:], label = 'SMA')
ax1.plot(upperline_values[100:], label = 'upperline')
ax1.plot(lowerline_values[100:], label = 'lowerline')
plt.text(155, 6375, "bollinger band window: "+ str(14) + ' candles')

plt.ylabel('Bitcoin price in USD')
plt.xlabel('15 minute candles')
plt.title('Bollinger bands')
plt.legend(loc = 'best')
plt.show()
#
ax0 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4)
ax0.plot(squeeze_values[100:], color = 'blue', linewidth=1.5, label = 'squeeze')
ax0.axhline(20, color='black')
ax0.axhline(80, color='black')
#ax0.fill_between(mfi2, len(mfi2), where=(mfi2>=80), facecolor='black', edgecolor='black', alpha=0.5)
#ax0.fill_between(mfi2, 20, where=(mfi2<=20), facecolor='black', edgecolor='black', alpha=0.5)
ax0.set_yticks([0,100])
ax0.yaxis.label.set_color("black")
ax0.spines['bottom'].set_color("black")
ax0.spines['top'].set_color("black")
ax0.spines['left'].set_color("black")
ax0.spines['right'].set_color("black")
ax0.tick_params(axis='y', colors='black')
ax0.tick_params(axis='x', colors='black')
plt.ylabel('squeeze')
plt.xlabel('15 minute candles')
plt.title('bollinger band squeeze')
plt.legend(loc = 'best')
plt.show()