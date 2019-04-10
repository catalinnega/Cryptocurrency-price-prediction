#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:55:13 2019

@author: catalin
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import mylib_TA as mta
import mpl_finance


def money_flow_index2(close_prices, high_prices, low_prices, volume, time_frame, feature_names = ['MFI']):
#The Money Flow Index (MFI) is a technical oscillator that uses price and volume for identifying overbought or oversold
    if('MFI' not in feature_names):
        return 0
    diff_circular_buffer_pos = np.zeros(time_frame)
    diff_circular_buffer_neg = np.zeros(time_frame)
    MFI_values = []
    raw_money_flow = 0
    diff = 0
    for i in range(len(close_prices)):
        typical_price = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
        raw_money_flow = typical_price * volume[i]
        if(i > 0):
            diff = close_prices[i] - close_prices[i-1]
        
        if(diff >= 0):
            diff_circular_buffer_pos = np.hstack((raw_money_flow, diff_circular_buffer_pos[:-1]))
            diff_circular_buffer_neg = np.hstack((0, diff_circular_buffer_neg[:-1]))
        else:
            diff_circular_buffer_pos = np.hstack((0, diff_circular_buffer_pos[:-1]))
            diff_circular_buffer_neg = np.hstack((raw_money_flow, diff_circular_buffer_neg[:-1]))            
        
        positive_money_flow = np.sum(diff_circular_buffer_pos)
        negative_money_flow = np.sum(diff_circular_buffer_neg)
        
        if(negative_money_flow < 0.0000001):
            MFI_values.append(100) ## happens 0.67% times in 144368 samples
        else:
            money_flow_ratio = positive_money_flow / negative_money_flow
            MFI_values.append(100 - (100/(1 + money_flow_ratio)))
    dict_MFI = {'MFI' : np.array(MFI_values)}
    return dict_MFI


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
#            
#
#mfi1 = mta.money_flow_index(close_prices, high_prices, low_prices, volume, 14, feature_names = ['MFI'])
#         
dict_MFI2 = money_flow_index2(close_prices, high_prices, low_prices, volume, 2, feature_names = ['MFI'])

mfi2 = dict_MFI2['MFI']
#
#plt.figure()
#plt.plot(close_prices, color = 'black', label = 'close prices')
##plt.plot(np.multiply(mfi2,100), color = 'blue', label = 'MFI')
##plt.legend(loc = 'best')
#
#
#plt.title('mfi')
###plt.plot(mn.denormalize_rescale(close_prices, data['min_prices'], data['max_prices']), label = 'close candles')
##plt.plot(mfi2, label = 'mfi')

a = np.array([utc_time, open_princes,close_prices, high_prices, low_prices])

fig = plt.figure()
ax1 = plt.subplot2grid((6,4), (1,0), rowspan=3, colspan=4)
mpl_finance.candlestick2_ochl(ax1, 
                              open_princes,
                              close_prices,
                              high_prices, 
                              low_prices, 
                              colorup = 'green', 
                              colordown = 'red', 
                              width = 0.6,
                              alpha = 0.9)
plt.ylabel('Price in USD')
plt.xlabel('15 minute candles')
plt.title('Bitcoin-USD historical data')
plt.show()

ax0 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4)
ax0.plot(mfi2, color = 'blue', linewidth=1.5)
ax0.axhline(70, color='black')
ax0.axhline(30, color='black')
#ax0.fill_between(mfi2, len(mfi2), where=(mfi2>=80), facecolor='black', edgecolor='black', alpha=0.5)
#ax0.fill_between(mfi2, 20, where=(mfi2<=20), facecolor='black', edgecolor='black', alpha=0.5)
ax0.set_yticks([20,80])
ax0.yaxis.label.set_color("black")
ax0.spines['bottom'].set_color("black")
ax0.spines['top'].set_color("black")
ax0.spines['left'].set_color("black")
ax0.spines['right'].set_color("black")
ax0.tick_params(axis='y', colors='black')
ax0.tick_params(axis='x', colors='black')
plt.ylabel('MFI')
plt.xlabel('15 minute candles')
plt.title('Money Flow Index')
plt.show()