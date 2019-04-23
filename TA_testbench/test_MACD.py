#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:55:13 2019

@author: catalin
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
#import mylib_TA as mta
import mpl_finance


def SMA_EMA(close_prices, periods, feature_names = ['SMA_12_day_values']):
    user_input_tags = ['SMA_12_day_values', 'EMA_12_day_values', 'SMA_26_day_values','EMA_26_day_values',
                       'MACD_line', 'MACD_signal_line', 'MACD_histogram', 'MACD_line_15_min', 
                       'MACD_signal_line_15_min', 'MACD_histogram_15_min', 'MACD_divergence_15_min',
                       'MACD_line_1h', 'MACD_signal_line_1h', 'MACD_histogram_1h', 'MACD_divergence_1h']
    ok = False
    for i in user_input_tags:
        if i in feature_names:
            ok = True
            break
    if(not ok):
        return 0,0
    
    SMA = 0
    SMA_values = []
    EMA_values = []
    time_frame = periods
    EMA_time_period = 2 / (periods + 1)
    EMA = 0
    for i in range(len(close_prices)):
        if(i < periods):
            ## add mean values
            SMA += close_prices[i] / periods
            EMA = SMA
        else:
            SMA = SMA - ((close_prices[i - time_frame] - close_prices[i]) / periods)     
            EMA = (close_prices[i] - EMA) * EMA_time_period + EMA
        SMA_values.append(SMA)
        EMA_values.append(EMA)
    return np.array(SMA_values), np.array(EMA_values)

def MACD(close_prices, ema_26_period = 26, ema_12_period = 12, feature_names = ['MACD_line_15_min']):
    user_input_tags = ['MACD_line', 'MACD_signal_line', 'MACD_histogram', 
                       'MACD_line_15_min', 'MACD_signal_line_15_min', 'MACD_histogram_15_min', 'MACD_divergence_15_min',
                       'MACD_line_1h', 'MACD_signal_line_1h', 'MACD_histogram_1h', 'MACD_divergence_1h']
    ok = False
    for i in user_input_tags:
        if(i in feature_names):
            ok = True
            break
    if(not ok):
        return 0,0,0,0
    
    s, EMA_26_day_values = SMA_EMA(close_prices, ema_26_period)
    s, EMA_12_day_values = SMA_EMA(close_prices, ema_12_period)
    MACD_line = np.zeros(len(EMA_26_day_values))
    for i in range(len(EMA_26_day_values)):
        MACD_line[i] = EMA_12_day_values[i] - EMA_26_day_values[i]
        if(MACD_line[i] > 10):
            MACD_line[i] = 0
    SMA_9_day_MACD, EMA_9_day_MACD = SMA_EMA(MACD_line, 9)
    
    MACD_histogram = np.zeros(len(EMA_9_day_MACD))
    for i in range(len(EMA_9_day_MACD)):
        MACD_histogram[i] = MACD_line[i] - EMA_9_day_MACD[i]
    
    ## get divergences
    divergence_values = 0
    MACD_signal_line = EMA_9_day_MACD
    return np.array(MACD_line), np.array(MACD_signal_line), np.array(MACD_histogram), np.array(divergence_values)


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
sma, ema = SMA_EMA(close_prices, 12)
macd_line, macd_signal_line, h ,d = MACD(close_prices, 26, 12)


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
ax1.plot(sma[100:], label = 'SMA')
ax1.plot(ema[100:], label = 'EMA')
plt.ylabel('Price in USD')
plt.xlabel('15 minute candles')
plt.title('Bitcoin-USD historical data')
plt.legend(loc = 'best')
plt.show()

ax0 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4)
ax0.plot(macd_line[100:], color = 'blue', linewidth=1.5, label = 'line')
ax0.plot(macd_signal_line[100:], color = 'red', linewidth=1.5, label = 'signal line')
ax0.axhline(5, color='black')
ax0.axhline(-5, color='black')
#ax0.fill_between(mfi2, len(mfi2), where=(mfi2>=80), facecolor='black', edgecolor='black', alpha=0.5)
#ax0.fill_between(mfi2, 20, where=(mfi2<=20), facecolor='black', edgecolor='black', alpha=0.5)
ax0.set_yticks([-5,5])
ax0.yaxis.label.set_color("black")
ax0.spines['bottom'].set_color("black")
ax0.spines['top'].set_color("black")
ax0.spines['left'].set_color("black")
ax0.spines['right'].set_color("black")
ax0.tick_params(axis='y', colors='black')
ax0.tick_params(axis='x', colors='black')
plt.ylabel('MACD')
plt.xlabel('15 minute candles')
plt.title('MACD')
plt.legend(loc = 'best')
plt.show()