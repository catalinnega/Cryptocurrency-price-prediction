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

def snr(close, window_length = 1):
    ### the higher the time frame the smoother the values
    buffer = np.zeros(window_length)
    S = 1
    nsr = []
    for i in range (len(close)):
        buffer = np.hstack(( close[i] , buffer[:-1]))
        M2 = ( 1 / buffer.shape[0] ) * ((buffer**2).sum())
        S = ( 1 / buffer.shape[0] ) * buffer.T.dot( np.tanh( np.sqrt(S) * buffer / ( M2 - 1 ) ) ) ** 2
        
        NSR_estimator = abs(20 * np.log10(S))
        nsr.append(NSR_estimator)
    return {'nsr' : np.array(nsr)}
                
            

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

NSR_estimator1, SNR_estimator, s = snr(close_prices, window_length = 2)

maxv = max(NSR_estimator1)
for i in range(len(NSR_estimator1)):
    NSR_estimator1[i] = NSR_estimator1[i] / maxv

NSR_estimator2, SNR_estimator, s = snr(close_prices, window_length = 4)

maxv = max(NSR_estimator2)
for i in range(len(NSR_estimator2)):
    NSR_estimator2[i] = NSR_estimator2[i] / maxv
#        

NSR_estimator3, SNR_estimator, s = snr(close_prices, window_length = 8)

maxv = max(NSR_estimator3)
for i in range(len(NSR_estimator3)):
    NSR_estimator3[i] = NSR_estimator3[i] / maxv
#        
NSR_estimator4, SNR_estimator, s = snr(close_prices, window_length = 12)

maxv = max(NSR_estimator4)
for i in range(len(NSR_estimator4)):
    NSR_estimator4[i] = NSR_estimator4[i] / maxv
#        
#            
#
#mfi1 = mta.money_flow_index(close_prices, high_prices, low_prices, volume, 14, feature_names = ['MFI'])
# 

#time_frame0 = 7
#atr_ema_7, atr_wilder_7 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame0)
#        
#time_frame1 = 14
#atr_ema_14, atr_wilder_14 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame1)
#
#time_frame2 = 28
#atr_ema_28, atr_wilder_28 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame2)
#
#time_frame3 = 56
#atr_ema_56, atr_wilder_56 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame3)
#
#
#
#
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
plt.ylabel('Price in USD')
plt.xlabel('15 minute candles')
plt.title('Bitcoin-USD historical data')
plt.legend(loc = 'best')
plt.show()

ax0 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4)
ax0.plot(NSR_estimator1[100:], color = 'blue', linewidth=1.5)
ax0.plot(NSR_estimator2[100:], color = 'orange', linewidth=1.5)
ax0.plot(NSR_estimator3[100:], color = 'green', linewidth=1.5)
ax0.plot(NSR_estimator4[100:], color = 'red', linewidth=1.5)
ax0.axhline(1, color='black')
ax0.axhline(0.999, color='black')
#ax0.fill_between(mfi2, len(mfi2), where=(mfi2>=80), facecolor='black', edgecolor='black', alpha=0.5)
#ax0.fill_between(mfi2, 20, where=(mfi2<=20), facecolor='black', edgecolor='black', alpha=0.5)
ax0.set_yticks([0.9996320677117784,1])
ax0.yaxis.label.set_color("black")
ax0.spines['bottom'].set_color("black")
ax0.spines['top'].set_color("black")
ax0.spines['left'].set_color("black")
ax0.spines['right'].set_color("black")
ax0.tick_params(axis='y', colors='black')
ax0.tick_params(axis='x', colors='black')
plt.ylabel('ATR')
plt.xlabel('15 minute candles')
plt.title('Average True Range')
plt.legend(loc = 'best')
plt.show()