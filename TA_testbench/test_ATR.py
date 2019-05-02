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

def ATR(close, high, low, time_frame = 14, feature_names = ['ATR_EMA']):
    ### the higher the time frame the smoother the values
    user_input_tags = ['ATR_EMA', 'ATR_EMA_Wilder']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0,0
    
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
    return np.array(ATR_EMA), np.array(ATR_EMA_Wilder)

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
time_frame00 = 3
atr_ema_3, atr_wilder_3 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame00)

time_frame0 = 7
atr_ema_7, atr_wilder_7 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame0)
        
time_frame1 = 14
atr_ema_14, atr_wilder_14 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame1)

time_frame2 = 28
atr_ema_28, atr_wilder_28 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame2)

time_frame3 = 56
atr_ema_56, atr_wilder_56 = ATR(close_prices, high_prices, low_prices, time_frame = time_frame3)




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
ax0.plot(atr_wilder_3[100:], color = 'red', linewidth=1.5, label = 'timeframe: '+ str(time_frame00) + ' candles')
ax0.plot(atr_wilder_7[100:], color = 'purple', linewidth=1.5, label = 'timeframe: '+ str(time_frame0) + ' candles')
ax0.plot(atr_wilder_14[100:], color = 'blue', linewidth=1.5, label = 'timeframe: '+ str(time_frame1) + ' candles')
ax0.plot(atr_wilder_56[100:], color = 'green', linewidth=1.5, label = 'timeframe: '+ str(time_frame3) + ' candles')
ax0.axhline(15, color='black')
ax0.axhline(5, color='black')
#ax0.fill_between(mfi2, len(mfi2), where=(mfi2>=80), facecolor='black', edgecolor='black', alpha=0.5)
#ax0.fill_between(mfi2, 20, where=(mfi2<=20), facecolor='black', edgecolor='black', alpha=0.5)
ax0.set_yticks([0,20])
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