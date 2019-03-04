#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 22:13:28 2018

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

close_prices = X_unprocessed[:,0]

def RSI(close_prices, time_frame = '15min'):
    if(time_frame == '15min'):
        time_window = 14
    elif(time_frame == '30min'):
        time_window = 14 * 2
    elif(time_frame == '1h'):
        time_window = 14 * 4
    elif(time_frame == '3h'):
        time_window = 14 * 4 * 3
    elif(time_frame == '6h'):
        time_window = 14 * 4 * 6
    elif(time_frame == '12h'):
        time_window = 14 * 4 * 12
    elif(time_frame == '1d'):
        time_window = 14 * 4 * 24
    elif(time_frame == '3d'):
        time_window = 14 * 4 * 24 * 3
    elif(time_frame == '1w'):
        time_window = 14 * 4 * 24 * 7
        
    adva = 0    
    decl = 0
    avg_gain = 0
    avg_loss = 0
    RSI_values = []
    for i in range(len(close_prices)):
        if(i <= 0):
            RSI_values.append(0)
        elif((i < time_window) and (i > 0)):
            RSI_values.append(0)
            if(close_prices[i] > close_prices[i - 1]):
                adva += close_prices[i] - close_prices[i-1]
            else:
                decl += close_prices[i - 1] - close_prices[i]
            if(i == (time_window - 1) ):
                avg_gain = adva / time_window
                avg_loss = decl / time_window
                RS = avg_gain / avg_loss
                adva = 0
                decl = 0
                RSI_values[-1] = 100 - 100 / (1 + RS)
        else:
            diff = close_prices[i] - close_prices[i - 1]
            if(diff > 0):
                current_gain = diff
                current_loss = 0
            else:
                current_loss = abs(diff)
                current_gain = 0
                
            avg_gain = (avg_gain * (time_window - 1) + current_gain) / time_window
            avg_loss = (avg_loss * (time_window - 1) + current_loss) / time_window
            RS = avg_gain / avg_loss
            RSI_values.append(100 - 100 / (1 + RS))
     
    return np.array(RSI_values)

def RSI_divergence(close_prices, RSI_values, divergence_window):
    RSI_lowest_low = 99999999999
    close_lowest_low = 99999999999
    close_highest_high = -9999999
    RSI_highest_high = -9999999
    bullish_div = []
    bearish_div = []
    RESET_TIMER = 100
    timer_bear = RESET_TIMER
    timer_bull = RESET_TIMER
    for i in range(len(RSI_values)):
        ## bullish div
        if(i <= divergence_window):
            bearish_div.append(0)
            bullish_div.append(0)
        else:
            close_low = np.min(close_prices[(i - divergence_window + 1): (i + 1)])
            RSI_low = np.min(RSI_values[(i - divergence_window + 1): (i + 1)])
            
            if(RSI_low < RSI_lowest_low):
                RSI_lowest_low = RSI_low
            
            if(close_low < close_lowest_low):
                close_lowest_low = close_low
                timer_bear = RESET_TIMER
                if(RSI_low > RSI_lowest_low):
                    bullish_div.append(1)
                else:
                    bullish_div.append(0)
            else:
                bullish_div.append(0)
                
            ## bearish div
            close_high = np.max(close_prices[(i - divergence_window + 1): (i + 1)])
            RSI_high = np.max(RSI_values[(i - divergence_window + 1): (i + 1)])
    
            if(RSI_high > RSI_highest_high):
                RSI_highest_high = RSI_high
                
            if(close_high > close_highest_high):
                close_highest_high = close_high
                timer_bull = RESET_TIMER
                if(RSI_high < RSI_highest_high):
                    bearish_div.append(-1)
                else:
                    bearish_div.append(0)
            else:
                bearish_div.append(0)
            
            if(timer_bear <= 0):
                RSI_lowest_low = RSI_low
                close_lowest_low = close_low
            if(timer_bull <= 0):
                RSI_highest_high = RSI_high
                close_highest_high = close_high
            timer_bear -= 1      
            timer_bull -= 1       
    divergence_values = np.add(bullish_div, bearish_div)    
    return divergence_values

def stochastic_RSI(RSI_values, time_frame):
    stochastic_RSI_values = []
    for i in range(len(RSI_values)):
        if(i <= time_frame):
            stochastic_RSI_values.append(0)
        else:
            lowest_low_RSI = np.min(RSI_values[(i - time_frame + 1) : (i + 1)])
            highest_high_RSI =  np.max(RSI_values[(i - time_frame + 1) : (i + 1)])
            diff = highest_high_RSI -  lowest_low_RSI
            if(diff < 0):
                print('wtf' + str(i))
                stochastic_RSI_values.append(1)
            else:
                if((RSI_values[i] - lowest_low_RSI) < 0):
                    print('wtf' + str(i))
                stochastic_RSI_values.append((RSI_values[i] - lowest_low_RSI) / (highest_high_RSI -  lowest_low_RSI))
    return np.multiply(np.array(stochastic_RSI_values), 100)
ceva1 = RSI(close_prices, '15min')
ceva2 = stochastic_RSI(ceva1, time_frame = 14)
#ceva2 = ta.RSI(close_prices)

from matplotlib.finance import candlestick_ochl
plt.close('all')
plt.figure()

plt.title('RSI')
plt.plot(mn.denormalize_rescale(close_prices, data['min_prices'], data['max_prices']), label = 'close candles')
plt.plot(RSI_values, label = 'RSI')

ceva = np.zeros([np.shape(X)[0], 6])
open_prices = X_unprocessed[:,3]
close_prices = X_unprocessed[:,0]
high_prices = X_unprocessed[:,1]
low_prices = X_unprocessed[:,2]
ceva[:,0] = np.arange(np.shape(X)[0]) 
ceva[:,1] = open_prices
ceva[:,2] = close_prices
ceva[:,3] = high_prices
ceva[:,4] = low_prices
ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
candlestick_ochl(ax1, ceva, width=0.2, colorup='green', colordown='red', alpha=1)
#candlestick(ax1, close_prices, width=.6, colorup='#53c156', colordown='#ff1717')

#ax1.plot(close_prices)
#ax1.plot(date[-SP:],Av1[-SP:],'#e1edf9',label=Label1, linewidth=1.5)
#ax1.plot(date[-SP:],Av2[-SP:],'#4ee6fd',label=Label2, linewidth=1.5)

ax1.grid(True, color='w')
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.yaxis.label.set_color("w")
ax1.spines['bottom'].set_color("#5998ff")
ax1.spines['top'].set_color("#5998ff")
ax1.spines['left'].set_color("#5998ff")
ax1.spines['right'].set_color("#5998ff")
ax1.tick_params(axis='y', colors='w')
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
ax1.tick_params(axis='x', colors='w')
plt.ylabel('Stock price and Volume')

ax0 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')

rsiCol = '#c1f9f7'
posCol = '#386d13'
negCol = '#8f2020'

ax0.plot(RSI_values, rsiCol, linewidth=1.5)
ax0.axhline(70, color=negCol)
ax0.axhline(30, color=posCol)
ax0.fill_between( RSI_values, 70, where=(RSI_values>=70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
ax0.fill_between( RSI_values, 30, where=(RSI_values<=30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
ax0.set_yticks([30,70])
ax0.yaxis.label.set_color("w")
ax0.spines['bottom'].set_color("#5998ff")
ax0.spines['top'].set_color("#5998ff")
ax0.spines['left'].set_color("#5998ff")
ax0.spines['right'].set_color("#5998ff")
ax0.tick_params(axis='y', colors='w')
ax0.tick_params(axis='x', colors='w')
plt.ylabel('RSI')
#plt.legend(loc='best')
#plt.show()