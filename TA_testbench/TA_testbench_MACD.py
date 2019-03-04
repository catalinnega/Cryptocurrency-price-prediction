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
    

plot = False
def MACD_divergence(close_prices, MACD_line, divergence_window):
    MACD_lowest_low = 99999999999
    close_lowest_low = 99999999999
    close_highest_high = -9999999
    MACD_highest_high = -9999999
    bullish_div = []
    bearish_div = []
    RESET_TIMER = 100
    timer_bear = RESET_TIMER
    timer_bull = RESET_TIMER
    for i in range(len(MACD_line)):
        ## bullish div
        if(i <= divergence_window):
            bearish_div.append(0)
            bullish_div.append(0)
        else:
            close_low = np.min(close_prices[(i - divergence_window): i])
            MACD_low = np.min(MACD_line[(i - divergence_window): i])
            
            if(MACD_low < MACD_lowest_low):
                MACD_lowest_low = MACD_low
            
            if(close_low < close_lowest_low):
                close_lowest_low = close_low
                timer_bear = RESET_TIMER
                if(MACD_low > MACD_lowest_low):
                    bullish_div.append(1)
                else:
                    bullish_div.append(0)
            else:
                bullish_div.append(0)
                
            ## bearish div
            close_high = np.max(close_prices[(i - divergence_window): i])
            MACD_high = np.max(MACD_line[(i - divergence_window): i])
    
            if(MACD_high > MACD_highest_high):
                MACD_highest_high = MACD_high
                
            if(close_high > close_highest_high):
                close_highest_high = close_high
                timer_bull = RESET_TIMER
                if(MACD_high < MACD_highest_high):
                    bearish_div.append(1)
                else:
                    bearish_div.append(0)
            else:
                bearish_div.append(0)
            
            if(timer_bear <= 0):
                MACD_lowest_low = MACD_low
                close_lowest_low = close_low
            if(timer_bull <= 0):
                MACD_highest_high = MACD_high
                close_highest_high = close_high
            timer_bear -= 1      
            timer_bull -= 1                
    return bullish_div, bearish_div

def SMA_EMA(close_prices, time_frame):
    SMA = 0
    SMA_values = []
    EMA_values = []
    EMA_time_period = 2 / (time_frame + 1)
    for i in range(len(close_prices)):
        if(i <= time_frame):
            ## add mean values
            SMA += close_prices[i] / time_frame
            EMA = SMA
        else:
            SMA = SMA - ((close_prices[i - time_frame] - close_prices[i]) / time_frame)     
            EMA = (close_prices[i] - EMA) * EMA_time_period + EMA
        SMA_values.append(SMA)
        EMA_values.append(EMA)
    return SMA_values , EMA_values

def MACD(close_prices, EMA_12_day_values, EMA_26_day_values, divergence_window):
    MACD_line = [EMA_12_day_values[i] - EMA_26_day_values[i] for i in range(len(EMA_26_day_values))] 
    SMA_9_day_MACD, EMA_9_day_MACD = SMA_EMA(MACD_line, 26)
    MACD_histogram = [MACD_line[i] - EMA_9_day_MACD[i] for i in range(len(EMA_9_day_MACD))]
    
    ## get divergences
    bullish_div, bearish_div = MACD_divergence(close_prices, MACD_line, divergence_window)
    MACD_signal_line = EMA_9_day_MACD
    return MACD_line, MACD_signal_line, MACD_histogram, bullish_div, bearish_div

def plot_vertical_lines(data, desired_color = 'brown'):
    ## even data samples
    data1 = [data[i] if ((i % 2) == 0) else 0 for i in range(len(data))]
    #odd data samples
    data2 = [data[i] if ((i % 2) != 0) else 0 for i in range(len(data))]
    #plot them
    plt.plot(data1, color = desired_color)
    plt.plot(data2, color = desired_color)

        

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
one_day_in_15_min_candles = 15 * 4 * 24
twelve_hrs_in_15_min_candles = 15 * 4 * 12
six_hrs_in_15_min_candles = 15 * 4 * 6
three_hrs_in_15_min_candles = 15 * 4 * 3
one_hr_in_15_min_candles = 15 * 4 


SMA_12_day_values, EMA_12_day_values = SMA_EMA(close_prices, 12 * 1)
SMA_26_day_values, EMA_26_day_values = SMA_EMA(close_prices, 26 * 1)

MACD_line, MACD_signal_line, MACD_histogram, bullish_div, bearish_div = MACD(close_prices, 
                                                           EMA_12_day_values, 
                                                           EMA_26_day_values,  
                                                           divergence_window = 100)
if(plot):
    plot_SMA = False
    plot_EMA = False
    plot_MACD = True
    
    plt.close('all')
    plt.figure()
    plt.title('Moving averages')
    plt.plot(close_prices, label = 'close prices')
    if(plot_SMA):
        plt.plot(SMA_12_day_values, label = 'SMA 12 days')
        plt.plot(SMA_26_day_values, label = 'SMA 26 days')
    if(plot_EMA or plot_MACD):
        if(plot_EMA):
            plt.plot(EMA_12_day_values, label = 'EMA 12 days')    
            plt.plot(EMA_26_day_values, label = 'EMA 26 days')
        plt.plot(MACD_signal_line, label = 'EMA of MACD line 9 days (Signal line)')
    if(plot_MACD):
        plt.plot(MACD_line, label = 'MACD line')
        plot_vertical_lines(MACD_histogram)
        #plt.hist2d(MACD_histogram, label = 'MACD histogram(MACD line - Signal line)')
        plt.plot(np.multiply(bullish_div, 0.2), label = 'bullish divergence signal')
        plt.plot(np.multiply(bearish_div, 0.2), label = 'bearish divergence signal')
    plt.legend(loc='best')
    plt.show()


    
        