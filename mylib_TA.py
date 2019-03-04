#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 01:08:14 2018

@author: catalin
"""
import numpy as np
import talib
import math
import pickle
from datetime import timedelta, date
import mylib_dataset as md

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def DCT_transform_II(vector):
	if vector.ndim != 1:
		raise ValueError()
	n = vector.size
	if n == 1:
		return vector.copy()
	elif n == 0 or n % 2 != 0:
		raise ValueError()
	else:
		half = n // 2
		gamma = vector[ : half]
		delta = vector[n - 1 : half - 1 : -1]
		alpha = DCT_transform_II(gamma + delta)
		beta  = DCT_transform_II((gamma - delta) / (np.cos(np.arange(0.5, half + 0.5) * (np.pi / n)) * 2.0))
		result = np.zeros_like(vector)
		result[0 : : 2] = alpha
		result[1 : : 2] = beta
		result[1 : n - 1 : 2] += beta[1 : ]
		return result

def Accum_distrib(dataset_dict, feature_names = ['ADL']):
    if('ADL' not in feature_names):
        return 0
    
    CLV = ( (dataset_dict['close'] - dataset_dict['low']) - (dataset_dict['high'] - dataset_dict['close']) )/ \
    (dataset_dict['high'] - dataset_dict['low'] + 0.001)
    
    MFV = CLV * np.array(dataset_dict['volume']) 
    
    ADL = np.zeros(len(MFV))
    for i in range (1,len(MFV)):
        ADL[i] = MFV[i] + ADL[i-1]
    
    return ADL

def Ichimoku_Cloud_Indicator(dataset_dict, 
                             time_span_Tenkan, 
                             time_span_Kijun, 
                             displacement, 
                             time_span_Span_B, 
                             displacement_Chikou,
                             feature_names = ['Tenkan_sen']):
    
    user_input_tags = ['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B',
                       'Chikou_Span','cloud_flags', 'tk_cross_flags', 'oc_cloud_flags']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        null_return = np.zeros(1000)
        return null_return,null_return,null_return,null_return,null_return,null_return,null_return,null_return
    
    displacement_Span_A = displacement_Span_B = displacement
    #time_span_Tenkan = 9 ## conversion line perios
    highest_high_days = [max(dataset_dict['high'][(i - time_span_Tenkan) : i]) for i in range (len(dataset_dict['high'])) if i >= time_span_Tenkan]
    highest_high_days = np.pad(highest_high_days, ( (len(dataset_dict['high'])  - len(highest_high_days)),0 ), 'constant')
    
    lowest_low_days = [min(dataset_dict['low'][(i - time_span_Tenkan) : i]) for i in range (len(dataset_dict['low'])) if i >= time_span_Tenkan]
    lowest_low_days = np.pad(lowest_low_days, ( (len(dataset_dict['low'])  - len(lowest_low_days)),0 ), 'constant')
    
    Tenkan_sen = (highest_high_days + lowest_low_days) / 2
    Tenkan_sen = np.array(Tenkan_sen)
    
    
    #time_span_Kijun = 26
    highest_high_days = [max(dataset_dict['high'][(i - time_span_Kijun) : i]) for i in range (len(dataset_dict['high'])) if i >= time_span_Kijun]
    highest_high_days = np.pad(highest_high_days, ( (len(dataset_dict['high'])  - len(highest_high_days)),0 ), 'constant')
    
    lowest_low_days = [min(dataset_dict['low'][(i - time_span_Kijun) : i]) for i in range (len(dataset_dict['low'])) if i >= time_span_Kijun]
    lowest_low_days = np.pad(lowest_low_days, ( (len(dataset_dict['low'])  - len(lowest_low_days)),0 ), 'constant')
    
    Kijun_sen = (highest_high_days + lowest_low_days) / 2
    Kijun_sen = np.array(Kijun_sen)

    #time_span_Span_A = 26
    #Senkou_Span_A = [((Tenkan_sen[i - time_span_Span_A] + Kijun_sen[i - time_span_Span_A]) / 2) for i in range (len(Tenkan_sen)) if i >= time_span_Span_A]
    #Senkou_Span_A = np.pad(Senkou_Span_A, ( (len(Tenkan_sen)  - len(Senkou_Span_A)),0 ), 'constant')
    #Senkou_Span_A = (Tenkan-sen + Kijun-sen) / 2
    
    #displacement_Span_A = 26
    Senkou_Span_A = (Tenkan_sen + Kijun_sen) / 2
    Senkou_Span_A = np.pad(Senkou_Span_A, ( displacement_Span_A, 0 ), 'constant')
    Senkou_Span_A = np.array(Senkou_Span_A)
    
    ### displacement = 26 (plotted ahead)
    
    #time_span_Span_B = 52
    #displacement_Span_B = 26
    
    highest_high_days = [max(dataset_dict['high'][(i - time_span_Span_B) : i]) for i in range (len(dataset_dict['high'])) if i >= time_span_Span_B]
    highest_high_days = np.pad(highest_high_days, ( (len(dataset_dict['high'])  - len(highest_high_days)),0 ), 'constant')
    
    lowest_low_days = [min(dataset_dict['low'][(i - time_span_Span_B) : i]) for i in range (len(dataset_dict['low'])) if i >= time_span_Span_B]
    lowest_low_days = np.pad(lowest_low_days, ( (len(dataset_dict['low'])  - len(lowest_low_days)),0 ), 'constant')
    
    
    Senkou_Span_B = (highest_high_days + lowest_low_days) / 2
    Senkou_Span_B = np.pad(Senkou_Span_B, ( displacement_Span_B, 0 ), 'constant')
    Senkou_Span_B = np.array(Senkou_Span_B)
    
    #displacement_Chikou = 22
    Chikou_Span = dataset_dict['close'][displacement_Chikou:]
    Chikou_Span = np.pad(Chikou_Span, ( 0, displacement_Chikou ), 'constant') ## plt before
    Chikou_Span = np.array(Chikou_Span)
    
    ######## optional bull/bear flags
    kumo_cloud = Senkou_Span_A - Senkou_Span_B
    cloud_flags = [1 if i > 0 else 0 for i in kumo_cloud] ## 1 is green
    cloud_flags = np.array(cloud_flags)
    
    ### flat_kumo_flags ## in progress
    
    tk_diff = Tenkan_sen - Kijun_sen  
    tk_cross_flags = np.add( [-1 if (tk_diff[i] > 0 and tk_diff[i-1] < 0 ) else 0 for i in range(1, len(tk_diff))], \
    [1 if (tk_diff[i] < 0 and tk_diff[i-1] > 0 ) else 0 for i in range(1, len(tk_diff))] )
    tk_cross_flags = np.insert(tk_cross_flags, 0, 0 ) ## insert zero at start
    tk_cross_flags = np.array(tk_cross_flags)
    
    oc_cloud_flags = np.add([1 if ( ( dataset_dict['open'][i] > Senkou_Span_A[i] and dataset_dict['open'][i] > Senkou_Span_B[i] ) \
                      and (dataset_dict['close'][i] < Senkou_Span_A[i] or dataset_dict['close'][i] < Senkou_Span_B[i] )) else 0 for i in range(len(dataset_dict['open'])) ], \
                     [-1 if ( ( dataset_dict['open'][i] < Senkou_Span_A[i] and dataset_dict['open'][i] < Senkou_Span_B[i] ) \
                      and (dataset_dict['close'][i] > Senkou_Span_A[i] or dataset_dict['close'][i] > Senkou_Span_B[i] )) else 0 for i in range(len(dataset_dict['open']))])
    oc_cloud_flags = np.array(oc_cloud_flags)                
    
    return Tenkan_sen, Kijun_sen, Senkou_Span_A, Senkou_Span_B, Chikou_Span, cloud_flags, tk_cross_flags, oc_cloud_flags


def support_levels(x, window_size, time_period):
    
    high_peaks =list(np.zeros(window_size))
    for i in range (window_size, len(x), time_period):
            tmp = i - window_size + list(x[ (i - window_size) : i]).index(np.max(x[ (i - window_size) : i ]))
            for j in range (time_period): high_peaks.append(tmp)
            
    high_peaks = high_peaks[:len(x)]
    persistant_high_peaks = [x[int(i)] for i in high_peaks]
    
    low_peaks =list(np.zeros(window_size))
    for i in range (window_size, len(x), time_period):
            tmp = i - window_size + list(x[ (i - window_size) : i]).index(np.min(x[ (i - window_size) : i ]))
            for j in range (time_period): low_peaks.append(tmp)
            
    low_peaks = low_peaks[:len(x)]
    persistant_low_peaks = [x[int(i)] for i in low_peaks]   
    
    return np.array(persistant_high_peaks), np.array(persistant_low_peaks)     

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
            close_low = np.min(close_prices[(i - divergence_window + 1) : (i + 1)])
            MACD_low = np.min(MACD_line[(i - divergence_window + 1) : (i + 1)])
            
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
            close_high = np.max(close_prices[(i - divergence_window + 1) : (i + 1)])
            MACD_high = np.max(MACD_line[(i - divergence_window + 1) : (i + 1)])
    
            if(MACD_high > MACD_highest_high):
                MACD_highest_high = MACD_high
                
            if(close_high > close_highest_high):
                close_highest_high = close_high
                timer_bull = RESET_TIMER
                if(MACD_high < MACD_highest_high):
                    bearish_div.append(-1)
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
    divergence_values = np.add(bullish_div, bearish_div)  
    return np.array(divergence_values)

def SMA_EMA(close_prices, days, feature_names = ['SMA_12_day_values']):
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
    time_frame = days * 4 * 24
    EMA_time_period = 2 / (days + 1)
    for i in range(len(close_prices)):
        if(i < time_frame):
            ## add mean values
            SMA += close_prices[i] / time_frame
            EMA = SMA
        else:
            SMA = SMA - ((close_prices[i - time_frame] - close_prices[i]) / days)     
            EMA = (close_prices[i] - EMA) * EMA_time_period + EMA
        SMA_values.append(SMA)
        EMA_values.append(EMA)
    return np.array(SMA_values), np.array(EMA_values)

def MACD(close_prices, EMA_12_day_values, EMA_26_day_values, divergence_window, feature_names = ['MACD_line_15_min']):
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

    MACD_line = [EMA_12_day_values[i] - EMA_26_day_values[i] for i in range(len(EMA_26_day_values))] 
    SMA_9_day_MACD, EMA_9_day_MACD = SMA_EMA(MACD_line, 26)
    MACD_histogram = [MACD_line[i] - EMA_9_day_MACD[i] for i in range(len(EMA_9_day_MACD))]
    
    ## get divergences
    divergence_values = MACD_divergence(close_prices, MACD_line, divergence_window)
    MACD_signal_line = EMA_9_day_MACD
    return np.array(MACD_line), np.array(MACD_signal_line), np.array(MACD_histogram), np.array(divergence_values)

def RSI(close_prices, time_frame = '15min',feature_names = ['RSI']):
    user_input_tags = ['divergence_RSI', 'RSI']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0
    
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
    RS = 0
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
                if(avg_loss == 0):
                    RSI_values[-1] = 100
                else:
                    RS = avg_gain / avg_loss
                    RSI_values[-1] = 100 - 100 / (1 + RS)
                adva = 0
                decl = 0   
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
            if(avg_loss == 0):
                RSI_values.append(100 - 100 / (1 + RS))
            else:
                RS = avg_gain / avg_loss
                RSI_values.append(100 - 100 / (1 + RS))     
    return np.array(RSI_values)

def stochastic_RSI(RSI_values, time_frame, feature_names = ['stochastic_RSI']):
    user_input_tags = ['divergence_RSI', 'stochastic_RSI']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0

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

def RSI_divergence(close_prices, RSI_values, divergence_window, feature_names = ['divergence_RSI']):
    if('divergence_RSI' not in feature_names):
        return 0
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
    return np.array(divergence_values)

def money_flow_index(close_prices, high_prices, low_prices, volume, time_frame, feature_names = ['MFI']):
    if('MFI' not in feature_names):
        return 0
    diff_circular_buffer = np.zeros(time_frame)
    MFI_values = []
    raw_money_flow = 0
    for i in range(len(close_prices)):
        typical_price = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
        previous_money_flow = raw_money_flow
        raw_money_flow = typical_price * volume[i]
        
        diff = raw_money_flow - previous_money_flow
        diff_circular_buffer = np.hstack((diff, diff_circular_buffer[:-1]))
        positive_money_flow = np.sum([i if (i > 0) else 0 for i in diff_circular_buffer])
        negative_money_flow = np.sum([-i if (i < 0) else 0 for i in diff_circular_buffer])
                
        money_flow_ratio = positive_money_flow / negative_money_flow
        MFI_values.append(100 - (100/(1 + money_flow_ratio)))
    return np.array(MFI_values)

def money_flow_divergence(close_prices, MFI_values, divergence_window, feature_names = ['divergence_MFI']):
    if('divergence_MFI' not in feature_names):
        return 0
    MFI_lowest_low = 99999999999
    close_lowest_low = 99999999999
    close_highest_high = -9999999
    MFI_highest_high = -9999999
    bullish_div = []
    bearish_div = []
    RESET_TIMER = 100
    timer_bear = RESET_TIMER
    timer_bull = RESET_TIMER
    for i in range(len(MFI_values)):
        ## bullish div
        if(i <= divergence_window):
            bearish_div.append(0)
            bullish_div.append(0)
        else:
            close_low = np.min(close_prices[(i - divergence_window + 1): (i + 1)])
            MFI_low = np.min(MFI_values[(i - divergence_window + 1): (i + 1)])
            
            if(MFI_low < MFI_lowest_low):
                MFI_lowest_low = MFI_low
            
            if(close_low < close_lowest_low):
                close_lowest_low = close_low
                timer_bear = RESET_TIMER
                if(MFI_low > MFI_lowest_low):
                    bullish_div.append(1)
                else:
                    bullish_div.append(0)
            else:
                bullish_div.append(0)
                
            ## bearish div
            close_high = np.max(close_prices[(i - divergence_window + 1): (i + 1)])
            MFI_high = np.max(MFI_values[(i - divergence_window + 1): (i + 1)])
    
            if(MFI_high > MFI_highest_high):
                MFI_highest_high = MFI_high
                
            if(close_high > close_highest_high):
                close_highest_high = close_high
                timer_bull = RESET_TIMER
                if(MFI_high < MFI_highest_high):
                    bearish_div.append(-1)
                else:
                    bearish_div.append(0)
            else:
                bearish_div.append(0)
            
            if(timer_bear <= 0):
                MFI_lowest_low = MFI_low
                close_lowest_low = close_low
            if(timer_bull <= 0):
                MFI_highest_high = MFI_high
                close_highest_high = close_high
            timer_bear -= 1      
            timer_bull -= 1       
    divergence_values = np.add(bullish_div, bearish_div)    
    return np.array(divergence_values)

def bollinger_bands(close_prices, time_period, feature_names = ['BB_SMA_values']):
    user_input_tags = ['BB_SMA_values', 'BB_upperline_values', 'BB_lowerline_values', 'BB_squeeze_values', 'BB_SMA_crossings',
                       'BB_SMA_values_12h', 'BB_upperline_values_12h', 'BB_lowerline_values_12h', 'BB_squeeze_values_12h', 'BB_SMA_crossings_12h',
                       'BB_SMA_values_1h', 'BB_upperline_values_1h', 'BB_lowerline_values_1h', 'BB_squeeze_values_1h', 'BB_SMA_crossings_1h'] 
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0,0,0,0,0
    
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
    return np.array(SMA_values), np.array(upperline_values), np.array(lowerline_values), np.array(squeeze_values), np.array(SMA_crossings)   

def commodity_channel_index(close_prices, high_prices, low_prices, time_period, feature_names = 'CCI'):
    user_input_tags = ['CCI', 'CCI_thresholds', 'CCI_12h', 'CCI_thresholds_12h', 'CCI_1h', 'CCI_thresholds_1h']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0,0
    
    typical_price_circular_buffer = np.zeros(time_period)
    CCI_values = []
    CCI_threshold_values = []
    constant = 0.015
    for i in range(len(close_prices)):
        typical_price = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
        typical_price_circular_buffer = np.hstack((typical_price, typical_price_circular_buffer[:-1]))
        TP_SMA = np.sum(typical_price_circular_buffer) / time_period
        mean_deviation = np.sum(abs(typical_price_circular_buffer - TP_SMA)) / time_period
        CCI_values.append((typical_price - TP_SMA) / (constant * mean_deviation))
        if(CCI_values[-1] > 190):
            CCI_threshold_values.append(1)
        elif(CCI_values[-1] < -190):
            CCI_threshold_values.append(-1)
        else:
             CCI_threshold_values.append(0)
    return np.array(CCI_values), np.array(CCI_threshold_values)


def slope_split_points(X, slope_window = 2, lam_att = 0.5, slope_diff = 0.051, clean_adjacent_points = 12):
    #### slope_diff = 0.051 , lam_att = 0.5, clean_adjacent_points = 12 , slope_window = 2
    slope = np.zeros(np.array(X.shape))
    for i in range(1,X.shape[0]):
            X[i] = X[i-1] * lam_att + (1 - lam_att) * X[i]
            if i >= slope_window:
                slope[i] = ( X[i] - X[i - slope_window] ) / (slope_window * X[i - slope_window])       
    max_slope = max(slope)
    min_slope = min(slope) 
    diff = max_slope - min_slope   
    slope = [ (i - min_slope) / diff for i in slope]        
    histogram = np.histogram(np.array(slope), bins = 1000)
    most_frequent_slope = histogram[1][ list(histogram[0]).index(max(histogram[0])) ]
    split_points = [ 1 if i >= (most_frequent_slope + slope_diff) else 0 for i in slope ] ## uptrend
    split_points = [ 1 if i <= (most_frequent_slope - slope_diff) else 0 for i in slope ] ## downtrend
        ## clean adjacent split points
    for i in range(len(split_points)):
        if split_points[i] > 0 :
            for j in range(1, clean_adjacent_points):
                if split_points[i+j] > 0:
                    split_points[i+j] = 0
                else:
                    break
    return split_points

def get_threshold_flags(data, upper_thresh, lower_thresh, lam = 0):
    try:
        a = len(data)
    except:
        return 0    
    thresh_flags = []
    for i  in range (len(data)):
        if(data[i] > upper_thresh):
            thresh_flags.append(1)
        elif(data[i] < lower_thresh):
            thresh_flags.append(-1)
        else:
            thresh_flags.append(0)
    if(lam):
        preprocessed_thresh = []
        for i in range(len(thresh_flags)):
            if(i==0):
                preprocessed_thresh.append(thresh_flags[i])
            else:
                preprocessed_thresh.append(preprocessed_thresh[-1] * lam + thresh_flags[i] * (1 - lam))
        return np.array(preprocessed_thresh)
    else:
        return np.array(thresh_flags)
            


def append_TA_descriptors_to_X(X_descriptors, 
                               MACD_flag, 
                               RSI_flag, 
                               RSI_threshold_flag, 
                               STOCH_RSI_flag, 
                               STOCH_RSI_threshold_flag, 
                               OBV_flag):
    if(MACD_flag):
        input_MACD = np.float64(X_descriptors[3]) # close prices
        MACD_descriptors = talib.MACD(input_MACD)
        for i  in range (len(MACD_descriptors)):
            for ii in range (len(MACD_descriptors[i])):
                if math.isnan(MACD_descriptors[i][ii]) == True:
                    MACD_descriptors[i][ii] = 0   
        MACD_descriptors = ( MACD_descriptors + abs(np.min(MACD_descriptors)) ) * 10 # shifting for positive values
        X_descriptors = np.concatenate([X_descriptors,MACD_descriptors])
    
    if(RSI_flag):
        input_RSI = np.float64(X_descriptors[3]) # close prices
        RSI_descriptors = RSI(input_RSI, '15min')
        RSI_thresh_descriptors = np.zeros(len(RSI_descriptors))
        stochastic_RSI_thresh_descriptors = np.zeros(len(RSI_descriptors))
        for i  in range (len(RSI_descriptors)):
            if math.isnan(RSI_descriptors[i]) == True:
                RSI_descriptors[i] = 0
            else:
                if(RSI_threshold_flag):
                    if(RSI_descriptors[i] < 30):
                        RSI_thresh_descriptors[i] = 0.50
                    if(RSI_descriptors[i] > 70):
                        RSI_thresh_descriptors[i] = 0.99
    
        if(STOCH_RSI_flag):
            STOCH_RSI_descriptors = stochastic_RSI(RSI_descriptors, time_frame = 14)
            for i  in range (len(STOCH_RSI_descriptors)):
                if math.isnan(STOCH_RSI_descriptors[i]) == True:
                    STOCH_RSI_descriptors[i] = 0
                if(STOCH_RSI_threshold_flag):
                    if(STOCH_RSI_descriptors[i] < 30):
                        stochastic_RSI_thresh_descriptors[i] = 0.50
                    if(STOCH_RSI_descriptors[i] > 70):
                        stochastic_RSI_thresh_descriptors[i] = 0.99
                    
                 
        print('len RSI: '+ str(np.shape(RSI_descriptors)))
        print('len RSI: '+ str(np.shape(X_descriptors)))
        RSI_descriptors = np.reshape(RSI_descriptors,(1,len(RSI_descriptors)))
        RSI_thresh_descriptors = np.reshape(RSI_thresh_descriptors,(1,len(RSI_thresh_descriptors)))
        X_descriptors = np.concatenate([X_descriptors,RSI_descriptors])
        if(RSI_threshold_flag):
             X_descriptors = np.concatenate([X_descriptors, RSI_thresh_descriptors])
        if(STOCH_RSI_flag):
            STOCH_RSI_descriptors = np.reshape(STOCH_RSI_descriptors,(1,len(STOCH_RSI_descriptors)))
            X_descriptors = np.concatenate([X_descriptors, STOCH_RSI_descriptors])
        if(STOCH_RSI_threshold_flag):
            stochastic_RSI_thresh_descriptors = np.reshape(stochastic_RSI_thresh_descriptors,(1,len(stochastic_RSI_thresh_descriptors)))
            X_descriptors = np.concatenate([X_descriptors, stochastic_RSI_thresh_descriptors])
        if(OBV_flag):
            input_OBV = [X_descriptors[3],X_descriptors[4]]
            OBV_descriptors = talib.OBV(input_OBV[0],input_OBV[1])
            OBV_descriptors = np.reshape(OBV_descriptors,(1,len(OBV_descriptors)))
            X_descriptors = np.concatenate([X_descriptors,OBV_descriptors])
            
    return X_descriptors
                
#def RSI_2(close_prices, time_frame = 14, time_window = 1, feature_names = ['RSI_2']):
#    user_inputs = ['RSI_2', 'RSI_timeframe_48', 'RSI_12h', 'RSI_1d', 'RSI_1w', 'RSI_1m', 'divergence_RSI']
#   
#    ok = False
#    for i in feature_names:
#        if i in user_inputs:
#            ok = True
#            break
#    if(not ok):
#        return 0
    
#    adva = 0    
#    decl = 0
#    avg_gain = 0
#    avg_loss = 0
#    RSI_values = []
#    index_cnt = 0
#    RS = 0
#    for i in range(0, len(close_prices), time_window):
#        if(index_cnt <= 0):
#            RSI_values.append(0)
#            for j in range(time_window - 1):
#                    RSI_values.append(RSI_values[-1])
#        elif((index_cnt < time_window) and (index_cnt > 0)):
#            RSI_values.append(0)
#            if(time_window > 1):
#                for j in range(time_window - 1):
#                    RSI_values.append(RSI_values[-1])
#            mean_window = np.mean(close_prices[(i - time_window):i])
#            if(close_prices[i] > mean_window):
#                adva += close_prices[i] - mean_window
#            else:
#                decl += mean_window - close_prices[i]
#            if(index_cnt == (time_window - 1) ):
#                avg_gain = adva / time_frame
#                avg_loss = decl / time_frame
#                if(avg_loss == 0):
#                    RSI_values[-1] = 100
#                else:
#                    RS = avg_gain / avg_loss
#                    RSI_values[-1] = 100 - 100 / (1 + RS)
#                adva = 0
#                decl = 0
#        else:
#            diff = close_prices[i] - close_prices[i - time_window]
#            if(diff > 0):
#                current_gain = diff
#                current_loss = 0
#            else:
#                current_loss = abs(diff)
#                current_gain = 0
#                
#            avg_gain = (avg_gain * (time_frame - 1) + current_gain) / time_frame
#            avg_loss = (avg_loss * (time_frame - 1) + current_loss) / time_frame
#            if(avg_loss == 0):
#                RSI_values.append(100)
#            else:
#                RS = avg_gain / avg_loss
#                RSI_values.append(100 - 100 / (1 + RS))
#            if(time_window > 1):
#                for j in range(time_window - 1):
#                    RSI_values.append(RSI_values[-1])
#        index_cnt += 1
#     
#    #print(index_cnt * time_window)    
#    for i in range(len(RSI_values) - len(close_prices)):
#        RSI_values = RSI_values[:-1]
#    return np.array(RSI_values)

def RSI_2(close_prices, days = 14, feature_names = ['RSI_2']):
    user_inputs = ['RSI_2', 'RSI_timeframe_48', 'RSI_14d', 'RSI_1d', 'RSI_1w', 'RSI_1m', 'divergence_RSI']
   
    ok = False
    for i in feature_names:
        if i in user_inputs:
            ok = True
            break
    if(not ok):
        return 0
    
    time_window = days * 4 * 24 ## scale from 15 min to 1 day
    k = 2 / (days + 1) ## EMA factor
    RSI = [0]
    EMA_gain = 0
    EMA_loss = 0
    for i in range(1,len(close_prices)):
        diff = close_prices[i] - close_prices[i-1]
        gain, loss = 0, 0
        if(diff >= 0):
            gain = diff
        else:
            loss = abs(diff)
        if(i <= time_window):
            EMA_gain += gain / days
            EMA_loss += loss / days
        else:
            EMA_gain = (gain - EMA_gain) * k + EMA_gain
            EMA_loss = (loss - EMA_loss) * k + EMA_loss

        if(EMA_loss == 0):
            RSI.append(100)
        else:          
            RS = EMA_gain / EMA_loss
            RSI.append(100 - (100 / (1 + RS)))
    return np.array(RSI)

def VBP(close_prices, volume, bins = 12, lam = 0.99, feature_names = ['volume_by_price']):
    user_input_tags = ['volume_by_price', 'slope_VBP', 'slope_VBP_smooth', 'volume_by_price_24', 'slope_VBP_24', 'slope_VBP_smooth_24']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0,0,0
    
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
                VBP.append(bin_volume[j-1])
                if(i == 0):
                    slope_VBP.append(0)
                    slope_VPB_smooth.append(0)
                else:
                    slope_VBP.append(VBP[-1] - VBP[-2])
                    slope_VPB_smooth.append(slope_VBP[-2] * lam + slope_VBP[-1] * (1-lam))
                break
    
    return np.array(VBP), np.array(slope_VBP), np.array(slope_VPB_smooth)

def NLMS_indicator(close, time_period = 128, nlms_step = 0.5, nlms_constant = 0.9, lam = 0.9, feature_names = ['nlms_indicator']):
    user_input_tags = ['nlms_indicator', 'nlms_smoothed_indicator']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0,0
    
    sample_circular_buffer = np.zeros(time_period)
    nlms_filter = np.zeros(time_period)
    nlms_indicator = []
    nlms_smoothed_indicator = []
    for i in range(len(close)):
         sample_circular_buffer = np.hstack((close[i], sample_circular_buffer[:-1]))
         filter_output = nlms_filter.dot(sample_circular_buffer.T)
         error = close[i] - filter_output 
         nlms_filter = nlms_filter + nlms_step * error * sample_circular_buffer / \
                     (sample_circular_buffer.T.dot(sample_circular_buffer) + nlms_constant)
         nlms_indicator.append(error)
         if(i==0):
             nlms_smoothed_indicator.append(nlms_indicator[-1])
         else:
             nlms_smoothed_indicator.append(nlms_smoothed_indicator[-1] * lam + (1-lam) * nlms_indicator[-1])
    return np.array(nlms_indicator), np.array(nlms_smoothed_indicator)

def RLS_indicator(close, time_period = 128, lam = 0.9, delta = 1, smoothing = 0.9, dct_transform = False, feature_names = ['rls_indicator_error']):
    user_input_tags = ['rls_indicator_error', 'rls_indicator_output', 'rls_smoothed_indicator']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0,0,0
    
    sample_circular_buffer = np.zeros(time_period)
    dct_sample_circular_buffer = np.zeros(time_period)
    rls_filter = np.zeros(time_period)
    rls_indicator_error = []
    rls_indicator_output = []
    rls_smoothed_indicator = []
    P = (delta**(-1)) * np.identity(time_period);
    for i in range(len(close)):
        if(dct_transform == "type II"):
            dct_sample_circular_buffer = np.hstack((close[i], dct_sample_circular_buffer[:-1]))
            sample_circular_buffer = DCT_transform_II(dct_sample_circular_buffer)
        else:
            sample_circular_buffer = np.hstack((close[i], sample_circular_buffer[:-1]))
        z = np.matmul(sample_circular_buffer,P)
        k = np.matmul(P, sample_circular_buffer.T) / (lam + np.matmul(z, sample_circular_buffer.T))
        y = np.matmul(rls_filter, sample_circular_buffer.T)
        error = close[i] - y
        rls_filter = rls_filter + np.multiply(k, np.conj(error))
        P = (1 / lam) * (P - np.multiply(np.matmul(z, k), P))
        rls_indicator_error.append(error)
        rls_indicator_output.append(y)
        if(i==0):
             rls_smoothed_indicator.append(rls_indicator_error[-1])
        else:
             rls_smoothed_indicator.append(rls_smoothed_indicator[-1] * smoothing + \
                                           (1 - smoothing) * rls_indicator_error[-1])
    return np.array(rls_indicator_error), np.array(rls_indicator_output) 


def ATR(close, high, low, time_frame = 14, feature_names = ['ATR_EMA']):
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

def Chaikin_money_flow(close, high, low, volume, time_period = 21, feature_names = ['CMF_12h']):
    user_input_tags = ['CMF_12h_2', 'CMF_12h']
    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        return 0
    
    MF_volume_circular_buffer = np.zeros(time_period)
    volume_circular_buffer = np.zeros(time_period)
    CMF = []
    for i in range(len(close)):
        MF_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        MF_volume = MF_multiplier * volume[i]
        MF_volume_circular_buffer = np.hstack((MF_volume, MF_volume_circular_buffer[:-1]))
        volume_circular_buffer = np.hstack((volume[i], volume_circular_buffer[:-1]))
        CMF.append(np.sum(MF_volume_circular_buffer) / np.sum(volume_circular_buffer))
    return np.array(CMF)

def get_sentiment_indicator_from_db(dataset_dict, path = '/home/catalin/databases/tweets/nltk_2014_2018_300_per_day.pkl'):
    with open(path, 'rb') as f:
        tweets = pickle.load(f)
        
    start_date = date(2014, 10, 2)
    end_date = date(2018, 11, 20)
    dates = []
    for single_date in daterange(start_date, end_date):
        dates.append(str(single_date))
        
    start_date = md.get_date_from_UTC_ms(dataset_dict['UTC'][0])
    end_date = md.get_date_from_UTC_ms(dataset_dict['UTC'][-1])
    
    start = False
    end = False
    sentiment_indicator_positive = np.zeros(len(dataset_dict['close']))
    sentiment_indicator_negative = np.zeros(len(dataset_dict['close']))
    step_1_day = 15*4*24
    for i in range(len(dates)):
        if(dates[i].find(end_date) != -1):
            end = True
        if(dates[i].find(start_date) != -1):
            start = True
        if(start and not end):
            ### offset with one step to adjust for real-time data feed.
            sentiment_indicator_positive[step_1_day * (i+1) : step_1_day * (i+2)] = tweets[dates[i]]['pos']
            sentiment_indicator_negative[step_1_day * (i+1) : step_1_day * (i+2)] = tweets[dates[i]]['neg']
    return np.array(sentiment_indicator_positive), np.array(sentiment_indicator_negative)