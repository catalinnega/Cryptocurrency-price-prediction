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
def get_dataset_with_descriptors(concatenate_datasets_preproc_flag, preproc_constant, normalization_method, dataset_directory, hard_coded_file_number):
    #########
    #mtaS 	int 	millisecond time stamp
    #OPEN 	float 	First execution during the time frame
    #CLOSE 	float 	Last execution during the time frame
    #HIGH 	float 	Highest execution during the time frame
    #LOW 	float 	Lowest execution during the timeframe
    #VOLUME 	float 	Quantity of symbol traded within the timeframe
    #########

    dictionary = ['UTC', 'open', 'close', 'high', 'low', 'volume']
    if(hard_coded_file_number):
        x = hard_coded_file_number
    else:      
        os.chdir(dataset_directory)
        x = subprocess.check_output(' ls | wc -l',shell = 'True')
        x = int(md.parser(str(x)))
        
    if(x == 0):
        print('No files found!')
    dataset = [0 for _ in range (x)]
    X1 = ['none']
    
    for index_proc in ['preprocessed','nonprocessed']:
        
        if index_proc == 'preprocessed' and not concatenate_datasets_preproc_flag:
            print("Skipping preprocess..")
            continue   
        
        for ii in range (0,x):## minus readme file
            try:
                dataset[ii] = pd.read_csv(dataset_directory + '/klines_15min_' +str(ii) +'.csv')
                dataset[ii] = dataset[ii].values
            except:
                #print(ii)
                dataset[ii] = np.zeros([246, 6])
            if ii == 0:
                 whole_dataset = dataset[ii]
            else:
                 whole_dataset = np.concatenate( [ whole_dataset, dataset[ii] ] )
        ############# fill missing
        
        ## remove zore values from the start.. these are caused by the listing time of the pairing.
        print('len dataset ' + str(len(whole_dataset)))
        for i in range(len(whole_dataset)):
            if(whole_dataset[i][1] > 0):
                print('debug '+ str(i) + ' data: ' + str(whole_dataset[i][0]))
                whole_dataset = whole_dataset[i:,:] 
                break
        print('len dataset2 ' + str(len(whole_dataset)))    
                
        for i in range (1, len(whole_dataset)):
            if whole_dataset[i][0] == 0:
                whole_dataset[i] = whole_dataset[i-1]
                whole_dataset[i][0] += 900000
        ##############
        whole_dataset = whole_dataset[:-3] ## remove last 3 candles
        whole_dataset = whole_dataset[:-1] ## remove last candle
                
        dataset_dict = {}
        for i in range (len(whole_dataset.T)):
            dataset_dict[dictionary[i]] = whole_dataset[:,i]
                   
        X = copy.copy(whole_dataset)
        X = X[:,1:] ## remove time column
        
        #### preprocess
        if index_proc == 'preprocessed':
#            lam = 0.5 
#            lam = 0.999 ## highest accuracy if all are preprocessed
            lam = preproc_constant
            for i in range(1,X.shape[0]):
                for j in range(X.shape[1]):
                    X[i][j] = X[i-1][j] * lam + (1 - lam) * X[i][j]
            #plt.plot(X[:,0])
            #### end preprocess
        
        MACD_flag = RSI_flag =  RSI_threshold_flag = True
        STOCH_RSI_flag = OBV_flag = False
        X = mta.append_TA_descriptors_to_X(X.T, MACD_flag, RSI_flag, RSI_threshold_flag,  STOCH_RSI_flag, OBV_flag)
        X = X.T
        
        Tenkan_sen, Kijun_sen, Senkou_Span_A, Senkou_Span_B, Chikou_Span, cloud_flags, tk_cross_flags, oc_cloud_flags \
            = mta.Ichimoku_Cloud_Indicator(dataset_dict, 9, 26, 26, 52, 22)
        
        ADL = mta.Accum_distrib(dataset_dict)
        
        close_prices = whole_dataset[:,1:][:,0]
        one_day_in_15_min_candles = 15 * 4 * 24
        twelve_hrs_in_15_min_candles = 15 * 4 * 12
        six_hrs_in_15_min_candles = 15 * 4 * 6
        three_hrs_in_15_min_candles = 15 * 4 * 3
        one_hr_in_15_min_candles = 15 * 4 

    
        SMA_12_day_values, EMA_12_day_values = mta.SMA_EMA(close_prices, 12 * one_day_in_15_min_candles)
        SMA_26_day_values, EMA_26_day_values = mta.SMA_EMA(close_prices, 26 * one_day_in_15_min_candles)
        
        MACD_line, MACD_signal_line, MACD_histogram, bullish_div, bearish_div = mta.MACD(close_prices, 
                                                                   EMA_12_day_values, 
                                                                   EMA_26_day_values,  
                                                                   divergence_window = 100)
    #    window_size = 96 * 30 ## 30 days
    #    window_size = 96 * 30 * 3 ## 30 * 3 days
    #    time_period = 96    
    #    support_high, support_low = mta.support_levels(dataset_dict['close'], window_size, time_period )
    #        
    
        X = md.concatenator([X,Tenkan_sen, 
                             Kijun_sen, 
                             Senkou_Span_A[:-26], 
                             Senkou_Span_B[:-26], 
                             Chikou_Span, 
                             cloud_flags[:-26], 
                             tk_cross_flags, 
                             oc_cloud_flags, 
                             ADL,
                             MACD_line,
                             MACD_signal_line,
                             MACD_histogram,
                             bullish_div,
                             bearish_div])
        
        if(normalization_method == "rescale"):
            X_norm_prices, min_prices, max_prices = mn.normalize_rescale(X[:,0:4])
            X_norm_volume, min_volume, max_volume = mn.normalize_rescale(X[:,4])
            if(MACD_flag):
                X_norm_MACD, junk, junk = mn.normalize_rescale(X[:,5:13])
            if(RSI_flag):
                X_norm_RSI, junk, junk = mn.normalize_rescale(X[:,13])
            
            X = md.concatenator([X_norm_prices,X_norm_volume, X_norm_MACD,X_norm_RSI])
            if index_proc == 'preprocessed':
                X1 = copy.copy(X)
            else:            
                return_dataset = {}
                return_dataset['preprocessed_data'] = X1
                return_dataset['data'] = X
                return_dataset['min_prices'] = min_prices
                return_dataset['min_volume'] = min_volume
                return_dataset['max_prices'] = max_prices
                return_dataset['max_volume'] = max_volume
                return_dataset['dataset_dict'] = dataset_dict
            
        elif(normalization_method == "norm"):
            X_norm_prices, X_norm_prices_denorm_constant = mn.normalize_norm(X[:,0:4])
            X_norm_volume, X_norm_volume_denorm_constant = mn.normalize_norm(X[:,4])
            if(MACD_flag):
                X_norm_MACD, X_norm_MACD_denorm_constant = mn.normalize_norm(X[:,5:8])
            if(RSI_flag):
                X_norm_RSI, X_norm_RSI_denorm_constant = mn.normalize_norm(X[:,8])
            X = md.concatenator([X_norm_prices,X_norm_volume, X_norm_MACD,X_norm_RSI])
            
            if index_proc == 'preprocessed':
                X1 = copy.copy(X)
            else:            
                return_dataset = {}
                return_dataset['preprocessed_data'] = X1
                return_dataset['data'] = X
                return_dataset['X_norm_prices_denorm_constant'] = X_norm_prices_denorm_constant
                return_dataset['X_norm_volume_denorm_constant'] = X_norm_volume_denorm_constant
                return_dataset['dataset_dict'] = dataset_dict

        if(normalization_method == "standardization"):
            X_norm_prices = mn.normalize_standardization(X[:,0:4])
            X_norm_volume = mn.normalize_standardization(X[:,4])
            if(MACD_flag):
                X_norm_MACD = mn.normalize_standardization(X[:,5:8])
            if(RSI_flag):
                X_norm_RSI = mn.normalize_standardization(X[:,8])
            
            X = md.concatenator([X_norm_prices,X_norm_volume, X_norm_MACD,X_norm_RSI])
            if index_proc == 'preprocessed':
                X1 = copy.copy(X)
            else:            
                return_dataset = {}
                return_dataset['preprocessed_data'] = X1
                return_dataset['data'] = X
                return_dataset['dataset_dict'] = dataset_dict            
#        elif concatenate_datasets_preproc_flag:
#            X = md.concatenator([X,X1])
            
    return return_dataset
        

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


SMA_12_day_values, EMA_12_day_values = SMA_EMA(close_prices, 12 * one_hr_in_15_min_candles)
SMA_26_day_values, EMA_26_day_values = SMA_EMA(close_prices, 26 * one_hr_in_15_min_candles)

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


    
        