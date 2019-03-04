#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 23:04:00 2018

@author: catalin
"""

import numpy as np
import copy
import subprocess
import os
import pandas as pd
import mylib_dataset as md
import mylib_normalize as mn
import mylib_TA as mta
import talib 
import datetime


#########
#MTS 	int 	millisecond time stamp
#OPEN 	float 	First execution during the time frame
#CLOSE 	float 	Last execution during the time frame
#HIGH 	float 	Highest execution during the time frame
#LOW 	float 	Lowest execution during the timeframe
#VOLUME 	float 	Quantity of symbol traded within the timeframe
#########

def var_name_to_string(**variables):
    return [x for x in variables][0]

def parser(a):
        index1 = []
        index2 = []
        index1 = a.find("'")
#        print(index1)
        if index1 == -1:
            if a.find("[") != -1:
                index1 = a.find("[")
            elif a.find("]") != -1:
                return a[:-1]
            else:
                return a## failed to find symbol
        if a[(index1 + 1):].find("\n") != -1 :
            index2 = a[(index1 + 1):].find("\n")
        if a[(index1 + 1):].find("'") != -1:
            index2 = a[(index1 + 1):].find("'")
        else:
            index2 = len(a)
            return a[index1+1:index2]
                
        if a[(index1+1):index2][0]=='[':
            return a[(index1+2):index2]
        else:
            return a[(index1+1):index2]
        
def create_dataset_binary_labels(dataset, descriptors_len,  look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i+look_back)])
        if dataset[i + look_back -1][- descriptors_len + 1] - dataset[i + look_back][- descriptors_len +1] > 0:
            dataY.append(0) ### close prices
        else:
            dataY.append(1) ### close prices
    return np.array(dataX), np.array(dataY)

def create_dataset_ternary_labels_percentages(dataset, descriptors_len, percentage, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - 2 * look_back):
        dataX.append(dataset[i:(i+look_back)])
     
        ### Condition:
        ## if any 'low' or 'high' price of the next batch is bigger
        ## than 'percentage'% of the last closing price of the batch 
    
        if True in (dataset[(i + look_back):(i + 2*look_back)][:, -descriptors_len + 2] \
                            > (((percentage + 100) / 100) * dataset[i + look_back][- descriptors_len + 1])):
            dataY.append(2)
        elif True in (dataset[(i + look_back):(i + 2*look_back)][:, -descriptors_len + 3] \
                            < (((100 - percentage) / 100) * dataset[i + look_back][- descriptors_len + 1])):
            dataY.append(1)
        else:
            dataY.append(0)
    return np.array(dataX), np.array(dataY)

#use unprocessed close values for Y
def create_dataset_labels_mean_percentages(dataset, dataset_unprocessed, window_length, train_data_flag):
    dataX, dataY = [], []
    mean_lows = []
    mean_highs = []
    data_open  = dataset_unprocessed[:,0]
    data_high = dataset_unprocessed[:,2]
    data_low  = dataset_unprocessed[:,3]
    
        ### Condition:
    ## if any 'low' or 'high' price of the next batch is bigger
    ## than 'percentage'% of the last closing price of the batch 
    for i in range(len(data_open) - window_length):
        dataX.append(dataset[i,:])
        
#        close = dataset[i + look_back][1]
        mean_high = np.mean(data_high[i:(i + window_length)])
        mean_low =  np.mean(data_low[i:(i + window_length)])
        
        dataY.append( [((mean_high - data_open[i]) / data_open[i]), ((data_open[i] - mean_low)/ data_open[i])] )
        mean_highs.append(mean_high)
        mean_lows.append(mean_low)
    if(train_data_flag):
        for i in range(window_length):
            dataX.append(dataset[i,:])
            dataY.append((0,0))
    return np.array(dataX), np.array(dataY), mean_highs, mean_lows

def get_date_from_UTC_ms(UTC_time):
    #UTC_time = data['dataset_dict']['UTC'][0]
    ceva = datetime.datetime.utcfromtimestamp(UTC_time/1000)       
    date = ceva.isoformat()[:ceva.isoformat().find('T')]
    print('UTC time:' + str(UTC_time) + ' --- date: ' + date)
    return date

def get_split(data,chunks):
    if not chunks: return np.array(data)
    try:
       data = np.split(data,chunks)
    except:
       data = data[(len(data) - int(len(data)/chunks)*chunks):]
       data = np.split(data,chunks)
    return np.array(data)

def get_split2(data,chunks):
    if not chunks: return np.array(data)
    try:
       data = np.split(data,chunks)
    except:
       data = data[:-(len(data) - int(len(data)/chunks)*chunks)]
       data = np.split(data,chunks)
    return np.array(data)

def concatenator(data):
    data = [np.reshape(data[i], (len(data[i]), 1) ) if len(data[i].shape) == 1 else data[i] for i in range (len(data)) ]
    return np.concatenate(data, axis = 1)

def boolarray_to_int(x):
    x = [int(i) for i in x]
    return np.array(x)

def arr(x):
    x = np.array(x)
    return x

def counter(x, value_to_count, percentage = False):
    x = np.array(x)
    if len(x.shape) > 1:
        x = np.reshape( x, ( np.prod(x.shape), ) )
    
    cnt = 0
    for i in x:
        if (i == value_to_count):
            cnt += 1
    if percentage:
        return cnt * 100 / len(x)
    else:
        return cnt

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

def bin_class_perf_indicators(results, reference):
    true_positive = true_negative = false_positive = false_negative = 0
    dict_indicators = {}
    
    ## confusion matrix
    for i in range (len(results)):
        if(results[i] == 1):
            if(results[i] == reference[i]):
                true_positive += 1
            else:
                false_positive += 1
                
        if(results[i] == 0):
            if(results[i] == reference[i]):
                true_negative += 1 
            else:
                false_negative += 1
                
    dict_indicators['fp'] = false_positive
    dict_indicators['fn'] = false_negative
    dict_indicators['tp'] = true_positive
    dict_indicators['tn'] = true_negative  
          
    if(false_positive + true_negative):
        dict_indicators['fpr'] = false_positive / (false_positive + true_negative)
    if(false_negative + true_positive):
        dict_indicators['fnr'] = false_negative / (false_negative + true_positive)
    if(false_negative + true_positive):
       dict_indicators['tpr'] = false_negative / (false_negative + true_positive)
    if(true_negative + false_positive):
       dict_indicators['tnr'] = true_negative / (true_negative + false_positive)
    if(true_positive + false_positive):
        dict_indicators['ppv'] = true_positive / (true_positive + false_positive)
    if(true_negative + false_negative):
        dict_indicators['npv'] = true_negative / (true_negative + false_negative)
    if(false_positive + true_positive):
       dict_indicators['fdr'] = false_positive / (false_positive + true_positive)
    if(false_negative + true_negative):
        dict_indicators['for'] = false_negative / (false_negative + true_negative)
    
    dict_indicators['accuracy'] = (true_positive + true_negative) / ( true_negative + true_positive + false_negative + false_positive )
    
    dict_indicators['f1_score'] = 2 * true_positive / ( (2 * true_positive) + false_negative + false_positive)
    dict_indicators['mcc'] = (true_positive * true_negative - (false_positive * false_negative)) \
                            / np.sqrt( (true_positive + false_positive) * (true_positive + false_negative) * \
                                      (true_negative + false_positive) * (true_negative + false_negative))
    
    if(('tpr' in dict_indicators) and ('tnr' in  dict_indicators)):
        dict_indicators['bm'] = dict_indicators['tpr'] - dict_indicators['tnr'] - 1
    if(('ppv' in dict_indicators) and ('npv' in dict_indicators)):
        dict_indicators['mk'] = dict_indicators['ppv'] - dict_indicators['npv'] - 1
    
    return dict_indicators

def print_dictionary(dictionary):
    for i in dictionary:
        print(i, ':', dictionary[i])


def get_dataset_with_descriptors(concatenate_datasets_preproc_flag,
                                  preproc_constant, 
                                  normalization_method,
                                  dataset_directory, 
                                  hard_coded_file_number,
                                  feature_names,
                                  lookback = 0):
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
        whole_dataset = whole_dataset[:-3]
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
            #### end preprocess

        if(normalization_method == "rescale"):
            X_norm_prices, min_prices, max_prices = mn.normalize_rescale(X[:,0:4])
            X_norm_volume, min_volume, max_volume = mn.normalize_rescale(X[:,4])
        if(normalization_method == "standardization"):
            X_norm_prices = mn.normalize_standardization(X[:,0:4])
            X_norm_volume = mn.normalize_standardization(X[:,4])
        
        
        X[:,0:4] = X_norm_prices
        X[:,4] = X_norm_volume
        if index_proc == 'preprocessed':
            close_prices = X[:,1]
            high_prices = X[:,2]
            low_prices = X[:,3]
            volume = X[:,4]
            
            
            RSI = mta.RSI(close_prices, '15min', feature_names = feature_names)
            RSI_threshold_flags = mta.get_threshold_flags(RSI,
                                                          upper_thresh = 80,
                                                          lower_thresh = 20
                                                          )
            divergence_RSI = mta.RSI_divergence(close_prices, RSI, 300, feature_names = feature_names)
            
            stochastic_RSI = mta.stochastic_RSI(RSI, time_frame = 14, feature_names = feature_names)
            stochastic_RSI_threshold_flags = mta.get_threshold_flags(stochastic_RSI,
                                                                     upper_thresh = 80,
                                                                     lower_thresh = 20
                                                                     )
            divergence_stochastic_RSI = mta.RSI_divergence(close_prices, stochastic_RSI, 100, feature_names = feature_names)
            
            RSI_timeframe_48 = mta.RSI_2(close_prices, 48, feature_names = feature_names)
            RSI_timeframe_48_threshold_flags = mta.get_threshold_flags(RSI_timeframe_48,
                                                                       upper_thresh = 80,
                                                                       lower_thresh = 20,
                                                                       lam = 0.99
                                                                       )
             
            #RSI_1d = mta.RSI_2(close_prices, 14, 4 * 24, feature_names = feature_names) #original
            RSI_1d = mta.RSI_2(close_prices, 1, feature_names = feature_names) # test
            RSI_1d_threshold_flags = mta.get_threshold_flags(RSI_1d,
                                                             upper_thresh = 80,
                                                             lower_thresh = 20,
                                                             lam = 0.99
                                                             )     
            RSI_1w = mta.RSI_2(close_prices, 7, feature_names = feature_names)
            RSI_1w_threshold_flags = mta.get_threshold_flags(RSI_1w,
                                                             upper_thresh = 80,
                                                             lower_thresh = 20,
                                                             lam = 0.99
                                                             )     
            RSI_14d = mta.RSI_2(close_prices, 14, feature_names = feature_names)
            RSI_14d_threshold_flags = mta.get_threshold_flags(RSI_14d,
                                                              upper_thresh = 80,
                                                              lower_thresh = 20,
                                                              lam = 0.99
                                                              )  
            RSI_1m = mta.RSI_2(close_prices, 30, feature_names = feature_names)
            RSI_1m_threshold_flags = mta.get_threshold_flags(RSI_1m,
                                                             upper_thresh = 80,
                                                             lower_thresh = 20,
                                                             lam = 0.99
                                                             )     

            MFI = mta.money_flow_index(close_prices, 
                                       high_prices, 
                                       low_prices, 
                                       volume, 
                                       time_frame = 14, 
                                       feature_names = feature_names)

            MFI_threshold_flags = mta.get_threshold_flags(MFI,
                                                          upper_thresh = 80,
                                                          lower_thresh = 20,
                                                          lam = 0.99
                                                          )
            divergence_MFI = mta.money_flow_divergence(close_prices, MFI, 300, feature_names = feature_names)  
            
            
            Tenkan_sen, Kijun_sen, Senkou_Span_A, Senkou_Span_B, Chikou_Span, cloud_flags, tk_cross_flags, oc_cloud_flags \
                = mta.Ichimoku_Cloud_Indicator(dataset_dict, 9, 26, 26, 52, 22, feature_names = feature_names)
            
            ADL = mta.Accum_distrib(dataset_dict, feature_names = feature_names)
        
            SMA_12_day_values, EMA_12_day_values = mta.SMA_EMA(close_prices, 12, feature_names = feature_names)
            SMA_26_day_values, EMA_26_day_values = mta.SMA_EMA(close_prices, 26, feature_names = feature_names)
            
            MACD_line, MACD_signal_line, MACD_histogram, MACD_divergence = mta.MACD(close_prices, 
                                                                                    EMA_12_day_values, 
                                                                                    EMA_26_day_values,  
                                                                                    divergence_window = 300,
                                                                                    feature_names = feature_names)
            ## get MACD for 15 min
            SMA_12_day_values, EMA_12_day_values = mta.SMA_EMA(close_prices, 12, feature_names = feature_names)
            SMA_26_day_values, EMA_26_day_values = mta.SMA_EMA(close_prices, 26, feature_names = feature_names)
            
            MACD_line_15_min, \
            MACD_signal_line_15_min, \
            MACD_histogram_15_min, \
            MACD_divergence_15_min = mta.MACD(close_prices, 
                                              EMA_12_day_values, 
                                              EMA_26_day_values,  
                                              divergence_window = 300,
                                              feature_names = feature_names)
            
            SMA_12_day_values, EMA_12_day_values = mta.SMA_EMA(close_prices, 12 * 15 * 4, feature_names = feature_names)
            SMA_26_day_values, EMA_26_day_values = mta.SMA_EMA(close_prices, 26 * 15 * 4, feature_names = feature_names)
            
            MACD_line_1h, \
            MACD_signal_line_1h, \
            MACD_histogram_1h, \
            MACD_divergence_1h = mta.MACD(close_prices, 
                                          EMA_12_day_values, 
                                          EMA_26_day_values,  
                                          divergence_window = 300,
                                          feature_names = feature_names)
            
            talib_MACD = talib.MACD(close_prices)
            talib_MACD1 = talib_MACD[0]
            talib_MACD2 = talib_MACD[1]
            talib_MACD3 = talib_MACD[2]
            
            BB_SMA_values, \
            BB_upperline_values,\
            BB_lowerline_values, \
            BB_squeeze_values, \
            BB_SMA_crossings = mta.bollinger_bands(close_prices, time_period = 20, feature_names = feature_names)
            
            BB_SMA_values_12h, \
            BB_upperline_values_12h, \
            BB_lowerline_values_12h,\
            BB_squeeze_values_12h, \
            BB_SMA_crossings_12h = mta.bollinger_bands(close_prices, time_period = 20 * 4 * 12, feature_names = feature_names)
            
            BB_SMA_values_1h, \
            BB_upperline_values_1h, \
            BB_lowerline_values_1h, \
            BB_squeeze_values_1h, \
            BB_SMA_crossings_1h = mta.bollinger_bands(close_prices, time_period = 20 * 4, feature_names = feature_names)
            
            CCI, CCI_thresholds = mta.commodity_channel_index(close_prices,
                                                              high_prices, 
                                                              low_prices, 
                                                              time_period = 20,
                                                              feature_names = feature_names)
            CCI_12h, CCI_thresholds_12h = mta.commodity_channel_index(close_prices, 
                                                                      high_prices, 
                                                                      low_prices, 
                                                                      time_period = 20 * 4 * 12, 
                                                                      feature_names = feature_names)
            CCI_1h, CCI_thresholds_1h = mta.commodity_channel_index(close_prices, 
                                                                    high_prices,
                                                                    low_prices, 
                                                                    time_period = 20 * 4, 
                                                                    feature_names = feature_names)
            
            volume_by_price, slope_VBP, slope_VBP_smooth = mta.VBP(close_prices, 
                                                                   volume, 
                                                                   bins = 12, 
                                                                   lam = 0.99, 
                                                                   feature_names = feature_names)
            volume_by_price_24, slope_VBP_24, slope_VBP_smooth_24 = mta.VBP(close_prices, 
                                                                            volume, 
                                                                            bins = 24,
                                                                            lam = 0.99,
                                                                            feature_names = feature_names)
            
            nlms_indicator, nlms_smoothed_indicator = mta.NLMS_indicator(close_prices, 
                                                                         time_period = 300, 
                                                                         nlms_step = 0.1, 
                                                                         nlms_constant = 0.5, 
                                                                         lam = 0.9, 
                                                                         feature_names = feature_names)
            
#            rls_indicator_error, rls_indicator_output, rls_smoothed_indicator = mta.RLS_indicator(close_prices,
#                                                                                                  time_period = 300, 
#                                                                                                  lam = 0.995, 
#                                                                                                  delta = 1/(np.var(close_prices)*100), 
#                                                                                                  smoothing = 0.95,
#                                                                                                  dct_transform = False,
#                                                                                                  feature_names = feature_names)
#            
            ATR_EMA, ATR_EMA_Wilder =  mta.ATR(close_prices,
                                               high_prices,
                                               low_prices, 
                                               time_frame = 300, 
                                               feature_names = feature_names)
            CMF_12h = mta.Chaikin_money_flow(close_prices, 
                                             high_prices,
                                             low_prices, 
                                             volume, 
                                             time_period = 4 * 12,
                                             feature_names = feature_names)
            CMF_12h_2 = mta.Chaikin_money_flow(close_prices, 
                                               high_prices, 
                                               low_prices,
                                               volume, 
                                               time_period = 4 * 12 * 21, 
                                               feature_names = feature_names)
            
            sentiment_indicator_positive, sentiment_indicator_negative = mta.get_sentiment_indicator_from_db(dataset_dict)
            
            dict_features = {}
            dict_features[var_name_to_string(RSI = RSI)] = RSI
            dict_features[var_name_to_string(RSI_threshold_flags = RSI_threshold_flags)] = RSI_threshold_flags
            dict_features[var_name_to_string(divergence_RSI = divergence_RSI)] = divergence_RSI
            dict_features[var_name_to_string(stochastic_RSI = stochastic_RSI)] = stochastic_RSI
            dict_features[var_name_to_string(stochastic_RSI_threshold_flags = stochastic_RSI_threshold_flags)] = stochastic_RSI_threshold_flags
            dict_features[var_name_to_string(divergence_stochastic_RSI = divergence_stochastic_RSI)] = divergence_stochastic_RSI
            dict_features[var_name_to_string(MFI = MFI)] = MFI
            dict_features[var_name_to_string(MFI_threshold_flags = MFI_threshold_flags)] = MFI_threshold_flags
            dict_features[var_name_to_string(divergence_MFI = divergence_MFI)] = divergence_MFI
            dict_features[var_name_to_string(Tenkan_sen = Tenkan_sen)] = Tenkan_sen
            dict_features[var_name_to_string(Kijun_sen = Kijun_sen)] = Kijun_sen
            dict_features[var_name_to_string(Senkou_Span_A = Senkou_Span_A)] = Senkou_Span_A[:-26]
            dict_features[var_name_to_string(Senkou_Span_B = Senkou_Span_B)] = Senkou_Span_B[:-26]
            dict_features[var_name_to_string(Chikou_Span = Chikou_Span)] = Chikou_Span
            dict_features[var_name_to_string(cloud_flags = cloud_flags)] = cloud_flags[:-26]
            dict_features[var_name_to_string(tk_cross_flags = tk_cross_flags)] = tk_cross_flags
            dict_features[var_name_to_string(oc_cloud_flags = oc_cloud_flags)] = oc_cloud_flags
            dict_features[var_name_to_string(ADL = ADL)] = ADL
            dict_features[var_name_to_string(SMA_12_day_values = SMA_12_day_values)] = SMA_12_day_values
            dict_features[var_name_to_string(EMA_12_day_values = EMA_12_day_values)] = EMA_12_day_values
            dict_features[var_name_to_string(SMA_26_day_values = SMA_26_day_values)] = SMA_26_day_values
            dict_features[var_name_to_string(EMA_26_day_values = EMA_26_day_values)] = EMA_26_day_values
            dict_features[var_name_to_string(MACD_line = MACD_line)] = MACD_line
            dict_features[var_name_to_string(MACD_signal_line = MACD_signal_line)] = MACD_signal_line
            dict_features[var_name_to_string(MACD_histogram = MACD_histogram)] = MACD_histogram
            dict_features[var_name_to_string(MACD_divergence = MACD_divergence)] = MACD_divergence
            dict_features[var_name_to_string(MACD_line_15_min = MACD_line_15_min)] = MACD_line_15_min
            dict_features[var_name_to_string(MACD_signal_line_15_min = MACD_signal_line_15_min)] = MACD_signal_line_15_min
            dict_features[var_name_to_string(MACD_histogram_15_min = MACD_histogram_15_min)] = MACD_histogram_15_min
            dict_features[var_name_to_string(MACD_divergence_15_min = MACD_divergence_15_min)] = MACD_divergence_15_min
            dict_features[var_name_to_string(MACD_line_1h = MACD_line_1h)] = MACD_line_1h
            dict_features[var_name_to_string(MACD_signal_line_1h = MACD_signal_line_1h)] = MACD_signal_line_1h
            dict_features[var_name_to_string(MACD_histogram_1h = MACD_histogram_1h)] = MACD_histogram_1h
            dict_features[var_name_to_string(MACD_divergence_1h = MACD_divergence_1h)] = MACD_divergence_1h
            dict_features[var_name_to_string(talib_MACD1 = talib_MACD1)] = talib_MACD1
            dict_features[var_name_to_string(talib_MACD2 = talib_MACD2)] = talib_MACD2
            dict_features[var_name_to_string(talib_MACD3 = talib_MACD3)] = talib_MACD3
            dict_features[var_name_to_string(BB_SMA_values = BB_SMA_values)] = BB_SMA_values
            dict_features[var_name_to_string(BB_upperline_values = BB_upperline_values)] = BB_upperline_values
            dict_features[var_name_to_string(BB_lowerline_values = BB_lowerline_values)] = BB_lowerline_values
            dict_features[var_name_to_string(BB_squeeze_values = BB_squeeze_values)] = BB_squeeze_values
            dict_features[var_name_to_string(BB_SMA_crossings = BB_SMA_crossings)] = BB_SMA_crossings
            dict_features[var_name_to_string(BB_SMA_values_12h = BB_SMA_values_12h)] = BB_SMA_values_12h
            dict_features[var_name_to_string(BB_upperline_values_12h = BB_upperline_values_12h)] = BB_upperline_values_12h
            dict_features[var_name_to_string(BB_lowerline_values_12h = BB_lowerline_values_12h)] = BB_lowerline_values_12h
            dict_features[var_name_to_string(BB_squeeze_values_12h = BB_squeeze_values_12h)] = BB_squeeze_values_12h
            dict_features[var_name_to_string(BB_SMA_crossings_12h = BB_SMA_crossings_12h)] = BB_SMA_crossings_12h
            dict_features[var_name_to_string(BB_SMA_values_1h = BB_SMA_values_1h)] = BB_SMA_values_1h
            dict_features[var_name_to_string(BB_upperline_values_1h = BB_upperline_values_1h)] = BB_upperline_values_1h
            dict_features[var_name_to_string(BB_lowerline_values_1h = BB_lowerline_values_1h)] = BB_lowerline_values_1h
            dict_features[var_name_to_string(BB_squeeze_values_1h = BB_squeeze_values_1h)] = BB_squeeze_values_1h
            dict_features[var_name_to_string(BB_SMA_crossings_1h = BB_SMA_crossings_1h)] = BB_SMA_crossings_1h
            dict_features[var_name_to_string(CCI = CCI)] = CCI
            dict_features[var_name_to_string(CCI_thresholds = CCI_thresholds)] = CCI_thresholds
            dict_features[var_name_to_string(CCI_12h = CCI_12h)] = CCI_12h
            dict_features[var_name_to_string(CCI_thresholds_12h = CCI_thresholds_12h)] = CCI_thresholds_12h
            dict_features[var_name_to_string(CCI_1h = CCI_1h)] = CCI_1h
            dict_features[var_name_to_string(CCI_thresholds_1h = CCI_thresholds_1h)] = CCI_thresholds_1h
            dict_features[var_name_to_string(RSI_timeframe_48 = RSI_timeframe_48)] = RSI_timeframe_48
            dict_features[var_name_to_string(RSI_timeframe_48_threshold_flags = RSI_timeframe_48_threshold_flags)] = RSI_timeframe_48_threshold_flags
            dict_features[var_name_to_string(RSI_1d = RSI_1d)] = RSI_1d
            dict_features[var_name_to_string(RSI_1d_threshold_flags = RSI_1d_threshold_flags)] = RSI_1d_threshold_flags
            dict_features[var_name_to_string(RSI_1w = RSI_1w)] = RSI_1w
            dict_features[var_name_to_string(RSI_1w_threshold_flags = RSI_1w_threshold_flags)] = RSI_1w_threshold_flags
            dict_features[var_name_to_string(RSI_14d = RSI_14d)] = RSI_14d
            dict_features[var_name_to_string(RSI_14d_threshold_flags = RSI_14d_threshold_flags)] = RSI_14d_threshold_flags
            dict_features[var_name_to_string(RSI_1m = RSI_1m)] = RSI_1m
            dict_features[var_name_to_string(RSI_1m_threshold_flags = RSI_1m_threshold_flags)] = RSI_1m_threshold_flags
            dict_features[var_name_to_string(volume_by_price = volume_by_price)] = volume_by_price
            dict_features[var_name_to_string(slope_VBP = slope_VBP)] = slope_VBP
            dict_features[var_name_to_string(slope_VBP_smooth = slope_VBP_smooth)] = slope_VBP_smooth
            dict_features[var_name_to_string(volume_by_price_24 = volume_by_price_24)] = volume_by_price_24
            dict_features[var_name_to_string(slope_VBP_24 = slope_VBP_24)] = slope_VBP_24
            dict_features[var_name_to_string(slope_VBP_smooth_24 = slope_VBP_smooth_24)] = slope_VBP_smooth_24
            dict_features[var_name_to_string(nlms_indicator = nlms_indicator)] = nlms_indicator
            dict_features[var_name_to_string(nlms_smoothed_indicator = nlms_smoothed_indicator)] = nlms_smoothed_indicator                                          
            dict_features[var_name_to_string(ATR_EMA = ATR_EMA)] = ATR_EMA
            dict_features[var_name_to_string(ATR_EMA_Wilder = ATR_EMA_Wilder)] = ATR_EMA_Wilder  
            dict_features[var_name_to_string(CMF_12h = CMF_12h)] = CMF_12h    
            dict_features[var_name_to_string(CMF_12h_2 = CMF_12h_2)] = CMF_12h_2
#            dict_features[var_name_to_string(rls_indicator_error = rls_indicator_error)] = rls_indicator_error
#            dict_features[var_name_to_string(rls_indicator_output = rls_indicator_output)] = rls_indicator_output                                         
#            dict_features[var_name_to_string(rls_smoothed_indicator = rls_smoothed_indicator)] = rls_smoothed_indicator                                         
            dict_features[var_name_to_string(sentiment_indicator_positive = sentiment_indicator_positive)] = sentiment_indicator_positive
            dict_features[var_name_to_string(sentiment_indicator_negative = sentiment_indicator_negative)] = sentiment_indicator_negative
            
            print('X initial shape ' + str(np.shape(X)))
            for i in feature_names:
                try:
                    X = md.concatenator([X, dict_features[i]])
                except:
                    print('not appending feature: ' + i)
                    pass
        if(lookback):
            X = get_lookback_data(X, lookback)
        dict_features[var_name_to_string(close_prices = close_prices)] = close_prices
        dict_features[var_name_to_string(high_prices = high_prices)] = high_prices
        dict_features[var_name_to_string(low_prices = low_prices)] = low_prices
        dict_features[var_name_to_string(volume = volume)] = volume
        
        MACD_flag = True
        RSI_flag = True
        if((normalization_method == "rescale") or (normalization_method == "standardization")):
            if index_proc == 'preprocessed':
                X1 = copy.copy(X)
            else:            
                return_dataset = {}
                return_dataset['preprocessed_data'] = X1
                return_dataset['data'] = X
                if(normalization_method != "standardization"):
                    return_dataset['min_prices'] = min_prices
                    return_dataset['min_volume'] = min_volume
                    return_dataset['max_prices'] = max_prices
                    return_dataset['max_volume'] = max_volume
                return_dataset['dataset_dict'] = dataset_dict
                return_dataset['features'] = dict_features
                
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
            
#        if(normalization_method == "standardization"):
#            X_norm_prices = mn.normalize_standardization(X[:,0:4])
#            X_norm_volume = mn.normalize_standardization(X[:,4])
#            if(MACD_flag):
#                X_norm_MACD = mn.normalize_standardization(X[:,5:8])
#            if(RSI_flag):
#                X_norm_RSI = mn.normalize_standardization(X[:,8])
#            
#            X = md.concatenator([X_norm_prices,X_norm_volume, X_norm_MACD,X_norm_RSI])
#            if index_proc == 'preprocessed':
#                X1 = copy.copy(X)
#            else:            
#                return_dataset = {}
#                return_dataset['preprocessed_data'] = X1
#                return_dataset['data'] = X
#                return_dataset['dataset_dict'] = dataset_dict            
#        elif concatenate_datasets_preproc_flag:
#            X = md.concatenator([X,X1])
            
    return return_dataset

def get_test_and_train_data(preprocessed_data, unprocessed_data, chunks, chunks_for_training, remove_chunks_from_start):
#    chunks = 11
#    chunks_for_training = 9
    if((remove_chunks_from_start + chunks_for_training) > chunks):
        print("Invalid chunk values")
        return 0
    X_chunks = md.get_split(preprocessed_data,chunks)
    X_chunks_unprocessed = md.get_split(unprocessed_data,chunks)
    train_data = X_chunks[remove_chunks_from_start : (remove_chunks_from_start + chunks_for_training)]
    train_data_unprocessed = X_chunks_unprocessed[remove_chunks_from_start : (remove_chunks_from_start + chunks_for_training)]
    test_data = X_chunks[(chunks_for_training + remove_chunks_from_start):]
    test_data_unprocessed = X_chunks_unprocessed[(chunks_for_training + remove_chunks_from_start):]
    
    train_data = np.reshape(train_data, ( train_data.shape[0] * train_data.shape[1], train_data.shape[2] ))
    train_data_unprocessed = np.reshape(train_data_unprocessed, ( train_data_unprocessed.shape[0] * train_data_unprocessed.shape[1], train_data_unprocessed.shape[2] ))
    test_data = np.reshape(test_data, ( test_data.shape[0] * test_data.shape[1], test_data.shape[2] ))
    test_data_unprocessed = np.reshape(test_data_unprocessed, ( test_data_unprocessed.shape[0] * test_data_unprocessed.shape[1], test_data_unprocessed.shape[2] ))

    dict_data = {}
    dict_data["test_data_preprocessed"] = test_data
    dict_data["test_data_unprocessed"] = test_data_unprocessed
    dict_data["train_data_preprocessed"] = train_data
    dict_data["train_data_unprocessed"] = train_data_unprocessed
    return dict_data

def get_feature_importances(feature_names, log_path):
    try:
        f = open(log_path, "r")
    except:
        print("opening "+ log_path + " failed")
        return {}
    text = f.read()
    f.close()
    index_features = text.find('feature_importance')
    new_text = text[index_features : index_features + 700]
    first_bracket_index = new_text.find('[')
    last_bracket_index = new_text.find(']')
    values_text = new_text[(first_bracket_index + 1) : last_bracket_index].replace('\n', '')
    
    ceva = values_text.split(' ')
    values = []
    for i in ceva:
        if i != '':
            values.append(float(i))
    if(len(values) != len(feature_names)):
        print('feature imp sizes do not match')
        return {}        
    dict_feature_importances = {}        
    for i in range(len(feature_names)):
        dict_feature_importances[feature_names[i]] = values[i]
    return dict_feature_importances

def get_lookback_data(X, lookback = 4):
    num_features = np.shape(X)[1]
    ceva = np.zeros(num_features * lookback)
    a = []
    for i in range(np.shape(X)[0]):
        ceva = np.hstack((X[i,:], ceva[:-num_features]))
        a.append(ceva)
    return np.array(a)