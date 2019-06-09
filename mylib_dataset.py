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
import mylib_normalize as mn
import mylib_TA as mta
#import mylib_blockchain as mb
#import talib 
import datetime
import calendar
import pickle

from sklearn import preprocessing
import matplotlib.pyplot as plt


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

def get_diff_raw(data_in, data, window):
    X = []
    Y = []
    for i in range(len(data)-window):
        X.append(data_in[i,:])
        Y.append((np.mean(data[i+1:i+window+1]) - data[i])/data[i])
        if(i<10):
            print('...',window, data[i+1:i+window+1], data[i], data[i+window])
            print(np.mean(data[i+1:i+window+1]), data[i], Y[-1])
    return np.array(X), np.array(Y)

def get_volatile_bool(data_in, data, window, target_movement = 'up', var_thresh_ratio = 0.5, apriori_window = 96):
    X = []
    Y = []
    label_weights = []
    cirbuf = np.zeros(apriori_window)
    print('dbg ',target_movement, window)
    for i in range(len(data)-window):
        X.append(data_in[i,:])
        cirbuf = np.hstack(((data[i] - data[i-1]) / data[i], cirbuf[:-1]))
        var_thr = np.sqrt(np.var(cirbuf)) * var_thresh_ratio
        if(target_movement == 'absolute'):
            next_percentage_change = np.abs(np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
        else:
            next_percentage_change = (np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
        if(target_movement == 'up'):
            if(next_percentage_change >= var_thr):
                Y.append(1)
                label_weights.append(1 + next_percentage_change)
            else:
                Y.append(-1)
                label_weights.append(1)
        elif(target_movement == 'down'):
            if(next_percentage_change <= -var_thr):
                Y.append(1)
                label_weights.append(1+next_percentage_change)
            else:
                Y.append(-1) 
                label_weights.append(1)
        elif(target_movement == 'absolute'):
            if(next_percentage_change >= var_thr):
                Y.append(1)
                label_weights.append(1+next_percentage_change)
            else:
                Y.append(-1) 
                label_weights.append(1)
    return np.array(X), np.asarray(Y), np.array(label_weights)

#def get_thr_trend(data_in, data, window):
#    X = []
#    Y = []
#    timespan = 96
#    cirbuf = np.zeros(timespan)
#    for i in range(len(data)-window):
#        X.append(data_in[i,:])
#        cirbuf = np.hstack(((data[i] - data[i-1]) / data[i], cirbuf[:-1]))
#        var_thr = np.sqrt(np.var(cirbuf))/2
#        next_percentage_change = (np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
#        if(next_percentage_change >= var_thr):
#            Y.append(1.0)
#        else:
#            Y.append(-1.0)
#    return np.array(X), np.array(Y)
##        
#def get_labels_mean_window(data_in, data, window_length, label_type = '', thresh_ratio = 1.2):
#    #in_data = data['dataset_dict']['close']
#    dataX, dataY = [], []
#    mean_lows = []
#    mean_highs = []
#    
#    for i in range(len(data) - window_length):
#        dataX.append(data_in[i,:])
#        up = [abs(data[i+j] - data[i])  if data[i+j] >= data[i] else 0 for j in range(1,window_length+1)]
#        down = [abs(data[i+j] - data[i])  if data[i+j] < data[i] else 0 for j in range(1,window_length+1)]
#        mean_up = np.mean(up)
#        mean_down = np.mean(down)
#        if(label_type):
#            if(label_type == 'bool_up'):
#                dataY.append(str(mean_up > mean_down * thresh_ratio))
#                #print('up:',up,'down:', down, 'meanup:', mean_up, 'meandown:',mean_down, 'y:', dataY[-1], 'i:',i)
#            elif(label_type == 'bool_down'):
#                dataY.append(str(mean_up * thresh_ratio < mean_down))
#            elif(label_type == 'ternary'):
#                if(mean_up > mean_down * thresh_ratio):
#                    dataY.append('bull')
#                elif (mean_up * thresh_ratio < mean_down):
#                    dataY.append('bear')
#                else:
#                    dataY.append('crab')
#            elif(label_type == 'ternary2'):
#                if(data[i + window_length] > data[i] * thresh_ratio):
#                    dataY.append('bull')
#                elif (data[i + window_length] * thresh_ratio < data[i]):
#                    dataY.append('bear')
#                else:
#                    dataY.append('crab')
#            elif(label_type == 'raw'):
#                dataY.append(data[i + window_length])
#            elif(label_type == 'diff'):
#                dataY.append(data[i + window_length] - data[i])
#            else:
#                print('Unknown label type: ', label_type)
#        else:
#            dataY.append( (mean_up, mean_down))
#        mean_highs.append(mean_up)
#        mean_lows.append(mean_down)
#    return np.array(dataX), np.array(dataY), mean_highs, mean_lows
def get_labels_mean_window(data_in, data, window_length, label_type = '', thresh_ratio = 1.2):
    #in_data = data['dataset_dict']['close']
    dataX, dataY = [], []
    mean_lows = []
    mean_highs = []
    
    for i in range(len(data) - window_length):
        dataX.append(data_in[i,:])
        up = [abs(data[i+j] - data[i])  if data[i+j] >= data[i] else 0 for j in range(1,window_length+1)]
        down = [abs(data[i+j] - data[i])  if data[i+j] < data[i] else 0 for j in range(1,window_length+1)]
        mean_up = np.mean(up)
        mean_down = np.mean(down)
        if(label_type):
            if(label_type == 'bool_up'):
                #dataY.append((data_in[i+window_length,0] - data_in[i,0])>0)
                    
                dataY.append((data_in[i+window_length,0] - data_in[i,0])>0)
                if(i<20):
                    print('i', i, 'win:', window_length, 'data[i]:',data[i],'data[i+win]:', data[i + window_length],'y:',dataY[-1]  )
                #print('up:',up,'down:', down, 'meanup:', mean_up, 'meandown:',mean_down, 'y:', dataY[-1], 'i:',i)
            elif(label_type == 'bool_down'):
                dataY.append(str(mean_up * thresh_ratio < mean_down))
            elif(label_type == 'ternary'):
                w_var = 16
                if(i >= w_var):
                    thr = np.sqrt(np.var(data[i-w_var:i]))
                else:
                    thr = 0
                a = data[i+window_length]
                b = data[i]
                if((a > (b - thr)) and (a < (b + thr))):
                    dataY.append('crab')
                elif(a >= (b + thr)):
                    dataY.append('bull')
                else:
                    dataY.append('bear')
                if((i<20) and (i>w_var)):
                    print('i',i, 'thr', thr, 'a',a,'b',b,'b+-thr',(b+thr,b-thr),'y', dataY[-1]) 
#                if(mean_up > mean_down * thresh_ratio):
#                    dataY.append('bull')
#                elif (mean_up * thresh_ratio < mean_down):
#                    dataY.append('bear')
#                else:
#                    dataY.append('crab')
            elif(label_type == 'ternary2'):
                if(data[i + window_length] > data[i] * thresh_ratio):
                    dataY.append('bull')
                elif (data[i + window_length] * thresh_ratio < data[i]):
                    dataY.append('bear')
                else:
                    dataY.append('crab')
            elif(label_type == 'raw'):
                dataY.append(data[i + window_length])
            elif(label_type == 'diff'):
                dataY.append(data[i + window_length] - data[i])
            else:
                print('Unknown label type: ', label_type)
        else:
            dataY.append( (mean_up, mean_down))
        mean_highs.append(mean_up)
        mean_lows.append(mean_down)
    return np.array(dataX), np.array(dataY), mean_highs, mean_lows

#def get_labels_mean_window2(data_in, data, window_length, label_type = '', thresh_ratio = 1.2):
#    #in_data = data['dataset_dict']['close']
#    dataX, dataY = [], []
#    mean_lows = []
#    mean_highs = []
#    
##    d_open = data['open']
##    d_close = data['open']
##    d_open = data['open']
##    d_open = data['open']
#    for i in range(len(data) - window_length):
#        dataX.append(data_in[i,:])
#        thr = np.mean(data[i-20:i])
#        if(data[i+1] > data[i] * thr):
#            
##    for i in range(len(data['high']) - window_length):
##        dataX.append(data_in[i,:])
##        dataY.append(str(np.sum(abs(data['high'][i:i+window_length] - np.repeat(data['close'][i],window_length))) \
##                     >= np.sum(abs(data['low'][i:i+window_length] - np.repeat(data['close'][i],window_length)))))
##   
#    
#    return np.array(dataX), np.array(dataY), mean_highs, mean_lows


def get_date_from_UTC_ms(UTC_time):
    #UTC_time = data['dataset_dict']['UTC'][0]
    date_dict = {}
    date_datetime = datetime.datetime.utcfromtimestamp(UTC_time/1000)       
    date = date_datetime.isoformat()[:date_datetime.isoformat().find('T')]
    date_dict['date_str'] = date
    date_dict['date_datetime'] = date_datetime
   # print('UTC time:' + str(UTC_time) + ' --- date: ' + date)
    return date_dict


def datetime_local_to_gmt_UTC(dt):
    return calendar.timegm(dt.utctimetuple())


def get_datetime_from_string(date_string):
    year = date_string[:date_string.find('-')]
    month = date_string[date_string.find('-'):][:date_string.find('-')].replace('-','')
    day = date_string[date_string.find('-'):][date_string.find('-'):][:date_string.find('-')][:2]
    if(date_string.find(':') is not -1):
        hour  = date_string[:date_string.find(':')][-2:]
        minutes = date_string[date_string.find(':'):][1:3]
    return datetime.datetime(int(year),int(month), int(day), int(hour), int(minutes))

def get_data_interval(data_UTC, start_date, end_date):
    start_UTC_ms = datetime_local_to_gmt_UTC(start_date) * 1000
    end_UTC_ms = datetime_local_to_gmt_UTC(end_date) * 1000
    start_index = 0
    end_index = 0
    for i in range(len(data_UTC)):
        if(data_UTC[i] == start_UTC_ms):
            start_index = i
        if(data_UTC[i] == end_UTC_ms):
            end_index = i
            break
    print('start_UTC_ms: ', start_UTC_ms, ' end_UTC_ms: ', end_UTC_ms) 
    print('start index: ', start_index, ' end_index: ', end_index) 
    return {'start':start_index,'end':end_index}

def get_split(data,chunks):
    if not chunks: return np.array(data)
    try:
       data = np.split(data,chunks)
    except:
       data = data[:-(len(data) - int(len(data)/chunks)*chunks)]
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

def bin_class_perf_indicators(results, reference, label_pos, label_neg):
    true_positive = true_negative = false_positive = false_negative = 0
    dict_indicators = {}
    
    ## confusion matrix
    for i in range (len(results)):
        if(results[i] == label_pos):
            if(results[i] == reference[i]):
                true_positive += 1
            else:
                false_positive += 1
                
        if(results[i] == label_neg):
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

def get_sentiment_indicators(dataset_path = '/home/catalin/databases/TA_SECURITY_SENTIMENT.csv',
                             symbol = 'BTC',
                             utc_data = None,
                             dict_param = {}):
    if(dict_param):
        if('skip' in dict_param):
            if(dict_param['skip'] == True):
                return {}, 0 , 0
    else:
        return {}, 0 , 0
        
        
    ### reshapes indicator by db timestamps
    try:
        dataset = pd.read_csv(dataset_path)
    except:
        print("Could not read file :", dataset_path)
    
    a = dataset.to_dict()
    db_keys = list(a.keys())
    vlen = len(list(a[db_keys[0]].values()))
    
    BTC_indexes = []
    for i in range(vlen):
        if(a['SYMBOL'][i] == symbol) and (a['TIME_INTERVAL'][i] == 'HOUR'):
            BTC_indexes.append(i)
            
    d_item = {}
    for key in db_keys:
        d_item[key] = []
    for i in BTC_indexes:
        for key in db_keys:
            d_item[key].append(a[key][i])
            
    
    
    ########### match UTC vals
    to_date = [datetime_local_to_gmt_UTC(get_datetime_from_string(i)) for i in list(d_item['DATE_TO'])]
    vals = np.zeros(len(utc_data))
    
    k = 1
    value = to_date[0]
    to_len = len(to_date)
    start, end = 0, 0
    
    return_keys = db_keys[6:-4] ## indicator keys
    return_dict = {}
    for i in return_keys:
        return_dict[i] = np.zeros(len(utc_data))
        
    for i in range(len(utc_data)):
        if(utc_data[i] >= to_date[0]):
            if(not start):
                start = i
            if(utc_data[i] <  to_date[k]):
                vals[i] = value
                for j in return_keys:
                    return_dict[j][i] = d_item[j][k]
            else:
                value = to_date[k]
                vals[i] = value
                for j in return_keys:
                    return_dict[j][i] = d_item[j][k]
                k += 1
                if(k >= to_len):
                    end = i
                    break
    ########### end match
                
    return return_dict, start, end

def get_dataset_multiple_csv(dataset_directory):
    print("Fetching data from ", dataset_directory, '...')
#    dictionary = ['UTC', 'open', 'close', 'high', 'low', 'volume']    
    os.chdir(dataset_directory)
    x = subprocess.check_output(' ls | wc -l',shell = 'True')
    x = int(parser(str(x)))
        
    if(x == 0):
        print('No files found!')
    dataset = [0 for _ in range (x)]
#    X1 = ['none']
               

    for ii in range (0,x):## minus readme file
        try:
            dataset[ii] = pd.read_csv(dataset_directory + '/klines_15min_' +str(ii) +'.csv')
            dataset[ii] = dataset[ii].values
            dataset[ii] = np.array(sorted(dataset[ii], key=lambda x: x[0])) ## fix for csv UTC time in descending order
        except:
            print('could not read: ',ii)
            dataset[ii] = np.zeros([246, 6])
        if ii == 0:
             whole_dataset = dataset[ii]
        else:
             whole_dataset = np.concatenate( [ whole_dataset, dataset[ii] ] )
    ############# fill missing
    
    ## remove zore values from the start.. these are caused by the listing time of the pairing.
    for i in range(len(whole_dataset)):
        if(whole_dataset[i][1] > 0):
#            print('debug '+ str(i) + ' data: ' + str(whole_dataset[i][0]))
            whole_dataset = whole_dataset[i:,:] 
            break  
    
    return whole_dataset
    
def get_dataset_single_csv(dataset_directory = '/home/catalin/git_workspace/disertatie/databases/btc_klines_2014_2019.csv'):
    ## still ochlv
    print("Fetching data from ", dataset_directory, '...')
    try:
        dataset = pd.read_csv(dataset_directory)
    except:
        print("Could not read file :", dataset_directory)
        return []
    dataset = dataset.as_matrix
    dataset = dataset()
    tmp = np.zeros((np.shape(dataset)[0],np.shape(dataset)[1]-1))
    for i in range(len(dataset)):
        tmp[i][:] = dataset[i][1:]
    whole_dataset = np.array(sorted(tmp, key=lambda x: x[0])) ## fix for csv UTC time in descending order
    
    return whole_dataset

def fill_missing_data(whole_dataset):
    for i in range (1, len(whole_dataset)):
        if whole_dataset[i][0] == 0:
            whole_dataset[i] = whole_dataset[i-1]
            whole_dataset[i][0] += 900000        
    return whole_dataset

def fill_missing_data2(whole_dataset):
    data = []
    cnt_missing_candles = 0
    for i in range (len(whole_dataset)):
        if(i>0):
            diff = whole_dataset[i][0] - whole_dataset[i-1][0]
            if(diff > 900000):
                #print('dbg diff missing utc',diff, 'start', whole_dataset[i-1][0],'end',whole_dataset[i][0], 'candles', int(diff / 900000))
                tmp = whole_dataset[i-1]
                for j in range(int(diff / 900000)-1):
                    tmp[0] = tmp[0] + 900000
                   # print('adding', tmp[0])
                    data.append(tmp)
                    cnt_missing_candles += 1
            data.append(whole_dataset[i])
        else:
            data.append(whole_dataset[i])
    print('cnt missing candles', cnt_missing_candles)
    return np.array(data)
        
def get_database_data(dataset_directory, 
                      normalization_method = 'rescale',
                      datetime_interval = {},
                      preproc_constant = 0, 
                      lookback = 0,
                      input_noise_debug = False,
                      dataset_type = 'dataset1'):
    print("\nDataset params: ",
          '\n\tdataset_type: ', dataset_type,
          '\n\tdatetime_interval:',datetime_interval,
          '\n\tlookback:', lookback,
          '\n\tnormalization_method:',normalization_method)
    #########
    #mtaS 	int 	millisecond time stamp
    #OPEN 	float 	First execution during the time frame
    #CLOSE 	float 	Last execution during the time frame
    #HIGH 	float 	Highest execution during the time frame
    #LOW 	float 	Lowest execution during the timeframe
    #VOLUME 	float 	Quantity of symbol traded within the timeframe
    #########
    dictionary = ['UTC', 'open', 'close', 'high', 'low', 'volume']   
    print('Dataset features:',
          '\n\t', dictionary)
    print('path:', dataset_directory)
    if(dataset_type == 'dataset1'): 
        whole_dataset = get_dataset_multiple_csv(dataset_directory)
    elif(dataset_type == 'dataset2'):
        whole_dataset = get_dataset_single_csv(dataset_directory)
    else:
        print('Unknown dataset type')
    
    whole_dataset = fill_missing_data(whole_dataset)
#    whole_dataset = fill_missing_data2(whole_dataset)
    ############# 

    if(np.shape(whole_dataset)[0] == 0):
        print('Could not get data. Aborting..')
        return {}
    
    if(datetime_interval):
        interval_index = get_data_interval(whole_dataset[:,0], 
                                           datetime_interval['start'],
                                           datetime_interval['end'])
        whole_dataset = whole_dataset[interval_index['start'] : interval_index['end']]
    #whole_dataset = whole_dataset[:-1] ## remove last candle
    
    dataset_dict = {}
    for i in range (len(whole_dataset.T)):
        dataset_dict[dictionary[i]] = whole_dataset[:,i]                    
    X = copy.copy(whole_dataset)
    X = X[:,1:] ## remove time column
    print('\nmost recent candle:', whole_dataset[-1,0], get_date_from_UTC_ms(whole_dataset[-1,0])['date_datetime'])
    ### preprocess
    if(preproc_constant):
        print('Preprocessing..')
        lam = preproc_constant
        for i in range(1,X.shape[0]):
            for j in range(X.shape[1]):
                X[i][j] = X[i-1][j] * lam + (1 - lam) * X[i][j]
        #### end preprocess

    if(normalization_method == "rescale"):
        print('Norming rescale..')
        X[:,0:4], min_prices, max_prices = mn.normalize_rescale(X[:,0:4])
        X[:,4], min_volume, max_volume = mn.normalize_rescale(X[:,4])

        
#    if(normalization_method == "standardization"):
#        print('Norming standarzation..')
#        from sklearn import preprocessing
#        scaler = preprocessing.StandardScaler()
#        X = scaler.fit_transform(X)
      
    if(input_noise_debug):
        print("input_noise_debug active. Replacing data with noise..")
        X = np.random.random(np.shape(X))
    
    dict_data = {'raw_dict': dataset_dict, 'X': X, 'raw_data': whole_dataset[:,1:], 'UTC':  whole_dataset[:,0]}
    print('\nDB dataset shape:', whole_dataset.shape)
    print('returning dataset dictionary with keys:',
          '\n\t',dict_data.keys(),
          "\n\t'raw_dict' -> ",dataset_dict.keys(),
          "\n\t'X' -> concatenated ochlv",
          "\n\t'raw_data' -> concatenated ochlv unprocessed"
          "\n\t'UTC' -> UTC timestamps"
          '\n')

    return dict_data


def append_ohclv_features(candle_data, feature_dicts):
    open_prices = candle_data[:,0]
    close_prices = candle_data[:,1]
    high_prices = candle_data[:,2]
    low_prices = candle_data[:,3]
    volume = candle_data[:,4]

    dict_candles  = {
                    'open': open_prices,
                    'close': close_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'volume': volume
                    }
    dict_ohclv = {}   
    
    
    keys = list(feature_dicts['ochlv'].keys())
    for key in keys:
        if(feature_dicts['ochlv'][key] == True):
            dict_ohclv.update({key : dict_candles[key]})
    
    return dict_ohclv
        

def get_features(candle_data, feature_dicts, blockchain_indicators_dicts, lookback = None, normalization = None, utc_time = None, reshape = None, 
                 dataset_path_sentiment = None):        
        
    dict_features = {}
    X = []
    open_prices = candle_data[:,0]
    close_prices = candle_data[:,1]
    high_prices = candle_data[:,2]
    low_prices = candle_data[:,3]
    volume = candle_data[:,4]
        
    print('Computing features...')
    
    dict_ohclv = append_ohclv_features(candle_data, feature_dicts)
    if(dict_ohclv):
        dict_features.update(dict_ohclv)
    
    if(blockchain_indicators_dicts):
        feature_dicts.update(blockchain_indicators_dicts)
        print('debug ', blockchain_indicators_dicts)
            
    dict_RSI = mta.RSI(close_prices, feature_dicts['RSI'])### returns dict of RSI timeframes
    if(dict_RSI):
        dict_features.update(dict_RSI)
        
    dict_snr = mta.snr(close_prices, feature_dicts['nsr'])
    if(dict_snr):
        dict_features.update(dict_snr)
        
    dict_MFI = mta.money_flow_index(close_prices, high_prices, low_prices, volume, feature_dicts['MFI'])
    if(dict_MFI):
        dict_features.update(dict_MFI)
            
    dict_ATR = mta.ATR2(close_prices, high_prices, low_prices, feature_dicts['ATR'])
    if(dict_ATR):
        dict_features.update(dict_ATR)
        
    dict_Ichimoku = mta.Ichimoku_Cloud_Indicator({'high':high_prices,
                                                   'low':low_prices,
                                                   'close':close_prices},
                                                   feature_dicts['IKH'])
    if(dict_Ichimoku):
        dict_features.update(dict_Ichimoku)
        
    dict_ADL = mta.Accum_distrib({'high':high_prices,
                                   'low':low_prices,
                                   'close':close_prices,
                                   'volume':volume}, 
                                    feature_dicts['ADL'])
    if(dict_ADL):
        dict_features.update(dict_ADL)
        
    dict_SMA_EMA = mta.SMA_EMA(close_prices, feature_dicts['SMA_EMA'])
    if(dict_SMA_EMA):
        dict_features.update(dict_SMA_EMA)
        
    dict_MACD = mta.MACD(close_prices, feature_dicts['MACD'])
    if(dict_MACD):
        dict_features.update(dict_MACD)
        
    dict_bollinger_bands = mta.bollinger_bands(close_prices, feature_dicts['BB'])
    if(dict_bollinger_bands):
        dict_features.update(dict_bollinger_bands)
        
    dict_CCI = mta.commodity_channel_index(close_prices, high_prices, low_prices, feature_dicts['CCI'])
    if(dict_CCI):
        dict_features.update(dict_CCI)
        
    dict_VBP = mta.VBP(close_prices, volume, feature_dicts['VBP'])
    if(dict_VBP):
        dict_features.update(dict_VBP)
        
    dict_NLMS = mta.NLMS_indicator(close_prices, feature_dicts['NLMS'])
    if(dict_NLMS):
        dict_features.update(dict_NLMS)
        
    dict_CMF = mta.Chaikin_money_flow(close_prices, high_prices, low_prices, volume, feature_dicts['CMF'])
    if(dict_CMF):
        dict_features.update(dict_CMF)
        
    dict_mean = mta.previous_mean(close_prices, feature_dicts['previous_mean'])
    if(dict_mean):
        dict_features.update(dict_mean)
        
    dict_var = mta.previous_var(close_prices, feature_dicts['previous_var'])
    if(dict_var):
        dict_features.update(dict_var)
        
    dict_adx = mta.ADX(close_prices, high_prices, low_prices, feature_dicts['ADX'])
    if(dict_adx):
        dict_features.update(dict_adx)
        
    dict_ema = mta.EMA(close_prices, feature_dicts['EMA'])
    if(dict_ema):
        dict_features.update(dict_ema)
    
    dict_VEMA = mta.EMA(volume, feature_dicts['volume_EMA'], identifier = 'volume')
    if(dict_VEMA):
        dict_features.update(dict_VEMA)
    
    dict_FRLS = mta.FRLS(close_prices, feature_dicts['FRLS'])
    if(dict_FRLS):
        dict_features.update(dict_FRLS)

    dict_OBV = mta.OBV(close_prices, volume, feature_dicts['OBV'])
    if(dict_OBV):
        dict_features.update(dict_OBV)

    dict_STO = mta.STO(close_prices, high_prices, low_prices, feature_dicts['STO'])
    if(dict_STO):
        dict_features.update(dict_STO)

    dict_TRIX = mta.TRIX(close_prices, feature_dicts['TRIX'])
    if(dict_TRIX):
        dict_features.update(dict_TRIX)

    dict_my_diff = mta.my_diff(close_prices, feature_dicts['MD'])
    if(dict_my_diff):
        dict_features.update(dict_my_diff)
 
    dict_ohcl_diff = mta.ohcl_diff(open_prices, close_prices, high_prices, low_prices, feature_dicts['ohcl_diff'])
    if(dict_ohcl_diff):
        dict_features.update(dict_ohcl_diff)
        
    dict_sentiment_db, start_utc, end_utc = get_sentiment_indicators(dataset_path = dataset_path_sentiment,
                                                                     symbol = 'BTC',
                                                                     utc_data = np.multiply(utc_time, 1/1000),
                                                                     dict_param =  feature_dicts['SENT'])
    
    dict_peaks = mta.thresholding_algo(close_prices, 50, 5, 0.9, feature_dicts['peaks'])
    if(dict_peaks):
        dict_features.update(dict_peaks)
    print('start',start_utc, end_utc)
    if(dict_sentiment_db):
        dict_features.update(dict_sentiment_db)
            
    index = 0
    X = np.zeros((len(open_prices), len(dict_features)))
    for key in dict_features:
        tmp = np.reshape(dict_features[key], len(dict_features[key]))
        X[:,index] = tmp
        index += 1
    if(lookback):
        X = get_lookback_data(X, lookback)  
    if(reshape):
        X = X[reshape['start'] : reshape['end'], :]
    
    if(normalization == 'standardization'):        
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    elif(normalization == 'minmax'):
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    else:
        scaler = None
#        for i in range(np.shape(X)[1]):
#            X[:,i] = X[:,i] / np.linalg.norm(X[:,i])
    return_dataset = {}
    return_dataset['data'] = X
    return_dataset['features'] = dict_features
    return_dataset['scaler'] = scaler
    return_dataset['UTC'] = utc_time
    
    
    print('Feature matrix shape:', X.shape)
    print('returning feature dictionary with keys:',
          '\n\t',return_dataset.keys(),
          "\n\t'data' -> ", 'concatenated features',
          "\n\t'features' -> feature dictionary"
          "\n\t'UTC' -> utc_time"
          '\n')

    return return_dataset

def cross_val_data_split(x1, x2, cross_vals):
    testing_split_index = cross_vals
    numbers_of_chunks, chunk_size, number_of_feats = np.shape(x1)
    train_data = x1[:testing_split_index,:,:]
    train_data_unprocessed = x2[:testing_split_index,:,:]
    
    train_data = np.concatenate([train_data, x1[testing_split_index+1:,:,:]])
    train_data = np.reshape(train_data, (train_data.shape[0] * train_data.shape[1], train_data.shape[2]))
    train_data_unprocessed = np.concatenate([train_data_unprocessed, x2[testing_split_index+1:,:,:]])
    train_data_unprocessed = np.reshape(train_data_unprocessed, (train_data_unprocessed.shape[0] * train_data_unprocessed.shape[1], train_data_unprocessed.shape[2]))

    test_data = x1[testing_split_index, :, :]
#    test_data = np.reshape(test_data, (test_data.shape[0] * test_data.shape[1], test_data.shape[2]))
    test_data_unprocessed = x2[testing_split_index, :, :]
#    test_data_unprocessed = np.reshape(test_data_unprocessed, (test_data_unprocessed.shape[0] * test_data_unprocessed.shape[1], test_data_unprocessed.shape[2]))
#    
    ret_dict = {}
    ret_dict['test_data_preprocessed'] = test_data
    ret_dict['test_data_unprocessed'] = test_data_unprocessed
    ret_dict['train_data_preprocessed'] = train_data
    ret_dict['train_data_unprocessed'] = train_data_unprocessed
    
    print('train ',np.shape(test_data))
    print('test ',np.shape(train_data))
    return ret_dict
    
    
def get_test_and_train_data(preprocessed_data, unprocessed_data, chunks, chunks_for_training, remove_chunks_from_start, cross_validation = None):
#    chunks = 11
#    chunks_for_training = 9
    if((remove_chunks_from_start + chunks_for_training) > chunks):
        print("Invalid chunk values")
        return 0
    X_chunks = get_split(preprocessed_data,chunks)
    X_chunks_unprocessed = get_split(unprocessed_data,chunks)
    if(cross_validation):
        dict_data = cross_val_data_split(X_chunks, X_chunks_unprocessed, cross_validation)
    else:        
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
    new_text = text[index_features : index_features + 1000]
    first_bracket_index = new_text.find('[')
    last_bracket_index = new_text.find(']')
    values_text = new_text[(first_bracket_index + 1) : last_bracket_index].replace('\n', '')
    
    ceva = values_text.split(' ')
    values = []
    for i in ceva:
        if i != '':
            values.append(float(i))
    if(len(values) != len(feature_names)):
        print('feature imp sizes do not match. parsed size: ',np.array(values).shape, ' features size: ', np.array(feature_names).shape)
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

def order_performance_dict(dict_features):
    ## key = feature name, value = list of performance results
    a = copy.copy(dict_features)
    a_k = list(a.keys())
    a_v = list(a.values())
    a_v_tmp = copy.copy(a_v)
    ordered_values = {'keys': [], 'values': []}
    while(1):
        try:
            max_value = max(a_v_tmp)
            index = a_v.index(max_value)
            ordered_values['keys'].append(a_k[index])
            ordered_values['values'].append(max_value)
            del a_v_tmp[a_v_tmp.index(max_value)]
        except:
            break
    return ordered_values


def get_feature_importances_mean(path = ''):

#with open('/home/catalin/git_workspace/disertatie/dict_perf_feats.pkl', 'rb') as f:
    with open(path, 'rb') as f:
       feature_performances = pickle.load(f)
       
    ### compute mean and variance
    mean_ft = {}
    for i in feature_performances:
        mean_ft[i] = np.mean(feature_performances[i])
        
    var_ft = {}
    for i in feature_performances:
        var_ft[i] = np.var(feature_performances[i])
    
    ### sort mean values in descending order with it's corresponding variance value
    
    ordered_values_mean = order_performance_dict(mean_ft)
    ordered_values_var = {'keys': [], 'values': []}
    for i in ordered_values_mean['keys']:
        ordered_values_var['keys'].append(i)
        ordered_values_var['values'].append(var_ft[i])
    return ordered_values_mean, ordered_values_var

def myalarm(path = ''):
    from pygame import mixer # Load the required library
    import time
    
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()
    time.sleep(6)
    mixer.music.stop()
    
    
def get_volatile_bool_debug(test_train_data,predictions, data, window, target_movement, var_thresh_ratio = 0.5, plot_predictions = True):
    Y = []
    timespan = 96
    cirbuf = np.zeros(timespan)
    for i in range(len(data)-window):
        cirbuf = np.hstack(((data[i] - data[i-1]) / data[i], cirbuf[:-1]))
        var_thr = np.sqrt(np.var(cirbuf)) * var_thresh_ratio
#        next_percentage_change = (np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
#        if(next_percentage_change >= var_thr):
#            Y.append([1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
#        else:
#            Y.append([-1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
        if(target_movement == 'absolute'):
            next_percentage_change = np.abs(np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
        else:
            next_percentage_change = (np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
            
        if(target_movement == 'up'):
            if(next_percentage_change >= var_thr):
                Y.append([1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
            else:
                Y.append([-1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
        elif(target_movement == 'down'):
            if(next_percentage_change <= -var_thr):
                Y.append([1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
            else:
                Y.append([-1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
        elif(target_movement == 'absolute'):
            if(next_percentage_change >= var_thr):
                Y.append([1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
            else:
                Y.append([-1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
    xx = np.array(Y)    

    x = test_train_data['X_test'][:,1]
    y_dict = dict(pred = xx[:,0],
                  per_change = xx[:,1],
                  mean = xx[:,2],
                  var_thr =  xx[:,3])
    plt.figure()
    plt.title('Etichetarea volatilităţii\n Lungimea ferestrei apriori = '+ str(timespan) +' eşantioane, \n Lungimea ferestrei aposteriori = '+ str(window) +' eşantioane, \n Factorul de scalare = '+str(var_thresh_ratio))
    plt.plot(y_dict['per_change'], label = 'media aposteriori raportată la preţul curent')
    plt.plot(np.multiply(y_dict['pred'], 1/100), label = 'eticheta asociată predicţiei')
    plt.plot(y_dict['var_thr'], label = 'varianţa apriori')
    plt.legend(loc = 'best')
    plt.xlabel('Timp')
    plt.ylabel('Amplitudine')
    plt.show()
    
    plt.figure()
    plt.title('Etichetarea volatilităţii: Valoarea medie aposteriori\n Lungimea ferestrei apriori = '+ str(timespan) +' eşantioane, \n Lungimea ferestrei aposteriori = '+ str(window) +' eşantioane, \n Factorul de scalare = '+str(var_thresh_ratio))
    plt.plot(x, label = 'Preţul real')
    plt.plot(y_dict['mean'], label = 'Valoarea medie aposteriori a preţului')
    plt.xlabel('Timp')
    plt.ylabel('Preţ')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.figure()
    plt.title('Etichetarea volatilităţii: varianţa apriori.\n Lungimea ferestrei apriori = '+ str(timespan) +' eşantioane, \n Lungimea ferestrei aposteriori = '+ str(window) +' eşantioane, \n Factorul de scalare = '+ str(var_thresh_ratio))
    plt.plot(x, label = 'Preţul real')
    plt.plot(np.multiply(y_dict['var_thr'],100000) + 3000, label = 'Varianţa apriori a preţului (scalată)')
    plt.xlabel('Timp')
    plt.ylabel('Preţ')
    plt.legend(loc = 'best')
    plt.show()
    
    #cir_buf_label = np.array(y_dict['pred'])
    #for i in range(12):
    #    cir_buf_label = np.hstack((0,cir_buf_label[:-1]))
    #taget_prices = [x[i] if(cir_buf_label[i] > 0) else None for i in range(len(cir_buf_label))]
    
    taget_prices = np.zeros(len(y_dict['pred']))
    for i in range(len(y_dict['pred'])):
        if(y_dict['pred'][i] > 0):
            taget_prices[i:i+window] = 1
    taget_prices = [x[i] if(taget_prices[i] > 0) else None for i in range(len(taget_prices))]
            
    mean_x = np.mean(x)
    plt.figure()
    plt.title('Etichetarea volatilităţii.\n Lungimea ferestrei apriori = '+ str(timespan) +' eşantioane, \n Lungimea ferestrei aposteriori = '+ str(window) +' eşantioane, \n Factorul de scalare = '+ str(var_thresh_ratio))
    plt.plot(x, label = 'Preţul real')
    plt.plot(np.multiply(y_dict['pred'], mean_x/12) + mean_x, label = 'Valoarea etichetei (scalată)')
    if(plot_predictions):
        plt.plot(np.multiply(predictions, mean_x/12) + mean_x, label = 'Valoarea predicţiei (scalată)')
    plt.plot(taget_prices, label = "Preţul 'ţintâ'", color = 'r')
    plt.xlabel('Timp')
    plt.ylabel('Preţ')
    plt.legend(loc = 'best')
    plt.show()
    #for i in range(len(x)):
    #    
