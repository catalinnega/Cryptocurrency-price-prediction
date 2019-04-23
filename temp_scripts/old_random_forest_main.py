#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 22:13:28 2018

@author: catalin
"""

import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import tensorflow as tf
import matplotlib.pyplot as plt

import mylib_dataset as md
import mylib_normalize as mn

from tensorflow.contrib.tensor_forest.client import random_forest
import pickle
import sys
import os

import datetime
proc_time_start = datetime.datetime.now()

##################### get log for estimator
import logging

compare_results_with_previous_run = True

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

user_path = os.path.dirname(os.path.realpath(__file__))

log_path = user_path + "/databases/temp_log"
log_dir = user_path + '/databases/'
filename = 'temp_log'
log_path = log_dir + filename + '.log'

try:
    os.remove(log_path)
except:
    pass

fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir, filename))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

##################### 

directory = user_path + '/databases/klines_2014-2018_15min/'

feature_names =                  [
                                 'open',
                                 'close',
                                 'high',
                                 'low',
                                 'volume',
#                                 'RSI',
#                                 'RSI_threshold_flags',
#                                 'divergence_RSI',
#                                 'stochastic_RSI',
#                                 'stochastic_RSI_threshold_flags',
#                                 'divergence_stochastic_RSI',
   #                              'MFI',
        #                         'MFI_threshold_flags',
     #                            'divergence_MFI',
                                 'Tenkan_sen', 
                                 'Kijun_sen', 
                                 'Senkou_Span_A', 
                                 'Senkou_Span_B', 
                                 'Chikou_Span', 
#                                 'cloud_flags', 
                                 'tk_cross_flags', 
                                 'oc_cloud_flags', 
                                 'ADL',
#                                 'SMA_12_day_values',
#                                 'EMA_12_day_values',
                                 'MACD_line',
                                 'MACD_signal_line',
                                 'MACD_histogram',
                                 'MACD_divergence',
#                                 'MACD_line_15_min',
#                                 'MACD_signal_line_15_min',
#                                 'MACD_histogram_15_min',
 #                                'MACD_divergence_15_min',
#                                 'MACD_line_1h', 
#                                 'MACD_signal_line_1h', 
#                                 'MACD_histogram_1h', 
#                                 'MACD_divergence_1h',
#                                 'talib_MACD1',
#                                 'talib_MACD2',
#                                 'talib_MACD3',
                                 'BB_SMA_values', 
                                 'BB_upperline_values', 
                                 'BB_lowerline_values', 
                                 'BB_squeeze_values', 
 #                                'BB_SMA_crossings',
                                 'BB_SMA_values_12h', 
                                 'BB_upperline_values_12h', 
                                 'BB_lowerline_values_12h',
                                 'BB_squeeze_values_12h', 
   #                              'BB_SMA_crossings_12h',
#                                 'BB_SMA_values_1h', 
#                                 'BB_upperline_values_1h', 
#                                 'BB_lowerline_values_1h',
#                                 'BB_squeeze_values_1h', 
#                                 'BB_SMA_crossings_1h',
                                 'CCI', 
     #                            'CCI_thresholds',
                                 'CCI_12h', 
       #                          'CCI_thresholds_12h',
                                 'CCI_1h', 
         #                        'CCI_thresholds_1h',
##                                 'RSI_timeframe_48',
##                                 'RSI_timeframe_48_threshold_flags',
#                                 'RSI_1d',
##                                 'RSI_1d_threshold_flags',
#                                 'RSI_1w',
##                                 'RSI_1w_threshold_flags',
                                 'RSI_14d',
                                 'RSI_14d_threshold_flags',
#                                 'RSI_1m',
#                                 'RSI_1m_threshold_flags',
                                 'volume_by_price',
                                 'slope_VBP',
                                 'slope_VBP_smooth',
#                                 'volume_by_price_24',
#                                 'slope_VBP_24',
#                                 'slope_VBP_smooth_24',
                                 'nlms_indicator',
                                 'nlms_smoothed_indicator',
#                                 'rls_indicator_error',
#                                 'rls_smoothed_indicator',
                                 'ATR_EMA',
                                 'ATR_EMA_Wilder',
                                 'CMF_12h',
                                 'CMF_12h_2',
#                                  'sentiment_indicator_positive',
#                                 'sentiment_indicator_negative'
                                 ]

data = md.get_dataset_with_descriptors(skip_preprocessing = False, 
                                       preproc_constant = 0.99, 
                                       normalization_method = "rescale",
                                       dataset_directory = directory,
                                       feature_names = feature_names)
X = data['preprocessed_data'] ## this will be used for training
X_unprocessed = data['data']
#sys.exit()
start_date = md.get_date_from_UTC_ms(data['dataset_dict']['UTC'][0])
end_date = md.get_date_from_UTC_ms(data['dataset_dict']['UTC'][-1])

#import random
#
#noise = np.array([np.array([random.uniform(0,1) for _ in range(X.shape[0])]) for i in range(5)])
#
#X = np.concatenate([X, noise.T], axis = 1)
#X_unprocessed = np.concatenate([X_unprocessed, noise.T], axis = 1)


test_and_train_data = md.get_test_and_train_data(preprocessed_data = X, 
                                                 unprocessed_data = X_unprocessed, 
                                                 chunks = 11 , 
                                                 chunks_for_training = 9,
                                                 remove_chunks_from_start = 0)

train_data = test_and_train_data["train_data_preprocessed"]
test_data = test_and_train_data["test_data_preprocessed"]
test_data_unprocessed = test_and_train_data["test_data_unprocessed"]
train_data_unprocessed = test_and_train_data["train_data_unprocessed"]

label_window = 100 * 3 ## if the dataset is smaller this windows has to be made smaller
X_train,Y_train, h, l = md.create_dataset_labels_mean_percentages(train_data, train_data_unprocessed, label_window, False) 
X_test, Y_test, h, l = md.create_dataset_labels_mean_percentages(test_data, test_data_unprocessed, label_window, True)

#### Random Forest 
num_features = X_train.shape[-1]
batch_size = 1 # The number of samples per batch

tf.reset_default_graph()

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
      regression = True,
      num_classes = 2, 
      num_features = num_features, 
      num_trees = num_features,## can double it
      max_nodes = 1000, 
      num_outputs = np.shape(Y_train)[1],
      max_fertile_nodes = 0, ## 100?
      #prune_every_samples = 300,
      #split_finish_name='basic',
  #    pruning_name='half'

      #model_name = 'all_sparse' ## default is all_dense
      #feature_bagging_fraction = 0.7
#      use_running_stats_method=True,
#      checkpoint_stats= True,
##      bagging_fraction=1
#      feature_bagg
      )
  
#random_forest.estimator.list_variables('/tmp/tmpzvdna1ol')
estimator = random_forest.TensorForestEstimator(params, report_feature_importances = True) 

estimator.config.save_checkpoints_steps
estimator.config.save_checkpoints_secs
## Fit
X_train = np.float32(X_train)
Y_train = np.float32(Y_train)
#fit = estimator.fit(input_fn = input_fn)

rootLogger.info(estimator.fit(X_train, Y_train))
dict_feature_importances = md.get_feature_importances(feature_names, log_path)
if(compare_results_with_previous_run):
    previous_log_path = log_path[:-4] + '_previous.log'
    dict_feature_importances_previous = md.get_feature_importances(feature_names, previous_log_path)
    dict_compare_feature_importances = {}
    for i in range(len(dict_feature_importances_previous)):
        dict_compare_feature_importances[feature_names[i]] = dict_feature_importances[feature_names[i]] \
                                                            - dict_feature_importances_previous[feature_names[i]]
os.rename(log_path, log_path[:-4] + '_previous.log')

# Evaluate
X_test = np.float32(X_test)
Y_test = np.float32(Y_test)
#evaluate = estimator.evaluate(X_test,Y_test)
  
## Predict returns an iterable of dicts.
results = list(estimator.predict(X_test))



################# Results

dictionary_percentages = ['up_percentage', 'down_percentage', 'up_reference', 'down_reference']
dict_percentages = {}
dict_percentages['up_percentage']   = [results[i]['scores'][0] for i in range (len(results))]
dict_percentages['down_percentage'] = [results[i]['scores'][1] for i in range (len(results))]
dict_percentages['up_reference']    = Y_test[:,0]
dict_percentages['down_reference'] = Y_test[:,1]

## check if there is which buy/sell stronger and if bool value matches the reference bool value
results_bool_up = np.array(dict_percentages['up_percentage']) > np.array(dict_percentages['down_percentage']) #going up
reference_bool_up = dict_percentages['up_reference'] > dict_percentages['down_reference']# going up

results_bool_up  = md.boolarray_to_int(results_bool_up)
reference_bool_up  = md.boolarray_to_int(reference_bool_up)

accurracy_bool_up = (np.array(results_bool_up) == np.array(reference_bool_up))
accurracy_bool_up  = md.boolarray_to_int(accurracy_bool_up)


up_percentage = md.counter(accurracy_bool_up, 1, percentage = True)
        
print('\n\n accuracy bool up:' + str(up_percentage) +' %')

## evaluation metrics dictionary
dict_indicators = md.bin_class_perf_indicators(results_bool_up, reference_bool_up)
md.print_dictionary(dict_indicators)

## start with an investment and see what will be the profit
initial_investment = 1000
trading_fee = 0.02

## find the first buy signal
for i in range(len(results_bool_up)):
    if(results_bool_up[i] == True):
        start_index_buy = i
        break
 
       
index_buy = start_index_buy
percentage_gain = 0
debug_gain = []
debug_index_buy = []
debug_index_sell = []
index_sell = -1
hangover = label_window
sell_flag = 0

close_candles = test_data_unprocessed[:,0]

investment = initial_investment
for i in range(len(results_bool_up)):
    if((results_bool_up[i] == False) and (results_bool_up[i-1] == True)):
        ##sell signal
        #set hangover. It gets decremented with every iteration if a buy signal is not met
        sell_flag = hangover
    if((results_bool_up[i] == True) and (results_bool_up[i-1] == False)):
        ##buy signal
        if ((not(sell_flag)) or (hangover == 1)):
            index_buy = i
        sell_flag = 0
        
    if(sell_flag):
        if(results_bool_up[i] == True):
            sell_flag = hangover
        else:
             sell_flag -= 1   
        if(sell_flag == 0):
            ## sell
            gain = ((close_candles[i] - close_candles[index_buy])/close_candles[index_buy]) * 100
            investment = investment  * (gain / 100) + investment - (investment * trading_fee)
            percentage_gain += gain
            debug_gain.append(gain)
            index_sell = i
        
    if(index_buy == i):   
        debug_index_buy.append(close_candles[i])
    else:
        debug_index_buy.append(0)
    if(index_sell == i):   
        debug_index_sell.append(close_candles[i])
    else:
        debug_index_sell.append(0)
        debug_gain.append(0)
        

print("percentage_gain: " + str(percentage_gain) + " %")
print("number of trades: " + str(len(np.array(np.where(np.array(debug_gain) > 2))[0]) * 2))
print("initial investment : " + str(initial_investment) + " <--> current amount: " + str(int(investment)) )

debug_index_buy = mn.denormalize_rescale(debug_index_buy, data['min_prices'], data['max_prices'])
debug_index_sell = mn.denormalize_rescale(debug_index_sell, data['min_prices'], data['max_prices'])
close_candles = mn.denormalize_rescale(close_candles, data['min_prices'], data['max_prices'])
 
ceva = mn.denormalize_rescale(X_test[:,0], data['min_prices'], data['max_prices'])
plot_flag = True
if(plot_flag):
    
    plt.close('all')
    plt.figure()
    plt.title("Bitcoin historical data")
    plt.plot(X_unprocessed[:,0])
    
    plt.figure()
    plt.title("Buy and sell orders relative to closing candle prices")
    plt.plot(close_candles)
    plt.plot(debug_index_buy, label = 'buy')
    plt.plot(debug_index_sell, label  = 'sell')
    #plt.plot(np.array(debug_gain)*0.015, label = 'gain from the trade', color = 'purple')
    plt.legend(loc='best')
    plt.ylabel('Close prices ($)')
    plt.xlabel('Samples(each represents a 15 min candle)')
    plt.show()  
    
    plt.figure()
    plt.title("Gain(%) relative to closing candle prices.\n (closing prices have been scaled for visualization)")
    plt.plot(close_candles/300, label = 'close prices')
    plt.plot(debug_gain, label = 'gain')
    plt.legend(loc='best')
    plt.ylabel('Close prices ($)')
    plt.xlabel('Samples(each represents a 15 min candle)')
    plt.show()
    
proc_time_end = datetime.datetime.now()
process_time = proc_time_end - proc_time_start
print('processing time: ',process_time)
    
