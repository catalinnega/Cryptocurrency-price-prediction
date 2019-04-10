#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:47:01 2019

@author: catalin
"""

import datetime

import mylib_dataset as md
import mylib_rf as rf
import feature_list
import numpy as np

import mylib_dataset as md
import tensorflow as tf
from tensorflow.contrib.tensor_forest.client import random_forest
import matplotlib.pyplot as plt
   
##################### get log for estimator
proc_time_start = datetime.datetime.now()


#start_date = md.get_datetime_from_string('2014-01-19')
#end_date = md.get_datetime_from_string('2018-05-1')


path_features_imp = '/home/catalin/git_workspace/disertatie/dict_perf_feats.pkl'
ordered_values_mean, ordered_values_var = md.get_feature_importances_mean(path_features_imp)
features = feature_list.get_feature_set()[0]
features.extend(ordered_values_mean['keys'][:30])   
features = feature_list.get_features_list()

rf_obj = rf.get_data_RF()

dataset_path, log_path, rootLogger = rf.init_paths()

data = rf_obj.get_input_data(database_path = dataset_path, 
                  feature_names = features, 
                  preprocessing_constant = 0.9, 
                  normalization_method = 'rescale',
                  skip_preprocessing = False,
                  #datetime_interval = {'start':start_date,'end':end_date},
                  datetime_interval = {},
                  blockchain_indicators = {},
                  lookback = 0 )

test_train_eth, h ,l = rf_obj.get_test_and_train_data(label_window = 12 , bool_labels = True)
 


#test_train_eth2 = rf_obj.normalize_features()

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor()
rf = RandomForestClassifier()

# Train the model on training data
rf.fit(test_train_eth['X_train'], test_train_eth['Y_train'])

# Use the forest's predict method on the test data
predictions = rf.predict(test_train_eth['X_test'])

#a = [test_train_eth['Y_test'][i] > test_train_eth['X_test'][i,0] for i in range(1,len(test_train_eth['Y_test']))]
#b = [predictions[i] > test_train_eth['X_test'][i,0] for i in range(1,len(predictions))]
#
cnt = 0
counter = 0
for i in predictions:
    if(i == test_train_eth['Y_test'][counter]):
        cnt+=1
    counter+=1
        
print(cnt*100/len(predictions))

md.myalarm('/home/catalin/Music/alarm.mp3')
proc_time_end = datetime.datetime.now()
print('proctime: ', proc_time_end - proc_time_start)


#b = a[4]
#c = data['features'][b]
#d = data['features']
#print(b)
#if(c):
#    plt.plot(c)
#    
#h = d['high_prices']
#l = d['low_prices']
#c = d['close_prices']
#
#c = test_train_data['X_train'][:,1]
#l = test_train_data['X_train'][:,3]
#h = test_train_data['X_train'][:,2]
#for i in range(len(h)):
#    if c[i] < l[i]:
#        print(i)
##
#plt.plot(a)
#a = mn.normalize_rescale(a)[0]
#plt.plot(a)
#    

#results_bool_up = np.array(predictions[:,0]) > np.array(predictions[:,1]) #going up
#reference_bool_up = test_train_eth['Y_test'][:,0] > test_train_eth['Y_test'][:,1]# going up
#
#results_bool_up  = md.boolarray_to_int(results_bool_up)
#reference_bool_up  = md.boolarray_to_int(reference_bool_up)
#
##accurracy_bool_up = (np.array(results_bool_up) == np.array(reference_bool_up))
#accurracy_bool_up  = results_bool_up - reference_bool_up
#
#
#up_percentage = md.counter(accurracy_bool_up, 0, percentage = True)
#        
#print('\n\n accuracy bool up:' + str(up_percentage) +' %')
#
#

##in_data = data['dataset_dict']['close']
#dataX, dataY = [], []
#mean_lows = []
#mean_highs = []
#
#window_length = 300
#for i in range(len(data) - window_length):
##    dataX.append(data[i,:])
#    up = [data[i+j] if data[i+j] >= data[i] else 0 for j in range(window_length)]
#    down = [data[i+j] if data[i+j] < data[i] else 0 for j in range(window_length)]
#    mean_up = np.mean(up)
#    mean_down = np.mean(down)
#    
#    dataY.append( (mean_up, mean_down))
#    mean_highs.append(mean_up)
#    mean_lows.append(mean_down)
#return np.array(dataX), np.array(dataY), mean_highs, mean_lows