#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:09:34 2019

@author: catalin
"""


import datetime

import mylib_dataset as md
import mylib_rf as mrf
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
blockchain_indicators = feature_list.get_blockchain_indicators()
#features = feature_list.get_features_list()

rf_obj = mrf.get_data_RF()

dataset_path, log_path, rootLogger = mrf.init_paths()

data = rf_obj.get_input_data(database_path = dataset_path, 
                              feature_names = features, 
                              preprocessing_constant = 1, 
                              normalization_method = 'rescale',
                              skip_preprocessing = False,
                              #datetime_interval = {'start':start_date,'end':end_date},
                              datetime_interval = {},
                              blockchain_indicators = {},
                              lookback = 0,
                              dataset_type = 'dataset2')

test_train_data, h ,l = rf_obj.get_test_and_train_data(label_window = 1 , label_type = 'bool_up', thresh_ratio = 1)
#rf_obj.normalize_features()

rf_obj.get_estimator_sklearn(n_estimators = 50, max_depth = 14, min_samples_leaf = 0.11) 
rf = rf_obj.fit_estimator_sklearn()
predictions = rf_obj.predict_estimator_sklearn()

accuracy = rf_obj.get_bool_accuracy_sklearn()


perf_dict = rf_obj.get_performance_sklearn()
for i in perf_dict:
    print(i, ":", perf_dict[i])
rf_obj.alarm()

print('accuracy: ', accuracy, '%')

proc_time_end = datetime.datetime.now()
print('proctime: ', proc_time_end - proc_time_start)

ft_imp = rf.feature_importances_
dict_ft_imp = {}
for i in range(len(rf.feature_importances_)):
    dict_ft_imp[features[i]] = ft_imp[i]
    

a = test_train_data['Y_test']

pos = 0
for i in a:
    if i == 'True':
        pos+=1
        
print('test pos: ', pos*100/len(a),'%')
    
a = predictions

pos = 0
for i in a:
    if i == 'True':
        pos+=1
        
print('predict pos: ', pos*100/len(a),'%')

rf_obj.plot_dendogram()

prob, loss = rf_obj.get_log_loss_sklearn()
prob_accuracy = 1/2**(loss)
print('log loss: ', loss , 'accuracy: ', prob_accuracy, '%')

cf_mtx = rf_obj.get_confusion_matrix_sklearn()
print(cf_mtx)

curve = rf_obj.get_ROC_curve(pos_label = 'True', plot = False)

auc = rf_obj.get_auc_score()
print('area under curve:', auc)
