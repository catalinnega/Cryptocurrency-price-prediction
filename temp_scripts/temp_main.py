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
    
##################### get log for estimator
proc_time_start = datetime.datetime.now()


start_date = md.get_datetime_from_string('2014-01-19')
end_date = md.get_datetime_from_string('2018-05-1')



path_features_imp = '/home/catalin/git_workspace/disertatie/dict_perf_feats.pkl'
ordered_values_mean, ordered_values_var = md.get_feature_importances_mean(path_features_imp)
features = feature_list.get_feature_set()[0]
features.extend(ordered_values_mean['keys'][:15])      

rf_obj = rf.get_data_RF()

dataset_path, log_path, rootLogger = rf.init_paths()

data = rf_obj.get_input_data(database_path = dataset_path, 
                  feature_names = features, 
                  preprocessing_constant = 0.99, 
                  normalization_method = 'rescale',
                  skip_preprocessing = False,
                  datetime_interval = {'start':start_date,'end':end_date},
#                  datetime_interval = {},
                  blockchain_indicators = {})

test_train_eth = rf_obj.get_test_and_train_data(label_window = 20)

rf_obj.estimator_fit(regression = True, 
                     num_classes = 2, 
                     num_trees = 2 * np.shape(test_train_eth['X_test'])[1], 
                    # max_nodes = 100000, 
                     max_fertile_nodes = 0, 
                     rootLogger = rootLogger) ##  creates a new estimator

ft,ft_c = rf_obj.get_feature_importances(log_path = log_path)
results = rf_obj.estimator_test()

dict_indicators,dict_percentages = rf_obj.get_results()
md.print_dictionary(dict_indicators)
rf_obj.simulate_investment(initial_investment = 1000, trading_fee = 0.02)


md.myalarm('/home/catalin/Music/alarm.mp3')
proc_time_end = datetime.datetime.now()
print('proctime: ', proc_time_end - proc_time_start)

##
##btc = get_data_RF()
##data_btc = btc.get_input_data(database_path = dataset_path_btc, 
##                              feature_names = feature_names, 
##                              preprocessing_constant = 0.99, 
##                              normalization_method = 'rescale',
##                              skip_preprocessing = False,
##                              datetime_interval = {'start':start_date,'end':end_date},
##                              blockchain_indicators = blockchain_indicators)
##
##test_train_btc = btc.get_test_and_train_data()
##btc.concatenate_datasets(test_train_eth)
#eth.estimator_fit()
#ft,ft_c = eth.get_feature_importances(log_path = log_path)
#results = eth.estimator_test()
#dict_indicators,dict_percentages = eth.get_results()
#
#
#md.print_dictionary(dict_indicators)
