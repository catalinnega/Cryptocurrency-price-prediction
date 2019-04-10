#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:47:01 2019

@author: catalin
"""

import mylib_dataset as md
import feature_list
import mylib_rf as rf
import datetime
    
##################### get log for estimator
proc_time_start = datetime.datetime.now()


#user_path = os.path.dirname(os.path.realpath(__file__))

    
from copy import copy
#
#start_date = md.get_datetime_from_string('2017-01-19')
#end_date = md.get_datetime_from_string('2018-05-1')

feature_set = feature_list.get_feature_set()
#blockchain_indicators = feature_list.get_blockchain_indicators()
feature_sets = feature_set[1:]
reference_group = feature_set[0]
dict_perf_feats = {}
break_flag = False
for analysis_set in feature_sets:
    if(break_flag):
        break
    for feature in analysis_set:
        dict_perf_feats[feature] = []
    for secondary_set in feature_sets:
        if(analysis_set != secondary_set):
            features = copy(reference_group)
            features.extend(analysis_set)
            features.extend(secondary_set)
            ##fit, return feature importances
            ## append features importance values to performance indicator
            ## at the end the results should be compared to reference group
            print('####              ', features)
            rf_obj = rf.get_data_RF()
            dataset_path, log_path, rootLogger = rf.init_paths()
            data = rf_obj.get_input_data(database_path = dataset_path, 
                                          feature_names = features, 
                                          preprocessing_constant = 0.99, 
                                          normalization_method = 'rescale',
                                          skip_preprocessing = False,
                                          datetime_interval = {},
                                          blockchain_indicators = {})
            test_train = rf_obj.get_test_and_train_data(label_window = 4, label_type = 'bool_up', thresh_ratio = 1)
            rf_obj.get_estimator_sklearn()
            rf_obj.fit_estimator_sklearn()
            ft_dict = rf_obj.get_ft_imp_sklearn()
            if(not ft_dict):
                print('ABORT. no feats.. probably could not create a requested feature')
                break_flag = True
                break
            for feature in analysis_set:
                dict_perf_feats[feature].append(ft_dict[feature]/ft_dict[reference_group[0]])

rf_obj.alarm()

proc_time_end = datetime.datetime.now()
print('proctime: ', proc_time_end - proc_time_start)