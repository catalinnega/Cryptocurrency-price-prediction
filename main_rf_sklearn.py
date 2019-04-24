#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:09:34 2019

@author: catalin
"""


import datetime

#import mylib_dataset as md
import mylib_rf as mrf
import feature_list
#import numpy as np

#import matplotlib.pyplot as plt
   

proc_time_start = datetime.datetime.now()


#start_date = md.get_datetime_from_string('2014-01-19')
#end_date = md.get_datetime_from_string('2018-05-1')

feature_dicts = feature_list.get_features_dicts()
#blockchain_indicators = feature_list.get_blockchain_indicators()

rf_obj = mrf.get_data_RF()
dataset_path, log_path, rootLogger = mrf.init_paths()

data = rf_obj.get_input_data(database_path = dataset_path, 
                              feature_dicts = feature_dicts, 
                              normalization_method = 'rescale',
                              dataset_type = 'dataset2',
                              normalization = None,
                              lookback = None
                              )



test_train_data, h ,l = rf_obj.get_test_and_train_data(label_window = 4, 
                                                       label_type = 'bool_up',
                                                       thresh_ratio = 1,
                                                       cross_validation = None)
#rf_obj.normalize_features()

rf_obj.get_estimator_sklearn(n_estimators = 1000,
                             max_depth = 14,
                             min_samples_leaf = 0.11,
                             random_state = 999 ### for reproducing results(debug)
                             ) 
rf = rf_obj.fit_estimator_sklearn()
predictions = rf_obj.predict_estimator_sklearn()

accuracy = rf_obj.get_bool_accuracy_sklearn()


perf_dict = rf_obj.get_performance_sklearn()
#for i in perf_dict:
#    print(i, ":", perf_dict[i])
rf_obj.alarm()

print('accuracy: ', accuracy, '%')

proc_time_end = datetime.datetime.now()
print('proctime: ', proc_time_end - proc_time_start)

ft_imp = rf.feature_importances_
dict_ft_imp = {}
features = rf_obj.feature_names
for i in range(len(ft_imp)):
    dict_ft_imp[features[i]] = ft_imp[i]
    
rf_obj.plot_dendogram()

prob, loss = rf_obj.get_log_loss_sklearn()
prob_accuracy = 1/2**(loss)
print('log loss: ', loss , 'accuracy: ', prob_accuracy, '%')

cf_mtx = rf_obj.get_confusion_matrix_sklearn()
print(cf_mtx)

curve = rf_obj.get_ROC_curve(pos_label = 'True', plot = False)

auc = rf_obj.get_auc_score()
print('area under curve:', auc)

dict_perm_ft_imp = rf_obj.permutation_importance()
#dict_drop_ft_imp = rf_obj.drop_column_importance()


##### plot historgram
#barWidth = 0.8
#plt.figure(figsize=(20, 1))
#plt.title('Feature importances')
#plt.bar(list(dict_perm_ft_imp.keys()), list(dict_perm_ft_imp.values()), align='edge', width=barWidth/2, label = 'permutation performance')
#plt.bar(list(dict_drop_ft_imp.keys()), list(dict_drop_ft_imp.values()), align='edge', width=barWidth/3, label = 'drop column performance')
#plt.bar(list(dict_ft_imp.keys()), list(dict_ft_imp.values()), align='edge', width=barWidth/4, label = 'impurity decrease performance')
#plt.xticks([r + barWidth for r in range(len(list(dict_perm_ft_imp.keys())))], list(dict_perm_ft_imp.keys()))
#plt.xticks(rotation=90)
#plt.ylabel('Feature importances')
#plt.legend(loc = 'best')
