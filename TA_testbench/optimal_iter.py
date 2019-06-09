#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:09:34 2019

@author: catalin
"""


import datetime

import mylib_rf as mrf
import feature_list_iter
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as sm
import pickle
   
dataset_path = '/home/catalin/git_workspace/disertatie/databases/btc_klines_2014_2019.csv'
feature_dicts = feature_list_iter.get_features_dicts()

feature_names = []
for i in feature_dicts:
    if 'periods' in feature_dicts[i]:
        feature_names.append(i)

learn_method = 'regression'
label_type = 'raw_diff'        
for feature_name in ['FRLS',]:
    for i in feature_dicts:
        if 'periods' in feature_dicts[i]:
            feature_dicts[i]['skip'] = True
    
    feature_dicts[feature_name]['skip'] = False

    xgb_obj = mrf.get_data_RF()
    tau_values = []
    for i in range(1,10):
        tau_values.append(i)
        
    
    results = {}
    for tau in tau_values:
        feature_dicts[feature_name]['periods'] = tau
        data = xgb_obj.get_input_data(database_path = dataset_path, 
                                      feature_dicts = feature_dicts, 
                                      normalization_method = None,
                                      dataset_type = 'dataset2',
                                      lookback = None,
                                      preproc_constant = None
                                      )
        
    
        test_train_data = xgb_obj.get_test_and_train_data(label_window = 4, 
                                                               label_type = label_type,
                                                               thresh_ratio = 1,
                                                               cross_validation = None)
        
        xgb_obj.get_estimator_xgb(
                                learning_rate = 0.3,
                                n_estimators = 30,
                                max_depth = 10,
                                gamma = 3.5,
                                min_child_weight = 8.1,
                                subsample = 0.84,
                                 random_state = 999, ### for reproducing results(debug),
                                learn_method = learn_method
                                 ) 
        rf = xgb_obj.fit_estimator_sklearn()
        predictions = xgb_obj.predict_estimator_sklearn()
        perf = xgb_obj.get_perf_regression()
        results[tau] = perf

#    with open('/home/catalin/git_workspace/disertatie/databases/'+str(feature_name) + 'diff_100.pkl', 'wb') as f:
#        pickle.dump(results, f)   
        
    mae_val = [results[i]['mae'] for i in list(results.keys())]
    mae_key =  list(results.keys())
    
    a = list(mae_val)
    key = mae_key[a.index(min(a))]
    val = mae_val[a.index(min(a))]
    print('n_estimators for max value in auc train: ', key )
    plt.figure()
    plt.title("Performanta indicatorului" + str(feature_name) + ".\n Minim = "+ str(val)+ ". " + str(feature_name) + " = " +  str(key))
    plt.plot(mae_key, mae_val, label = 'Eroarea medie absoluta')
    #plt.xticks(rotation=90)
    plt.xlabel('Lungimea ferestrei')
    plt.ylabel('Valoarea MAE')
    #plt.axvline(keys_auc_train[a.index(max(a))], label = 'max value', color = 'r')
    #plt.annotate('Minim. ' + str(feature_name) + ' = ' +  str(key), xy=(key, val), xytext=(key*4.5, val*1.002),
    #            arrowprops=dict(facecolor='black',shrink=0.00001)
    #            )
    plt.legend(loc = 'best')
    plt.show()


