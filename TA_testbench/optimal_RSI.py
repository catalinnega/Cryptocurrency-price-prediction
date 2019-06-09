#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:09:34 2019

@author: catalin
"""


import datetime

#import mylib_dataset as md
import mylib_rf as mrf
import feature_list_test
#import numpy as np

import matplotlib.pyplot as plt
   
import sklearn.metrics as sm
   

proc_time_start = datetime.datetime.now()

feature_dicts = feature_list_test.get_features_dicts()

xgb_obj = mrf.get_data_RF()
dataset_path, log_path, rootLogger = mrf.init_paths()

tau_values = []
for i in range(5):
    tau_values.append(i)
    
    
learn_method = 'regression'
label_type = 'raw'
results = {}
for tau in tau_values:
    feature_dicts['RSI']['days'][0] = int(tau)
    data = xgb_obj.get_input_data(database_path = dataset_path, 
                                  feature_dicts = feature_dicts, 
                                  normalization_method = None,
                                  dataset_type = 'dataset2',
                                  lookback = None,
                                  preproc_constant = None
                                  )
    

    test_train_data, h ,l = xgb_obj.get_test_and_train_data(label_window = 4, 
                                                           label_type = label_type,
                                                           thresh_ratio = 1,
                                                           cross_validation = None)
    
    xgb_obj.get_estimator_xgb(
                             n_estimators = 60,
                             max_depth = 7,
                             gamma = 3.5,
                             min_child_weight = 22.22,
                             subsample = 0.9,
                             learning_rate = 0.09,
                             random_state = 999, ### for reproducing results(debug),
                             learn_method = learn_method
                             ) 
    rf = xgb_obj.fit_estimator_sklearn()
    predictions = xgb_obj.predict_estimator_sklearn()
    perf = xgb_obj.get_perf_regression()
    results[tau] = perf


xgb_obj.alarm()
proc_time_end = datetime.datetime.now()
#
#mse_val = [results[i]['mse'] for i in list(results.keys())]
#mse_key =  list(results.keys())

mae_val = [results[i]['mae'] for i in list(results.keys())]
mae_key =  list(results.keys())
#mse_val = np.array(list(mse_vals.values()))
#mse_key = np.array(list(mse_vals.keys()))
#
#mse_val = np.array(list(mae_vals.values()))
#mse_key = np.array(list(mae_vals.keys()))

#plt.figure()
#plt.title("AUC scores")
#plt.plot(mse_key, mse_val, label = 'train')
#plt.legend(loc = 'best')
#plt.xlabel('min samples for split')
#plt.ylabel('AUC value')
#plt.show()

#
#dict_params = {'mae_key':mae_key, 'mae_val' : mae_val}
#import pickle
#with open('rsi5_300.pkl', 'wb') as f:
#    pickle.dump(dict_params, f)

with open('rsi5_300.pkl', 'rb') as f:
    a = pickle.load(f)
key = np.concatenate([mae_key, a['mae_key']])
val = np.concatenate([mae_val, a['mae_val']])
mae_key = key
mae_val = val
a = list(mae_val)
key = mae_key[a.index(min(a))]
val = mae_val[a.index(min(a))]
print('n_estimators for max value in auc train: ', key )
plt.figure()
plt.title("Performanta indicatorului RSI")
plt.plot(mae_key, mae_val, label = 'Eroarea medie absoluta')
plt.xlabel('Lungimea ferestrei(zile)')
plt.ylabel('Valoarea MAE')
#plt.axvline(keys_auc_train[a.index(max(a))], label = 'max value', color = 'r')
plt.annotate('Minim. Lungimea ferestrei = ' +  str(key) + ' zi', xy=(key, val), xytext=(key*9, val),
            arrowprops=dict(facecolor='black',shrink=0.00001)
            )
plt.legend(loc = 'best')
plt.show()

import numpy as np
c = np.zeros(len(mae_val))
#plt.figure()
for i in range(2,len(mae_val)):
    if((mae_val[i-2] > mae_val[i-1]) and (mae_val[i-1] < mae_val[i])):
        ##valley detected
        ##backwards search for peak
        for j in range(2,len(mae_val[:i-1])):
            index = i - 1 - j
            if((mae_val[index] < mae_val[index - 1]) and (mae_val[index - 1] < mae_val[index - 2])):
                ##slope found   
                c[i-1] = abs(mae_val[index-1] - mae_val[i-1])
    if(c[i-1] > 0.1):
        print(i-1+5)
        plt.axvline(i-1+5)

plt.figure()
plt.plot(c)

