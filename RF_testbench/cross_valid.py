#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:51:03 2019

@author: catalin
"""

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

import tensorflow as tf
from tensorflow.contrib.tensor_forest.client import random_forest
import matplotlib.pyplot as plt
   
##################### get log for estimator
proc_time_start = datetime.datetime.now()


#start_date = md.get_datetime_from_string('2014-01-19')
#end_date = md.get_datetime_from_string('2018-05-1')

features = feature_list.get_features_list()
blockchain_indicators = feature_list.get_blockchain_indicators()

results_perf = []
rf_obj = mrf.get_data_RF()
for cross_val_index in range(1,50):
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
    
    test_train_data, h ,l = rf_obj.get_test_and_train_data(label_window = 1 , 
                                                           label_type = 'bool_up',
                                                           thresh_ratio = 1,
                                                           chunks = 50,
                                                           cross_validation = cross_val_index)
    #rf_obj.normalize_features()
    
    rf_obj.get_estimator_sklearn(n_estimators = 50, max_depth = 14, min_samples_leaf = 0.11) 
    rf = rf_obj.fit_estimator_sklearn()
    predictions = rf_obj.predict_estimator_sklearn()
    
    accuracy = rf_obj.get_bool_accuracy_sklearn()
    
    
    perf_dict = rf_obj.get_performance_sklearn()
    for i in perf_dict:
        print(i, ":", perf_dict[i])
    
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
    
    
    prob, loss = rf_obj.get_log_loss_sklearn()
    prob_accuracy = 1/2**(loss)
    print('log loss: ', loss , 'accuracy: ', prob_accuracy, '%')
    
    cf_mtx = rf_obj.get_confusion_matrix_sklearn()
    print(cf_mtx)
    
    curve = rf_obj.get_ROC_curve(pos_label = 'True', plot = False)
    
    auc = rf_obj.get_auc_score()
    print('area under curve:', auc)
    results_perf.append((perf_dict, auc))


import statistics 
accuracy, auc = 0, 0
acc_list, auc_list = [], []
for i in range(len(results_perf)):
    accuracy += results_perf[i][0]['accuracy']
    acc_list.append(results_perf[i][0]['accuracy'])
    auc += results_perf[i][1]
    auc_list.append(results_perf[i][1])
accuracy /= len(results_perf)
auc /= len(results_perf)

std_acc = statistics.stdev(acc_list)
std_auc = statistics.stdev(auc_list)

plt.figure()
plt.title('Cross validation accuracy and AUC')
plt.ylabel('Performace scores')
plt.xlabel('Index of split set used for validation')
plt.plot(acc_list, label = 'accuracy')
plt.plot(auc_list, label = 'area under ROC curve')
plt.text(30, 0.65, "accuracy mean: "+ str(accuracy*100)[:4] + '%, standard deviation: +/-' +str(std_acc*100)[:4] + '%')
plt.text(30, 0.62, "AUC mean: "+ str(accuracy*100)[:4] + '%, standard deviation: +/-' +str(std_auc*100)[:4] + '%')
plt.text(30, 0.59, "number of validation labels per set: "+ str(np.shape(test_train_data['Y_test'])[0]))
plt.legend(loc = 'best')
plt.show()
#print(np.mean(results_perf[1][:]))