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
from copy import copy

import mylib_dataset as md
import tensorflow as tf
from tensorflow.contrib.tensor_forest.client import random_forest
import matplotlib.pyplot as plt
   
##################### get log for estimator
proc_time_start = datetime.datetime.now()

##### COMMENT
## train data has low variance, better choose from test data
####


path_features_imp = '/home/catalin/git_workspace/disertatie/dict_perf_feats.pkl'
ordered_values_mean, ordered_values_var = md.get_feature_importances_mean(path_features_imp)
features = feature_list.get_feature_set()[0]
features.extend(ordered_values_mean['keys'][:30])   
features = feature_list.get_features_list()
blockchain_indicators = feature_list.get_blockchain_indicators()
#features = feature_list.get_features_list()
import pickle
with open('/home/catalin/python/force_data.pickle', 'rb') as handle:
    force_data = pickle.load(handle)

min_samples_leafs = []
for i in range (1,30):
    min_samples_leafs.append(i/60)

roc_curves = {}
auc_scores = {}
for min_samples_leaf in min_samples_leafs:   
    rf_obj = mrf.get_data_RF()
    
    dataset_path, log_path, rootLogger = mrf.init_paths()
    
    data = rf_obj.get_input_data(force_data = force_data)
    
    test_train_data, h ,l = rf_obj.get_test_and_train_data(force_data = force_data)
    #rf_obj.normalize_features()
    
    rf_obj.get_estimator_sklearn(n_estimators = 10, max_depth = None, min_samples_leaf = min_samples_leaf) 
    rf = rf_obj.fit_estimator_sklearn()
    
    ### train eval
    curve = rf_obj.predict_proba_train()
    curve_train = rf_obj.get_ROC_curve(pos_label = 'True', plot = False)
    auc_train = rf_obj.get_auc_score()
    
    
    ### test eval
    rf_obj.get_log_loss_sklearn()
    curve_test = rf_obj.get_ROC_curve(pos_label = 'True', plot = False)
    auc_test = rf_obj.get_auc_score()
    auc_scores[min_samples_leaf] = (auc_train, auc_test)
    roc_curves[min_samples_leaf] = (curve_train, curve_test)
rf_obj.alarm()


keys_auc_train = list(auc_scores.keys())
values_auc_train = np.array(list(auc_scores.values()))[:,0]
keys_auc_test = list(auc_scores.keys())
values_auc_test = np.array(list(auc_scores.values()))[:,1]

plt.figure()
plt.title("AUC scores")
plt.plot(keys_auc_train, values_auc_train, label = 'train')
plt.plot(keys_auc_test, values_auc_test, label = 'test')
plt.legend(loc = 'best')
plt.xlabel('min samples for leafs')
plt.ylabel('AUC value')
plt.show()


a = list(values_auc_train)
key = keys_auc_train[a.index(max(a))]
val = values_auc_train[a.index(max(a))]
print('n_estimators for max value in auc train: ', key )
plt.figure()
plt.title("AUC score for train data")
plt.plot(keys_auc_train, values_auc_train, label = 'AUC train scores')
plt.xlabel('min samples for leafs')
plt.ylabel('AUC value')
#plt.axvline(keys_auc_train[a.index(max(a))], label = 'max value', color = 'r')
plt.annotate('local max: ' + str(val), xy=(key, val), xytext=(key+700, val-0.000005),
            arrowprops=dict(facecolor='black',shrink=0.00001)
            )
plt.legend(loc = 'best')
plt.show()

a = list(values_auc_test)
key = keys_auc_test[a.index(max(a))]
val = values_auc_test[a.index(max(a))]
print('n_estimators for max value in auc test: ', key )
#plt.figure()
plt.figure()
plt.title("AUC score for test data")
line, = plt.plot(keys_auc_test, values_auc_test, label = 'AUC test scores')
plt.xlabel('min samples for leafs')
plt.ylabel('AUC value')
#plt.axvline(key, label = 'max value', color = 'r')
plt.annotate('local max: ' + str(val), xy=(key, val), xytext=(key+500, val+0.0005),
            arrowprops=dict(facecolor='black',shrink=0.001)
            )
#ax.set(xlim=(0, 5), ylim=(0, 5))
plt.legend(loc = 'best')
plt.show()


#### plot historgram
#barWidth = 0.3
#plt.figure(figsize=(20, 1))
#plt.title('Mean feature importances')
#plt.bar(keys_auc, values_auc, align='edge', width=barWidth, label = 'mean')
#plt.xticks([r + barWidth for r in range(len(values_auc))], keys_auc)
#plt.xticks(rotation=90)
#plt.ylabel('Feature importances')
#plt.legend(loc = 'best')


    
