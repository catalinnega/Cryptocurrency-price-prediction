#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:09:34 2019

@author: catalin
"""


import datetime

import mylib_dataset as md
import mylib_rf as mrf
import feature_list_iter_xgb
import numpy as np

import matplotlib.pyplot as plt
import sklearn.metrics as sm 
import numpy as np

import xgboost

proc_time_start = datetime.datetime.now()

feature_dicts = feature_list_iter_xgb.get_features_dicts()
k = list(feature_dicts.keys())
for i in k:
   if('skip' in feature_dicts[i]):
       if(feature_dicts[i]['skip'] == False):
           feature_dicts[i]['skip'] = True
           
rf_obj = mrf.get_data_RF()
dataset_path = '/home/catalin/git_workspace/disertatie/databases/btc_klines_2014_2019.csv'

### Specify which features to use
reshape_data = None
#reshape_data = {'start': 182145, 'end': 207913} 
#feature_dicts['SENT']['skip'] = False  
feature_dicts['FRLS']['skip'] = False ##good
feature_dicts['NLMS']['skip'] = False ##good
feature_dicts['peaks']['skip'] = False
feature_dicts['BB']['skip'] = False ##good
feature_dicts['IKH']['skip'] = False
feature_dicts['VBP']['skip'] = False ##good
feature_dicts['STO']['skip'] = False ##good
feature_dicts['MD']['skip'] = False  ##good
feature_dicts['nsr']['skip'] = False  ##good
feature_dicts['ohcl_diff']['skip'] = False


## get dataset
data = rf_obj.get_input_data(database_path = dataset_path, 
                              feature_dicts = feature_dicts, 
                              normalization_method = None,
                              dataset_type = 'dataset2',
                              lookback = None,
                              preproc_constant = None,
                              reshape = reshape_data
                              )


## get test train data
label_type = 'volatile_bool'
learn_method = 'classification'
target_movement = 'up'
var_thresh_ratio = 1
aposteriori_window = 30
test_train_data = rf_obj.get_test_and_train_data(label_window = aposteriori_window, 
                                                                 label_type = label_type,
                                                                 thresh_ratio = 1,
                                                                 cross_validation = None,
                                                                 target_movement = target_movement, 
                                                                 var_thresh_ratio = var_thresh_ratio)
rf_obj.get_dates_test_train()
class_distributions = rf_obj.get_class_distribution()
scale_pos_weight = class_distributions[0] ## largest population

## get estimator, fit and predict
rf_obj.get_estimator_xgb(
                        learning_rate = 0.5,
                        n_estimators = 300,
                        max_depth = 5,
                        gamma = 0.95,
#                        max_delta_step = 1,
                        min_child_weight = 0.1,
                        subsample = 0.95,
                        random_state = 999, ### for reproducing results(debug),
                        scale_pos_weight = scale_pos_weight,
                        verbosity = 1,
#                        objective= 'binary:logitraw',
#                        objective= 'multi:softmax', num_class = 2,
                        objective= 'binary:logistic', #better than logitraw
                        disable_default_eval_metric = 1,
                        eval_metric = 'auc',
                       # num_class = 1,
          #              booster = 'dart',
                  #      col_sample = {'colsample_bytree':1, 'colsample_bylevel':1, 'colsample_bynode':1},
                        colsample_bytree = 1,
                        colsample_bylevel = 1,
                        colsample_bynode = 1,
                        learn_method = learn_method,
                        base_score = 0.5
#                        num_class = 2
                        ) 

#w = rf_obj.label_weights
#rf = rf_obj.fit_estimator_sklearn(weights = w)
rf = rf_obj.fit_estimator_sklearn()
predictions = rf_obj.predict_estimator_sklearn()






## plot volatile labels and other 
plot_volatile_debug = True
if(plot_volatile_debug):
    md.get_volatile_bool_debug(test_train_data, 
                               predictions,
                               test_train_data['X_test'][:,1],
                               aposteriori_window,
                               target_movement,
                               var_thresh_ratio = var_thresh_ratio,
                               plot_predictions = False)


rf_obj.alarm()
print('proctime: ', datetime.datetime.now() - proc_time_start)



### plot precision-recall curves
probs = rf.predict_proba(test_train_data['X_test']) 
aucpr = rf_obj.get_binary_aucpr(plot = 'normal',probs = probs)	


## Precision Recall adapted evaluation metric
score = rf_obj.eval_volatility_PR(max_precision_thresh = 0.8)

### simulate investment strategy
simultate_investment = False
if(simultate_investment):
    predictions_new = [1 if probs[:,1][i] > 0.98 else -1 for i in range(len(predictions))]
    print('precision', sm.precision_score(test_train_data['Y_test'], predictions_new))
    
    alpha_values = []
    profits = []
    for i in range(200):
        alpha_values.append(i/10 + 1)
        
    for alpha in alpha_values:
        initial_investment = 1000
        investment = initial_investment
        buy_flag = False
        over_thr_flag = False
        timeout_counter = 0
        buy_fee = 0.002
        sell_fee = 0.001
        close = test_train_data['X_test'][:,1]
        buy_value = 0
        buy = np.zeros(len(predictions_new))
        over_thr = np.zeros(len(predictions_new))
        sell1 = np.zeros(len(predictions_new))
        sell2 = np.zeros(len(predictions_new))
        sell3 = np.zeros(len(predictions_new))
        wlen = len(predictions_new)
        apriori_window = 96
        cirbuf = np.zeros(apriori_window)
        buy_cnt = 0
        for i in range(len(predictions_new[:wlen])):
            cirbuf = np.hstack(((close[i] - close[i-1]) / close[i], cirbuf[:-1]))
            if(i > apriori_window):
                if(predictions_new[i] > 0):
                    var_thr = np.sqrt(np.var(cirbuf)) * var_thresh_ratio
                    if(var_thr > 0.003):
                        if(not buy_flag):
                            buy_cnt += 1
                            buy_flag = True
                            over_thr_flag = False
                            buy_value = close[i]
                            #var_thr = 0.031
                            timeout_counter = aposteriori_window
                            investment -= investment * buy_fee
                            buy[i] = 1
    #                            print('buy at \n\t buy value', buy_value,
    #                                  '\n\t investment', investment)
                        else:
                            #var_thr = np.sqrt(np.var(close[i-apriori_window:i])) * var_thresh_ratio
                            # = 0.031
                            timeout_counter = aposteriori_window
                            
                if((close[i] >= (buy_value + buy_value * var_thr)) and buy_flag and not over_thr_flag):
                    over_thr_flag = True
                    over_thr[i] = 1
                elif(close[i] < (buy_value + buy_value * var_thr) and buy_flag and over_thr_flag):
                    buy_flag = False
                    over_thr_flag = False
                    sell1[i] = 1
                    timeout_counter = 0
                    investment = investment * (((buy_value + buy_value * var_thr) / buy_value) - sell_fee)
    #                print('sell1 at \n\t buy value', buy_value,
    #                      '\n\t investment', investment, 
    #                      '\n\t profit', ((buy_value + buy_value * var_thr) / buy_value) - sell_fee,
    #                      '\n\t sell value', close[i],
    #                      )
                    
                if((close[i] >= (buy_value + buy_value * var_thr * alpha)) and buy_flag):
                    buy_flag = False
                    over_thr_flag = False
                    sell2[i] = 1
                    timeout_counter = 0
                    investment = investment * ((close[i] / buy_value) - sell_fee)
    #                print('sell2 at \n\t buy value', buy_value,
    #                      '\n\t investment', investment, 
    #                      '\n\t profit', close[i] / buy_value,
    #                      '\n\t sell value', close[i],
    #                      )
                
                        
                if((timeout_counter == 0) and buy_flag) and (not over_thr_flag):
                    buy_flag = False
                    over_thr_flag = False
                    sell3[i] = 1
                    timeout_counter = 0
                    investment = investment * ((close[i] / buy_value) - sell_fee)
    #                print('sell3 at \n\t buy value', buy_value,
    #                      '\n\t investment', investment, 
    #                      '\n\t profit', close[i] / buy_value,
    #                      '\n\t sell value', close[i],
    #                      )
                if(timeout_counter):
                    timeout_counter -= 1
        profits.append(investment / initial_investment)
    
    plt.figure()
    plt.title('Profitabilitatea strategiei de vânzare-cumpărare în funcţie de factorul alfa')
    plt.plot(alpha_values , np.multiply(np.array(profits)-1, 100), label = 'Profitul înregistrat')
    plt.xlabel('Valorile parametrului alfa')
    plt.ylabel('Profitul procentual')
    plt.show()
    
    
        
    #    plt.figure()
    #    plt.plot(close[:wlen])
    #    plt.plot(np.multiply(predictions_new[:wlen], np.mean(close[:wlen])/30) + np.mean(close[:wlen]),color = 'black')
    #    plt.plot(np.multiply(buy[:wlen], np.mean(close[:wlen])/30) + np.mean(close[:wlen]), color = 'green')
    #    plt.plot(np.multiply(over_thr[:wlen], np.mean(close[:wlen])/30) + np.mean(close[:wlen]), color = 'grey')
    #    plt.plot(np.multiply(sell1[:wlen], np.mean(close[:wlen])/30) + np.mean(close[:wlen]), color = 'red')
    #    plt.plot(np.multiply(sell2[:wlen], np.mean(close[:wlen])/30) + np.mean(close[:wlen]), color = 'purple')
    #    plt.plot(np.multiply(sell3[:wlen], np.mean(close[:wlen])/30) + np.mean(close[:wlen]), color = 'orange')

eval_feature_importance = False  ## drop column takes a lot of processing..
if(eval_feature_importance):
    dict_ft_imp = rf_obj.get_dict_ft_imp() 
    dict_drop_ft_imp = rf_obj.drop_column_importance(eval_function = rf_obj.eval_volatility_PR, max_precision_thresh = 0.8)
    rf_obj.plot_dict_drop_ft_imp()
    
get_variable_correlations = True
if(get_variable_correlations):
    rf_obj.plot_corr_heatmap()
    rf_obj.plot_dendogram()
    



