#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:34:49 2019

@author: catalin
"""

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Train
X_train = test_train_eth['X_train'][:50]
Y_train = test_train_eth['Y_train'][:50]
model.fit(X_train, Y_train)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = Y_train,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = "/home/catalin/git_workspace/disertatie/tree.png")



#xgboost.fit(data, label)
#predictions = rf_obj.predict_estimator_sklearn()
#
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
#import feature_list_test2
#import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import xgboost
   

proc_time_start = datetime.datetime.now()


#start_date = md.get_datetime_from_string('2014-01-19')
#end_date = md.get_datetime_from_string('2018-05-1')
#
#feature_dicts = feature_list.get_features_dicts()
feature_dicts = feature_list_iter_xgb.get_features_dicts()
#blockchain_indicators = feature_list.get_blockchain_indicators()
feature_dicts = feature_list_iter_xgb.get_features_dicts()
k = list(feature_dicts.keys())
for i in k:
   if('skip' in feature_dicts[i]):
       if(feature_dicts[i]['skip'] == False):
           feature_dicts[i]['skip'] = True
           
rf_obj = mrf.get_data_RF()
#dataset_path, log_path, rootLogger = mrf.init_paths()
dataset_path = '/home/catalin/git_workspace/disertatie/databases/btc_klines_2014_2019.csv'

#import os
#os.system('python3 /home/catalin/bfx/bitfinex-ohlc-import-master/bitfinex/main.py')    
#os.system('sqlite3 bitfinex.sqlite3_eth -header -csv "select * from candles;" > data_tmp_eth.csv') 
#os.system('cp data_tmp_eth.csv /home/catalin/git_workspace/disertatie/databases/eth_klines_15min.csv') 

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
#feature_dicts['TRIX']['skip'] = False  ##not good
#feature_dicts['OBV']['skip'] = False  ##not goog
#feature_dicts['ADX']['skip'] = False  ##not goog
data = rf_obj.get_input_data(database_path = dataset_path, 
                              feature_dicts = feature_dicts, 
                              normalization_method = None,
                              dataset_type = 'dataset2',
                              lookback = 1,
                              preproc_constant = None,
                              reshape = reshape_data
                              )


learn_method = 'regression'
label_type = 'raw_diff'
#label_type = 'volatile regressor'
#label_type = 'raw'
label_type = 'volatile_bool'
learn_method = 'classification'
target_movement = 'up'
test_train_data = rf_obj.get_test_and_train_data(label_window = 12, 
                                                                 label_type = label_type,
                                                                 thresh_ratio = 1,
                                                                 cross_validation = None,
                                                                 target_movement = target_movement)
rf_obj.get_dates_test_train()
rf_obj.get_class_distribution()
#
#cnt = 0
#for i in range(len(predictions)):
#    if(predictions[i] * test_train_data['Y_test'][i] > 0):
#        cnt+=1
#for i in range(1,len(predictions)):
#    if(np.sign((predictions[i] - predictions[i-1])) * np.sign((test_train_data['Y_test'][i] - test_train_data['Y_test'][i-1])) > 0):
#        cnt+=1
#print(cnt/len(predictions))
#    def get_estimator_sklearn(self, 
#                              oob_score = True,
#                              n_estimators = 10,
#                              max_depth = None,
#                              min_samples_split = 2,
#                              min_samples_leaf = 1,
#                              max_features = 'auto',
#                              random_state = None,
#                              learn_method = 'classifier'):
#        # Instantiate model with 1000 decision trees
#        if(learn_method == 'regression'):
#            self.model_type = 'RF regressor'
#            print('model type:', self.model_type)
#            rf = RandomForestRegressor(oob_score = oob_score,
#                                        n_estimators = n_estimators,
#                                        max_depth = max_depth,
#                                        min_samples_split = min_samples_split, 
#                                        min_samples_leaf = min_samples_leaf,
#                                        max_features = max_features,
#                                        random_state = random_state)
#        elif(learn_method == 'ET_regression'):
#rf_obj.get_estimator_sklearn(learn_method = learn_method,n_estimators = 100)
#rf_obj.get_estimator_xgb(
#                        learning_rate = 0.4,
#                        n_estimators = 100,
#                        max_depth = 10,
#                        gamma = 0.001,
#                        min_child_weight = 0.1,
#                        subsample = 0.89,
#                        random_state = 999, ### for reproducing results(debug),
#                        learn_method = learn_method
#                        ) 
#rf_obj.get_estimator_xgb(
#                        learning_rate = 0.1,
#                        n_estimators = 590,
#                        max_depth = 3,
#                        gamma = 0.1,
#                        min_child_weight = 0.001,
#                        subsample = 0.1,
#                        random_state = 999, ### for reproducing results(debug),
#                        scale_pos_weight = 1-0.93,#0.0510541773745214245,
#                        verbosity = 3,
#                        #objective= 'binary:logitraw',
#                        #objective= 'multi:softmax', num_class = 2,
#                        objective= 'binary:logistic', #better than logitraw
#                        eval_metric = 'auc',
#          #              booster = 'dart',
#                        col_sample = {'colsample_bytree':1, 'colsample_bylevel':1, 'colsample_bynode':1},
#                        learn_method = learn_method
#                        ) 

a = test_train_data
y = a['Y_train']

cnt = 0
for i in range(len(y)):
    if(y[i] == True):
        cnt+=1
pos_ratio = cnt/len(y)
print('true population', pos_ratio)
        
rf_obj.get_estimator_xgb(
                        learning_rate = 0.1,
                        n_estimators = 500,
                        max_depth = 3,
                        gamma = 0.01,
#                        max_delta_step = 1,
                        min_child_weight = 0.01,
                        subsample = 0.1,
                        random_state = 999, ### for reproducing results(debug),
                        scale_pos_weight = 0.35,#0.0510541773745214245, 0.35
                        verbosity = 3,
                        #objective= 'binary:logitraw',
                        #objective= 'multi:softmax', num_class = 2,
                        objective= 'binary:logistic', #better than logitraw
                        eval_metric = 'auc',
          #              booster = 'dart',
                        col_sample = {'colsample_bytree':1, 'colsample_bylevel':1, 'colsample_bynode':1},
                        learn_method = learn_method
                        ) 
rf = rf_obj.fit_estimator_sklearn(
                 #                 early_stopping_rounds = 30
                                 )
predictions = rf_obj.predict_estimator_sklearn()
md.get_volatile_bool_debug(test_train_data, predictions, test_train_data['X_test'][:,1], 12, target_movement)

#
#xgmat = xgb.DMatrix( train_test_data['X_train'], label=train_test_data['Y_train'], missing = -999.0, weight=weight )
#
#

#pos = aaa[:,0]
#predictions = [1 if i < 0.5 else -1 for i in pos]


cnt_p, cnt_f = 0, 0
for i in range(len(predictions)):
    if(predictions[i] >0):
        if(predictions[i] * test_train_data['Y_test'][i] >= 0):
            cnt_p += 1
        else:
            cnt_f += 1

print('peak acc', cnt_p / (cnt_p + cnt_f))
print('true pred population', (cnt_p + cnt_f)/len(y))




#b = pd.DataFrame(data['features'])
#c = b.corr()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(c,cmap='coolwarm', vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticks = np.arange(0,len(b.columns),1)
#ax.set_xticks(ticks)
#plt.xticks(rotation=90)
#ax.set_yticks(ticks)
#ax.set_xticklabels(c.columns)
#ax.set_yticklabels(c.columns)
#plt.show()
plt.figure()
plt.plot(test_train_data['Y_test'])
plt.plot(predictions)
plt.plot(test_train_data['X_test'][:,0])
plt.plot(np.multiply(test_train_data['Y_test'], 3000) + 3000)




rf_obj.alarm()
proc_time_end = datetime.datetime.now()
print('proctime: ', proc_time_end - proc_time_start)
accuracy = rf_obj.get_bool_accuracy_sklearn()
print('accuracy: ', accuracy, '%')
ft_imp = rf.feature_importances_
dict_ft_imp = {}
features = rf_obj.feature_names
for i in range(len(ft_imp)):
    dict_ft_imp[features[i]] = ft_imp[i]
    
#params = rf.get_xgb_params()
#data = test_train_data['X_train']
#label = test_train_data['Y_train'] /2 + 0.5
#dtrain = xgboost.DMatrix( data, label=label, missing = None, weight=np.ones(len(data)) )
#a = xgboost.cv(params,
#               dtrain,
#               num_boost_round = params['n_estimators'],
#               nfold=5,
#               early_stopping_rounds = 50,
#               metrics={'auc'},
#               seed = 0,
#               shuffle = False)
##rf.set_params(n_estimators=a.shape[0])
#
#
#param_test1 = {
# 'max_depth': (3,10,2),
# 'min_child_weight':(1,6,2)
#}
#
#from sklearn.grid_search import GridSearchCV
#gsearch1 = GridSearchCV(estimator = rf, param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(data,label)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#

#
#import sklearn.metrics as sm
#mse = sm.regression.mean_squared_error(test_train_data['Y_test'], predictions)
#mae = sm.regression.mean_absolute_error(test_train_data['Y_test'], predictions)
#print('mse', mse)
#print('mae', mae)
#print('oob', rf.oob_score_)
    
#if(learn_method == 'classification'):
#
#    accuracy = rf_obj.get_bool_accuracy_sklearn()
#    
#    print('proctime: ', proc_time_end - proc_time_start)
#    #
#    print('accuracy: ', accuracy, '%')
#    ft_imp = rf.feature_importances_
#    dict_ft_imp = {}
#    features = rf_obj.feature_names
#    for i in range(len(ft_imp)):
#        dict_ft_imp[features[i]] = ft_imp[i]
#        
##    rf_obj.plot_dendogram()
#    #
#    prob, loss = rf_obj.get_log_loss_sklearn()
#    prob_accuracy = 1/2**(loss)
#    print('log loss: ', loss , 'accuracy: ', prob_accuracy, '%')
#    
#    #cf_mtx = rf_obj.get_confusion_matrix_sklearn()
#    #print(cf_mtx)
#    
#    #curve = rf_obj.get_ROC_curve(pos_label = 'True', plot = False)
#    #
#    #auc = rf_obj.get_auc_score()
#    #print('area under curve:', auc)
#    #a
#    #dict_perm_ft_imp = rf_obj.permutation_importance()
#    #
#
#    
#    a = test_train_data['Y_test']
#    cnt = 0
#    for i in range(len(predictions)):
#        if predictions[i] == a[i]:
#            cnt+=1
#    print('accuracy', cnt*100/len(predictions),'%')
#    
#    cnt_bull = 0
#    cnt = 0
#    for i in range(len(predictions)):
#        if(predictions[i] == 'bull'):
#            cnt_bull += 1
#            if predictions[i] == a[i]:
#                cnt+=1
#    print('bull accuracy', cnt*100/cnt_bull,'%')
#
#    cnt_bear = 0
#    cnt = 0
#    for i in range(len(predictions)):
#        if(predictions[i] == 'bear'):
#            cnt_bear += 1
#            if predictions[i] == a[i]:
#                cnt+=1
#    print('bear accuracy', cnt*100/cnt_bear,'%')
#    
#    
#    cnt_bear = 0
#    cnt = 0
#    for i in range(len(predictions)):
#        if(predictions[i] == 'bear'):
#            cnt_bear += 1
#
#    print('bear', cnt_bear*100/len(predictions),'%')
#    
#    cnt_bull = 0
#    cnt = 0
#    for i in range(len(predictions)):
#        if(predictions[i] == 'bull'):
#            cnt_bull += 1
#
#    print('bull', cnt_bull*100/len(predictions),'%')
#  
#    pos, neg = 0, 0
#    for i in a:
#        if i == 'True':
#            pos+=1
#        elif i == 'False':
#            neg+=1
#            
#    print('test pos: ', pos*100/len(a),'%')
#    print('test neg: ', neg*100/len(a),'%')
#      
#    a = predictions
#    pos, neg = 0, 0
#    for i in a:
#        if i == 'True':
#            pos+=1
#        elif i == 'False':
#            neg+=1
#            
#    print('predict pos: ', pos*100/len(a),'%')
#    print('predict neg: ', neg*100/len(a),'%')
#elif(learn_method == 'regression'):   
#    import sklearn.metrics as sm
#    mse = sm.regression.mean_squared_error(test_train_data['Y_test'], predictions)
#    mae = sm.regression.mean_absolute_error(test_train_data['Y_test'], predictions)
#    print('mse', mse)
#    print('mae', mae)
#    a = rf_obj.get_perf_regression()
#    ft_imp = rf.feature_importances_
#    dict_ft_imp = {}
#    features = rf_obj.feature_names
#    for i in range(len(ft_imp)):
#        dict_ft_imp[features[i]] = ft_imp[i]
# 
#    
#    
#
#cnt = 0
#for i in range(len(predictions)):
#    if(predictions[i] * test_train_data['Y_test'][i] > 0):
#        cnt+=1
#print(cnt/len(predictions))
##
#x = test_train_data['X_test'][:,0]
#r1,r0,r11,r00,r111,r000 = 0,0,0,0,0,0
#lam_a = 0.99
#c_a= 0.00001
#dtd1, dtd2, dtd3, aa = [], [], [], []
#for n in range(len(x)):
#    r1 = lam_a * r1 + (x[n] * x[n-1])
#    r0 = lam_a * r0 + (x[n]**2)
#    a1 = r1/(r0 + c_a)
#       
#    r11 = lam_a * r11 + (x[n-1] * x[n-2])
#    r00 = lam_a * r00 + (x[n-1]**2)
#    a2 = r11/(r00 + c_a)
#       
#    r111 = lam_a * r111 + (x[n-2] * x[n-3])
#    r000 = lam_a * r000 + (x[n-2]**2)
#    a3 = r111/(r000 + c_a)
#    dtd1.append(a1)
#    dtd2.append(a2)
#    dtd3.append(a3)
#    a = max(abs(a1), abs(a2), abs(a3))
#    aa.append(a)
#
#diff = np.array(dtd1) - np.array(dtd2)    
#diff1 =  np.array(dtd1) - np.array(dtd2)  
#diff2 =  np.array(dtd1) - np.array(dtd3)     
#diff0 =  diff1 - diff2
#diff0[:7] = 0
#diff[:2] = 0   

#        
#from sklearn.model_selection import validation_curve
#validation_curve(rf, test_train_data['X_test'], test_train_data['Y_test'], param_name = 'gamma', param_range = [0,1])
#
#import mylib_TA as mta
#a = mta.slope_split_points(test_train_data['X_test'][:,0])
#        
#rf_obj.plot_dendogram()
#dict_drop_ft_imp = rf_obj.drop_column_importance()

#
#sorted_dict = np.array(sorted(dict_drop_ft_imp.items(), key=lambda x: x[1]))
#keys = sorted_dict[:,0]
#vals = [float(i) for i in sorted_dict[:,1]]
#
#vals_ft_imp = [dict_ft_imp[i] for i in keys]
##### plot historgram
#barWidth = 1
#plt.figure(figsize=(20, 1))
#plt.title('Importanta parametrilor: Metoda eliminarii coloanelor')
##plt.bar(list(dict_perm_ft_imp.keys()), list(dict_perm_ft_imp.values()), align='edge', width=barWidth/2, label = 'permutation performance')
##plt.bar(list(sorted_dict.keys()), list(sorted_dict.values()), align='edge', width=barWidth/3, label = 'drop column performance')
#plt.bar(keys, vals, align='edge', width=barWidth/3, label = 'Diferenta de performanta: Metoda eliminarii coloanelor')
#plt.bar(keys, np.multiply(vals_ft_imp,2), align='edge', width=barWidth/3, label = 'Diferenta de performanta: Metoda XGB - scalata')
##plt.bar(list(dict_ft_imp.keys()), list(dict_ft_imp.values()), align='edge', width=barWidth/4, label = 'impurity decrease performance')
#plt.xticks([r for r in range(len(keys))], keys)
#plt.xticks(rotation=90)
#plt.ylabel('Diferenta de performanta')
#plt.legend(loc = 'best')


#def get_volatile_bool_debug(test_train_data,predictions, data, window):
#    Y = []
#    timespan = 96
#    cirbuf = np.zeros(timespan)
#    for i in range(len(data)-window):
#        cirbuf = np.hstack(((data[i] - data[i-1]) / data[i], cirbuf[:-1]))
#        var_thr = np.sqrt(np.var(cirbuf))/2
#        next_percentage_change = (np.mean(data[i+1:i+window+1]) - data[i]) / data[i]
#        if(next_percentage_change >= var_thr):
#            Y.append([1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
#        else:
#            Y.append([-1.0, next_percentage_change, np.mean(data[i+1:i+window+1]), var_thr])
#    
#    xx = np.array(Y)    
#
#    x = test_train_data['X_test'][:,0]
#    y_dict = dict(pred = xx[:,0],
#                  per_change = xx[:,1],
#                  mean = xx[:,2],
#                  var_thr =  xx[:,3])
#    plt.figure()
#    plt.title('Etichetarea volatilităţii')
#    plt.plot(y_dict['per_change'], label = 'media aposteriori raportată la preţul curent')
#    plt.plot(np.multiply(y_dict['pred'], 1/100), label = 'eticheta asociată predicţiei')
#    plt.plot(y_dict['var_thr'], label = 'varianţa apriori')
#    plt.legend(loc = 'best')
#    plt.xlabel('Timp')
#    plt.ylabel('Amplitudine')
#    plt.show()
#    
#    plt.figure()
#    plt.title('Etichetarea volatilităţii: Valoarea medie aposteriori')
#    plt.plot(x, label = 'Preţul real')
#    plt.plot(y_dict['mean'], label = 'Valoarea medie aposteriori a preţului')
#    plt.xlabel('Timp')
#    plt.ylabel('Preţ')
#    plt.legend(loc = 'best')
#    plt.show()
#    
#    plt.figure()
#    plt.title('Etichetarea volatilităţii: varianţa apriori. (Lungimea ferestrei = 96 eşantioane)')
#    plt.plot(x, label = 'Preţul real')
#    plt.plot(np.multiply(y_dict['var_thr'],100000) + 3000, label = 'Varianţa apriori a preţului (scalată)')
#    plt.xlabel('Timp')
#    plt.ylabel('Preţ')
#    plt.legend(loc = 'best')
#    plt.show()
#    
#    #cir_buf_label = np.array(y_dict['pred'])
#    #for i in range(12):
#    #    cir_buf_label = np.hstack((0,cir_buf_label[:-1]))
#    #taget_prices = [x[i] if(cir_buf_label[i] > 0) else None for i in range(len(cir_buf_label))]
#    
#    taget_prices = np.zeros(len(y_dict['pred']))
#    for i in range(len(y_dict['pred'])):
#        if(y_dict['pred'][i] > 0):
#            taget_prices[i:i+12] = 1
#    taget_prices = [x[i] if(taget_prices[i] > 0) else None for i in range(len(taget_prices))]
#            
#            
#    plt.figure()
#    plt.title('Etichetarea volatilităţii. (Lungimea ferestrei de predicţie = 12 eşantioane)')
#    plt.plot(x, label = 'Preţul real')
#    plt.plot(np.multiply(y_dict['pred'],500) + 6000, label = 'Valoarea etichetei (scalată)')
#    plt.plot(np.multiply(predictions,500) + 6000, label = 'Valoarea etichetei (scalată)')
#    plt.plot(taget_prices, label = "Preţul 'ţintâ'", color = 'r')
#    plt.xlabel('Timp')
#    plt.ylabel('Preţ')
#    plt.legend(loc = 'best')
#    plt.show()
#    #for i in range(len(x)):
#    #    
