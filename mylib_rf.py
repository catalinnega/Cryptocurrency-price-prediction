#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:02:49 2019

@author: catalin
"""

import numpy as np
import matplotlib.pyplot as plt
import mylib_normalize as mn
import mylib_TA as mta
import pandas as pd


import datetime

from scipy.interpolate import interp1d

from copy import copy

import mylib_dataset as md

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc, regression
import sklearn.metrics as sm   
import xgboost as xgb
#from sklearn.grid_search import GridSearchCV


class get_data_RF():
    def __init__(self):   
        self.cross_validation = False
        self.feature_names = None
        self.dict_indicators = {}
        self.dict_regression = {}
        self.forest_model = None
        self.scaler = None
        self.xgb_args = None
        self.dict_test_and_train = {}
    
    def get_input_data(self, 
                       normalization_method = 'rescale', 
                       database_path = None,
                       feature_dicts = {}, 
                       datetime_interval = None,
                       blockchain_indicators_dicts = {},
                       lookback = 0, 
                       dataset_type = 'dataset1',
                       force_data = None,
                       preproc_constant = 0,
                       reshape = None,
                       dataset_path_sentiment = '/home/catalin/databases/TA_SECURITY_SENTIMENT.csv'
                       ):
        
        dict_data = md.get_database_data(dataset_directory = database_path, 
                                          normalization_method = normalization_method,
                                          datetime_interval = {},
                                          preproc_constant = preproc_constant, 
                                          lookback = lookback,
                                          input_noise_debug = False,
                                          dataset_type = dataset_type)
        data = md.get_features(candle_data = dict_data['X'],
                                feature_dicts = feature_dicts, 
                                blockchain_indicators_dicts = {}, 
                                lookback = lookback,
                                normalization = normalization_method,
                                utc_time = dict_data['UTC'],
                                reshape = reshape,
                                dataset_path_sentiment = dataset_path_sentiment)
        
        data['preprocessed_data'] = data['data']
        if(reshape):
            data['unprocessed_data'] = dict_data['raw_data'][reshape['start'] : reshape['end'],:]
            data['UTC'] = dict_data['UTC'][reshape['start'] : reshape['end']]
        else:
            data['unprocessed_data'] = dict_data['raw_data']
        self.data_ochlv = dict_data['raw_dict']
        self.scaler = data['scaler']
        
        self.data = data
        self.feature_names = list(data['features'].keys())
        self.normalization = normalization_method
        print(self.feature_names)
        return self.data
    
    def get_features_from_data(self, candle_data = None, feature_dicts = None, normalization = None, dataset_path_sentiment = None):
        data = md.get_features(candle_data = candle_data,
                                feature_dicts = feature_dicts, 
                                blockchain_indicators_dicts = {}, 
                                lookback = None,
                                normalization = normalization,
                                dataset_path_sentiment = dataset_path_sentiment) 
        return data
    
    def get_test_and_train_data(self, 
                                label_window = 100 * 3,
                                chunks = 11, 
                                chunks_for_training = 9, 
                                remove_chunks_from_start = 0, 
                                label_type = '', 
                                thresh_ratio = 1,
                                force_data = None,
                                cross_validation = None,
                                target_movement = None,
                                var_thresh_ratio = 0.5,
                                apriori_window = 96):
        if(force_data):
            self.dict_test_and_train = force_data['test_train_data']
            self.Y_train = force_data['test_train_data']['Y_train']
            self.Y_test = force_data['test_train_data']['Y_test']
            self.X_train = force_data['test_train_data']['X_train']
            self.X_test = force_data['test_train_data']['X_test']
            self.label_window = force_data['label_window']
            h, l = 0, 0
            print('Using input data.')
        else:
            test_and_train_data = md.get_test_and_train_data(preprocessed_data = self.data['preprocessed_data'], 
                                                             unprocessed_data = self.data['unprocessed_data'], 
                                                             chunks = chunks , 
                                                             chunks_for_training = chunks_for_training,
                                                             remove_chunks_from_start = remove_chunks_from_start,
                                                             cross_validation = cross_validation
                                                             )
            train_data = test_and_train_data["train_data_preprocessed"]
            test_data = test_and_train_data["test_data_preprocessed"]
            test_data_unprocessed = test_and_train_data["test_data_unprocessed"]
            train_data_unprocessed = test_and_train_data["train_data_unprocessed"]
    #        X_train,Y_train, h, l = md.create_dataset_labels_mean_percentages(train_data, train_data_unprocessed, label_window) 
    #        X_test, Y_test, h, l = md.create_dataset_labels_mean_percentages(test_data, test_data_unprocessed, label_window)
    
    #        X_train,Y_train, h, l = md.create_dataset_labels_mean_percentages_bool(train_data, train_data_unprocessed, label_window) 
    #        X_test, Y_test, h, l = md.create_dataset_labels_mean_percentages_bool(test_data, test_data_unprocessed, label_window)
            if(label_type == 'raw_diff'):
                X_train,Y_train = md.get_diff_raw(train_data, train_data_unprocessed[:,1], label_window) 
                X_test, Y_test = md.get_diff_raw(test_data, test_data_unprocessed[:,1], label_window)
            elif(label_type == 'volatile_bool'):
                X_train,Y_train, label_weights = md.get_volatile_bool(train_data,
                                                                      train_data_unprocessed[:,1],
                                                                      label_window,
                                                                      target_movement = target_movement,
                                                                      var_thresh_ratio = var_thresh_ratio,
                                                                      apriori_window = apriori_window) 
                X_test, Y_test, label_weights = md.get_volatile_bool(test_data,
                                                                     test_data_unprocessed[:,1],
                                                                     label_window,
                                                                     target_movement = target_movement,
                                                                     var_thresh_ratio = var_thresh_ratio,
                                                                     apriori_window = apriori_window)
            else:
                X_train,Y_train, h, l = md.get_labels_mean_window(train_data, train_data_unprocessed[:,1], label_window, label_type, thresh_ratio) 
                X_test, Y_test, h, l = md.get_labels_mean_window(test_data, test_data_unprocessed[:,1], label_window, label_type, thresh_ratio)
#                X_train,Y_train, h, l = md.get_labels_mean_window(train_data, self.data_ochlv, label_window, label_type, thresh_ratio) 
#                X_test, Y_test, h, l = md.get_labels_mean_window(test_data, self.data_ochlv, label_window, label_type, thresh_ratio)
#                 
            dict_test_and_train = {'X_test': X_test,
                                   'Y_test': Y_test,
                                   'X_train': X_train,
                                   'Y_train': Y_train}
            
            self.dict_test_and_train = dict_test_and_train
            self.label_window = label_window
            self.Y_train = Y_train
            self.Y_test = Y_test
            self.X_train = X_train
            self.X_test = X_test
            self.label_weights = label_weights
        self.cross_validation = cross_validation
        return self.dict_test_and_train

    def get_estimator_sklearn(self, 
                              oob_score = True,
                              n_estimators = 10,
                              max_depth = None,
                              min_samples_split = 2,
                              min_samples_leaf = 1,
                              max_features = 'auto',
                              random_state = None,
                              learn_method = 'classifier'):
        # Instantiate model with 1000 decision trees
        if(learn_method == 'regression'):
            self.model_type = 'RF regressor'
            print('model type:', self.model_type)
            rf = RandomForestRegressor(oob_score = oob_score,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_samples_split = min_samples_split, 
                                        min_samples_leaf = min_samples_leaf,
                                        max_features = max_features,
                                        random_state = random_state)
        elif(learn_method == 'ET_regression'):
            self.model_type = 'ET regressor'
            print('model type:', self.model_type)
            rf = ExtraTreesRegressor(oob_score = oob_score,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_samples_split = min_samples_split, 
                                        min_samples_leaf = min_samples_leaf,
                                        max_features = max_features,
                                        random_state = random_state)
        elif(learn_method == 'classification'):
            self.model_type = 'RF classifier'
            print('model type:', self.model_type)
            rf = RandomForestClassifier(oob_score = oob_score,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_samples_split = min_samples_split, 
                                        min_samples_leaf = min_samples_leaf,
                                        max_features = max_features,
                                        random_state = random_state)
            
        elif(learn_method == 'ET_classification'):
            self.model_type = 'ET_classifier'
            print('model type:', self.model_type)
            rf = ExtraTreesClassifier(oob_score = oob_score,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_samples_split = min_samples_split, 
                                        min_samples_leaf = min_samples_leaf,
                                        max_features = max_features,
                                        random_state = random_state)   
        elif(learn_method == 'XGB_classification'):
            self.model_type = 'classifier'
            print('model type:', self.model_type)
            rf = xgb.XGBClassifier(     n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        random_state = random_state) 
                        
        self.rf = rf
        self._rf_oob_score = oob_score
        self._rf_n_estimators = n_estimators
        self._rf_max_depth = max_depth
        self._rf_min_samples_split = min_samples_split
        self._rf_min_samples_leaf = min_samples_leaf
        self._rf_max_features = max_features
        self._rf_random_state = random_state
        self.forest_model = 'rf'
        return self.rf
    
    
    def get_estimator_xgb(self, learn_method = 'classification',
                          **args
#                          n_estimators = 100,
#                          max_depth = 3,
#                          random_state = None,
#                          gamma = 0,
#                          min_child_weight = 1,
#                    #      max_delta_step = 0,
#                          subsample = 1,
#                       #   colsample_bytree = 1, #0.4
#                          reg_alpha = 0, #0.75
#                          reg_lambda = 1,  #0.45
#                          scale_pos_weight = 1,
#                          verbosity = 1,
#                          objective='binary:logistic',
#                          disable_default_eval_metric = 0,
#                          eval_metric = 'auc',
#                          booster = 'gbtree',
#                          col_sample = {'colsample_bytree':1, 'colsample_bylevel':1, 'colsample_bynode':1},
#                          num_class = 1,
#                          learning_rate = 0.1
                            ):
        
        oob_score, min_samples_split, min_samples_leaf, max_features = None, None, None, None
        
        if(learn_method == 'classification'):
            self.model_type = 'classifier'
            print('model type:', self.model_type)
            rf = xgb.XGBClassifier(
                                    **args
#                                   n_estimators = n_estimators,
#                                   max_depth = max_depth,
#                                   gamma = gamma,
#                                   min_child_weight = min_child_weight,
#                              #     max_delta_step = max_delta_step,
#                                   subsample = subsample,
#                                   seed = random_state,
#                                   reg_alpha = reg_alpha,
#                                   reg_lambda = reg_lambda,
#                                   scale_pos_weight = scale_pos_weight,
#                                   verbosity = verbosity,
#                                   objective = objective,
#                                   eval_metric = eval_metric,
#                                   disable_default_eval_metric = disable_default_eval_metric,
#                                   booster = booster,
#                                   colsample_bytree = col_sample['colsample_bytree'],
#                                   colsample_bylevel = col_sample['colsample_bylevel'],
#                                   colsample_bynode = col_sample['colsample_bynode'],
#                                   num_class = num_class,
#                                   learning_rate = learning_rate
                                   ) 
        elif(learn_method == 'regression'):
            self.model_type = 'regressor'
            print('model type:', self.model_type)
            rf = xgb.XGBRegressor(
                                    **args
#                                  n_estimators = n_estimators,
#                                   max_depth = max_depth,
#                                   gamma = gamma,
#                                   min_child_weight = min_child_weight,
#                                   subsample = subsample,
#                                   seed = random_state,
#                                   learning_rate = learning_rate,
#                                   reg_alpha = reg_alpha,
#                                   reg_lambda = reg_lambda,
#                                   scale_pos_weight = scale_pos_weight,
#                                   verbosity = verbosity,
#                                   colsample_bytree = col_sample['colsample_bytree'],
#                                   colsample_bylevel = col_sample['colsample_bylevel'],
#                                   colsample_bynode = col_sample['colsample_bynode'],
##                                   colsample_bytree = colsample_bytree
                                   ) 
        self.rf = rf
        self._rf_oob_score = oob_score
#        self._rf_n_estimators = n_estimators
#        self._rf_max_depth = max_depth
        self._rf_min_samples_split = min_samples_split
        self._rf_min_samples_leaf = min_samples_leaf
        self._rf_max_features = max_features
#        self._rf_random_state = random_state
#        
 
        self.learn_method = learn_method
#        self.n_estimators = n_estimators
#        self.max_depth = max_depth
#        self.random_state = random_state
#        self.gamma = gamma
#        self.min_child_weight = min_child_weight
#        self.subsample = subsample
#        self.learning_rate = learning_rate
        self.xgb_args = args
        self.forest_model = 'xgb'
                        
    
    def fit_estimator_sklearn(self, force_data = None, force_model = None, early_stopping_rounds = None, weights = None):    
        # Train the model on training data
        if force_data is None:
            self.rf.fit(self.dict_test_and_train['X_train'], self.dict_test_and_train['Y_train'], sample_weight = weights)
        else:
            if force_model is None:
                self.rf.fit(force_data[0], force_data[1])
            else:
                force_model.fit(force_data[0], force_data[1])
                force_model.X_train = force_data[0]
                force_model.Y_train = force_data[1]
                return force_model
        return self.rf
        
    def predict_estimator_sklearn(self, force_data = None, force_model = None): 
        # Use the forest's predict method on the test data
        if force_data is None:
            test_data = self.dict_test_and_train['X_test']
            predictions = self.rf.predict(test_data)
        else:
            if force_model is None:
                predictions = self.rf.predict(force_data)
            else:
                predictions = force_model.predict(force_data)
                return predictions
        self.predictions = predictions
        return self.predictions
        
    def get_bool_accuracy_sklearn(self): 
        predictions = self.predictions
        cnt = 0
        counter = 0
        for i in predictions:
            if(i == self.dict_test_and_train['Y_test'][counter]):
                cnt+=1
            counter+=1
                
        return cnt * 100 / len(predictions)
    
    def get_performance_sklearn(self, force_predictions = None):
        if force_predictions is None:
            predictions = self.predictions
        else:
            predictions = force_predictions
        
        unique_labels =  list(np.unique(predictions))
        label_neg = unique_labels[0]
        label_pos = unique_labels[1]
        if(len(unique_labels) == 2):
            if('False' in unique_labels):
                label_neg = 'False'
                label_pos = 'True'
            elif('bull' in unique_labels):
                label_neg = 'bear'
                label_pos = 'bull'
            else:
                print('unknown label type')
            dict_indicators = md.bin_class_perf_indicators(predictions, self.dict_test_and_train['Y_test'], label_pos = label_pos, label_neg = label_neg)
        else:
            print("not implemented when the number of labels is not 2. Labels are:",unique_labels)
        self.dict_indicators = dict_indicators
        return self.dict_indicators

    def get_perf_regression(self, force_predictions = None):
        if force_predictions is None:
            predictions = self.predictions
        else:
            predictions = force_predictions
        
        
        if(self.normalization == 'standardization'):
            y = self.dict_test_and_train['Y_test'] * np.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]
            y_pred = predictions * np.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]
            dict_regression = {
                  'mse': regression.mean_squared_error(y, y_pred),
                  'mae': regression.mean_absolute_error(y, y_pred)
                  }
        elif(self.normalization == 'minmax'):
#            y = self.scaler.inverse_transform(self.dict_test_and_train['Y_test'])
#            y_pred = self.scaler.inverse_transform(predictions)
#            
            y = mn.denormalize_rescale(self.dict_test_and_train['Y_test'], self.scaler.data_max_[0], self.scaler.data_min_[0])
            y_pred = mn.denormalize_rescale(predictions, self.scaler.data_min_[0], self.scaler.data_max_[0])
            dict_regression = {
                  'mse': regression.mean_squared_error(y, y_pred),
                  'mae': regression.mean_absolute_error(y, y_pred)
                  }
        else:
            dict_regression = {
                              'mse': regression.mean_squared_error(self.dict_test_and_train['Y_test'], predictions),
                              'mae': regression.mean_absolute_error(self.dict_test_and_train['Y_test'], predictions)
                              }
        self.dict_regression = dict_regression
        return dict_regression
    
    def get_ft_imp_sklearn(self):
        print('ft imp debug:',self.feature_names)
        ft_imp = self.rf.feature_importances_
        dict_ft_imp = {}
        for i in range(len(self.rf.feature_importances_)):
            dict_ft_imp[self.feature_names[i]] = ft_imp[i]
        
    def get_log_loss_sklearn(self):
        clf_probs = self.rf.predict_proba(self.dict_test_and_train['X_test'])
        score = log_loss(self.dict_test_and_train['Y_test'], clf_probs)
        self.predict_data = self.dict_test_and_train['Y_test']
        self.prob = clf_probs
        return clf_probs, score
    
    def predict_proba_train(self):
        clf_probs = self.rf.predict_proba(self.dict_test_and_train['X_train'])
        score = log_loss(self.dict_test_and_train['Y_train'], clf_probs)
        self.predict_data = self.dict_test_and_train['Y_train']
        self.prob = clf_probs ## classes order: false,true
        return clf_probs, score
    
    def get_confusion_matrix_sklearn(self):       
        cf_mtx = confusion_matrix(self.predictions, self.dict_test_and_train['Y_test'])
        return cf_mtx
    
    def get_ROC_curve(self, pos_label = 'True', plot = True):   
        curve_tuple = roc_curve(self.predict_data, self.prob[:,1], pos_label = pos_label)
        fp = curve_tuple[0]
        tp = curve_tuple[1]
        if(plot):
            decreasing_thresh = curve_tuple[2]
            
            f2 = interp1d(fp, tp, kind='linear',assume_sorted=True)
            xnew = np.linspace(0, 1, num=100, endpoint=True)

            plt.figure()
            plt.title('ROC curve')
            #plt.plot(curve_tuple[1], curve_tuple[0], label="ROC curve")
            plt.plot(tp,fp, 'o', f2(xnew), xnew,'--')
            for i in range(len(tp)):
                plt.annotate(decreasing_thresh[i],(tp[i],fp[i]))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.show()
        self.fp = fp
        self.tp = tp
        return curve_tuple

    def get_auc_score(self, force_fp = None, force_tp = None):
        if(force_fp is not None):
            fp = force_fp
        else:
            fp = self.fp
        if(force_tp is not None):
            tp = force_tp
        else:
            tp = self.tp
        roc_auc = auc(fp, tp)
        return roc_auc
    
    
    def plot_dendogram(self):
        all_features = self.data['features']
        features = self.feature_names        
#        features.append('open_prices') ### ugly af 
#        features.append('close_prices') ### ugly af 
#        features.append('high_prices') ### ugly af 
#        features.append('low_prices') ### ugly af 
#        features = features[4:] ### ugly af 
        
        used_features = {}
        for i in features:
            try:
                used_features[i] = all_features[i][-3000:]
            except:
                continue
            
        feature_matrix = np.array(list(used_features.values())) 
        mta.plot_dendrogram(feature_matrix.T, features)
        
    def plot_corr_heatmap(self):
        b = pd.DataFrame(self.data['features'])
        c = b.corr()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(c,cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(b.columns),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(b.columns)
        ax.set_yticklabels(b.columns)
        plt.show()
    
    def permutation_importance(self):
        ft_imp = []
        print('dbg', self.model_type)
        if(self.model_type == 'regressor'):
            if(self.dict_regression):
                reference_perf = self.dict_regression['mse']
            else:
                self.dict_regression = self.get_perf_regression()
                reference_perf = self.dict_regression['mse']
        else:
            if(not self.dict_indicators):
                self.predict_estimator_sklearn()
                self.get_performance_sklearn()      
            reference_perf = self.dict_indicators['accuracy']
        x_test = self.dict_test_and_train['X_test']
        for i in range(len(x_test[0,:])):
            data = copy(x_test)
            data[:,i] = np.random.permutation(data[:,i])
            self.predict_estimator_sklearn(data)
            if(self.model_type == 'regressor'):
                dict_regression = self.get_perf_regression()
                perf = dict_regression['mse']
                ft_imp.append(perf - reference_perf)
            else:
                self.get_performance_sklearn()
                perf = self.dict_indicators['accuracy']
                ft_imp.append(reference_perf - perf)
     #   print('dbg', self.feature_names)
        feature_keys = list(self.data['features'].keys())
        dict_ft_imp = {}
        for i in range(len(ft_imp)):
        #    print(feature_keys[i])
            dict_ft_imp[feature_keys[i]] = ft_imp[i]
        return dict_ft_imp
    
    def drop_column_importance(self, eval_function, **eval_args):
        x_test = self.X_test
        x_train = self.X_train
        y_train = self.Y_train
        y_test = self.Y_test
        rf = None
        dummy_obj = get_data_RF()
        dummy_obj.X_test = self.X_test
        dummy_obj.Y_test = self.Y_test
        dummy_obj.predictions = self.predictions
        dummy_obj.rf = self.rf
        try:
            self.predictions[0]
            predictions = self.predictions
        except:
            rf = self.get_estimator_xgb(**self.xgb_args)
            self.fit_estimator_sklearn(force_data = (x_train, y_train), force_model = rf)
            predictions = self.predict_estimator_sklearn(x_test, force_model = rf)
        baseline_perf = eval_function(predictions, y_test, **eval_args, model_obj = self, force_model = rf)
        
        print(np.shape(x_train), np.shape(y_train), np.shape(x_test))
        
        ft_imp = []
        for i in range(len(x_test[0,:])):
            test_data = copy(x_test)
            train_data = copy(x_train)
            test_data = np.delete(test_data, i, 1) ## delete 'i' column from axis 1
            train_data = np.delete(train_data, i, 1) ## delete 'i' column from axis 1
            print(np.shape(train_data))
            
            rf = self.get_estimator_xgb(**self.xgb_args)
            rf = self.fit_estimator_sklearn(force_data = (train_data, y_train), force_model = rf)
            predictions = self.predict_estimator_sklearn(test_data, force_model = rf)
            dummy_obj.X_test = test_data
            dummy_obj.predictions = predictions
            dummy_obj.Y_test = y_test
            perf = eval_function(predictions, y_test, **eval_args, model_obj = dummy_obj, force_model = rf)
            ft_imp.append(baseline_perf - perf)
            
        dict_ft_imp = {}
        for i in range(len(ft_imp)):
            dict_ft_imp[self.feature_names[i]] = ft_imp[i]
        
        self.dict_drop_ft_imp = dict_ft_imp
        return dict_ft_imp
  
    def reshape_by_UTC(self, utc_dict):
        self_utc_vals = self.data['UTC'] 
        start_index, end_index = -1, -1
        
        p_d = np.empty((0,len(self.data['preprocessed_data'][0,:])),float)
        u_d = np.empty((0,len(self.data['unprocessed_data'][0,:])),float)
        print('utc dict', utc_dict, 'start', utc_dict['start'], 'end', utc_dict['end'], 'len_self_utc', len(self_utc_vals))
        for i in range(len(self_utc_vals)):
            if(self_utc_vals[i] >= utc_dict['start']):
                if(start_index == -1):
                    start_index = i
                p_d = np.vstack((self.data['preprocessed_data'][i,:], p_d))
                u_d = np.vstack((self.data['unprocessed_data'][i,:], u_d))
            if((self_utc_vals[i] > utc_dict['end']) and (end_index == -1)):
                end_index = i
                break
            if(i == len(self_utc_vals) - 1):
                end_index = i
        
        self.data['preprocessed_data'] = p_d
        self.data['unprocessed_data'] = u_d
        print('indexes',start_index, end_index, end_index - start_index, len(p_d), len(u_d))
        return self.data

    def reshape_by_UTC2(self, utc_dict_preproc, utc_dict_unproc, utc_vals):
        data_preproc = []
        data_unproc = []
        for i in utc_vals:
            if(i in utc_dict_preproc):
                data_preproc.append(utc_dict_preproc[i])
                data_unproc.append(utc_dict_unproc[i])
            else:
                data_preproc.append(data_preproc[-1])
                data_unproc.append(data_unproc[-1])
#            else:
#                lag = 900000
#                val = i
#                cnt = 0
#                while(val not in utc_dict_preproc):
#                    val -= lag
#                    cnt += 1
#                for _ in range(cnt):
#                    val += lag
#                    utc_dict_preproc.update({val : utc_dict_preproc[i - cnt * lag]})
#                    utc_dict_unproc.update({val : utc_dict_unproc[i - cnt * lag]})
#                    data_preproc.append(utc_dict_preproc[i - cnt * lag])
#                    data_unproc.append(utc_dict_unproc[i - cnt * lag])
#                print('dbg1', i, val)
        
        self.data['preprocessed_data'] = np.array(data_preproc)
        self.data['unprocessed_data'] = np.array(data_unproc)
        return self.data

    def concatenate_datasets(self, in_data, update_labels = False):
        if(self.dict_test_and_train):
            print('debug shapes', in_data['X_test'].shape, self.dict_test_and_train['X_test'].shape)
            print('debug shapes', in_data['X_train'].shape, self.dict_test_and_train['X_train'].shape)
            self.dict_test_and_train['X_train'] = np.concatenate([self.dict_test_and_train['X_train'], in_data['X_train']], axis = 1) 
            self.dict_test_and_train['X_test'] = np.concatenate([self.dict_test_and_train['X_test'], in_data['X_test']], axis = 1)
            self.X_train = self.dict_test_and_train['X_train']
            self.X_test = self.dict_test_and_train['X_test']
        else:
            self.dict_test_and_train['X_test'] = in_data['X_test']
            self.dict_test_and_train['X_train'] = in_data['X_train']
            self.X_train = self.dict_test_and_train['X_train']
            self.X_test = self.dict_test_and_train['X_test']            
        if(update_labels):
            self.dict_test_and_train['Y_test'] = in_data['Y_test']
            self.dict_test_and_train['Y_train'] = in_data['Y_train']
            self.Y_test = self.dict_test_and_train['Y_test']
            self.Y_train = self.dict_test_and_train['Y_train']            
#    def get_start_end_date(self): 
#        print(self.data['start_end_date'])
#        start = md.get_date_from_UTC_ms(self.data['start_end_date']['start'])
#        end = md.get_date_from_UTC_ms(self.data['start_end_date']['end'])
#        return start, end
    
#    def normalize_features(self):
#        for i in range(len(self.dict_test_and_train['X_train'][0,:]) - 5):
#            self.dict_test_and_train['X_train'][:, i + 5] = mn.normalize_rescale(self.dict_test_and_train['X_train'][:, i + 5])[0]
#            self.dict_test_and_train['X_test'][:, i + 5] = mn.normalize_rescale(self.dict_test_and_train['X_test'][:, i + 5])[0]
#        return self.dict_test_and_train
#    
    def xgb_cv(self, nfold = 5, early_stopping_rounds = 50, metrics = {'auc'}):
        rf = self.rf
        train_data = self.dict_test_and_train
        
        params = rf.get_xgb_params()
        data = train_data['X_train']
        label = train_data['Y_train'] /2 + 0.5
        dtrain = xgb.DMatrix( data, label=label, missing = None, weight=np.ones(len(data)) )
        cv_results = xgb.cv(params,
                            dtrain,
                            num_boost_round = params['n_estimators'],
                            nfold = nfold,
                            early_stopping_rounds = early_stopping_rounds,
                            metrics = metrics,
                            seed = 0,
                            shuffle = False)
        self._cv_results = cv_results
        return cv_results
    
#    def grid_search(self, param_dict, scoring = 'roc_auc', n_jobs=4, cv=5):
#        train_data = self.dict_test_and_train
#        data = train_data['X_train']
#        label = train_data['Y_train'] /2 + 0.5
#        
#        gsearch1 = GridSearchCV(estimator = self.rf, 
#                                param_grid = param_dict,
#                                scoring = scoring,
#                                n_jobs = n_jobs,
#                                iid = False,
#                                cv = cv)
#        gsearch1.fit(data,label)
#        results = dict(scores = gsearch1.grid_scores_, best_params = gsearch1.best_params_, best_score = gsearch1.best_score_)
#        self._grid_results = results
#        return results

    def get_dates_test_train(self):
        utc_vals = self.data['UTC']
        train_start = md.get_date_from_UTC_ms(utc_vals[0])
        train_end = md.get_date_from_UTC_ms(utc_vals[len(self.dict_test_and_train['X_train'])])
        test_end = md.get_date_from_UTC_ms(utc_vals[-1])
        print('Training starts from:\n\t', train_start['date_str'], '\ntill: \n\t', train_end['date_str'],'\nTesting until:\n\t', test_end['date_str'])
        return None
    
    def get_class_distribution(self, plot = False):
        y_train = self.Y_train
        y_valid = self.Y_test
        unique, counts = np.unique(y_valid, return_counts=True)
        barWidth = 0.2
        plt.figure()
        plt.bar(unique, counts/len(y_valid), width=barWidth, label = 'Datele de testare')
        unique, counts = np.unique(y_train, return_counts=True)
        plt.bar(unique, counts/len(y_train), width=barWidth/3, label = 'Datele de antrenare')
        plt.title('Probabilitatea de apariţie a etichetelor binare')
        plt.xlabel('Clasele binare')
        plt.ylabel('Probabilitatea')
        plt.legend(loc = 'best')
        plt.show()
        return counts/len(y_train)
        
    def get_dict_ft_imp(self):
        ft_imp = self.rf.feature_importances_
        dict_ft_imp = {}
        features = self.feature_names
        for i in range(len(ft_imp)):
            dict_ft_imp[features[i]] = ft_imp[i]
        self.dict_ft_imp = dict_ft_imp
        return dict_ft_imp
            
    def plot_dict_drop_ft_imp(self, dict_drop_ft_imp = None, dict_ft_imp = None):
        try:
            dict_drop_ft_imp.items()
        except:
            dict_drop_ft_imp = self.dict_drop_ft_imp
        
        try:
            dict_ft_imp.items()
        except:
            dict_ft_imp = self.dict_ft_imp
            
        sorted_dict = np.array(sorted(dict_drop_ft_imp.items(), key=lambda x: x[1]))
        keys = sorted_dict[:,0]
        vals = [float(i) for i in sorted_dict[:,1]]
        
        vals_ft_imp = [dict_ft_imp[i] for i in keys]
        #### plot historgram
        barWidth = 1
        plt.figure(figsize=(20, 1))
        plt.title('Importanţa parametrilor: Metoda eliminării coloanelor')
        #plt.bar(list(dict_perm_ft_imp.keys()), list(dict_perm_ft_imp.values()), align='edge', width=barWidth/2, label = 'permutation performance')
        #plt.bar(list(sorted_dict.keys()), list(sorted_dict.values()), align='edge', width=barWidth/3, label = 'drop column performance')
        plt.bar(keys, vals, width=barWidth*2/3, label = 'Diferenţa de performanţă: Metoda eliminării coloanelor')
        plt.bar(keys, np.multiply(vals_ft_imp,2), width=barWidth/3, label = 'Diferenţa de performanţă: Metoda XGB - scalată')
        #plt.bar(list(dict_ft_imp.keys()), list(dict_ft_imp.values()), align='edge', width=barWidth/4, label = 'impurity decrease performance')
        plt.xticks([r for r in range(len(keys))], keys)
        plt.xticks(rotation=90)
        plt.ylabel('Diferenţa de performanţă')
        plt.legend(loc = 'best')
        
    def get_binary_aucpr(self, y_test = None, predictions = None, probs = None, plot = False, pos_weight = None, label = ''):
        try:
            y_test[0]
        except:
            y_test = self.Y_test
        try:
            predictions[0]
        except:
            predictions = self.predictions  
        try:
            probs[0]
        except:
            probs = self.probs
        labels = np.unique(predictions)
        #print('dbg',labels, len(labels))
        if(len(labels) > 1):
            pos_label = labels[1]
                      
            if(pos_weight):
                weights = [pos_weight if (y_test[i] == pos_label) else (1-pos_weight) for i in range(len(predictions))]
                print('weighted f1',  sm.f1_score(y_test, predictions, sample_weight = weights))  
                print('weighted precision', sm.precision_score(y_test, predictions,sample_weight = weights))
                print('weighted recall',  sm.recall_score(y_test, predictions, sample_weight = weights))
            else:
                weights = None
                print('precision', sm.precision_score(y_test, predictions,sample_weight = None))
                print('f1',  sm.f1_score(y_test, predictions, sample_weight = None))
                print('recall',  sm.recall_score(y_test, predictions, sample_weight = None))
                
            #probs = rf.predict_proba(test_train_data['X_test']) 
            precision, recall, thresholds  = sm.precision_recall_curve(y_test, probs[:,1], sample_weight = weights)
            
            #plt.figure()
            if(plot == 'normal'):
                plt.figure()
                plt.title('Precizia în funcţie de recall')
                plt.plot(recall, precision)
                plt.xlabel('Recall')
                plt.ylabel('Precizie')
                plt.legend(loc = 'best')
                plt.show()
                
                plt.figure()
                plt.title('Valorile preciziei şi a recall-ului în funcţie de pragurile atribuite probabilităţilor de decizie')
                plt.plot(thresholds, precision[1:], label = 'Precizie ' + label)
                plt.plot(thresholds, recall[1:], label = 'Recall' + label)
                plt.xlabel('Valorile pragurilor')
                plt.ylabel('Amplitudine')
                plt.legend(loc = 'best')
                plt.show()
                
                plt.figure()
                plt.title('Recall Probabiltiy threshold')
                plt.plot(thresholds, recall[1:])
                plt.xlabel('thresholds')
                plt.ylabel('Recall')
                plt.legend(loc = 'best')
                plt.show()
            elif(plot == 'iter'):
                plt.title('Valorile preciziei şi a recall-ului în funcţie de pragurile atribuite probabilităţilor de decizie')
                plt.plot(thresholds, precision[1:], label = 'Precizie ' + label)
                plt.plot(thresholds, recall[1:], label = 'Recall' + label)
                plt.xlabel('Valorile pragurilor')
                plt.ylabel('Amplitudine')
                plt.legend(loc = 'best')
                plt.show()
            return dict(precision = precision, recall = recall, thresholds = thresholds)
        else:
            return None
            
    def eval_volatility_PR(self, *_, max_precision_thresh = 0.7, resolution = 0.05, model_obj = None, force_model = None):
        if(model_obj):
            if(force_model):
                probs = force_model.predict_proba(model_obj.X_test) 
                aucpr = model_obj.get_binary_aucpr(plot = False, probs = probs)
                pred = model_obj.predictions
                y = model_obj.Y_test
            else:
                probs = model_obj.rf.predict_proba(model_obj.X_test) 
                aucpr = model_obj.get_binary_aucpr(plot = False, probs = probs)
                pred = model_obj.predictions
                y = model_obj.Y_test
        else:
            probs = self.rf.predict_proba(self.X_test) 
            aucpr = self.get_binary_aucpr(plot = False, probs = probs)
            pred = self.predictions
            y = self.Y_test

        if(aucpr):
            precision  = aucpr['precision']
            recall  = aucpr['recall']
            precision = precision[:-1]
            max_val = np.max(precision)
            print('max_precision', max_val)
            if(max_val > max_precision_thresh):
                #peaks_len = int(len(precision) * resolution)
                peaks_len = len(precision[precision > max_precision_thresh])
                high = np.sort(precision)[-peaks_len:]
                high = np.array(high)
                pr = []
                precision = list(precision)
                for h in high:
                    pr.append(h * recall[precision.index(h)])
                #print('pow', np.sum(np.power(pr,2)))
                max_ratio = np.max(pr)
                high = list(high)
                precision_max_ratio = high[pr.index(max_ratio)]
                recall_max_ratio = recall[precision.index(precision_max_ratio)]
                score = np.sum(np.power(pr,2))/ peaks_len
                print('-------------Results PR thresholds',
                      '\n\t PR threshold score:', score,
                      '\n\t best ratio:', max_ratio,
                      '\n\t precision: ', precision_max_ratio,
                      '\n\t recall:', recall_max_ratio,
                      '\n\t number of vals:',peaks_len,
                      '\n-------------')
                return score
            else:
                cnt_p, cnt_f = 0, 0
                for i in range(len(pred)):
                    if(pred[i] >0):
                        if(pred[i] * y[i] >= 0):
                            cnt_p += 1
                        else:
                            cnt_f += 1
            
                pr = cnt_p / (cnt_p + cnt_f)
                print('dbg lower than max', pr, -(1 - pr))
                pr = -np.power((1 - pr), 2)
                return pr
        else:
            return 0
    def plot_tree_xgb(self):
        a = self.rf.get_booster()
        b = self.feature_names
        a.feature_names = b
        xgb.plot_tree(a)

    def alarm(self, path = '/home/catalin/alarm.mp3'): 
        md.myalarm(path)
    
    def setter(**agrs):
        self.args = args
        
        
            
            
            
    
            
        