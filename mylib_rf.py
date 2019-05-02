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


import logging
import datetime

import os
import sys
from scipy.interpolate import interp1d

from copy import copy

import mylib_dataset as md

#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc


def init_paths():
    user_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    #user_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(user_path)
    os.chdir('/home/catalin/git_workspace/disertatie')
    user_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    dataset_path = user_path + '/databases/klines_2014-2018_15min/'  
    dataset_path = '/home/catalin/git_workspace/disertatie/databases/klines_2014-2018_15min/'
    user_path = '/home/catalin/git_workspace/disertatie'
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    
    log_path = user_path + "/databases/temp_log"
    log_dir = user_path + '/databases/'
    filename = 'temp_log'
    log_path = log_dir + filename + '.log'
    
    try:
        os.remove(log_path)
    #    os.system('touch '+ log_path)
    except:
        pass
    
    
    fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir, filename))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    
    return dataset_path, log_path, rootLogger

##################### get log for estimator
proc_time_start = datetime.datetime.now()

#user_path = os.path.dirname(os.path.realpath(__file__))

class get_data_RF():
    def __init__(self):   
        self.cross_validation = False
        self.feature_names = None
        self.dict_indicators = {}
    
    def get_input_data(self, 
                       normalization_method = 'rescale', 
                       database_path = None,
                       feature_dicts = {}, 
                       datetime_interval = None,
                       blockchain_indicators_dicts = {},
                       lookback = 0, 
                       dataset_type = 'dataset1',
                       force_data = None,
                       normalization = None):
        
        dict_data = md.get_database_data(dataset_directory = database_path, 
                                          normalization_method = 'rescale',
                                          datetime_interval = {},
                                          preproc_constant = 0, 
                                          lookback = lookback,
                                          input_noise_debug = False,
                                          dataset_type = dataset_type)
        data = md.get_features(candle_data = dict_data['X'],
                                feature_dicts = feature_dicts, 
                                blockchain_indicators_dicts = {}, 
                                lookback = lookback,
                                normalization = normalization)
        
        data['preprocessed_data'] = data['data']
        self.data = data
        self.feature_names = list(data['features'].keys())
        print(self.feature_names)
        return self.data
    
    def get_features_from_data(self, candle_data = None, feature_dicts = None, normalization = None):
        data = md.get_features(candle_data = candle_data,
                                feature_dicts = feature_dicts, 
                                blockchain_indicators_dicts = {}, 
                                lookback = None,
                                normalization = normalization) 
        return data
    
    def get_test_and_train_data(self, 
                                label_window = 100 * 3,
                                chunks = 11, 
                                chunks_for_training = 9, 
                                remove_chunks_from_start = 0, 
                                label_type = '', 
                                thresh_ratio = 1,
                                force_data = None,
                                cross_validation = None):
        if(force_data):
            self.dict_test_and_train = force_data['test_train_data']
            self.label_window = force_data['label_window']
            h, l = 0, 0
            print('Using input data.')
        else:
            test_and_train_data = md.get_test_and_train_data(preprocessed_data = self.data['preprocessed_data'], 
                                                             unprocessed_data = self.data['data'], 
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
            X_train,Y_train, h, l = md.get_labels_mean_window(train_data, train_data_unprocessed[:,0], label_window, label_type, thresh_ratio) 
            X_test, Y_test, h, l = md.get_labels_mean_window(test_data, test_data_unprocessed[:,0], label_window, label_type, thresh_ratio)
    
            dict_test_and_train = {'X_test': X_test,
                                   'Y_test': Y_test,
                                   'X_train': X_train,
                                   'Y_train': Y_train}
            
            self.dict_test_and_train = dict_test_and_train
            self.label_window = label_window
        self.cross_validation = cross_validation
        return self.dict_test_and_train, h, l
    
            
    def get_estimator_sklearn(self, 
                              oob_score = True,
                              n_estimators = 10,
                              max_depth = None,
                              min_samples_split = 2,
                              min_samples_leaf = 1,
                              max_features = 'auto',
                              random_state = None):
        # Instantiate model with 1000 decision trees
        #rf = RandomForestRegressor()
        rf = RandomForestClassifier(oob_score = oob_score,
                                    n_estimators = n_estimators,
                                    max_depth = max_depth,
                                    min_samples_split = min_samples_split, 
                                    min_samples_leaf = min_samples_leaf,
                                    max_features = max_features,
                                    random_state = random_state)
        self.rf = rf
        self._rf_oob_score = oob_score
        self._rf_n_estimators = n_estimators
        self._rf_max_depth = max_depth
        self._rf_min_samples_split = min_samples_split
        self._rf_min_samples_leaf = min_samples_leaf
        self._rf_max_features = max_features
        self._rf_random_state = random_state
        return self.rf
    
    def fit_estimator_sklearn(self, force_data = None, force_model = None):    
        # Train the model on training data
        if force_data is None:
            self.rf.fit(self.dict_test_and_train['X_train'], self.dict_test_and_train['Y_train'])
        else:
            if force_model is None:
                self.rf.fit(force_data[0], force_data[1])
            else:
                force_model.fit(force_data[0], force_data[1])
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
            print("not implemented when the number of labels is not 2")
        self.dict_indicators = dict_indicators
        return self.dict_indicators
    
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

    def get_auc_score(self):
        roc_auc = auc(self.fp, self.tp)
        return roc_auc
    
    
    def plot_dendogram(self):
        all_features = self.data['features']
        features = self.feature_names        
        features.append('open_prices') ### ugly af 
        features.append('close_prices') ### ugly af 
        features.append('high_prices') ### ugly af 
        features.append('low_prices') ### ugly af 
        features = features[4:] ### ugly af 
        
        used_features = {}
        for i in features:
            try:
                used_features[i] = all_features[i][-3000:]
            except:
                continue
            
        feature_matrix = np.array(list(used_features.values())) 
        mta.plot_dendrogram(feature_matrix.T, features)
    
    def permutation_importance(self):
        if(not self.dict_indicators):
            self.predict_estimator_sklearn()
            self.get_performance_sklearn()
            
        ft_imp = []
        baseline_accuracy = self.dict_indicators['accuracy']
        x_test = self.dict_test_and_train['X_test']
        for i in range(len(x_test[0,:])):
            data = copy(x_test)
            data[:,i] = np.random.permutation(data[:,i])
            self.predict_estimator_sklearn(data)
            self.get_performance_sklearn()
            accuracy = self.dict_indicators['accuracy']
            ft_imp.append(baseline_accuracy - accuracy)
        dict_ft_imp = {}
        for i in range(len(ft_imp)):
            dict_ft_imp[self.feature_names[i]] = ft_imp[i]
        return dict_ft_imp
    
    def drop_column_importance(self):
        if(not self.dict_indicators):
            self.predict_estimator_sklearn()
            self.get_performance_sklearn()
            
        ft_imp = []
        baseline_accuracy = self.dict_indicators['accuracy']
        x_test = self.dict_test_and_train['X_test']
        x_train = self.dict_test_and_train['X_train']
        y_train = self.dict_test_and_train['Y_train']
        for i in range(len(x_test[0,:])):
            test_data = copy(x_test)
            train_data = copy(x_train)
            test_data = np.delete(test_data, i, 1) ## delete 'i' column from axis 1
            train_data = np.delete(train_data, i, 1) ## delete 'i' column from axis 1
            
            rf = self.get_estimator_sklearn(oob_score = self._rf_oob_score,
                                            n_estimators = self._rf_n_estimators,
                                            max_depth = self._rf_max_depth,
                                            min_samples_split = self._rf_min_samples_split,
                                            min_samples_leaf = self._rf_min_samples_leaf,
                                            max_features = self._rf_max_features,
                                            random_state = self._rf_random_state)
            rf = self.fit_estimator_sklearn(force_data = (train_data, y_train), force_model = rf)
            predictions = self.predict_estimator_sklearn(test_data, force_model = rf)
            indicators = self.get_performance_sklearn(force_predictions = predictions)
            accuracy = indicators['accuracy']
            ft_imp.append(baseline_accuracy - accuracy)
        dict_ft_imp = {}
        for i in range(len(ft_imp)):
            dict_ft_imp[self.feature_names[i]] = ft_imp[i]
        return dict_ft_imp
        
    def concatenate_datasets(self, in_data):
        #print('aa', in_data['X_test'].shape, self.dict_test_and_train['X_test'].shape)
        self.dict_test_and_train['X_test'] = np.concatenate([self.dict_test_and_train['X_test'], in_data['X_test']], axis = 1)
        self.dict_test_and_train['X_train'] = np.concatenate([self.dict_test_and_train['X_train'], in_data['X_train']], axis = 1)
                    
    def get_start_end_date(self):
        print(self.data['start_end_date'])
        start = md.get_date_from_UTC_ms(self.data['start_end_date']['start'])
        end = md.get_date_from_UTC_ms(self.data['start_end_date']['end'])
        return start, end
    
    def normalize_features(self):
        for i in range(len(self.dict_test_and_train['X_train'][0,:]) - 5):
            self.dict_test_and_train['X_train'][:, i + 5] = mn.normalize_rescale(self.dict_test_and_train['X_train'][:, i + 5])[0]
            self.dict_test_and_train['X_test'][:, i + 5] = mn.normalize_rescale(self.dict_test_and_train['X_test'][:, i + 5])[0]
        return self.dict_test_and_train
    
    def alarm(self, path = '/home/catalin/alarm.mp3'): 
        md.myalarm(path)
        
        
        
            
            
            
    
            
        