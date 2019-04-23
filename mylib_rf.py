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


import mylib_dataset as md
import tensorflow as tf
from tensorflow.contrib.tensor_forest.client import random_forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc



    
##################### get log for estimator
proc_time_start = datetime.datetime.now()


#user_path = os.path.dirname(os.path.realpath(__file__))


class get_data_RF():
#    def __init__(self):   
#
#    
    def get_input_data(self, skip_preprocessing = False, 
                       preprocessing_constant = 0,
                       normalization_method = 'rsscale', 
                       database_path = None,
                       feature_names = None, 
                       datetime_interval = None,
                       blockchain_indicators = None,
                       lookback = 0, 
                       dataset_type = 'dataset1',
                       force_data = None):
        if(force_data):
            self.data = force_data['data']
            self.feature_names = force_data['feature_names']
        else:
            data = md.get_dataset_with_descriptors(skip_preprocessing = skip_preprocessing, 
                                                   preproc_constant = preprocessing_constant, 
                                                   normalization_method = normalization_method,
                                                   dataset_directory = database_path,
                                                   feature_names = feature_names,
                                                   datetime_interval = datetime_interval,
                                                   blockchain_indicators = blockchain_indicators,
                                                   lookback = lookback,
                                                   dataset_type = dataset_type)
            
            self.data = data
            self.feature_names = feature_names
        
        return self.data
    
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
        return self.dict_test_and_train, h, l
    
    def estimator_fit(self,
                      regression = True,
                      num_classes = 2, 
                      num_trees = None,
                      max_nodes = 1000, 
                      max_fertile_nodes = 0, 
                      rootLogger = None):
        #### Random Forest 
        X_train = np.float32(self.dict_test_and_train['X_train'])
        Y_train = np.float32(self.dict_test_and_train['Y_train'])
        num_features = X_train.shape[-1]
        
        if(num_trees == None):
            num_trees = num_features
            
        tf.reset_default_graph()
        
        params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(regression = regression,
                                                                             num_classes = num_classes, 
                                                                             num_features = num_features, 
                                                                             num_trees = num_trees,## can double it
                                   #                                          max_nodes = max_nodes, 
                                                                             num_outputs = np.shape(Y_train)[1],
                                                                             max_fertile_nodes = max_fertile_nodes, ## 100?
                                                                              #prune_every_samples = 300,
                                                                              #split_finish_name='basic',
                                                                          #    pruning_name='half'
                                                                        
                                                                              #model_name = 'all_sparse' ## default is all_dense
                                                                              #feature_bagging_fraction = 0.7
                                                                        #      use_running_stats_method=True,
                                                                        #      checkpoint_stats= True,
                                                                        ##      bagging_fraction=1
                                                                        #      feature_bagg
                                                                              )
        estimator = random_forest.TensorForestEstimator(params, report_feature_importances = True) 
        estimator.config.save_checkpoints_steps
        estimator.config.save_checkpoints_secs
        
        #with tf.Session() as session:
          #input_fn_train = tf.estimator.inputs.numpy_input_fn(X_train, X_train, batch_size=1, shuffle=False, num_epochs=1)
          #input_fn_test = tf.estimator.inputs.numpy_input_fn(X_test, batch_size=1, shuffle=False, num_epochs=1)
        rootLogger.info(estimator.fit(X_train, Y_train))
          #rootLogger.info(estimator.fit(input_fn_train))
        self.estimator = estimator
        #self.session = session
         # self.input_fn_test = input_fn_test
         # self.input_fn_train = input_fn_train
    
    
        
    def get_feature_importances(self, compare_results_with_previous_run = True, log_path = None):
        dict_feature_importances = md.get_feature_importances(self.feature_names, log_path)
        if(compare_results_with_previous_run):
            previous_log_path = log_path[:-4] + '_previous.log'
            dict_feature_importances_previous = md.get_feature_importances(self.feature_names, previous_log_path)
            dict_compare_feature_importances = {}
            for i in range(len(dict_feature_importances_previous)):
                dict_compare_feature_importances[self.feature_names[i]] = dict_feature_importances[self.feature_names[i]] \
                                                                    - dict_feature_importances_previous[self.feature_names[i]]
        os.rename(log_path, log_path[:-4] + '_previous.log')
        self.dict_feature_importances = dict_feature_importances
        self.dict_compare_feature_importances = dict_compare_feature_importances
        return self.dict_feature_importances, self.dict_compare_feature_importances
    
    def estimator_test(self):
        #evaluate = estimator.evaluate(X_test,Y_test)
          
        ## Predict returns an iterable of dicts.
        X_test = np.float32(self.dict_test_and_train['X_test'])
        results = list(self.estimator.predict(X_test))
        #results = list(self.estimator.predict(self.input_fn_train))
        #print(results)
        self.results = results
        return self.results
    
    def get_results(self):
        Y_test = np.float32(self.dict_test_and_train['Y_test'])
            
        dict_percentages = {}
        dict_percentages['up_percentage']   = [self.results[i]['scores'][0] for i in range (len(self.results))]
        dict_percentages['down_percentage'] = [self.results[i]['scores'][1] for i in range (len(self.results))]
        dict_percentages['up_reference']    = Y_test[:,0]
        dict_percentages['down_reference'] = Y_test[:,1]
        
        results_bool_up = np.array(dict_percentages['up_percentage']) > np.array(dict_percentages['down_percentage']) #going up
        reference_bool_up = dict_percentages['up_reference'] > dict_percentages['down_reference']# going up
        
        results_bool_up  = md.boolarray_to_int(results_bool_up)
        reference_bool_up  = md.boolarray_to_int(reference_bool_up)
        
        accurracy_bool_up = (np.array(results_bool_up) == np.array(reference_bool_up))
        accurracy_bool_up  = md.boolarray_to_int(accurracy_bool_up)
        
        
        up_percentage = md.counter(accurracy_bool_up, 1, percentage = True)
                
        print('\n\n accuracy bool up:' + str(up_percentage) +' %')
        
        ## evaluation metrics dictionary
        dict_indicators = md.bin_class_perf_indicators(results_bool_up, reference_bool_up, label_pos = 1, label_neg = 0)
        
        self.dict_indicators = dict_indicators
        self.buy_bool_indexes = results_bool_up
        return self.dict_indicators, dict_percentages
    
    def concatenate_datasets(self, in_data):
        print('aa', in_data['X_test'].shape, self.dict_test_and_train['X_test'].shape)
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
    
    def simulate_investment(self, initial_investment = 1000, trading_fee = 0.02):
#        initial_investment = 1000
#        trading_fee = 0.02
        buy_bool_indexes = self.buy_bool_indexes
        data = self.data
        
        ## find the first buy signal
        for i in range(len(buy_bool_indexes)):
            if(buy_bool_indexes[i] == True):
                start_index_buy = i
                break
         
               
        index_buy = start_index_buy
        percentage_gain = 0
        debug_gain = []
        debug_index_buy = []
        debug_index_sell = []
        index_sell = -1
        hangover = self.label_window
        sell_flag = 0
        
        close_candles = data["test_data_unprocessed"][:,0]
        
        investment = initial_investment
        for i in range(len(buy_bool_indexes)):
            if((buy_bool_indexes[i] == False) and (buy_bool_indexes[i-1] == True)):
                ##sell signal
                #set hangover. It gets decremented with every iteration if a buy signal is not met
                sell_flag = hangover
            if((buy_bool_indexes[i] == True) and (buy_bool_indexes[i-1] == False)):
                ##buy signal
                if ((not(sell_flag)) or (hangover == 1)):
                    index_buy = i
                sell_flag = 0
                
            if(sell_flag):
                if(buy_bool_indexes[i] == True):
                    sell_flag = hangover
                else:
                     sell_flag -= 1   
                if(sell_flag == 0):
                    ## sell
                    gain = ((close_candles[i] - close_candles[index_buy])/close_candles[index_buy]) * 100
                    investment = investment  * (gain / 100) + investment - (investment * trading_fee)
                    percentage_gain += gain
                    debug_gain.append(gain)
                    index_sell = i
                
            if(index_buy == i):   
                debug_index_buy.append(close_candles[i])
            else:
                debug_index_buy.append(0)
            if(index_sell == i):   
                debug_index_sell.append(close_candles[i])
            else:
                debug_index_sell.append(0)
                debug_gain.append(0)
                
        
        print("percentage_gain: " + str(percentage_gain) + " %")
        print("number of trades: " + str(len(np.array(np.where(np.array(debug_gain) > 2))[0]) * 2))
        print("initial investment : " + str(initial_investment) + " <--> current amount: " + str(int(investment)) )
        
        debug_index_buy = mn.denormalize_rescale(debug_index_buy, data['min_prices'], data['max_prices'])
        debug_index_sell = mn.denormalize_rescale(debug_index_sell, data['min_prices'], data['max_prices'])
        close_candles = mn.denormalize_rescale(close_candles, data['min_prices'], data['max_prices'])
         
        plot_flag = True
        if(plot_flag):
            
            plt.close('all')
            plt.figure()
            plt.title("Bitcoin historical data")
            plt.plot(data["train_data_unprocessed"][:,0][:,0])
            
            plt.figure()
            plt.title("Buy and sell orders relative to closing candle prices")
            plt.plot(close_candles)
            plt.plot(debug_index_buy, label = 'buy')
            plt.plot(debug_index_sell, label  = 'sell')
            #plt.plot(np.array(debug_gain)*0.015, label = 'gain from the trade', color = 'purple')
            plt.legend(loc='best')
            plt.ylabel('Close prices ($)')
            plt.xlabel('Samples(each represents a 15 min candle)')
            plt.show()  
            
            plt.figure()
            plt.title("Gain(%) relative to closing candle prices.\n (closing prices have been scaled for visualization)")
            plt.plot(close_candles/300, label = 'close prices')
            plt.plot(debug_gain, label = 'gain')
            plt.legend(loc='best')
            plt.ylabel('Close prices ($)')
            plt.xlabel('Samples(each represents a 15 min candle)')
            plt.show()
            
    def get_estimator_sklearn(self, 
                              oob_score = True,
                              n_estimators = 10,
                              max_depth = None,
                              min_samples_split = 2,
                              min_samples_leaf = 1,
                              max_features = 'auto'):
        # Instantiate model with 1000 decision trees
        #rf = RandomForestRegressor()
        rf = RandomForestClassifier(oob_score = oob_score,
                                    n_estimators = n_estimators,
                                    max_depth = max_depth,
                                    min_samples_split = min_samples_split, 
                                    min_samples_leaf = min_samples_leaf,
                                    max_features = max_features)
        self.rf = rf
        return self.rf
    
    def fit_estimator_sklearn(self):    
        # Train the model on training data
        self.rf.fit(self.dict_test_and_train['X_train'], self.dict_test_and_train['Y_train'])
        return self.rf
        
    def predict_estimator_sklearn(self): 
        # Use the forest's predict method on the test data
        predictions = self.rf.predict(self.dict_test_and_train['X_test'])
        self.predictions = predictions
        return self.predictions
        
    def get_bool_accuracy_sklearn(self): 
        #a = [test_train_eth['Y_test'][i] > test_train_eth['X_test'][i,0] for i in range(1,len(test_train_eth['Y_test']))]
        #b = [predictions[i] > test_train_eth['X_test'][i,0] for i in range(1,len(predictions))]
        #
        predictions = self.predictions
        cnt = 0
        counter = 0
        for i in predictions:
            if(i == self.dict_test_and_train['Y_test'][counter]):
                cnt+=1
            counter+=1
                
#        print(cnt*100/len(predictions))
        return cnt * 100 / len(predictions)
    
    def get_performance_sklearn(self):
        unique_labels =  list(np.unique(self.predictions))
        if(len(unique_labels) == 2):
            if('False' in unique_labels):
                label_neg = 'False'
                label_pos = 'True'
            elif('bull' in unique_labels):
                label_neg = 'bear'
                label_pos = 'bull'
            else:
                print('unknown label type')
            dict_indicators = md.bin_class_perf_indicators(self.predictions, self.dict_test_and_train['Y_test'], label_pos = label_pos, label_neg = label_neg)
        else:
            print("not implemented when the number of labels is not 2")
        self.dict_indicators = dict_indicators
        return self.dict_indicators
    
    def get_ft_imp_sklearn(self):
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
    
#    def get_oob_score(self,)
    
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
    
    def alarm(self, path = '/home/catalin/alarm.mp3'): 
        md.myalarm(path)
        
        
        
            
            
            
    
            
        