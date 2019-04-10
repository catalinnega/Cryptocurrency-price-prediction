#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:47:01 2019

@author: catalin
"""
import numpy as np
import matplotlib.pyplot as plt

import logging
import datetime

import pickle

import os
import sys

user_path = os.path.dirname(os.path.realpath(sys.argv[0]))
#user_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(user_path)
os.chdir('/home/catalin/git_workspace/disertatie')
user_path = os.path.dirname(os.path.realpath(sys.argv[0]))
dataset_path = user_path + '/databases/klines_2014-2018_15min/'  
dataset_path = '/home/catalin/git_workspace/disertatie/databases/klines_2014-2018_15min/'
user_path = '/home/catalin/git_workspace/disertatie'

import mylib_dataset as md
import mylib_normalize as mn
import feature_list
import tensorflow as tf
from tensorflow.contrib.tensor_forest.client import random_forest


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

    
##################### get log for estimator
proc_time_start = datetime.datetime.now()


#user_path = os.path.dirname(os.path.realpath(__file__))


class get_data_RF():
#    def __init__(self):   
#
#    
    def get_input_data(self, skip_preprocessing, preprocessing_constant, normalization_method, database_path, feature_names, datetime_interval, blockchain_indicators):
        data = md.get_dataset_with_descriptors(skip_preprocessing = skip_preprocessing, 
                                               preproc_constant = preprocessing_constant, 
                                               normalization_method = normalization_method,
                                               dataset_directory = database_path,
                                               feature_names = feature_names,
                                               datetime_interval = datetime_interval,
                                               blockchain_indicators = blockchain_indicators)
        
        self.data = data
        self.feature_names = feature_names
        
        return self.data
    
    def get_test_and_train_data(self, label_window = 100 * 3, chunks = 11, chunks_for_training = 9, remove_chunks_from_start = 0):
        test_and_train_data = md.get_test_and_train_data(preprocessed_data = self.data['preprocessed_data'], 
                                                         unprocessed_data = self.data['data'], 
                                                         chunks = chunks , 
                                                         chunks_for_training = chunks_for_training,
                                                         remove_chunks_from_start = remove_chunks_from_start)
        train_data = test_and_train_data["train_data_preprocessed"]
        test_data = test_and_train_data["test_data_preprocessed"]
        test_data_unprocessed = test_and_train_data["test_data_unprocessed"]
        train_data_unprocessed = test_and_train_data["train_data_unprocessed"]
        X_train,Y_train, h, l = md.create_dataset_labels_mean_percentages(train_data, train_data_unprocessed, label_window, False) 
        X_test, Y_test, h, l = md.create_dataset_labels_mean_percentages(test_data, test_data_unprocessed, label_window, True)
        
        dict_test_and_train = {'X_test': X_test,
                               'Y_test': Y_test,
                               'X_train': X_train,
                               'Y_train': Y_train}
        
        self.dict_test_and_train = dict_test_and_train
        return self.dict_test_and_train
    
    def estimator_fit(self, regression = True, num_classes = 2, num_trees = None, max_nodes = 1000, max_fertile_nodes = 0):
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
                                                                             max_nodes = max_nodes, 
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
        dict_indicators = md.bin_class_perf_indicators(results_bool_up, reference_bool_up)
        
        self.dict_indicators = dict_indicators
        return self.dict_indicators
    
    def concatenate_datasets(self, in_data):
        print('aa', in_data['X_test'].shape, self.dict_test_and_train['X_test'].shape)
        self.dict_test_and_train['X_test'] = np.concatenate([self.dict_test_and_train['X_test'], in_data['X_test']], axis = 1)
        self.dict_test_and_train['X_train'] = np.concatenate([self.dict_test_and_train['X_train'], in_data['X_train']], axis = 1)
                    
    def get_start_end_date(self):
        print(self.data['start_end_date'])
        start = md.get_date_from_UTC_ms(self.data['start_end_date']['start'])
        end = md.get_date_from_UTC_ms(self.data['start_end_date']['end'])
        return start, end




        
#a =  data_eth['dataset_dict']['UTC']
#b =  data_btc['dataset_dict']['UTC']
#for i in range(len(a)):
#    print(get_date_from_UTC_ms(a[i])['date_datetime'])
#    if(a[i] != b[i]):
#        break

feature_names = feature_list.get_features_list()
blockchain_indicators = feature_list.get_blockchain_indicators()


start_date = md.get_datetime_from_string('2017-01-19')
end_date = md.get_datetime_from_string('2018-05-19')
dataset_path_btc = dataset_path
dataset_path_eth = '/home/catalin/git_workspace/disertatie/databases/BTCETH_klines_2014-2018_15min/'

eth = get_data_RF()
data_eth = eth.get_input_data(database_path = dataset_path_eth, 
                              feature_names = feature_names, 
                              preprocessing_constant = 0.99, 
                              normalization_method = 'rescale',
                              skip_preprocessing = False,
                              datetime_interval = {'start':start_date,'end':end_date},
                              blockchain_indicators = None)

start, end = eth.get_start_end_date()

test_train_eth = eth.get_test_and_train_data()


btc = get_data_RF()
data_btc = btc.get_input_data(database_path = dataset_path_btc, 
                              feature_names = feature_names, 
                              preprocessing_constant = 0.99, 
                              normalization_method = 'rescale',
                              skip_preprocessing = False,
                              datetime_interval = {'start':start_date,'end':end_date},
                              blockchain_indicators = blockchain_indicators)

test_train_btc = btc.get_test_and_train_data()
btc.concatenate_datasets(test_train_eth)
btc.estimator_fit()
#ft,ft_c = btc.get_feature_importances(log_path = log_path)
btc.estimator_test()
dict_indicators = btc.get_results()


md.print_dictionary(dict_indicators)


