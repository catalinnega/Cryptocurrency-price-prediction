#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:06:49 2019

@author: catalin
"""
import pickle 
import numpy as np
from datetime import timedelta, date
import mylib_dataset as md
import matplotlib.pyplot as plt
import copy

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_sentiment_indicator_from_db(dataset_dict, path = '/home/catalin/databases/tweets/nltk_2014_2018_300_per_day.pkl'):
    with open(path, 'rb') as f:
        tweets = pickle.load(f)
        
    start_date = date(2014, 10, 2)
    end_date = date(2018, 11, 20)
    dates = []
    for single_date in daterange(start_date, end_date):
        dates.append(str(single_date))
        
    start_date = md.get_date_from_UTC_ms(dataset_dict['dataset_dict']['UTC'][0])
    end_date = md.get_date_from_UTC_ms(dataset_dict['dataset_dict']['UTC'][-1])
    
    start = False
    end = False
    sentiment_indicator_positive = np.zeros(len(dataset_dict['data']))
    sentiment_indicator_negative = np.zeros(len(dataset_dict['data']))
    step_1_day = 15*4*24
    for i in range(len(dates)):
        if(dates[i].find(end_date) != -1):
            end = True
        if(dates[i].find(start_date) != -1):
            start = True
        if(start and not end):
            print(i)
            ### offset with one step to adjust for real-time data feed.
            sentiment_indicator_positive[step_1_day * (i+1) : step_1_day * (i+2)] = tweets[dates[i]]['pos']
            sentiment_indicator_negative[step_1_day * (i+1) : step_1_day * (i+2)] = tweets[dates[i]]['neg']
    return np.array(sentiment_indicator_positive), np.array(sentiment_indicator_negative)

directory = '/home/catalin/databases/klines_2014-2018_15min/'
data = md.get_dataset_with_descriptors2(concatenate_datasets_preproc_flag = True, 
                                       preproc_constant = 0.99, 
                                       normalization_method = "rescale",
                                       dataset_directory = directory,
                                       hard_coded_file_number = 0,
                                       feature_names = ['']) 
sentiment_pos, sentiment_neg, _ = get_sentiment_indicator_from_db(data)

prices = data['preprocessed_data']
X = copy.copy(prices)
X = X[:,1:] ## remove time column
close_prices = X[:,1]
cumulative_pos = [0]
cumulative_neg = [0]
crossing = []
for i in range(len(sentiment_pos)):
    cumulative_pos.append(sentiment_pos[i] + cumulative_pos[-1])
    cumulative_neg.append(sentiment_neg[i] + cumulative_neg[-1])
    crossing.append(cumulative_pos[-1] - cumulative_neg[-1])
cumulative_pos = cumulative_pos[1:]
cumulative_neg = cumulative_neg[1:]

plt.figure()
plt.plot(sentiment_pos, label = 'positive sentiment')
plt.plot(sentiment_neg, label = 'negative sentiment')
plt.plot(np.multiply(close_prices, 100), label = 'scaled prices')
plt.legend(loc = 'best')
plt.title('nltk sentiment indicator. 300 tweets per day')
plt.xlabel('time in samples')
plt.ylabel('amplitude')
plt.show()