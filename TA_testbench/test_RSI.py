#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:55:13 2019

@author: catalin
"""

import numpy as np
import pickle

def RSI_2(close_prices, days = 14, feature_names = ['RSI_2']):
    user_inputs = ['RSI_2', 'RSI_timeframe_48', 'RSI_14d', 'RSI_1d', 'RSI_1w', 'RSI_1m', 'divergence_RSI']
   
    ok = False
    for i in feature_names:
        if i in user_inputs:
            ok = True
            break
    if(not ok):
        return 0
    
    time_window = days * 4 * 24 ## scale from 15 min to 1 day
    k = 2 / (days + 1) ## EMA factor
    RSI = [0]
    EMA_gain = 0
    EMA_loss = 0
    debug_cnt = 0
    for i in range(1,len(close_prices)):
        diff = close_prices[i] - close_prices[i-1]
        gain, loss = 0, 0
        if(diff >= 0):
            gain = diff
        else:
            loss = abs(diff)
        if(i <= time_window):
            EMA_gain += gain / days
            EMA_loss += loss / days
        else:
            EMA_gain = (gain - EMA_gain) * k + EMA_gain
            EMA_loss = (loss - EMA_loss) * k + EMA_loss

        if(EMA_loss == 0):
            RSI.append(100)
            debug_cnt+=1
        else:          
            RS = EMA_gain / EMA_loss
            RSI.append(100 - (100 / (1 + RS)))
    return np.array(RSI), debug_cnt


with open('/home/catalin/git_workspace/disertatie/unprocessed_btc_data.pkl', 'rb') as f:
    data = pickle.load(f)


a = test_train_data['X_test'][:,0]
rsi ,b = RSI_2(a, 1,feature_names = ['RSI_2'])