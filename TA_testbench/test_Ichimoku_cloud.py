#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:48:56 2019

@author: catalin
"""

import pickle
import numpy as np
import mylib_TA as mta
import matplotlib.pyplot as plt
from copy import copy

def apply_linear_decay_flags(input_data, span = 20):
    data = copy(input_data)
    ### flags need to have 1 or 0 values only.
    for i in range(len(data)):
        if(data[i] == 1):
            for j in range(span):
                if((i + j) < len(data)):
                    data[i + j] = 1 - j / span
                if((i + j + 1) < len(data)):
                    if(data[i + j + 1] == 1):
                        i += j
                        break
    return data
    


def Ichimoku_Cloud_Indicator(dataset_dict, 
                             time_span_Tenkan = 9, 
                             time_span_Kijun = 26, 
                             displacement = 26, 
                             time_span_Span_B = 52, 
                             displacement_Chikou = 26,
                             flag_decay_span = 20,
                             feature_names = ['Tenkan_sen']):
    
    user_input_tags = ['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B',
                       'Chikou_Span','cloud_flags', 'close_over_cloud', 'close_under_cloud',
                       'cross_over_Kijun', 'cross_under_Kijun', 'cross_over_Tenkan',
                       'cross_under_Tenkan']

    ok = False
    for i in feature_names:
        if i in user_input_tags:
            ok = True
            break
    if(not ok):
        null_return = np.zeros(1000)
        return null_return,null_return,null_return,null_return,null_return,null_return,null_return,null_return
    
    displacement_Span_A = displacement_Span_B = displacement
    
    #time_span_Tenkan = 9 ## conversion line perios
    highest_high_days = [max(dataset_dict['high'][(i - time_span_Tenkan) : i]) for i in range (len(dataset_dict['high'])) if i >= time_span_Tenkan]
    highest_high_days = np.pad(highest_high_days, ( (len(dataset_dict['high'])  - len(highest_high_days)),0 ), 'constant')
    
    lowest_low_days = [min(dataset_dict['low'][(i - time_span_Tenkan) : i]) for i in range (len(dataset_dict['low'])) if i >= time_span_Tenkan]
    lowest_low_days = np.pad(lowest_low_days, ( (len(dataset_dict['low'])  - len(lowest_low_days)),0 ), 'constant')
    
    Tenkan_sen = (highest_high_days + lowest_low_days) / 2
    Tenkan_sen = np.array(Tenkan_sen)
    
    
    #time_span_Kijun = 26
    highest_high_days = [max(dataset_dict['high'][(i - time_span_Kijun) : i]) for i in range (len(dataset_dict['high'])) if i >= time_span_Kijun]
    highest_high_days = np.pad(highest_high_days, ( (len(dataset_dict['high'])  - len(highest_high_days)),0 ), 'constant')
    
    lowest_low_days = [min(dataset_dict['low'][(i - time_span_Kijun) : i]) for i in range (len(dataset_dict['low'])) if i >= time_span_Kijun]
    lowest_low_days = np.pad(lowest_low_days, ( (len(dataset_dict['low'])  - len(lowest_low_days)),0 ), 'constant')
    
    Kijun_sen = (highest_high_days + lowest_low_days) / 2
    Kijun_sen = np.array(Kijun_sen)

    #time_span_Span_A = 26
    #Senkou_Span_A = [((Tenkan_sen[i - time_span_Span_A] + Kijun_sen[i - time_span_Span_A]) / 2) for i in range (len(Tenkan_sen)) if i >= time_span_Span_A]
    #Senkou_Span_A = np.pad(Senkou_Span_A, ( (len(Tenkan_sen)  - len(Senkou_Span_A)),0 ), 'constant')
    #Senkou_Span_A = (Tenkan-sen + Kijun-sen) / 2
    
    #displacement_Span_A = 26
    Senkou_Span_A = (Tenkan_sen + Kijun_sen) / 2
    Senkou_Span_A = np.pad(Senkou_Span_A, ( displacement_Span_A, 0 ), 'constant')
    Senkou_Span_A = np.array(Senkou_Span_A[:-displacement_Span_A])
    
    ### displacement = 26 (plotted ahead)
    
    #time_span_Span_B = 52
    #displacement_Span_B = 26
    
    highest_high_days = [max(dataset_dict['high'][(i - time_span_Span_B) : i]) for i in range (len(dataset_dict['high'])) if i >= time_span_Span_B]
    highest_high_days = np.pad(highest_high_days, ( (len(dataset_dict['high'])  - len(highest_high_days)),0 ), 'constant')
    
    lowest_low_days = [min(dataset_dict['low'][(i - time_span_Span_B) : i]) for i in range (len(dataset_dict['low'])) if i >= time_span_Span_B]
    lowest_low_days = np.pad(lowest_low_days, ( (len(dataset_dict['low'])  - len(lowest_low_days)),0 ), 'constant')
    
    
    Senkou_Span_B = (highest_high_days + lowest_low_days) / 2
    Senkou_Span_B = np.pad(Senkou_Span_B, ( displacement_Span_B, 0 ), 'constant')
    Senkou_Span_B = np.array(Senkou_Span_B[:-displacement_Span_B])
    
    #displacement_Chikou = 22
    Chikou_Span = dataset_dict['close'][displacement_Chikou:]
    Chikou_Span = np.pad(Chikou_Span, ( 0, displacement_Chikou ), 'constant') ## plt before
    Chikou_Span = np.array(Chikou_Span)
    
    ######## optional bull/bear flags
    kumo_cloud = Senkou_Span_A - Senkou_Span_B
    cloud_reversal_bull = [1 if (kumo_cloud[i] >= 0 and kumo_cloud[i-1] < 0)  else 0 for i in range(len(kumo_cloud))]
    cloud_reversal_bear = [1 if (kumo_cloud[i] < 0 and kumo_cloud[i-1] > 0)  else 0 for i in range(len(kumo_cloud))] 
    cloud_reversal_bull = np.array(cloud_reversal_bull)
    cloud_reversal_bear = np.array(cloud_reversal_bear)
    
    close_over_cloud = [1 if ((dataset_dict['close'][i] > Senkou_Span_A[i] and kumo_cloud[i] >= 0) \
                        or (dataset_dict['close'][i] > Senkou_Span_B[i] and kumo_cloud[i] < 0)) \
                        else 0 for i in range(len(dataset_dict['close']))]
    close_under_cloud = [1 if( (dataset_dict['close'][i] < Senkou_Span_A[i] and kumo_cloud[i] < 0) \
                         or (dataset_dict['close'][i] < Senkou_Span_B[i] and kumo_cloud[i] >= 0)) \
                         else 0 for i in range(len(dataset_dict['close']))]
        
    cross_over_Kijun = [1 if (dataset_dict['close'][i] > Kijun_sen[i] and dataset_dict['close'][i-1] < Kijun_sen[i-1]) else 0 for i in range(len(dataset_dict['close']))]
    cross_under_Kijun = [1 if (dataset_dict['close'][i] < Kijun_sen[i] and dataset_dict['close'][i-1] > Kijun_sen[i-1]) else 0 for i in range(len(dataset_dict['close']))]
   
    cross_over_Tenkan = [1 if (Tenkan_sen[i] >= Kijun_sen[i] and Tenkan_sen[i-1] < Kijun_sen[i-1]) else 0 for i in range(len(dataset_dict['close']))]
    cross_under_Tenkan = [1 if (Tenkan_sen[i] < Kijun_sen[i] and Tenkan_sen[i-1] >= Kijun_sen[i-1]) else 0 for i in range(len(dataset_dict['close']))]
       ### flat_kumo_flags ## in progress
    
    if(flag_decay_span):
        close_over_cloud = apply_linear_decay_flags(close_over_cloud, span = 20)
        close_under_cloud = apply_linear_decay_flags(close_under_cloud, span = 20)
        cross_over_Kijun = apply_linear_decay_flags(cross_over_Kijun, span = 20)
        cross_under_Kijun = apply_linear_decay_flags(cross_under_Kijun, span = 20)
        cross_over_Tenkan = apply_linear_decay_flags(cross_over_Tenkan, span = 20)
        cross_under_Tenkan = apply_linear_decay_flags(cross_under_Tenkan, span = 20)

    dict_results = { 'Tenkan_sen': Tenkan_sen,
                     'Kijun_sen': Kijun_sen,
                     'Senkou_Span_A': Senkou_Span_A,
                     'Senkou_Span_B': Senkou_Span_B,
                     'Chikou_Span': Chikou_Span,
                     'close_over_cloud': close_over_cloud,
                     'close_under_cloud': close_under_cloud,
                     'cross_over_Kijun': cross_over_Kijun,
                     'cross_under_Kijun': cross_under_Kijun,
                     'cross_over_Tenkan': cross_over_Tenkan,
                     'cross_under_Tenkan': cross_under_Tenkan
                    }                      
    return dict_results


with open('/home/catalin/git_workspace/disertatie/unprocessed_btc_data.pkl', 'rb') as f:
    data = pickle.load(f)
    

data = data['dataset_dict']
dict_ichimoku    = Ichimoku_Cloud_Indicator(data, 
                                               time_span_Tenkan = 9, 
                                               time_span_Kijun = 26, 
                                               displacement =  26, 
                                               time_span_Span_B = 52, 
                                               displacement_Chikou = 22)          

Senkou_Span_A = dict_ichimoku['Senkou_Span_A']
Senkou_Span_B = dict_ichimoku['Senkou_Span_B']
Tenkan_sen = dict_ichimoku['Tenkan_sen']
Kijun_sen = dict_ichimoku['Kijun_sen']
Chikou_Span = dict_ichimoku['Chikou_Span']
close_over_cloud = dict_ichimoku['close_over_cloud']
close_under_cloud = dict_ichimoku['close_over_cloud']
cross_over_Kijun = dict_ichimoku['close_over_cloud']
cross_under_Kijun = dict_ichimoku['close_over_cloud']
cross_over_Tenkan = dict_ichimoku['close_over_cloud']
cross_under_Tenkan = dict_ichimoku['close_over_cloud']


plt.figure()
plt.title('Ichimoku cloud')
plt.plot(Senkou_Span_A, color = 'green')
plt.plot(Senkou_Span_B, color = 'red')
plt.fill_between(np.arange(len(Senkou_Span_A)),
                 Senkou_Span_A, 
                 Senkou_Span_B, 
                 where=Senkou_Span_A >= Senkou_Span_B,
                 facecolor='green',
                 interpolate=True,
                 label = 'Bullish cloud')
plt.fill_between(np.arange(len(Senkou_Span_A)),
                 Senkou_Span_A, 
                 Senkou_Span_B, 
                 where=Senkou_Span_A < Senkou_Span_B,
                 facecolor='red',
                 interpolate=True,
                 label = 'Bearish cloud')
plt.legend(loc = 'best')
plt.show()

plot_close_over_cloud = True
if(plot_close_over_cloud):
    interval_start = 126000
    interval_length = 600
    close = data['close'][interval_start: interval_start+interval_length]
    A = Senkou_Span_A[interval_start: interval_start+interval_length]
    B = Senkou_Span_B[interval_start: interval_start+interval_length]
    flags = close_over_cloud[interval_start: interval_start+interval_length]
    
    plt.figure()
    plt.title('Close over cloud flags')
    plt.plot(np.multiply(flags, 10000), label = 'close over cloud flags')
    plt.plot(close, color = 'black', label = 'close values')
    plt.plot(A, color = 'green')
    plt.plot(B, color = 'red')
    plt.fill_between(np.arange(len(A)),
                     A, 
                     B, 
                     where=A >= B,
                     facecolor='green',
                     interpolate=True,
                     label = 'Bullish cloud')
    plt.fill_between(np.arange(len(A)),
                     A, 
                     B, 
                     where=A < B,
                     facecolor='red',
                     interpolate=True,
                     label = 'Bearish cloud')
    plt.xlabel('15 min time intervals')
    plt.ylabel('Bitcoin price in dollars')
    plt.legend(loc = 'best')
    plt.show()
    

plot_cross_over_Kijun = True
if(plot_cross_over_Kijun):
    interval_start = 126000
    interval_length = 600
    close = data['close'][interval_start: interval_start+interval_length]
    flags_over = cross_over_Kijun[interval_start: interval_start+interval_length]
    flags_under = cross_under_Kijun[interval_start: interval_start+interval_length]
    Kijun = Kijun_sen[interval_start: interval_start+interval_length]
    
    plt.figure()
    plt.title('Cross over Kijun-sen flags')
    plt.plot(np.multiply(flags_over, 10100), color = 'green', label = 'cross over Kijun-sen')
    plt.plot(np.multiply(flags_under, 10000), color = 'red', label = 'cross under Kijun-sen')
    plt.plot(close, color = 'black', label = 'close values')
    plt.plot(Kijun, color = 'orange', label  = 'Kijun sen(Base line)')
    plt.xlabel('15 min time intervals')
    plt.ylabel('Bitcoin price in dollars')
    plt.legend(loc = 'best')
    plt.show()
    
plot_cross_over_Tenkan = True
if(plot_cross_over_Kijun):
    interval_start = 126000
    interval_length = 600
    close = data['close'][interval_start: interval_start+interval_length]
    flags_over = cross_over_Tenkan[interval_start: interval_start+interval_length]
    flags_under = cross_under_Tenkan[interval_start: interval_start+interval_length]
    Kijun = Kijun_sen[interval_start: interval_start+interval_length]
    Tenkan = Tenkan_sen[interval_start: interval_start+interval_length]
    
    plt.figure()
    plt.title('Tenkan sen/Kijun sen cross flags')
    plt.plot(np.multiply(flags_over, 10100), color = 'green', label = 'Kijun-sen cross over Tenkan-sen')
    plt.plot(np.multiply(flags_under, 10000), color = 'red',label = 'Kijun-sen cross under Tenkan-sen')
    plt.plot(close, color = 'black', label = 'close values')
    plt.plot(Kijun, color = 'orange', label  = 'Kijun sen(Base line)')
    plt.plot(Tenkan, color = 'purple', label  = 'Tenkan sen(Base line)')
    plt.xlabel('15 min time intervals')
    plt.ylabel('Bitcoin price in dollars')
    plt.legend(loc = 'best')
    plt.show()

a = apply_linear_decay_flags(flags_over, 50)
# should add decay(fade-out)
