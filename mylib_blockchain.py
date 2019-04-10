#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:19:18 2019

@author: catalin
"""
import blockchain
import numpy as np

def get_blockchain_historical_data(chart_type, time_span, rolling_average, api_code = None):
    ## https://github.com/blockchain/api-v1-client-python/blob/master/blockchain/statistics.py
    chart_obj = blockchain.statistics.get_chart
    values_obj = chart_obj(chart_type = chart_type, time_span = time_span, rolling_average = rolling_average, api_code = api_code)
    point_type_values = values_obj.values
    utc_time, data = [], []
    for i in range(len(point_type_values)):
        utc_time.append(point_type_values[i].x)
        data.append(point_type_values[i].y)
    return {'utc_time': np.multiply(utc_time, 1000), 'data': data}
        

def rescale_blockchain_indicator(blockchain_indicator, UTC_data):
    ## TO DO: protection if UTC_data starts earlier or ends later than blockchain indicator
    bc_utc = blockchain_indicator['utc_time']
    bc_data = blockchain_indicator['data']
    for i in range(len(bc_utc)):
        if(UTC_data[0] <= bc_utc[i]):
            index_start_utc = i
            break
    
    rescaled_indicator = []
    cnt = 0
    for i in range(len(UTC_data)):
        rescaled_indicator.append(bc_data[index_start_utc + cnt])
        if(bc_utc[index_start_utc + cnt] <= UTC_data[i]):
            cnt += 1
    return np.array(np.reshape(rescaled_indicator, len(rescaled_indicator)))
#
#raw_hash_rate = get_blockchain_historical_data(chart_type = 'hash-rate',
#                                           time_span = 'all',
#                                           rolling_average= None, 
#                                           api_code = '44649562764')
#
#data_UTC = altceva['dataset_dict']['UTC']
#hashrate = rescale_blockchain_indicator(raw_hash_rate, data_UTC)