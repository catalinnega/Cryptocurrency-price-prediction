#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:22:39 2019

@author: catalin
"""
#def get_features_list():
#    feature_names =                  [
#                                 'open',
#                                 'close',
#                                 'high',
#                                 'low',
#                                 'volume',
#                                 'RSI',
##                                 'RSI_threshold_flags',
##                                 'divergence_RSI',
##                                 'stochastic_RSI',
##                                 'stochastic_RSI_threshold_flags',
##                                 'divergence_stochastic_RSI',
#                                 'MFI',
#                                 'MFI_threshold_flags',
##                                 'divergence_MFI',
#                                 'Tenkan_sen', 
#                                 'Kijun_sen', 
#                                 'Senkou_Span_A', 
#                                 'Senkou_Span_B', 
#                                 'Chikou_Span', 
##                                 'cloud_flags', 
#                                 'tk_cross_flags', 
#                                 'oc_cloud_flags', 
#                                 'ADL',
#                                 'SMA_12_day_values',
#                                 'EMA_12_day_values',
#                                 'MACD_line',
#                                 'MACD_signal_line',
#                                 'MACD_histogram',
##                                 'MACD_divergence',
#                                 'MACD_line_15_min',
#                                 'MACD_signal_line_15_min',
#                                 'MACD_histogram_15_min',
#                                 'MACD_divergence_15_min',
#                                 'MACD_line_1h', 
#                                 'MACD_signal_line_1h', 
#                                 'MACD_histogram_1h', 
#                                 'MACD_divergence_1h',
#                                 'talib_MACD1',
#                                 'talib_MACD2',
#                                 'talib_MACD3',
#                                 'BB_SMA_values', 
#                                 'BB_upperline_values', 
#                                 'BB_lowerline_values', 
#                                 'BB_squeeze_values', 
#                                 'BB_SMA_crossings',
#                                 'BB_SMA_values_12h', 
#                                 'BB_upperline_values_12h', 
#                                 'BB_lowerline_values_12h',
#                                 'BB_squeeze_values_12h', 
##                                 'BB_SMA_crossings_12h',
#                                 'BB_SMA_values_1h', 
#                                 'BB_upperline_values_1h', 
#                                 'BB_lowerline_values_1h',
#                                 'BB_squeeze_values_1h', 
##                                 'BB_SMA_crossings_1h',
#                                 'CCI', 
##                                 'CCI_thresholds',
#                                 'CCI_12h', 
#                                 'CCI_thresholds_12h',
#                                 'CCI_1h', 
#                                 'CCI_thresholds_1h',
#                                 'RSI_timeframe_48',
#                                 'RSI_timeframe_48_threshold_flags',
##                                 'RSI_1d',
#                                 'RSI_1d_threshold_flags',
#                                 'RSI_1w',
#                                 'RSI_1w_threshold_flags',
#                                 'RSI_14d',
#                                 'RSI_14d_threshold_flags',
#                                 'RSI_1m',
#                                 'RSI_1m_threshold_flags',
#                                 'volume_by_price',
#                                 'slope_VBP',
#                                 'slope_VBP_smooth',
#                                 'volume_by_price_24',
#                                 'slope_VBP_24',
#                                 'slope_VBP_smooth_24',
#                                 'nlms_indicator',
#                                 'nlms_smoothed_indicator',
##                                 'rls_indicator_error',
##                                 'rls_smoothed_indicator',
#                                 'ATR_EMA',
#                                 'ATR_EMA_Wilder',
#                                 'CMF_12h',
#                                 'CMF_12h_2',
##                                  'sentiment_indicator_positive',
#
#                                 ]
#    return feature_names

def get_feature_set():
    feature_set =[
                                 ['open',
                                 'close',
                                 'high',
                                 'low',
                                 'volume'],
                                 ['RSI',
                                 'RSI_threshold_flags',
                                 'divergence_RSI',
                                 'stochastic_RSI',
                                 'stochastic_RSI_threshold_flags',
                                 'divergence_stochastic_RSI'],
                                 ['MFI',
                                 'MFI_threshold_flags',
                                 'divergence_MFI'],
                                  ['Tenkan_sen', 
                                   'Kijun_sen',
                                   'Senkou_Span_A',
                                   'Senkou_Span_B',
                                   'Chikou_Span',
                                   'cloud_flags',
                                   'close_over_cloud', 
                                   'close_under_cloud',
                                   'cross_over_Kijun', 
                                   'cross_under_Kijun', 
                                   'cross_over_Tenkan',
                                   'cross_under_Tenkan'],
                                 ['ADL'],
                                 ['SMA_12_day_values',
                                 'EMA_12_day_values'],
                                 ['MACD_line',
                                 'MACD_signal_line',
                                 'MACD_histogram',
                                 'MACD_divergence'],
                                 ['MACD_line_15_min',
                                 'MACD_signal_line_15_min',
                                 'MACD_histogram_15_min',
                                 'MACD_divergence_15_min'],
                                 ['MACD_line_1h', 
                                 'MACD_signal_line_1h', 
                                 'MACD_histogram_1h', 
                                 'MACD_divergence_1h'],
                                 ['talib_MACD1',
                                 'talib_MACD2',
                                 'talib_MACD3'],
                                 ['BB_SMA_values', 
                                 'BB_upperline_values', 
                                 'BB_lowerline_values', 
                                 'BB_squeeze_values', 
                                 'BB_SMA_crossings'],
                                 ['BB_SMA_values_12h', 
                                 'BB_upperline_values_12h', 
                                 'BB_lowerline_values_12h',
                                 'BB_squeeze_values_12h', 
                                 'BB_SMA_crossings_12h'],
                                 ['BB_SMA_values_1h', 
                                 'BB_upperline_values_1h', 
                                 'BB_lowerline_values_1h',
                                 'BB_squeeze_values_1h', 
                                 'BB_SMA_crossings_1h'],
                                 ['CCI', 
                                 'CCI_thresholds'],
                                 ['CCI_12h', 
                                 'CCI_thresholds_12h'],
                                 ['CCI_1h', 
                                 'CCI_thresholds_1h'],
                                 ['RSI_timeframe_48',
                                 'RSI_timeframe_48_threshold_flags'],
                                 ['RSI_1d',
                                 'RSI_1d_threshold_flags'],
                                 ['RSI_1w',
                                 'RSI_1w_threshold_flags'],
                                 ['RSI_14d',
                                 'RSI_14d_threshold_flags'],
                                 ['RSI_1m',
                                 'RSI_1m_threshold_flags'],
                                 ['volume_by_price',
                                 'slope_VBP',
                                 'slope_VBP_smooth'],
                                 ['volume_by_price_24',
                                 'slope_VBP_24',
                                 'slope_VBP_smooth_24'],
                                 ['nlms_indicator',
                                 'nlms_smoothed_indicator'],
#                                 'rls_indicator_error',
#                                 'rls_smoothed_indicator',
                                 ['ATR_EMA',
                                 'ATR_EMA_Wilder'],
                                 ['CMF_12h',
                                 'CMF_12h_2']
#                                  'sentiment_indicator_positive',
#                                 'sentiment_indicator_negative',
                ]
    return feature_set
    

# =============================================================================
# 
# import itertools
# import random
# 
# ## get separate combinations for 2, 5, 10 groups of features sets respectively.
# ## random pick 10 times from each 
# ## get values for the 30 total picks
# ceva = list(itertools.combinations(feature_set[1:], 5)) 
# random.choice(ceva)
# =============================================================================

def get_features_list():
    feature_names =                  [
                                 'open',
                                 'close',
                                 'high',
                                 'low',
                                 'volume',
#                                 'RSI',
#                                 'RSI_threshold_flags',
#                                 'divergence_RSI',
#                                 'stochastic_RSI',
#                                 'stochastic_RSI_threshold_flags',
#                                 'divergence_stochastic_RSI',
                                 'SMA_12_day_values',
                                 'EMA_12_day_values',
                                 'MACD_line',
                                 'MACD_signal_line',
                                 'MACD_histogram',
                                 'MFI',
#                                 'MFI_threshold_flags',
#                                 'divergence_MFI',
                                 'RSI_1d',
#                                 'RSI_1d_threshold_flags',
                                 'RSI_1w',
#                                 'RSI_1w_threshold_flags',
                                 'RSI_14d',
#                                 'RSI_14d_threshold_flags',
                                 'RSI_1m',
#                                 'RSI_1m_threshold_flags',
                                 'volume_by_price',
                                 'slope_VBP',
                                 'slope_VBP_smooth',
#                                 'volume_by_price_24',
#                                 'slope_VBP_24',
#                                 'slope_VBP_smooth_24',
                                 'nlms_indicator',
                                 'nlms_smoothed_indicator',
                                 'Tenkan_sen', 
                                   'Kijun_sen',
                                   'Senkou_Span_A',
                                   'Senkou_Span_B',
                                   'Chikou_Span',
                                   'close_over_cloud', 
                                   'close_under_cloud',
                                   'cross_over_Kijun', 
                                   'cross_under_Kijun', 
                                   'cross_over_Tenkan',
                                   'cross_under_Tenkan',
#                                 'RSI_1m_threshold_flags',
                                 'ADL',
#                                 'MACD_line_15_min',
#                                 'MACD_signal_line_15_min',
#                                 'MACD_histogram_15_min',
#                                 'MACD_divergence_15_min',
#                                 'MACD_line_1h', 
#                                 'MACD_signal_line_1h', 
#                                 'MACD_histogram_1h', 
#                                 'MACD_divergence_1h',
#                                 'talib_MACD1',
#                                 'talib_MACD2',
#                                 'talib_MACD3',
#                                 'CCI', 
#                                 'CCI_thresholds',
#                                 'CCI_12h', 
#       #                          'CCI_thresholds_12h',
#                                 'CCI_1h', 
#                                 'CCI_thresholds_1h',
##                                 'RSI_timeframe_48',
##                                 'RSI_timeframe_48_threshold_flags',
#                                 'volume_by_price_24',
#                                 'slope_VBP_24',
#                                 'slope_VBP_smooth_24',
#                                 'rls_indicator_error',
#                                 'rls_smoothed_indicator',
                                 'CMF_12h',
                                 'CMF_12h_2',
#                                  'sentiment_indicator_positive',
#                                 'sentiment_indicator_negative',
                                 #'hash_rate',
                                 #'difficulty'
                                 ]
    return feature_names  


#def get_features_list():
#    feature_names =                  [
#                                 'open',
#                                 'close',
#                                 'high',
#                                 'low',
#                                 'volume',
##                                 'RSI',
##                                 'RSI_threshold_flags',
##                                 'divergence_RSI',
##                                 'stochastic_RSI',
##                                 'stochastic_RSI_threshold_flags',
##                                 'divergence_stochastic_RSI',
#   #                              'MFI',
#        #                         'MFI_threshold_flags',
#     #                            'divergence_MFI',
#                                   'Tenkan_sen', 
#                                   'Kijun_sen',
#                                   'Senkou_Span_A',
#                                   'Senkou_Span_B',
#                                   'Chikou_Span',
#                                   'close_over_cloud', 
#                                   'close_under_cloud',
#                                   'cross_over_Kijun', 
#                                   'cross_under_Kijun', 
#                                   'cross_over_Tenkan',
#                                   'cross_under_Tenkan',
##                                 'ADL',
#                                 'SMA_12_day_values',
#                                 'EMA_12_day_values',
#                                 'MACD_line',
#                                 'MACD_signal_line',
#                                 'MACD_histogram',
##                                 'MACD_divergence',
##                                 'MACD_line_15_min',
##                                 'MACD_signal_line_15_min',
##                                 'MACD_histogram_15_min',
# #                                'MACD_divergence_15_min',
##                                 'MACD_line_1h', 
##                                 'MACD_signal_line_1h', 
##                                 'MACD_histogram_1h', 
##                                 'MACD_divergence_1h',
##                                 'talib_MACD1',
##                                 'talib_MACD2',
##                                 'talib_MACD3',
#                                 'BB_SMA_values', 
#                                 'BB_upperline_values', 
#                                 'BB_lowerline_values', 
#                                 'BB_squeeze_values', 
#                                 'BB_SMA_crossings',
#                                 'BB_SMA_values_12h', 
#                                 'BB_upperline_values_12h', 
#                                 'BB_lowerline_values_12h',
#                                 'BB_squeeze_values_12h', 
#                                 'BB_SMA_crossings_12h',
#                                 'BB_SMA_values_1h', 
#                                 'BB_upperline_values_1h', 
#                                 'BB_lowerline_values_1h',
#                                 'BB_squeeze_values_1h', 
#                                 'BB_SMA_crossings_1h',
##                                 'CCI', 
#     #                            'CCI_thresholds',
#                                 'CCI_12h', 
#       #                          'CCI_thresholds_12h',
#                                 'CCI_1h', 
#         #                        'CCI_thresholds_1h',
###                                 'RSI_timeframe_48',
###                                 'RSI_timeframe_48_threshold_flags',
#                                 'RSI_1d',
##                                 'RSI_1d_threshold_flags',
#                                 'RSI_1w',
##                                 'RSI_1w_threshold_flags',
#                                 'RSI_14d',
#                                 'RSI_14d_threshold_flags',
#                                 'RSI_1m',
##                                 'RSI_1m_threshold_flags',
#                                 'volume_by_price',
#                                 'slope_VBP',
#                                 'slope_VBP_smooth',
##                                 'volume_by_price_24',
##                                 'slope_VBP_24',
##                                 'slope_VBP_smooth_24',
#                                 'nlms_indicator',
#                                 'nlms_smoothed_indicator',
##                                 'rls_indicator_error',
##                                 'rls_smoothed_indicator',
#                                 'ATR_EMA',
#                                 'ATR_EMA_Wilder',
##                                 'CMF_12h',
##                                 'CMF_12h_2',
##                                  'sentiment_indicator_positive',
##                                 'sentiment_indicator_negative',
#                                 #'hash_rate',
#                                 #'difficulty'
#                                 ]
#    return feature_names


def get_blockchain_indicators():
    names = ['hashrate',
             'difficulty',
#             'total_btc_sent',
             'estimated_transaction_volume_usd',
#             'miners_revenue_btc'
             ] 
    return names