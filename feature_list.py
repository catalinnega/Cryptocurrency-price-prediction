#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:22:39 2019

@author: catalin
"""

def get_features_dicts():
    feature_names ={
                                 'ochlv': {
                                          'open' : True,
                                          'close': True,
                                          'high': True,
                                          'low': True,
                                          'volume': True
                                          },
                                         
                                 'RSI': {
                                         'skip': False,
                                         'days': (1,
                                                  2,
                                                  3),
                                         'threshold_flags': {
                                                             'lower_thresh': 20,
                                                             'upper_thresh': 80,
                                                             'lam': False
                                                             },
                                         'stoch_RSI': {
                                                       'skip': True,
                                                       'period': 4
                                                      }
                                        },
                                         
                                 'MFI': {
                                         'skip': True,
                                         'timeframes': (2,
                                                        4),
                                         'threshold_flags': {
                                                             'lower_thresh': 20,
                                                             'upper_thresh': 80,
                                                             'lam': False
                                                             }                                         
                                        },
                                        
                                 'ATR': {
                                         'skip': True,
                                         'timeframes': (2,
                                                        4 * 24,
#                                                        4 * 24 * 2,
#                                                        4 * 24 * 7,
                                                        4 * 24 * 30 ),
                                        },
                                        
                                 'IKH': {
                                         'skip': True,
                                         'time_span_Tenkan': (9 * 4 * 24, 
                                                              9 * 4 * 24 * 2),
                                         'time_span_Kijun': (26 * 4 * 24,
                                                             26 * 4 * 24 * 2),
                                         'time_span_Span_B': (54 * 4 * 24,
                                                              54 * 4 * 24 * 2),
                                         'displacement_Chikou': (26 * 4 * 24,
                                                                 26 * 4 * 24 * 2),
                                         'displacement':  (26 * 4 * 24,
                                                           26 * 4 * 24 * 2),
                                         'flag_decay_span': None
                                        },
                                         
                                 'ADL': {
                                         'skip': True
                                        },
                                         
                                 'SMA_EMA': {
                                             'skip': True,
                                             'periods': (2,
                                                         4,
                                                         6,
#                                                         12,
#                                                         24,
#                                                         48,
#                                                         128,
#                                                         256,
                                                         512)
                                            },
                                 
                                 'MACD': {
                                         'skip': True,
                                         'period_pair': ((2,6),
                                                         (4,12),
                                                         (12,26),
                                                         (42,52),
                                                         (48,104)
                                                         ),
                                         },
                                         
                                 'BB': {
                                        'skip': False,
                                        'periods': (4,
                                                    8, 
#                                                    14, 
#                                                    14*2,
#                                                    14*4,
#                                                    14*4*12
                                                    ),
                                       },
                                 
                                 'CCI':{
                                        'skip': True,
                                        'periods': (2,
                                                    4,
#                                                    8, 
#                                                    14, 
#                                                    14*2,
#                                                    14*4,
#                                                    14*4*12
                                                    ),
                                        'constant': 0.015
                                       },
                                         
                                 'VBP':{
                                         'skip': True,
                                         'bins': (
                                                 3,
                                                 5,
                                                 8,
#                                                 12,
#                                                 24,
#                                                 42,
                                                 ),
                                         'smoothing': 0.99
                                       },
                                
                                'CMF':{
                                        'skip': True,
                                        'periods': (
                                                   2,
                                                   3,
                                                   4,
                                                   5,
                                                   7,
                                                   8,
                                                   12,
                                                   21,
                                                   42,
#                                                   64,
#                                                   128,
#                                                   256
                                                   )
                                        
                                      },
                                         
                                 'NLMS': {
                                         'skip': True,
                                         'params': (
#                                                    (128, 0.5, 0.9),
#                                                    (64, 0.5, 0.9),
#                                                    (32, 0.8, 0.9),
#                                                    (24, 0.8, 0.9),
#                                                    (24, 0.8, 0.99),
#                                                    (24, 0.8, 0.85),
#                                                    (20, 0.8, 0.9),
##                                                    (21, 0.8, 0.9),
##                                                    (22, 0.8, 0.9),
#                                                    (19, 0.8, 0.9),
#                                                    (18, 0.8, 0.9),
                                                    (17, 0.8, 0.9),
                                                    (16, 0.8, 0.9),
                                                    
#                                                    (32, 0.8, 0.8),
#                                                    (12, 0.3, 0.9),
#                                                    (12, 0.7, 0.9),
 #                                                   (12, 0.8, 0.9),
#                                                    (12, 0.9, 0.9),
#                                                    (12, 0.2, 0.9)
                                                    ),
                                         'smoothing': None
                                         },         
                                
                                 'nsr': {
                                        'skip': True,
                                        'period': 4*12
                                        },
                                         
                                 'previous_mean': {
                                                  'skip': True,
                                                  'window': (1,2,3,4,5,12,24)
                                                  },
                                         
                                 'previous_var': {
                                                 'skip': True,
                                                 'window': (1,2,3,4,12,24)
                                                 }
                                 }
    print('Feature dict:', '\n\t', feature_names)
    return feature_names  


def get_blockchain_indicators():
    names = ['hashrate',
             'difficulty',
#             'total_btc_sent',
             'estimated_transaction_volume_usd',
#             'miners_revenue_btc'
             ] 
    return names