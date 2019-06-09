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
                                         'periods': 11,
                                         'threshold_flags': {
                                                             'skip': True,
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
                                         'skip': False,
                                         'periods': (
                                                        61
                                                        ),
                                         'threshold_flags': {
                                                             'skip': True,
                                                             'lower_thresh': 20,
                                                             'upper_thresh': 80,
                                                             'lam': False
                                                             }                                         
                                        },
                                        
                                 'ATR': {
                                         'skip': False,
                                         'periods': (
                                                         3
                                                    ),
                                         'method': {
                                                    'standard' : False,
                                                    'Wilder': True
                                                   }
                                        },
                                        
                                 'IKH': {
                                         'skip': False,
                                         'time_span_Tenkan': (
                                                             9 , 
#                                                             9 * 4 * 24 * 2
                                                             ),
                                         'time_span_Kijun': (
                                                             26 ,
#                                                             26 * 4 * 24 * 2
                                                             ),
                                         'time_span_Span_B': (
                                                              54 ,
#                                                              5 * 4 * 24 * 2
                                                              ),
                                         'displacement_Chikou': (
                                                                 26,
#                                                                 26 * 4 * 24 * 2
                                                                 ),
                                         'displacement':  (
                                                           26,
#                                                           26 * 4 * 24 * 2
                                                           ),
                                                 
                                         'flag_decay_span': None,
                                         
                                         'specify_features':
                                             {
                                                     'skip': True,
                                                     'features':
                                                      {
                                                              1 : {
                                                                   'key': 'Tenkan_sen',
                                                                   'value': 9 * 4
                                                                   }
                                                      }
                                             }
                                        },
                                         
                                 'ADL': {
                                         'skip': False
                                        },
                                         
                                 'SMA_EMA': {
                                             'skip': False,
                                             'periods': (
                                                         29
                                                         )
                                            },
                                 
                                 'MACD': {
                                         'skip': True,
                                         'period_pair': (
                                                         [1,2,9]
#                                                         (4,12),
#                                                         (12,26),
#                                                         (42,52),
#                                                         (96,208)
                                                         ),
                                         },
                                         
                                 'BB': {
                                        'skip': False,
                                        'periods': (
                                                    4
#                                                    8, 
#                                                    14, 
#                                                    14*2,
#                                                    14*4,
#                                                    14*4*12
                                                    ),
                                         'specify_features':
                                                    {
                                                     'skip': True,
                                                     'features':
                                                             {
                                                              1 : {
                                                                   'key': 'BB_SMA',
                                                                   'value': 4
                                                                   },
                                                              2 : {
                                                                   'key': 'BB_upperline',
                                                                   'value': 4
                                                                   },
                                                              3 : {
                                                                   'key': 'BB_lowerline',
                                                                   'value': 4
                                                                   },
                                                              4 : {
                                                                   'key': 'BB_squeeze',
                                                                   'value': 4
                                                                   },
                                                              5 : {
                                                                   'key': 'BB_SMA_crossings',
                                                                   'value': 4
                                                                   }
                                                             }
                                                    }
                                       },
                                 
                                 'CCI':{
                                        'skip': False,
                                        'periods': (
                                                    1
                                                    ),
                                        'constant': 0.015
                                       },
                                         
                                 'VBP':{
                                         'skip': False,
                                         'bins': (
                                                 8,
                                                 ),
                                         'smoothing': 0
                                       },
                                
                                'CMF':{
                                        'skip': False,
                                        'periods': (
                                                     52
                                                   )
                                        
                                      },
                                         
                                 'NLMS': {
                                         'skip': False,
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
#                                                    (17, 0.8, 0.9, 0),
 #                                                   (16, 0.8, 0.9, 0), ##classifier
                                                    [41, 0.86, 0.15, 0], ##regressor
                                                    
#                                                    (32, 0.8, 0.8),
#                                                    (12, 0.3, 0.9),
#                                                    (12, 0.7, 0.9),
 #                                                   (12, 0.8, 0.9),
#                                                    (12, 0.9, 0.9),
#                                                    (12, 0.2, 0.9)
                                                    ),
                                         'smoothing': None
                                         },    
                                         
                                'FRLS': {
                                         'skip': False,
                                         'periods': (
                                                    55
                                                    

                                                    ),
                                         'forward window': 0,
                                         'smoothing': None
                                         },     
                                     
                                 'ADX': {
                                        'skip': False,
                                        'periods': (
                                                    16
                                                    )
                                        }, 
                                 
                                 'OBV': {
                                         'skip': False 
                                        },

                                 'STO': {
                                         'skip': False,
                                         'periods': (
                                                     3#6
                                                   )
                                        },
                                 
                                 'TRIX': {
                                         'skip': False,
                                         'periods': (
                                                     1
                                                   ) 
                                         },
                                 
                                 'MD': {
                                         'skip': False,
                                         },   
                                   
                                 'nsr': {
                                        'skip': False,
                                        'periods': 44
                                        },
                                         
                                 'EMA': {
                                         'skip': False,
                                         'periods': (
                                                 16
                                                 )
                                         },
                                 
                                 'volume_EMA': {
                                         'skip': False,
                                         'periods': (
                                                    85
                                                    )
                                         },
    
                                 'previous_mean': {
                                                  'skip': True,
                                                  'window': (1,2,3,4,5,12,24)
                                                  },
                                         
                                 'previous_var': {
                                                 'skip': True,
                                                 'window': (1,2,3,4,12,24)
                                                 },
                                 
                                 'SENT': {
                                                 'skip': True
                                                 },
                                 'peaks': {
                                                 'skip': True
                                                 },
                                 'ohcl_diff': {
                                             'skip': True,
                                             'specify_features':
                                                  {
                                                     'skip': True,
                                                     'features':       
                                                         {
                                                                 'c_diff',
                                                                 'c_o',
                                                                 'h_o',
                                                                 'l_o',
                                                                 'h_l'
                                                         }
                                                  }
                                              }
                                 }
    return feature_names  

def use_only_ohclv(feature_dicts):
    k = list(feature_dicts.keys())
    for i in k:
       if('skip' in feature_dicts[i]):
           if(feature_dicts[i]['skip'] == False):
               feature_dicts[i]['skip'] = True
    return feature_dicts
