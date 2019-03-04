
import numpy as np
import matplotlib.pyplot as plt

import mylib_dataset as md
import mylib_normalize as mn

#from matplotlib.finance import candlestick
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import talib as ta
        

directory = '/home/catalin/databases/klines_2014-2018_15min/'
hard_coded_file_number = 0

data = md.get_dataset_with_descriptors(concatenate_datasets_preproc_flag = True, 
                                       preproc_constant = 0.99, 
                                       normalization_method = "rescale",
                                       dataset_directory = directory,
                                       hard_coded_file_number = hard_coded_file_number)
X = data['preprocessed_data'] ## this will be used for training
X_unprocessed = data['data']

open_prices = X_unprocessed[:,0]
close_prices = X_unprocessed[:,1]
high_prices = X_unprocessed[:,2]
low_prices = X_unprocessed[:,3]
volume = X_unprocessed[:,4]

time_period = 20
constant = 0.015
def commodity_channel_index(close_prices, high_prices, low_prices, time_period):
    typical_price_circular_buffer = np.zeros(time_period)
    CCI_values = []
    CCI_threshold_values = []
    for i in range(len(close_prices)):
        typical_price = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
        typical_price_circular_buffer = np.hstack((typical_price, typical_price_circular_buffer[:-1]))
        TP_SMA = np.sum(typical_price_circular_buffer) / time_period
        mean_deviation = np.sum(abs(typical_price_circular_buffer - TP_SMA)) / time_period
        CCI_values.append((typical_price - TP_SMA) / (constant * mean_deviation))
        if(CCI_values[-1] > 190):
            CCI_threshold_values.append(1)
        elif(CCI_values[-1] < -190):
            CCI_threshold_values.append(-1)
        else:
             CCI_threshold_values.append(0)
    return CCI_values, CCI_threshold_values

CCI_values, CCI_threshold_values = commodity_channel_index(close_prices, high_prices, low_prices, time_period)

plt.plot(CCI_threshold_values)
    
    
    