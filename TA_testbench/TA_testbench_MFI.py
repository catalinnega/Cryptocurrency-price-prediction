
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

open_prices = X_unprocessed[:,3]
close_prices = X_unprocessed[:,0]
high_prices = X_unprocessed[:,1]
low_prices = X_unprocessed[:,2]
volume = X_unprocessed[:,4]

def money_flow_index(close_prices, high_prices, low_prices, volume, time_frame):
    diff_circular_buffer = np.zeros(time_frame)
    MFI_values = []
    raw_money_flow = 0
    for i in range(len(close_prices)):
        typical_price = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
        previous_money_flow = raw_money_flow
        raw_money_flow = typical_price * volume[i]
        
        diff = raw_money_flow - previous_money_flow
        diff_circular_buffer = np.hstack((diff, diff_circular_buffer[:-1]))
        positive_money_flow = np.sum([i if (i > 0) else 0 for i in diff_circular_buffer])
        negative_money_flow = np.sum([-i if (i < 0) else 0 for i in diff_circular_buffer])
                
        money_flow_ratio = positive_money_flow / negative_money_flow
        MFI_values.append(100 - (100/(1 + money_flow_ratio)))
    return np.array(MFI_values)

def money_flow_divergence(close_prices, MFI_values, divergence_window):
    MFI_lowest_low = 99999999999
    close_lowest_low = 99999999999
    close_highest_high = -9999999
    MFI_highest_high = -9999999
    bullish_div = []
    bearish_div = []
    RESET_TIMER = 100
    timer_bear = RESET_TIMER
    timer_bull = RESET_TIMER
    for i in range(len(MFI_values)):
        ## bullish div
        if(i <= divergence_window):
            bearish_div.append(0)
            bullish_div.append(0)
        else:
            close_low = np.min(close_prices[(i - divergence_window + 1): (i + 1)])
            MFI_low = np.min(MFI_values[(i - divergence_window + 1): (i + 1)])
            
            if(MFI_low < MFI_lowest_low):
                MFI_lowest_low = MFI_low
            
            if(close_low < close_lowest_low):
                close_lowest_low = close_low
                timer_bear = RESET_TIMER
                if(MFI_low > MFI_lowest_low):
                    bullish_div.append(1)
                else:
                    bullish_div.append(0)
            else:
                bullish_div.append(0)
                
            ## bearish div
            close_high = np.max(close_prices[(i - divergence_window + 1): (i + 1)])
            MFI_high = np.max(MFI_values[(i - divergence_window + 1): (i + 1)])
    
            if(MFI_high > MFI_highest_high):
                MFI_highest_high = MFI_high
                
            if(close_high > close_highest_high):
                close_highest_high = close_high
                timer_bull = RESET_TIMER
                if(MFI_high < MFI_highest_high):
                    bearish_div.append(-1)
                else:
                    bearish_div.append(0)
            else:
                bearish_div.append(0)
            
            if(timer_bear <= 0):
                MFI_lowest_low = MFI_low
                close_lowest_low = close_low
            if(timer_bull <= 0):
                MFI_highest_high = MFI_high
                close_highest_high = close_high
            timer_bear -= 1      
            timer_bull -= 1       
    divergence_values = np.add(bullish_div, bearish_div)    
    return divergence_values


ceva = money_flow_index(close_prices, high_prices, low_prices, volume, time_frame = 14)
ceva1 = money_flow_divergence(close_prices, ceva, divergence_window = 100)
plt.plot(ceva1)