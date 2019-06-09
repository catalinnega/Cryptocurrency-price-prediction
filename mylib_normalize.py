#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:17:53 2018

@author: catalin
"""
import numpy as np

def normalize_rescale(data):
    min_data = np.min(abs(data))
    max_data = np.max(abs(data))
    range_data = max_data - min_data
    data = [ ((i - min_data) / range_data) for i in data ]
    return np.array(data), min_data, max_data


def normalize_mean(data):
    min_data = np.min(abs(data))
    max_data = np.max(abs(data))
    average_data = np.average(abs(data))
    range_data = max_data - min_data
    data = [ ((abs(i) - average_data) / range_data) for i in data ]
    return np.array(data), min_data, max_data, average_data
 
def normalize_norm(data):
    norm_data = np.linalg.norm(data)
    data = [( i / norm_data) for i in data]
    return np.array(data), norm_data

def normalize_standardization(data):
    new_data = []
    try:
        if(np.shape(data)[1]):
            print('matrix standardization')
            for j in range(np.shape(data)[1]-1):
                mean = np.mean(data[:, j])
                for i in range(np.shape(data)[0]):
#                    mean = np.mean(data[:i, j])
                    #std = np.sqrt((1/(i + 1)) * np.sum([(data[k, j] - mean) ** 2 for k in range(i + 1)] ))
                    std = np.std(data[:i, j], ddof = 1)
                    new_data[i, j] = (data[i, j] - mean) / std
            return np.array(new_data)
    except:
        print('vector standardization')
        for i in range(len(data)):
            mean = np.mean(data[:i])
#            std = np.sqrt((1/(i + 1)) * np.sum([(data[j] - mean) ** 2 for j in range(i + 1)] ))
            std = np.std(data[:i], ddof = 1)
            new_data.append((data[i] - mean) / std)
        return np.array(new_data)

def denormalize_rescale(data, min_data, max_data):
    range_data = max_data - min_data
    data = [ i * range_data + min_data for i in data ]
    return np.array(data)

def denormalize_mean(data, min_data, max_data, average_data):
    range_data = max_data - min_data
    data = [ i * range_data + average_data for i in data ]
    return np.array(data)

def denormalize_norm(data, norm_data):
    data = [ i * norm_data for i in data ]
    return np.array(data)