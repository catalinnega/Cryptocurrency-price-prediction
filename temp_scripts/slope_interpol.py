#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:11:34 2019

@author: catalin
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scp

#y = [ 815.97,  815.95,  815.98,  ...,  815.9] #60 elements
#x = [405383892, 405383894, 405383895, ..., 405383896] #60 elements
x = curve[1]
y = curve[0]
mediana = np.median(x)
f = scp.interp1d(x,y,bounds_error=False,fill_value=0.,kind='cubic')
new_x_array = np.arange(0,1,0.001)
plt.figure()    
plt.plot(new_x_array,f(new_x_array),'ro')
plt.plot([x[0],x[3]], [mediana,mediana], 'g-')
plt.plot(x, y, 'o')

roc = f(new_x_array)
#roc = 1-roc
#roc = 1/roc
res_idx = []
min_diff = 99999999
a = test_train_data['Y_test']

pos = 0
for i in a:
    if i == 'True':
        pos+=1
        
print('test pos: ', pos*100/len(a),'%')
    
slope_rate = len(predictions) / pos - 1

slope = np.zeros(len(new_x_array))
for i in range(len(new_x_array)):
    slope[i] = new_x_array[i] * slope_rate
    
alfa = 0
idx_offset = 0
a = 0
for i in range(len(roc)):
    a = i/len(roc)
    newx = slope + a
    tmp_res_idx = []
    for j in range(len(newx)-1):
#        if newx[j-1] < roc[j] and newx[j] >= roc[j]:
#            tmp_res_idx.append(j)
        if((roc[j] - newx[j])*(roc[j+1] - newx[j+1]) < 0 ):
            tmp_res_idx.append(j)
    res_idx.append(tmp_res_idx)
#    if(tmp_res_idx):
#        print('i ',i,'idx ',tmp_res_idx, 'diff ',tmp_res_idx[1] - tmp_res_idx[0])
    if(len(tmp_res_idx) == 1):
        print('perfect tangent bias: ', i)
    elif(len(tmp_res_idx) == 2):
        diff = tmp_res_idx[1] - tmp_res_idx[0]
        if(diff < min_diff):
            min_diff = diff
            min_offset = a
            idx_offset = i
            alfa = newx
print('offset : ', a, 'idx', idx_offset)
plt.plot(roc)
plt.plot(alfa)