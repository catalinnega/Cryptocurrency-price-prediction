#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:07:19 2019

@author: catalin
"""



import matplotlib.pyplot as plt
import mylib_dataset as md

path = '/home/catalin/git_workspace/disertatie/dict_perf_feats.pkl'
ordered_values_mean, ordered_values_var = md.get_feature_importances_mean(path)
### plot historgram
barWidth = 0.3
plt.figure(figsize=(20, 1))
plt.title('Mean feature importances')
plt.bar(ordered_values_mean['keys'], ordered_values_mean['values'], align='edge', width=barWidth, label = 'mean')
plt.bar(ordered_values_var['keys'], ordered_values_var['values'], align='edge', width=barWidth, label = 'variance')
plt.xticks([r + barWidth for r in range(len(ordered_values_var['keys']))], ordered_values_var['keys'])
plt.xticks(rotation=90)
plt.ylabel('Feature importances')
plt.legend(loc = 'best')