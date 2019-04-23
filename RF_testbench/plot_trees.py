#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:34:49 2019

@author: catalin
"""

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Train
X_train = test_train_eth['X_train'][:50]
Y_train = test_train_eth['Y_train'][:50]
model.fit(X_train, Y_train)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = Y_train,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = "/home/catalin/git_workspace/disertatie/tree.png")