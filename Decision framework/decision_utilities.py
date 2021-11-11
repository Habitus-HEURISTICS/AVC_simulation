#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:11:44 2021

@author: prcohen
"""


import copy

import numpy as np
import numpy.ma as ma
from numpy.random import default_rng
rng = default_rng(1108) # stuff for random sampling; fix random seed
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import ternary


#______________________ Useful Vectorized Functions _________________________

def ps (x):
    ''' print a table of reals to p places precision '''
    for i in range(len(x)):
        print(*[f"{j:.3f}\t" for j in x[i,:]])
    print()
        
def row_sums (a):
    return np.sum(a,axis = 1)

def col_sums (a):
    return np.sum(a,axis = 1)

def row_margins (a):
    return np.sum(a,axis=1)/a.shape[1]

def col_margins (a):
    return np.sum(a,axis=0)/a.shape[0]

def index2onehot (index, shape):
    ''' index is an np.array of length r that holds indices between 0 and c - 1.  
    This returns an array of shape r,c that contains a one-hot encoding of the 
    column indicated by index; e.g., for c = 3 and index = np.array([0,2,1]), 
    index2onehot(index) -> [[1. 0. 0.],[0. 0. 1.],[0. 1. 0.]] '''
    zeros = np.zeros(shape)
    zeros[np.indices(index.shape)[0], index]=1
    return zeros

def onehot2index (x):
    return np.argmax(x,axis=1)

def row_min_onehot (scores):
    ''' One-hot encoding of the column that holds the minimum score.  If
    two columns hold the same minimum, this takes the first.'''
    return index2onehot(np.argmin(scores,axis=1),scores.shape)

def row_max_onehot (scores):
    ''' One-hot encoding of the column that holds the maximum score.  If
    two columns hold the same maximum, this takes the first.'''
    return index2onehot(np.argmax(scores,axis=1),scores.shape)

def row_sample (probs):
    ''' probs is a 2D array in which each row is a multinomial distribution. This
    returns a one-hot encoding of the colum selected by sampling from each row. 
    
    For machine learning purposes, this choice must run fast.  Parts of the solution are 
    https://bit.ly/3AXSWJV, https://bit.ly/3peWVzv, https://bit.ly/3G3aSq3 ''' 
    
    chosen_col = (probs.cumsum(1) > rng.random(probs.shape[0])[:,None]).argmax(1)
    return index2onehot(chosen_col,probs.shape)


def ged (p0, p1):
    ''' Generalized euclidean distance. p0 and p1 must both have the same number 
    of columns, c, and p0 must be broadcastable to p1.  Each row is treated as a 
    point in c-dimensional space. This returns an array of distances of shape r,c, 
    where r is the number of rows in p1. It uses numpy linalg.norm, which allows for 
    different distance measures than euclidean distance but one could also use 
    the more familiar np.sum((p0-p1)**2,axis=1)**.5 . '''
    
    return np.linalg.norm(a0-a1, axis = 1)


def plot_ternary (vertex_labels, points, special_points = None, color_by=None, 
                  color_by_label=None, bounds = None, figsize = (4,4)):
    mpl.rcParams['figure.dpi'] = 200
    mpl.rcParams['figure.figsize'] = figsize
    
    ### Scatter Plot
    scale = 1.0
    fontsize = 8
    offset = 0.03
    figure, tax = ternary.figure(scale=scale)
    #tax.set_title("Decision Space", fontsize=12)
    tax.boundary(linewidth= .5)
    
    tax.left_corner_label(vertex_labels[0], fontsize=12)
    tax.top_corner_label(vertex_labels[1], fontsize=12)
    tax.right_corner_label(vertex_labels[2], fontsize=12)
    tax.get_axes().axis('off')
    
    tax.gridlines(multiple=0.2, color="black")
    tax.clear_matplotlib_ticks()
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, fontsize=fontsize, offset=offset, tick_formats="%.1f")
    
    
    if color_by is not None:
        cmap = plt.cm.RdYlGn
        if bounds is not None:
            norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
        else:
            norm = mpl.colors.Normalize(vmin=np.min(color_by), vmax=np.max(color_by))
        color = cmap(norm(list(color_by)))
        
        figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            orientation='horizontal', label=color_by_label)
        tax.scatter(points, marker='o', color=color, s = 3, cmap=cmap)
    else:
        tax.scatter(points, marker='o', color='black', s = 3)
    
    if special_points is not None:
        for p,c in zip(special_points[0],special_points[1]):       
            tax.scatter([p], marker='s', color = [c], s = 10)
        
    tax.gridlines(multiple=5, color="blue")
    #tax.legend(loc = 'upper right',cmap=cmap)
    #tax.ticks(axis='lbr', linewidth=1, multiple=5)
    
    
    tax.show()
    ternary.plt.show()
    
