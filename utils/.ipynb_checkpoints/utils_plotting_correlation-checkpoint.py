"""
Utilization function for correlation plotting
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_plotting

def plot_correlation_line(dataset, group_variable, group_value, group_color, feat_x, feat_y, scatter_size, ax):
    
    ax = sns.regplot(data=dataset[dataset[group_variable]==group_value], 
                     x=feat_x, y=feat_y, 
                     scatter=False, color=group_color,
                     ax=ax)
    
    ax = sns.scatterplot(data=dataset[dataset[group_variable]==group_value], alpha=0.25, 
                         x=feat_x, y=feat_y, color=group_color, s=scatter_size, ax=ax)
    
    ax.tick_params(axis='x', labelsize=utils_plotting.LABEL_SIZE_label)
    ax.tick_params(axis='y', labelsize=utils_plotting.LABEL_SIZE_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    return ax
