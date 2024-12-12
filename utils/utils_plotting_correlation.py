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

def fill_empty_grid(df, n_bins, fill_with):
    
    all_grid_bins     = list(range(n_bins)) # list of grid_bin values from 0 to n_bins
    missing_grid_bins = set(all_grid_bins) - set(df.index) # get missing grid_bin values

    # create new rows for the missing grid_bin values (fill with NaN)
    empty_rows        = pd.DataFrame(fill_with, columns=df.columns, index=list(missing_grid_bins)) 

    return pd.concat([df, empty_rows])

def plot_grid_wise_correlation(df_correlation, features_order, n_bins, axis):
    
    df_corr   = df_correlation.pivot(index="grid_bin", columns="feature_1", values="correlation")
    df_corr   = df_corr[features_order]
    df_corr   = fill_empty_grid(df_corr, n_bins=n_bins, fill_with=np.nan)
    df_corr   = df_corr.sort_index(ascending=False)
    
    df_pvalue = df_correlation.pivot(index="grid_bin", columns="feature_1", values="significant")
    df_pvalue = df_pvalue[features_order]
    df_pvalue = fill_empty_grid(df_pvalue, n_bins=n_bins, fill_with=False)
    df_pvalue = df_pvalue.sort_index(ascending=False)
    

    
    # Plot heatmap on the provided axis
    axis  = sns.heatmap(df_corr, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Coefficient'},
                      annot_kws={"fontsize":5},linewidths=0, linecolor='black', mask=~df_pvalue,
                      vmin=-0.5, vmax=0.5, ax=axis)

    axis.spines.left.set_visible(False)
    axis.spines.bottom.set_visible(False)
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    axis.xaxis.set_ticks_position('none')  # Hide the x-axis ticks
    axis.xaxis.set_ticks_position('none')  # Hide the x-axis ticks

    axis.set_xlabel("")
    axis.set_xticklabels(axis.get_xticklabels(), fontsize=utils_plotting.LABEL_SIZE)
    axis.set_yticklabels(axis.get_yticklabels(), fontsize=utils_plotting.LABEL_SIZE)

    return axis
