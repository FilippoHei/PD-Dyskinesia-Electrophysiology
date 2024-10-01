"""
Utilisation function for plotting
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')

import utils_plotting

import utils_plotting

# Custom transformation function for x-axis
def custom_transform(x):
    if x < 30:
        return x
    elif x < 60:
        return 30 + (x - 30) / 3  # Compressing 30-60 to a smaller range
    else:
        return 40 + (x - 60)  # Shifting 60+ to the right

# Apply the inverse of the custom transformation to set the ticks correctly
def inverse_transform(x):
    if x < 30:
        return x
    elif x < 40:
        return 30 + (x - 30) * 3  # Expanding the compressed range back to 30-60
    else:
        return 60 + (x - 40)  # Shifting the right part back

def plot_adjusted_coherence(freq, coherence_mean, coherence_error, color, axis):
    
    # Plot the data using the custom transformation
    freq_transformed = [custom_transform(xi) for xi in freq]
    
    axis.plot(freq_transformed, coherence_mean, label='mean', c=color, linewidth=1)
    #######################################################################################
    #######################################################################################
    axis.axvspan(custom_transform(35), custom_transform(60), color='white', zorder=100) 
    #######################################################################################
    #######################################################################################
    axis.fill_between(freq_transformed, coherence_mean - coherence_error, coherence_mean + coherence_error, alpha=0.2, color=color)
    
    
    # Set the custom ticks and labels
    x_ticks = [4, 12, 20, 35, 60, 70, 80, 90]
    axis.set_xticks([custom_transform(xt) for xt in x_ticks])
    axis.set_xticklabels(x_ticks)
    axis.set_ylim([-25, 50])
    axis.set_yticks([-25,0,25,50])
    axis.set_ylim([-25, 75])
    axis.set_yticks([-25,0,25,50,75])
    axis.set_xlim([custom_transform(4),custom_transform(90)])
    
    # Apply the custom transformation to the x-axis
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: inverse_transform(x)))
    
def plot_coherence_panel(freq, coherence_array, error_type, color, axis):

    # get mean coherence
    coherence_mean = np.nanmean(coherence_array, axis=0)
    
    # based on the selected error bar, find either the standard deviation or standard error around the average coherence for each frequency
    if(error_type=="sd"):
        coherence_error       = np.nanstd(coherence_array, axis=0)
    elif(error_type=="se"):
        coherence_error       = 2 * (np.nanstd(coherence_array, axis=0) / np.sqrt(len(coherence_array)))
    
    plot_adjusted_coherence(freq, coherence_mean=coherence_mean, coherence_error=coherence_error, color=color, axis=axis)
    utils_plotting.set_axis(axis)
    axis.set_xticklabels([xt for xt in [4, 12, 20, 35, 60, 70, 80, 90]]) 
        
    return axis

def plot_LID_vs_noLID_coherence(dataset_LID, dataset_noLID, segment="event", error_type="se", figure_name=""):

    if(segment=="event"):
        coherence_feature = "event_coherence"
    elif(segment=="pre_event"):
        coherence_feature = "pre_event_coherence"
    else:
        coherence_feature = "post_event_coherence"

    # get the coherence array of selected event segment
    coherence_LID_array   = dataset_LID[coherence_feature].to_list()
    coherence_noLID_array = dataset_noLID[coherence_feature].to_list()
    freq                  = np.linspace(4,100,97) # fixed

    # plot
    plt = utils_plotting.get_figure_template()
    ax  = plt.subplot2grid((77, 66), (0, 0) , colspan=20, rowspan=15)
    
    try:
        plot_coherence_panel(freq, coherence_LID_array, error_type=error_type, color=utils_plotting.colors["voluntary"]["severe"], axis=ax)
    except:
        pass

    try:
        plot_coherence_panel(freq, coherence_noLID_array, error_type=error_type, color=utils_plotting.colors["no_LID"], axis=ax)
    except:
        pass
    
    ax.set_title(segment, fontsize=utils_plotting.LABEL_SIZE_label)
    utils_plotting.set_axis(ax)
    
    save_path = os.path.dirname(figure_name + ".png")
    os.makedirs(save_path, exist_ok=True) # Create directories if they don't exist
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)

def plot_dyskinesia_severity_coherence(dataset_LID, dataset_noLID, segment="event", error_type="se", figure_name=""):

    if(segment=="event"):
        coherence_feature = "event_coherence"
    elif(segment=="pre_event"):
        coherence_feature = "pre_event_coherence"
    else:
        coherence_feature = "post_event_coherence"

    # get the coherence array of selected event segment
    freq                  = np.linspace(4,100,97) # fixed
    
    try:
        plot_coherence_panel(freq, dataset_LID[dataset_LID.dyskinesia_arm == "mild"][coherence_feature].to_list(), 
                             error_type=error_type, color=utils_plotting.colors["tapping"]["mild"], axis=ax)
    except:
        pass
    try:
        plot_coherence_panel(freq, dataset_LID[dataset_LID.dyskinesia_arm == "moderate"][coherence_feature].to_list(), 
                             error_type=error_type, color=utils_plotting.colors["tapping"]["moderate"], axis=ax)
    except:
        pass

    try:
        plot_coherence_panel(freq, dataset_noLID[dataset_noLID.event_start_time<30][coherence_feature].to_list(), 
                             error_type=error_type, color=utils_plotting.colors["no_LID_no_DOPA"], axis=ax)
    except:
        pass
    try:
        plot_coherence_panel(freq, dataset_noLID[dataset_noLID.event_start_time>=30][coherence_feature].to_list(), 
                             error_type=error_type, color=utils_plotting.colors["no_LID_DOPA"], axis=ax)
    except:
        pass
    
    ax.set_title(segment, fontsize=utils_plotting.LABEL_SIZE_label)
    utils_plotting.set_axis(ax)

    save_path = os.path.dirname(figure_name + ".png")
    os.makedirs(save_path, exist_ok=True) # Create directories if they don't exist
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)
