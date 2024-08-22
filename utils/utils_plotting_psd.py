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

from lib_LFP import LFP

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

def plot_adjusted_psd(freq, psd_mean, psd_error, color, axis):
    
    # Plot the data using the custom transformation
    freq_transformed = [custom_transform(xi) for xi in freq]
    
    axis.plot(freq_transformed, psd_mean, label='mean', c=color, linewidth=1)
    #######################################################################################
    #######################################################################################
    axis.axvspan(custom_transform(35), custom_transform(60), color='white', zorder=100) 
    #######################################################################################
    #######################################################################################
    axis.fill_between(freq_transformed, psd_mean - psd_error, psd_mean + psd_error, alpha=0.2, color=color)
    
    
    # Set the custom ticks and labels
    x_ticks = [4, 12, 20, 30, 60, 70, 80, 90]
    axis.set_xticks([custom_transform(xt) for xt in x_ticks])
    axis.set_xticklabels(x_ticks)
    
    # Apply the custom transformation to the x-axis
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: inverse_transform(x)))

def plot_power_spectra_panel(freq, psd_array, error_type, color, axis):

    # get mean of the psd
    psd_mean = np.nanmean(psd_array, axis=0)
    
    # based on the selected error bar, find either the standard deviation or standard error around the average PSD for each frequency
    if(error_type=="sd"):
        psd_error       = np.nanstd(psd_array, axis=0)
    elif(error_type=="se"):
        psd_error       = 2 * (np.nanstd(psd_array, axis=0) / np.sqrt(len(psd_array)))
    
    if(len(freq)!=0):
        plot_adjusted_psd(freq, psd_mean=psd_mean, psd_error=psd_error, color=color, axis=axis)
        utils_plotting.set_axis(axis)
        axis.set_ylim([-50,100])
        #######################################################################################
        #######################################################################################
        axis.set_xlim([-4,custom_transform(90)])
        #######################################################################################
        #######################################################################################
        axis.set_xticklabels([xt for xt in [4, 12, 20, 30, 60, 70, 80, 90]]) 
        
    return axis

def plot_LID_vs_noLID_psd(dataset_LID, dataset_noLID, segment="event", error_type="se", figure_name=""):

    if(segment=="event"):
        psd_feature = "event_psd"
    elif(segment=="pre_event"):
        psd_feature = "pre_event_psd"
    else:
        psd_feature = "post_event_psd"

    # get the PSD array of selected event segment
    psd_LID_array   = dataset_LID[psd_feature].to_list()
    psd_noLID_array = dataset_noLID[psd_feature].to_list()
    freq            = np.linspace(4,100,97) # fixed

    # plot
    plt             = utils_plotting.get_figure_template()
    ax              = plt.subplot2grid((77, 66), (0, 0) , colspan=20, rowspan=15)
    
    try:
        plot_power_spectra_panel(freq, psd_LID_array, error_type=error_type, color=utils_plotting.colors["voluntary"]["severe"], axis=ax)
    except:
        pass

    try:
        plot_power_spectra_panel(freq, psd_noLID_array, error_type=error_type, color=utils_plotting.colors["no_LID"], axis=ax)
    except:
        pass
    
    ax.set_title(segment, fontsize=utils_plotting.LABEL_SIZE_label)
    
    save_path = os.path.dirname(figure_name + ".png")
    os.makedirs(save_path, exist_ok=True) # Create directories if they don't exist
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)

def plot_LID_severity_psd(dataset_LID, segment="event", dyskinesia_strategy="dyskinesia_arm", error_type="se", figure_name=""):

    if(segment=="event"):
        psd_segment = "event_psd"
    elif(segment=="pre_event"):
        psd_segment = "pre_event_psd"
    else:
        psd_segment = "post_event_psd"

    # get the PSD array of selected event segment
    psd_LID_mild     = dataset_LID[dataset_LID[dyskinesia_strategy]=="mild"][psd_segment].to_list()
    psd_LID_moderate = dataset_LID[dataset_LID[dyskinesia_strategy]=="moderate"][psd_segment].to_list()
    psd_LID_severe   = dataset_LID[dataset_LID[dyskinesia_strategy]=="severe"][psd_segment].to_list()
    
    freq            = np.linspace(4,100,97) # fixed

    # plot
    plt             = utils_plotting.get_figure_template()
    ax              = plt.subplot2grid((77, 66), (0, 0) , colspan=20, rowspan=15)

    try:
        plot_power_spectra_panel(freq, psd_LID_mild, error_type=error_type, color=utils_plotting.colors["voluntary"]["mild"], axis=ax)
    except:
        pass
    try:
        plot_power_spectra_panel(freq, psd_LID_moderate, error_type=error_type, color=utils_plotting.colors["voluntary"]["moderate"], axis=ax)
    except:
        pass
    try:
        plot_power_spectra_panel(freq, psd_LID_severe, error_type=error_type, color=utils_plotting.colors["voluntary"]["severe"], axis=ax)
    except:
        pass
    ax.set_title(segment, fontsize=utils_plotting.LABEL_SIZE_label)

    save_path = os.path.dirname(figure_name + ".png")
    os.makedirs(save_path, exist_ok=True) # Create directories if they don't exist
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)

def plot_noLID_states_psd(dataset_noLID, segment="event", error_type="se", figure_name=""):

    if(segment=="event"):
        psd_feature = "event_psd"
    elif(segment=="pre_event"):
        psd_feature = "pre_event_psd"
    else:
        psd_feature = "post_event_psd"

    # get the PSD array of selected event segment
    psd_noLID_noDOPA_array = dataset_noLID[dataset_noLID["event_start_time"]<=30][psd_feature].to_list()
    psd_noLID_DOPA_array = dataset_noLID[dataset_noLID["event_start_time"]>30][psd_feature].to_list()
    freq                     = np.linspace(4,100,97) # fixed

    # plot
    plt             = utils_plotting.get_figure_template()
    ax              = plt.subplot2grid((77, 66), (0, 0) , colspan=20, rowspan=15)
    
    plot_power_spectra_panel(freq, psd_noLID_noDOPA_array, error_type=error_type, color=utils_plotting.colors["no_LID_no_DOPA"], axis=ax)
    plot_power_spectra_panel(freq, psd_noLID_DOPA_array, error_type=error_type, color=utils_plotting.colors["no_LID_DOPA"], axis=ax)
    
    ax.set_title(segment, fontsize=utils_plotting.LABEL_SIZE_label)

    save_path = os.path.dirname(figure_name + ".png")
    os.makedirs(save_path, exist_ok=True) # Create directories if they don't exist
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)
