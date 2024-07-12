"""
Utilisation function for plotting
"""

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

from lib_LFP import LFP

LABEL_SIZE       = 5
LABEL_SIZE_label = 6 
LABEL_SIZE_title = 7 

# color dataframe
colors                               = {}
colors["tapping"]                    = {}
colors["involuntary_movement"]       = {}

colors["tapping"]["LID_none"]        = "#FFEA00"
colors["tapping"]["LID_mild"]        = "#EF8A06"
colors["tapping"]["LID_moderate"]    = "#DC2F02"
colors["tapping"]["LID_severe"]      = "#9D0208"
colors["tapping"]["LID_extreme"]     = "#370617"

colors["involuntary_movement"]["LID_none"]     = "#70D8EB"
colors["involuntary_movement"]["LID_mild"]     = "#00AACC"
colors["involuntary_movement"]["LID_moderate"] = "#006AA3"
colors["involuntary_movement"]["LID_severe"]   = "#023579"
colors["involuntary_movement"]["LID_extreme"]  = "#03045E"

colors_segment               = {}
colors_segment["pre_event"]  = "#f37660"
colors_segment["event"]      = "#c8246b"
colors_segment["post_event"] = "#07b2a0" 



def get_figure_template():
    
    plt.rc('font', serif="Neue Haas Grotesk Text Pro")
    fig = plt.figure()
    fig.tight_layout()
    cm = 1/2.54  # centimeters in inches
    plt.subplots(figsize=(18.5*cm, 21*cm))
    return plt

def set_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=LABEL_SIZE)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE)
    ax.yaxis.offsetText.set_fontsize(LABEL_SIZE)

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
    axis.fill_between(freq_transformed, psd_mean - psd_error, psd_mean + psd_error, alpha=0.2, color=color)
    
    # Set the custom ticks and labels
    x_ticks = [4, 12, 20, 30, 60, 70, 80, 90]
    axis.set_xticks([custom_transform(xt) for xt in x_ticks])
    axis.set_xticklabels(x_ticks)
    
    # Apply the custom transformation to the x-axis
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: inverse_transform(x)))

def plot_LFP_power_spectra_panel(freq, psd_mean, psd_error, color, axis):
    
    if(len(freq)!=0):
        plot_adjusted_psd(freq, psd_mean=psd_mean, psd_error=psd_error, color=color, axis=axis)
        set_axis(axis)
        axis.set_ylim([-0.1,2.5])
        axis.set_xticklabels([xt for xt in [4, 12, 20, 30, 60, 70, 80, 90]]) 
        
    return axis

def plot_LFP_PSD_for_hemisphere_and_event_category(dataset, hemisphere, event_category, segment):
    
    plt = get_figure_template()

    ax_32_i = plt.subplot2grid((77, 40), (0, 0), colspan=12, rowspan=10)
    ax_32_c = plt.subplot2grid((77, 40), (0, 12), colspan=12, rowspan=10)
    ax_43_i = plt.subplot2grid((77, 40), (13, 0), colspan=12, rowspan=10)
    ax_43_c = plt.subplot2grid((77, 40), (13, 12), colspan=12, rowspan=10)
    ax_54_i = plt.subplot2grid((77, 40), (26, 0), colspan=12, rowspan=10)
    ax_54_c = plt.subplot2grid((77, 40), (26, 12), colspan=12, rowspan=10)
    ax_65_i = plt.subplot2grid((77, 40), (39, 0), colspan=12, rowspan=10)
    ax_65_c = plt.subplot2grid((77, 40), (39, 12), colspan=12, rowspan=10)
    ax_76_i = plt.subplot2grid((77, 40), (52, 0), colspan=12, rowspan=10)
    ax_76_c = plt.subplot2grid((77, 40), (52, 12), colspan=12, rowspan=10)
    ax_87_i = plt.subplot2grid((77, 40), (65, 0), colspan=12, rowspan=10)
    ax_87_c = plt.subplot2grid((77, 40), (65, 12), colspan=12, rowspan=10)
    
    channel = "32"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_32_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_32_c)
    
    channel = "43"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_43_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_43_c)
    
    channel = "54"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_54_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_54_c)
    
    channel = "65"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_65_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_65_c)
    
    channel = "76"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_76_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_76_c) 
    
    channel = "87"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_87_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_87_c)
        
    ax_32_i.set_ylabel("Channel 32", fontsize=LABEL_SIZE, weight="bold")
    ax_43_i.set_ylabel("Channel 43", fontsize=LABEL_SIZE, weight="bold")
    ax_54_i.set_ylabel("Channel 54", fontsize=LABEL_SIZE, weight="bold")
    ax_65_i.set_ylabel("Channel 65", fontsize=LABEL_SIZE, weight="bold")
    ax_76_i.set_ylabel("Channel 76", fontsize=LABEL_SIZE, weight="bold")
    ax_87_i.set_ylabel("Channel 87", fontsize=LABEL_SIZE, weight="bold")
    
    ax_32_i.set_title("Ipsilateral Events", fontsize=LABEL_SIZE, weight="bold")
    ax_32_c.set_title("Contralateral Events", fontsize=LABEL_SIZE, weight="bold")
    
    ax_32_c.set_yticklabels("")
    ax_43_c.set_yticklabels("")
    ax_54_c.set_yticklabels("")
    ax_65_c.set_yticklabels("")
    ax_76_c.set_yticklabels("")
    ax_87_c.set_yticklabels("")
    
    plt.show()

def plot_LFP_PSD_for_event_category(dataset, event_category, segment):
    
    plt = get_figure_template()

    ax_32_i = plt.subplot2grid((77, 40), (0, 0), colspan=12, rowspan=10)
    ax_32_c = plt.subplot2grid((77, 40), (0, 12), colspan=12, rowspan=10)
    ax_43_i = plt.subplot2grid((77, 40), (13, 0), colspan=12, rowspan=10)
    ax_43_c = plt.subplot2grid((77, 40), (13, 12), colspan=12, rowspan=10)
    ax_54_i = plt.subplot2grid((77, 40), (26, 0), colspan=12, rowspan=10)
    ax_54_c = plt.subplot2grid((77, 40), (26, 12), colspan=12, rowspan=10)
    ax_65_i = plt.subplot2grid((77, 40), (39, 0), colspan=12, rowspan=10)
    ax_65_c = plt.subplot2grid((77, 40), (39, 12), colspan=12, rowspan=10)
    ax_76_i = plt.subplot2grid((77, 40), (52, 0), colspan=12, rowspan=10)
    ax_76_c = plt.subplot2grid((77, 40), (52, 12), colspan=12, rowspan=10)
    ax_87_i = plt.subplot2grid((77, 40), (65, 0), colspan=12, rowspan=10)
    ax_87_c = plt.subplot2grid((77, 40), (65, 12), colspan=12, rowspan=10)
    
    channel = "32"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_32_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_32_c)
    
    channel = "43"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_43_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_43_c)
    
    channel = "54"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_54_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_54_c)
    
    channel = "65"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_65_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_65_c)
    
    channel = "76"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_76_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_76_c) 
    
    channel = "87"
    for severity in ["none","mild","moderate","severe","extreme"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity)
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors[event_category]["LID_" + severity], axis=ax_87_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors[event_category]["LID_" + severity], axis=ax_87_c)
        
    ax_32_i.set_ylabel("Channel 32", fontsize=LABEL_SIZE, weight="bold")
    ax_43_i.set_ylabel("Channel 43", fontsize=LABEL_SIZE, weight="bold")
    ax_54_i.set_ylabel("Channel 54", fontsize=LABEL_SIZE, weight="bold")
    ax_65_i.set_ylabel("Channel 65", fontsize=LABEL_SIZE, weight="bold")
    ax_76_i.set_ylabel("Channel 76", fontsize=LABEL_SIZE, weight="bold")
    ax_87_i.set_ylabel("Channel 87", fontsize=LABEL_SIZE, weight="bold")
    
    ax_32_i.set_title("Ipsilateral Events", fontsize=LABEL_SIZE, weight="bold")
    ax_32_c.set_title("Contralateral Events", fontsize=LABEL_SIZE, weight="bold")
    
    ax_32_c.set_yticklabels("")
    ax_43_c.set_yticklabels("")
    ax_54_c.set_yticklabels("")
    ax_65_c.set_yticklabels("")
    ax_76_c.set_yticklabels("")
    ax_87_c.set_yticklabels("")
    
    plt.show()

def plot_LFP_PSD_for_event_category_and_segment(dataset, event_category):
    
    plt = get_figure_template()

    ax_32_i = plt.subplot2grid((77, 40), (0, 0), colspan=12, rowspan=10)
    ax_32_c = plt.subplot2grid((77, 40), (0, 12), colspan=12, rowspan=10)
    ax_43_i = plt.subplot2grid((77, 40), (13, 0), colspan=12, rowspan=10)
    ax_43_c = plt.subplot2grid((77, 40), (13, 12), colspan=12, rowspan=10)
    ax_54_i = plt.subplot2grid((77, 40), (26, 0), colspan=12, rowspan=10)
    ax_54_c = plt.subplot2grid((77, 40), (26, 12), colspan=12, rowspan=10)
    ax_65_i = plt.subplot2grid((77, 40), (39, 0), colspan=12, rowspan=10)
    ax_65_c = plt.subplot2grid((77, 40), (39, 12), colspan=12, rowspan=10)
    ax_76_i = plt.subplot2grid((77, 40), (52, 0), colspan=12, rowspan=10)
    ax_76_c = plt.subplot2grid((77, 40), (52, 12), colspan=12, rowspan=10)
    ax_87_i = plt.subplot2grid((77, 40), (65, 0), colspan=12, rowspan=10)
    ax_87_c = plt.subplot2grid((77, 40), (65, 12), colspan=12, rowspan=10)
    
    channel = "32"
    for segment in ["pre_event", "event", "post_event"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity="all")
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors_segment[segment], axis=ax_32_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors_segment[segment], axis=ax_32_c)
    
    channel = "43"
    for segment in ["pre_event", "event", "post_event"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity="all")
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors_segment[segment], axis=ax_43_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors_segment[segment], axis=ax_43_c)
    
    channel = "54"
    for segment in ["pre_event", "event", "post_event"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity="all")
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors_segment[segment], axis=ax_54_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors_segment[segment], axis=ax_54_c)
    
    channel = "65"
    for segment in ["pre_event", "event", "post_event"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity="all")
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors_segment[segment], axis=ax_65_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors_segment[segment], axis=ax_65_c)
    
    channel = "76"
    for segment in ["pre_event", "event", "post_event"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity="all")
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors_segment[segment], axis=ax_76_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors_segment[segment], axis=ax_76_c) 
    
    channel = "87"
    
    for segment in ["pre_event", "event", "post_event"]:
        freq_i, mean_i, error_i, freq_c, mean_c, error_c = LFP.measure_LFP_power_spectra_with_laterality(dataset, "", channel, event_category, segment, severity="all")
        plot_LFP_power_spectra_panel(freq_i, mean_i, error_i, color=colors_segment[segment], axis=ax_87_i)
        plot_LFP_power_spectra_panel(freq_c, mean_c, error_c, color=colors_segment[segment], axis=ax_87_c)

    c_pre   = mpatches.Patch(color=colors_segment["pre_event"], label='pre-event')
    c_event = mpatches.Patch(color=colors_segment["event"], label='event')
    c_post  = mpatches.Patch(color=colors_segment["post_event"], label='post-event')
    
    ax_87_c.legend(handles=[c_pre, c_event, c_post], prop={'size': LABEL_SIZE}, loc='best')
    
    ax_32_i.set_ylabel("Channel 32", fontsize=LABEL_SIZE, weight="bold")
    ax_43_i.set_ylabel("Channel 43", fontsize=LABEL_SIZE, weight="bold")
    ax_54_i.set_ylabel("Channel 54", fontsize=LABEL_SIZE, weight="bold")
    ax_65_i.set_ylabel("Channel 65", fontsize=LABEL_SIZE, weight="bold")
    ax_76_i.set_ylabel("Channel 76", fontsize=LABEL_SIZE, weight="bold")
    ax_87_i.set_ylabel("Channel 87", fontsize=LABEL_SIZE, weight="bold")
    
    ax_32_i.set_title("Ipsilateral Events", fontsize=LABEL_SIZE, weight="bold")
    ax_32_c.set_title("Contralateral Events", fontsize=LABEL_SIZE, weight="bold")
    
    ax_32_c.set_yticklabels("")
    ax_43_c.set_yticklabels("")
    ax_54_c.set_yticklabels("")
    ax_65_c.set_yticklabels("")
    ax_76_c.set_yticklabels("")
    ax_87_c.set_yticklabels("")
    
    plt.show()