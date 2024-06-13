"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable

LABEL_SIZE       = 5
LABEL_SIZE_label = 6 
LABEL_SIZE_title = 7 

# color dataframe
colors                                         = {}
colors["tapping"]                              = {}
colors["involuntary_movement"]                 = {}

colors["tapping"]["LID_none"]                  = "#FFEA00"
colors["tapping"]["LID_mild"]                  = "#EF8A06"
colors["tapping"]["LID_moderate"]              = "#DC2F02"
colors["tapping"]["LID_severe"]                = "#9D0208"
colors["tapping"]["LID_extreme"]               = "#370617"

colors["involuntary_movement"]["LID_none"]     = "#70D8EB"
colors["involuntary_movement"]["LID_mild"]     = "#00AACC"
colors["involuntary_movement"]["LID_moderate"] = "#006AA3"
colors["involuntary_movement"]["LID_severe"]   = "#023579"
colors["involuntary_movement"]["LID_extreme"]  = "#03045E"

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

def plot_accelerometer_events(data, fs, color, axis, error_bar= "se", padding_for="onset"):

    if(len(data)!=0):
        
        assert error_bar in ["sd", "se"], f'Please choose error bar as standard deviation:"sd" or standard error: "se"'
        assert padding_for in ["onset", "offset"], f'Please choose padding_for as "onset" or "offset"'
        
        # Find the maximum length among all arrays
        max_length  = max(len(arr) for arr in data) 
    
        # Pad arrays to make them all the same length
        # padding the end
        if(padding_for=="onset"):
            data_padded = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in data]
            t  = np.linspace(-1, max_length/fs, max_length)
        elif(padding_for=="offset"):
            data_padded = [np.pad(arr, (max_length - len(arr), 0), mode='constant') for arr in data]
            t  = np.linspace(-max_length/fs, 1, max_length)
        
        # Compute the mean and error_bar (standard deviation or standard error)
        mean_data   = np.mean(data_padded, axis=0)

        # define upper and lower bondaries of y axis
        y_axis_upper = max(mean_data) * 1.25
        y_axis_lower = min(mean_data) * 1.25
    
        if(error_bar=="sd"):
            error       = np.std(data_padded, axis=0)
            error_label = "standard deviation"
        elif(error_bar=="se"):
            error       = 2 * np.std(data_padded, axis=0) / np.sqrt(len(data))
            error_label = "standard error"
        
        # plot
        axis.plot(t, mean_data, label='mean', c=color, linewidth=1)
        axis.fill_between(t, mean_data - error, mean_data + error, alpha=0.2, color=color, label=error_label)
        axis.grid(False)
        axis.set_ylim([y_axis_lower, y_axis_upper])
        
        return axis
    else:
        return axis

def plot_accelerometer_events_by_category(accelerometer_events, kinematics, colors, figure_name):
    
    plt  = get_figure_template()
    
    ax1  = plt.subplot2grid((75, 40), (0, 0)  , colspan=18, rowspan=15)
    ax2  = plt.subplot2grid((75, 40), (0, 25) , colspan=18, rowspan=15)
    ax3  = plt.subplot2grid((75, 40), (20, 0) , colspan=18, rowspan=15)
    ax4  = plt.subplot2grid((75, 40), (20, 25), colspan=18, rowspan=15)

    # onset aligned
    cat  = "tapping"
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"], kinematics.fs, axis=ax1, 
                                             color=colors[cat]["LID_moderate"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["offset"], kinematics.fs, axis=ax3, 
                                             color=colors[cat]["LID_moderate"], padding_for="offset")
    
    cat  = "involuntary_movement"
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"], kinematics.fs, axis=ax2, 
                                             color=colors[cat]["LID_moderate"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["offset"], kinematics.fs, axis=ax4, 
                                             color=colors[cat]["LID_moderate"], padding_for="offset")
    
    set_axis(ax1)
    set_axis(ax2)
    set_axis(ax3)
    set_axis(ax4)

    ax1.set_title("Tapping"    , fontsize=LABEL_SIZE_title, weight="bold")
    ax2.set_title("Involuntary Movements"  , fontsize=LABEL_SIZE_title, weight="bold")
    ax1.set_ylabel("Accelerometer", fontsize=LABEL_SIZE_title)
    ax3.set_ylabel("Accelerometer", fontsize=LABEL_SIZE_title)
    
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax3.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax4.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    
    ax1.set_xlim([-1,3])
    ax2.set_xlim([-1,3])
    ax3.set_xlim([-3,1])
    ax4.set_xlim([-3,1])
    
    ax1.set_ylim([0, 2.5e-06])
    ax2.set_ylim([0, 2.5e-06])
    ax3.set_ylim([0, 2.5e-06])
    ax4.set_ylim([0, 2.5e-06])
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)


def plot_accelerometer_events_by_dyskinesia(accelerometer_events, kinematics, colors, figure_name):
    
    plt  = get_figure_template()
    ax1  = plt.subplot2grid((75, 40), (0, 0)  , colspan=18, rowspan=15)
    ax2  = plt.subplot2grid((75, 40), (0, 25) , colspan=18, rowspan=15)
    ax3  = plt.subplot2grid((75, 40), (20, 0) , colspan=18, rowspan=15)
    ax4  = plt.subplot2grid((75, 40), (20, 25), colspan=18, rowspan=15)

    event_category = "tapping"

    # onset aligned
    for severity in ["LID_extreme", "LID_severe", "LID_moderate", "LID_mild", "LID_none"]:
        plot_accelerometer_events(accelerometer_events[event_category][severity]["onset"], kinematics.fs, 
                                  axis=ax1, color=colors[event_category][severity], padding_for="onset")

    # offset aligned
    for severity in ["LID_extreme", "LID_severe", "LID_moderate", "LID_mild", "LID_none"]:
        plot_accelerometer_events(accelerometer_events[event_category][severity]["offset"], kinematics.fs, 
                                  axis=ax3, color=colors[event_category][severity], padding_for="offset")

    event_category = "involuntary_movement"
    
    # onset aligned
    for severity in ["LID_extreme", "LID_severe", "LID_moderate", "LID_mild", "LID_none"]:
        plot_accelerometer_events(accelerometer_events[event_category][severity]["onset"], kinematics.fs, 
                                  axis=ax2, color=colors[event_category][severity], padding_for="onset")

    # offset aligned
    for severity in ["LID_extreme", "LID_severe", "LID_moderate", "LID_mild", "LID_none"]:
        plot_accelerometer_events(accelerometer_events[event_category][severity]["offset"], kinematics.fs, 
                                  axis=ax4, color=colors[event_category][severity], padding_for="offset")
    
    
    ax1.set_title("Tapping"              , fontsize=LABEL_SIZE_title, weight="bold")
    ax2.set_title("Involuntary Movements", fontsize=LABEL_SIZE_title, weight="bold")
    ax1.set_ylabel("Accelerometer"       , fontsize=LABEL_SIZE_title)
    ax3.set_ylabel("Accelerometer"       , fontsize=LABEL_SIZE_title)
    
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax3.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax4.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
   
    ax1.set_xlim([-1,3])
    ax2.set_xlim([-1,3])
    ax3.set_xlim([-3,1])
    ax4.set_xlim([-3,1])
    
    # aling y axes
    ax1.set_ylim([0, 2.5e-06])
    ax2.set_ylim([0, 2.5e-06])
    ax3.set_ylim([0, 2.5e-06])
    ax4.set_ylim([0, 2.5e-06])
    
    set_axis(ax1)
    set_axis(ax2)
    set_axis(ax3)
    set_axis(ax4)

    # add legend
    c_LID_none     = mpatches.Patch(color=colors["tapping"]["LID_none"], label='LID none')
    c_LID_mild     = mpatches.Patch(color=colors["tapping"]["LID_mild"], label='LID mild')
    c_LID_moderate = mpatches.Patch(color=colors["tapping"]["LID_moderate"], label='LID moderate')
    c_LID_severe   = mpatches.Patch(color=colors["tapping"]["LID_severe"], label='LID severe')
    c_LID_extreme  = mpatches.Patch(color=colors["tapping"]["LID_extreme"], label='LID extreme')
    ax3.legend(handles=[c_LID_none, c_LID_mild, c_LID_moderate, c_LID_severe, c_LID_extreme], 
               prop={'size': LABEL_SIZE_title}, loc='lower center', bbox_to_anchor=(0.5,-1))

    # add legend
    c_LID_none     = mpatches.Patch(color=colors["involuntary_movement"]["LID_none"], label='LID none')
    c_LID_mild     = mpatches.Patch(color=colors["involuntary_movement"]["LID_mild"], label='LID mild')
    c_LID_moderate = mpatches.Patch(color=colors["involuntary_movement"]["LID_moderate"], label='LID moderate')
    c_LID_severe   = mpatches.Patch(color=colors["involuntary_movement"]["LID_severe"], label='LID severe')
    c_LID_extreme  = mpatches.Patch(color=colors["involuntary_movement"]["LID_extreme"], label='LID extreme')
    ax4.legend(handles=[c_LID_none, c_LID_mild, c_LID_moderate, c_LID_severe, c_LID_extreme], 
               prop={'size': LABEL_SIZE_title}, loc='lower center', bbox_to_anchor=(0.5,-1))
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)

def plot_single_event_and_spectogram(event, time_vector, fs, alignment):

    event = np.array(event)
    
    plt  = get_figure_template()
    ax1  = plt.subplot2grid((75, 40), (0, 0) , colspan=18, rowspan=10)
    ax2  = plt.subplot2grid((75, 40), (11, 0) , colspan=18, rowspan=15)

    # plot event
    ax1.plot(time_vector, event)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

    # plot spectogram
    f, t, Sxx = spectrogram(event, fs=fs, nperseg=int(fs/10))
    Sxx       = 10 * np.log10(Sxx) # 'Power/Frequency (dB/Hz)'

    if(alignment=="onset"):
        time_vec_spectogram = t - 1
        cax = ax2.pcolormesh(time_vec_spectogram, f, Sxx, shading='gouraud')
        ax1.set_xlim([-1,(len(event)/fs)-1])
        ax2.set_xlim([-1,(len(event)/fs)-1])
    else:
        time_vec_spectogram = (t - max(t)) + 1
        cax = ax2.pcolormesh(time_vec_spectogram, f, Sxx, shading='gouraud')
        ax1.set_xlim([1-(len(event)/fs),1])
        ax2.set_xlim([1-(len(event)/fs),1])
        
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='-', color="dimgrey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='-', color="white")
    ax1.text(-0.1, ax1.get_ylim()[1]*1.1, 'offset', c="k", fontsize = LABEL_SIZE)
    
    # Add the colorbar to the new axis
    divider = make_axes_locatable(ax2)
    cbar_ax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = plt.colorbar(cax, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Power/Frequency (dB/Hz)', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)

    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [s]')
    set_axis(ax1)
    set_axis(ax2)
    ax2.set_ylim([0,100])

def plot_average_spectogram_for_event_category(accelerometer_events, time_vector, fs):
    
    plt  = get_figure_template()

    ax1  = plt.subplot2grid((75, 40), (0, 0)   , colspan=18, rowspan=10)
    ax2  = plt.subplot2grid((75, 40), (0, 20)  , colspan=18, rowspan=10)
    ax3  = plt.subplot2grid((75, 40), (15, 0)  , colspan=18, rowspan=10)
    ax4  = plt.subplot2grid((75, 40), (15, 20) , colspan=18, rowspan=10)

    get_average_spectogram_panel(accelerometer_events["tapping"]["all"]["onset"], time_vector, fs, "onset", ax=ax1,
                                 ylabel='Frequency [Hz]',colorbar_visibility=False)
    
    get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["all"]["onset"], time_vector, fs, "onset", ax=ax2, 
                                 ylabel='',colorbar_visibility=False)

    get_average_spectogram_panel(accelerometer_events["tapping"]["all"]["offset"], time_vector, fs, "offset", ax=ax3, 
                                 ylabel='Frequency [Hz]', colorbar_visibility=False)
    
    get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["all"]["offset"],time_vector, fs, "offset", ax=ax4, 
                                 ylabel='', colorbar_visibility=False)

    ax1.set_title("Tapping"              , fontsize=LABEL_SIZE_title, weight="bold")
    ax2.set_title("Involuntary Movements", fontsize=LABEL_SIZE_title, weight="bold")
