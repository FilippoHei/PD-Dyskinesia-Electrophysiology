"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_plotting, utils_accelerometer

from lib_data import DATA_IO


def plot_accelerometer_events(data, color, axis, error_bar= "se"):

    if(len(data)!=0):
        
        assert error_bar in ["sd", "se"], f'Please choose error bar as standard deviation:"sd" or standard error: "se"'
        
        # Compute the mean and error_bar (standard deviation or standard error)
        mean_data   = np.mean(data, axis=0)

        # define upper and lower bondaries of y axis
        y_axis_upper = max(mean_data) * 1.25
        y_axis_lower = min(mean_data) * 1.25
    
        if(error_bar=="sd"):
            error       = np.std(data, axis=0)
            error_label = "standard deviation"
        elif(error_bar=="se"):
            error       = 2 * np.std(data, axis=0) / np.sqrt(len(data))
            error_label = "standard error"

        t  = np.linspace(-2, 2, 4*512)
        # plot
        axis.plot(t, mean_data, label='mean', c=color, linewidth=1)
        axis.fill_between(t, mean_data - error, mean_data + error, alpha=0.2, color=color, label=error_label)
        axis.grid(False)
        axis.set_ylim([y_axis_lower, y_axis_upper])
        
        return axis
    else:
        return axis
        
def plot_accelerometer_events_for_event_category(dataset, event_category, figure_name):

    plt  = utils_plotting.get_figure_template()
    ax1  = plt.subplot2grid((75, 40), (0, 0)  , colspan=19, rowspan=15)
    ax2  = plt.subplot2grid((75, 40), (0, 21) , colspan=19, rowspan=15)

    dataset_noLID        = dataset[(dataset.event_category == event_category) & (dataset.dyskinesia_total == "none")]
    dataset_LID          = dataset[(dataset.event_category == event_category) & (dataset.dyskinesia_total != "none")]
    
    dataset_noLID_onset  = []
    dataset_noLID_offset = []
    dataset_LID_onset    = []   
    dataset_LID_offset   = []
    
    for index, row in dataset_LID.iterrows():
        dataset_LID_onset.append(row["event_onset_aligned"])
        dataset_LID_offset.append(row["event_offset_aligned"])

    for index, row in dataset_noLID.iterrows():
        dataset_noLID_onset.append(row["event_onset_aligned"])
        dataset_noLID_offset.append(row["event_offset_aligned"])

    ax1  = plot_accelerometer_events(dataset_noLID_onset, axis=ax1, color=utils_plotting.colors["no_LID"])
    ax1  = plot_accelerometer_events(dataset_LID_onset, axis=ax1, color=utils_plotting.colors[event_category]["severe"])

    ax2  = plot_accelerometer_events(dataset_noLID_offset, axis=ax2, color=utils_plotting.colors["no_LID"])
    ax2  = plot_accelerometer_events(dataset_LID_offset, axis=ax2, color=utils_plotting.colors[event_category]["severe"])

    utils_plotting.set_axis(ax1)
    utils_plotting.set_axis(ax2)

    ax1.set_title("Tapping [Onset Aligned]", fontsize=utils_plotting.LABEL_SIZE_label, weight="bold")
    ax2.set_title("Tapping [Offset Aligned]", fontsize=utils_plotting.LABEL_SIZE_label, weight="bold")
    
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax1.set_xlim([-2,2])
    ax2.set_xlim([-2,2])
    ax1.set_ylim([0, 1e-06])
    ax2.set_ylim([0, 1e-06])

    ax2.set_yticklabels("")
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)

def plot_accelerometer_events_for_dyskinesia_severity(dataset, event_category, dyskinesia_strategy, figure_name):

    plt  = utils_plotting.get_figure_template()
    ax1  = plt.subplot2grid((75, 40), (0, 0)  , colspan=19, rowspan=15)
    ax2  = plt.subplot2grid((75, 40), (0, 21) , colspan=19, rowspan=15)

    if(dyskinesia_strategy == "total"):
        
        for severity in dataset.dyskinesia_total.unique():
            dataset_severity        = dataset[(dataset.event_category == event_category) & (dataset.dyskinesia_total == severity)]
            dataset_severity_onset  = []
            dataset_severity_offset = []
    
            for index, row in dataset_severity.iterrows():
                dataset_severity_onset.append(row["event_onset_aligned"])
                dataset_severity_offset.append(row["event_offset_aligned"])

            ax1  = plot_accelerometer_events(dataset_severity_onset , axis=ax1, color=utils_plotting.colors[event_category][severity])
            ax2  = plot_accelerometer_events(dataset_severity_offset, axis=ax2, color=utils_plotting.colors[event_category][severity])

    else:
        
        for severity in dataset.dyskinesia_arm.unique():
            dataset_severity        = dataset[(dataset.event_category == event_category) & (dataset.dyskinesia_arm == severity)]
            dataset_severity_onset  = []
            dataset_severity_offset = []
    
            for index, row in dataset_severity.iterrows():
                dataset_severity_onset.append(row["event_onset_aligned"])
                dataset_severity_offset.append(row["event_offset_aligned"])

            ax1  = plot_accelerometer_events(dataset_severity_onset , axis=ax1, color=utils_plotting.colors[event_category][severity])
            ax2  = plot_accelerometer_events(dataset_severity_offset, axis=ax2, color=utils_plotting.colors[event_category][severity])

        
    utils_plotting.set_axis(ax1)
    utils_plotting.set_axis(ax2)

    ax1.set_title("Tapping [Onset Aligned]", fontsize=utils_plotting.LABEL_SIZE_label, weight="bold")
    ax2.set_title("Tapping [Offset Aligned]", fontsize=utils_plotting.LABEL_SIZE_label, weight="bold")
    
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax1.set_xlim([-2,2])
    ax2.set_xlim([-2,2])
    ax1.set_ylim([0, 1e-06])
    ax2.set_ylim([0, 1e-06])

    ax2.set_yticklabels("")
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)


##########################################################################################
##########################################################################################
##########################################################################################

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

def plot_CDRS_evolution_panel(CDRS_time, CDRS_score, CDRS_type, ax):
    
    for i in range(len(CDRS_time)):
    
        period = CDRS_time[i]
        score  = CDRS_score[i]

        if(CDRS_type=="arm"):
            if(score==0):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["none"], alpha=0.25)
            elif(score==1):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["mild"], alpha=1)
            elif(score==2):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["moderate"], alpha=1)
            elif(score==3):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["severe"], alpha=1)
            elif(score==4):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["extreme"], alpha=1)
                
        if(CDRS_type=="total"):

            if(score==0):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["none"], alpha=0.25)
            elif((score>0) & (score<=4)):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["mild"], alpha=1)
            elif((score>4) & (score<=8)):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["moderate"], alpha=1)
            elif((score>8) & (score<=12)):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["severe"], alpha=1)
            elif(score>12):
                ax.axvspan(period[0], period[1], color=utils_plotting.colors["voluntary"]["extreme"], alpha=1)

def plot_patient_activity(EVENTS_HISTORY, SUB):

    # check if the personal figure directory of the patient does exist, if not create
    directory = DATA_IO.path_figure + SUB
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # get all the indices
    indices                      = {}
    t_period                     = {}
    scores                       = {}

    indices["tapping"]           = utils_accelerometer.find_event_segments_indices(EVENTS_HISTORY.period_tap)
    indices["rest"]              = utils_accelerometer.find_event_segments_indices(EVENTS_HISTORY.period_rest)
    indices["free"]              = utils_accelerometer.find_event_segments_indices(EVENTS_HISTORY.period_free)
    t_period["tapping"]          = utils_accelerometer.find_timepoint_from_indices(EVENTS_HISTORY.times, indices["tapping"] )
    t_period["rest"]             = utils_accelerometer.find_timepoint_from_indices(EVENTS_HISTORY.times, indices["rest"] )
    t_period["free"]             = utils_accelerometer.find_timepoint_from_indices(EVENTS_HISTORY.times, indices["free"] )

    # plotting part

    plt        = utils_plotting.get_figure_template()
    
    ax_task    = plt.subplot2grid((80, 40), (0, 0), colspan=40, rowspan=2)
    
    ax_CDRS    = list(range(10))
    ax_CDRS[0] = plt.subplot2grid((80, 40), (3, 0), colspan=40, rowspan=2)
    ax_CDRS[1] = plt.subplot2grid((80, 40), (5, 0), colspan=40, rowspan=2)
    ax_CDRS[2] = plt.subplot2grid((80, 40), (7, 0), colspan=40, rowspan=2)  
    
    ax_vol_r   = plt.subplot2grid((80, 40), (10, 0), colspan=40, rowspan=3)
    ax_vol_l   = plt.subplot2grid((80, 40), (13, 0), colspan=40, rowspan=3)
    ax_invol_r = plt.subplot2grid((80, 40), (16, 0), colspan=40, rowspan=3)
    ax_invol_l = plt.subplot2grid((80, 40), (19, 0), colspan=40, rowspan=3)
    
    # task periods
    for period in t_period["tapping"]:
        ax_task.axvspan(period[0]/60, period[1]/60, color=utils_plotting.colors["tapping"], alpha=1)
    
    for period in t_period["rest"]:
        ax_task.axvspan(period[0]/60, period[1]/60, color=utils_plotting.colors["rest"], alpha=1)
    
    for period in t_period["free"]:
        ax_task.axvspan(period[0]/60, period[1]/60, color=utils_plotting.colors["free"], alpha=1)
    
    # right hand voluntary movements
    ax_vol_r = sns.lineplot(x=EVENTS_HISTORY.times/60, y=EVENTS_HISTORY.right_voluntary_movements, ax=ax_vol_r, 
                           color=utils_plotting.colors["voluntary"]["moderate"])
    
    # left hand voluntary movements
    ax_vol_l = sns.lineplot(x=EVENTS_HISTORY.times/60, y=EVENTS_HISTORY.left_voluntary_movements, ax=ax_vol_l, 
                           color=utils_plotting.colors["voluntary"]["moderate"])
    
    # right hand voluntary movements
    ax_invol_r = sns.lineplot(x=EVENTS_HISTORY.times/60, y=EVENTS_HISTORY.right_involuntary_movements, ax=ax_invol_r, 
                              color=utils_plotting.colors["involuntary"]["moderate"])
    
    # left hand voluntary movements
    ax_invol_l = sns.lineplot(x=EVENTS_HISTORY.times/60, y=EVENTS_HISTORY.left_involuntary_movements, ax=ax_invol_l, 
                              color=utils_plotting.colors["involuntary"]["moderate"])
    
    plot_CDRS_evolution_panel(EVENTS_HISTORY.CDRS_right_hand_indexes, EVENTS_HISTORY.CDRS_right_hand_scores, "arm", ax_CDRS[0])
    plot_CDRS_evolution_panel(EVENTS_HISTORY.CDRS_left_hand_indexes , EVENTS_HISTORY.CDRS_left_hand_scores, "arm", ax_CDRS[1])
    plot_CDRS_evolution_panel(EVENTS_HISTORY.CDRS_total_indexes     , EVENTS_HISTORY.CDRS_total_scores, "total", ax_CDRS[2])
    
    ax_task.set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_task.set_xticklabels("")
    ax_task.set_yticklabels("")
    ax_task.set_ylabel("task", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_CDRS[0].set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_CDRS[0].set_xticklabels("")
    ax_CDRS[0].set_yticklabels("")
    ax_CDRS[0].set_ylabel("CDRS right arm", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_CDRS[1].set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_CDRS[1].set_xticklabels("")
    ax_CDRS[1].set_yticklabels("")
    ax_CDRS[1].set_ylabel("CDRS left arm", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_CDRS[2].set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_CDRS[2].set_xticklabels("")
    ax_CDRS[2].set_yticklabels("")
    ax_CDRS[2].set_ylabel("CDRS total", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_vol_r.set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_vol_r.set_xticklabels("")
    ax_vol_r.set_yticklabels("")
    ax_vol_r.set_ylabel("right \n tapping", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_vol_l.set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_vol_l.set_xticklabels("")
    ax_vol_l.set_yticklabels("")
    ax_vol_l.set_ylabel("left \n tapping", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_invol_r.set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_invol_r.set_xticklabels("")
    ax_invol_r.set_yticklabels("")
    ax_invol_r.set_ylabel("right \n involuntary", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_invol_l.set_xlim([np.min(EVENTS_HISTORY.times)/60, np.max(EVENTS_HISTORY.times)/60])
    ax_invol_l.set_xticklabels(ax_invol_l.get_xticklabels(), fontsize=utils_plotting.LABEL_SIZE_label)
    ax_invol_l.set_yticklabels("")
    ax_invol_l.set_ylabel("left \n involuntary", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    ax_invol_l.set_xlabel("time (min)", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    utils_plotting.set_axis(ax_task)
    utils_plotting.set_axis(ax_vol_r)
    utils_plotting.set_axis(ax_vol_l)
    utils_plotting.set_axis(ax_invol_r)
    utils_plotting.set_axis(ax_invol_l)
    utils_plotting.set_axis(ax_CDRS[0])
    utils_plotting.set_axis(ax_CDRS[1])
    utils_plotting.set_axis(ax_CDRS[2])

    plt.savefig(directory + "/recording_session.png", dpi=300)

