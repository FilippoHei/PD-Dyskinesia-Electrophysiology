"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

LABEL_SIZE       = 5
LABEL_SIZE_label = 6 
LABEL_SIZE_title = 7 

# color dataframe
colors                                 = {}
colors["voluntary_tapping"]            = {}
colors["involuntary_tapping"]          = {}
colors["involuntary_movement"]         = {}

colors["voluntary_tapping"]["dys0"]    = "#12F1D7"
colors["voluntary_tapping"]["dys1"]    = "#00B982"
colors["voluntary_tapping"]["dys2"]    = "#02483C"
colors["voluntary_tapping"]["dys3"]    = "#00140F"

colors["involuntary_tapping"]["dys0"]  = "#C70098"
colors["involuntary_tapping"]["dys1"]  = "#8100CC"
colors["involuntary_tapping"]["dys2"]  = "#2E0074"
colors["involuntary_tapping"]["dys3"]  = "#020028"

colors["involuntary_movement"]["dys0"] = "#FFC31F"
colors["involuntary_movement"]["dys1"] = "#CC6D00"
colors["involuntary_movement"]["dys2"] = "#B50021"
colors["involuntary_movement"]["dys3"] = "#52000F"

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

def plot_accelerometer_events_by_dyskinesia(accelerometer_events, kinematics, event_category, colors, figure_name):
    
    plt  = get_figure_template()
    ax1  = plt.subplot2grid((75, 45), (0 , 0) , colspan=20, rowspan=20)
    ax2  = plt.subplot2grid((75, 45), (25, 0) , colspan=20, rowspan=20)
    ax3  = plt.subplot2grid((75, 45), (50, 0) , colspan=20, rowspan=20)
    ax4  = plt.subplot2grid((75, 45), (0,  25) , colspan=20, rowspan=20)
    ax5  = plt.subplot2grid((75, 45), (25, 25) , colspan=20, rowspan=20)
    ax6  = plt.subplot2grid((75, 45), (50, 25) , colspan=20, rowspan=20)

    # onset aligned
    plot_accelerometer_events(accelerometer_events[event_category]["dys0"]["onset"]["X"], kinematics.fs, axis=ax1, color=colors[event_category]["dys0"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys1"]["onset"]["X"], kinematics.fs, axis=ax1, color=colors[event_category]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys2"]["onset"]["X"], kinematics.fs, axis=ax1, color=colors[event_category]["dys2"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys3"]["onset"]["X"], kinematics.fs, axis=ax1, color=colors[event_category]["dys3"], padding_for="onset")
    
    plot_accelerometer_events(accelerometer_events[event_category]["dys0"]["onset"]["Y"], kinematics.fs, axis=ax2, color=colors[event_category]["dys0"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys1"]["onset"]["Y"], kinematics.fs, axis=ax2, color=colors[event_category]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys2"]["onset"]["Y"], kinematics.fs, axis=ax2, color=colors[event_category]["dys2"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys3"]["onset"]["Y"], kinematics.fs, axis=ax2, color=colors[event_category]["dys3"], padding_for="onset")
  
    plot_accelerometer_events(accelerometer_events[event_category]["dys0"]["onset"]["Z"], kinematics.fs, axis=ax3, color=colors[event_category]["dys0"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys1"]["onset"]["Z"], kinematics.fs, axis=ax3, color=colors[event_category]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys2"]["onset"]["Z"], kinematics.fs, axis=ax3, color=colors[event_category]["dys2"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys3"]["onset"]["Z"], kinematics.fs, axis=ax3, color=colors[event_category]["dys3"], padding_for="onset")
  
    # offset aligned
    plot_accelerometer_events(accelerometer_events[event_category]["dys0"]["offset"]["X"], kinematics.fs, axis=ax4, color=colors[event_category]["dys0"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys1"]["offset"]["X"], kinematics.fs, axis=ax4, color=colors[event_category]["dys1"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys2"]["offset"]["X"], kinematics.fs, axis=ax4, color=colors[event_category]["dys2"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys3"]["offset"]["X"], kinematics.fs, axis=ax4, color=colors[event_category]["dys3"], padding_for="offset")
    
    plot_accelerometer_events(accelerometer_events[event_category]["dys0"]["offset"]["Y"], kinematics.fs, axis=ax5, color=colors[event_category]["dys0"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys1"]["offset"]["Y"], kinematics.fs, axis=ax5, color=colors[event_category]["dys1"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys2"]["offset"]["Y"], kinematics.fs, axis=ax5, color=colors[event_category]["dys2"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys3"]["offset"]["Y"], kinematics.fs, axis=ax5, color=colors[event_category]["dys3"], padding_for="offset")
  
    plot_accelerometer_events(accelerometer_events[event_category]["dys0"]["offset"]["Z"], kinematics.fs, axis=ax6, color=colors[event_category]["dys0"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys1"]["offset"]["Z"], kinematics.fs, axis=ax6, color=colors[event_category]["dys1"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys2"]["offset"]["Z"], kinematics.fs, axis=ax6, color=colors[event_category]["dys2"], padding_for="offset")
    plot_accelerometer_events(accelerometer_events[event_category]["dys3"]["offset"]["Z"], kinematics.fs, axis=ax6, color=colors[event_category]["dys3"], padding_for="offset")
    
    ax1.set_ylabel("Accelerometer X Axis", fontsize=LABEL_SIZE_title)
    ax2.set_ylabel("Accelerometer Y Axis", fontsize=LABEL_SIZE_title)
    ax3.set_ylabel("Accelerometer Z Axis", fontsize=LABEL_SIZE_title)
    
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax3.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax4.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax5.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax6.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    
    ax1.set_xlim([-1,3])
    ax2.set_xlim([-1,3])
    ax3.set_xlim([-1,3])
    ax4.set_xlim([-3,1])
    ax5.set_xlim([-3,1])
    ax6.set_xlim([-3,1])
    
    # aling y axes
    ax1.set_ylim([min(ax1.get_ylim()[0], ax4.get_ylim()[0]), max(ax1.get_ylim()[1], ax4.get_ylim()[1])])
    ax2.set_ylim([min(ax2.get_ylim()[0], ax5.get_ylim()[0]), max(ax2.get_ylim()[1], ax5.get_ylim()[1])])
    ax3.set_ylim([min(ax3.get_ylim()[0], ax6.get_ylim()[0]), max(ax3.get_ylim()[1], ax6.get_ylim()[1])])
    ax4.set_ylim([min(ax1.get_ylim()[0], ax4.get_ylim()[0]), max(ax1.get_ylim()[1], ax4.get_ylim()[1])])
    ax5.set_ylim([min(ax2.get_ylim()[0], ax5.get_ylim()[0]), max(ax2.get_ylim()[1], ax5.get_ylim()[1])])
    ax6.set_ylim([min(ax3.get_ylim()[0], ax6.get_ylim()[0]), max(ax3.get_ylim()[1], ax6.get_ylim()[1])])
    
    set_axis(ax1)
    set_axis(ax2)
    set_axis(ax3)
    set_axis(ax4)
    set_axis(ax5)
    set_axis(ax6)
    
    ax4.set_yticklabels("")
    ax5.set_yticklabels("")
    ax6.set_yticklabels("")
    
    ax1.set_title("Onset Aligned", fontsize= LABEL_SIZE_label, weight="bold")
    ax4.set_title("Offset Aligned", fontsize= LABEL_SIZE_label, weight="bold")

    # add legend
    c_dys0 = mpatches.Patch(color=colors[event_category]["dys0"], label='dys=0')
    c_dys1 = mpatches.Patch(color=colors[event_category]["dys1"], label='dys=1')
    c_dys2 = mpatches.Patch(color=colors[event_category]["dys2"], label='dys=2')
    c_dys3 = mpatches.Patch(color=colors[event_category]["dys3"], label='dys=3')
    ax5.legend(handles=[c_dys0, c_dys1, c_dys2, c_dys3], prop={'size': LABEL_SIZE_title}, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)


def plot_accelerometer_events_by_category(accelerometer_events, kinematics, colors, figure_name):
    
    plt  = get_figure_template()
    
    ax1  = plt.subplot2grid((75, 40), (0, 0) , colspan=12, rowspan=20)
    ax2  = plt.subplot2grid((75, 40), (25, 0) , colspan=12, rowspan=20)
    ax3  = plt.subplot2grid((75, 40), (50, 0) , colspan=12, rowspan=20)
    ax4  = plt.subplot2grid((75, 40), (0, 14) , colspan=12, rowspan=20)
    ax5  = plt.subplot2grid((75, 40), (25, 14) , colspan=12, rowspan=20)
    ax6  = plt.subplot2grid((75, 40), (50, 14) , colspan=12, rowspan=20)
    ax7  = plt.subplot2grid((75, 40), (0, 28) , colspan=12, rowspan=20)
    ax8  = plt.subplot2grid((75, 40), (25, 28) , colspan=12, rowspan=20)
    ax9  = plt.subplot2grid((75, 40), (50, 28) , colspan=12, rowspan=20)

    # onset aligned
    cat  = "voluntary_tapping"
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["X"], kinematics.fs, axis=ax1, color=colors[cat]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["Y"], kinematics.fs, axis=ax2, color=colors[cat]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["Z"], kinematics.fs, axis=ax3, color=colors[cat]["dys1"], padding_for="onset")

    cat  = "involuntary_tapping"
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["X"], kinematics.fs, axis=ax4, color=colors[cat]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["Y"], kinematics.fs, axis=ax5, color=colors[cat]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["Z"], kinematics.fs, axis=ax6, color=colors[cat]["dys1"], padding_for="onset")
    
    cat  = "involuntary_movement"
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["X"], kinematics.fs, axis=ax7, color=colors[cat]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["Y"], kinematics.fs, axis=ax8, color=colors[cat]["dys1"], padding_for="onset")
    plot_accelerometer_events(accelerometer_events[cat]["all"]["onset"]["Z"], kinematics.fs, axis=ax9, color=colors[cat]["dys1"], padding_for="onset")

    set_axis(ax1)
    set_axis(ax2)
    set_axis(ax3)
    set_axis(ax4)
    set_axis(ax5)
    set_axis(ax6)
    set_axis(ax7)
    set_axis(ax8)
    set_axis(ax9)

    ax1.set_title("Voluntary Tapping"    , fontsize=LABEL_SIZE_title, weight="bold")
    ax4.set_title("Involuntary Tapping"  , fontsize=LABEL_SIZE_title, weight="bold")
    ax7.set_title("Involuntary Movement" , fontsize=LABEL_SIZE_title, weight="bold")
    ax1.set_ylabel("Accelerometer X Axis", fontsize=LABEL_SIZE_title)
    ax2.set_ylabel("Accelerometer Y Axis", fontsize=LABEL_SIZE_title)
    ax3.set_ylabel("Accelerometer Z Axis", fontsize=LABEL_SIZE_title)
    
    ax4.set_yticks([])
    ax7.set_yticks([])
    ax5.set_yticks([])
    ax8.set_yticks([])
    ax6.set_yticks([])
    ax9.set_yticks([])
    
    ax1.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax2.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax3.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax4.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax5.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax6.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax7.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax8.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    ax9.axvline(x=0, ymin=-1, ymax=1, ls='--', color="grey")
    
    ax1.set_xlim([-1,3])
    ax2.set_xlim([-1,3])
    ax3.set_xlim([-1,3])
    ax4.set_xlim([-1,3])
    ax5.set_xlim([-1,3])
    ax6.set_xlim([-1,3])
    ax7.set_xlim([-1,3])
    ax8.set_xlim([-1,3])
    ax9.set_xlim([-1,3])
    
    ax1.set_ylim([-5e-07, 5e-07])
    ax4.set_ylim([-5e-07, 5e-07])
    ax7.set_ylim([-5e-07, 5e-07])
    ax2.set_ylim([-5e-07, 5e-07])
    ax5.set_ylim([-5e-07, 5e-07])
    ax8.set_ylim([-5e-07, 5e-07])
    ax3.set_ylim([-5e-07, 5e-07])
    ax6.set_ylim([-5e-07, 5e-07])
    ax9.set_ylim([-5e-07, 5e-07])
    
    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)
