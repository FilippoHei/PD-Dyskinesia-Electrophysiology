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
colors                                 = {}
colors["voluntary_tapping"]            = {}
colors["involuntary_movement"]         = {}

colors["voluntary_tapping"]["LID_none"]        = "#FFEA00"
colors["voluntary_tapping"]["LID_mild"]        = "#EF8A06"
colors["voluntary_tapping"]["LID_moderate"]    = "#DC2F02"
colors["voluntary_tapping"]["LID_severe"]      = "#9D0208"
colors["voluntary_tapping"]["LID_extreme"]     = "#370617"

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
        cax = ax2.pcolormesh(time_vec_spectogram, f, Sxx, shading='gouraud', vmin=-230, vmax=-150)
        ax1.set_xlim([-1,(len(event)/fs)-1])
        ax2.set_xlim([-1,(len(event)/fs)-1])
    else:
        time_vec_spectogram = (t - max(t)) + 1
        cax = ax2.pcolormesh(time_vec_spectogram, f, Sxx, shading='gouraud', vmin=-230, vmax=-150)
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


def get_average_spectogram(event_array, fs):

    """
        Description
            The method measured the average spectrogram for a given set of events based on the Short Time Fourier Series (STFS).

        Input
            event_array : A double list contains events of a particular event category/condition. Each element corresponds to an event and is 
                          represented by a list. The duration of each event in this array is the same.
            fs          : An integer value represents the sampling frequency of events.

        Output
            avg_Sxx     : A double list, it contains the average spectrogram across N event inevent_array.
            t           : A double list, time vector corresponds to the average spectrogram.
            freq        : A double list, frequency list in which average spectrogram created for
    """
    
    # measure average power spectrum
    Sxx_array = []
    for i in range(len(event_array)):
        event           = np.array(event_array[i])
        freq, time, Sxx = spectrogram(event, fs=fs, nperseg=int(fs/20)) # spectorgrams in 100ms segments
        Sxx             = 10 * np.log10(Sxx)                            # 'Power/Frequency (dB/Hz)'     
        Sxx_array.append(Sxx)
        
    avg_Sxx = np.average(Sxx_array,axis=0)
    
    return avg_Sxx, time, freq
    
def get_average_spectogram_panel(event_array, time_vector, fs, alignment, ax, ylabel, colorbar_visibility=False):

    """
        Description
            This method plots the average spectrogram of a particular event category based on a provided set of events stored in "event_array" for a given axis. 
            The event data is sampled with fs frequency and all events should already aligned based on their onset and offset.

        Input
            event_array        : A double list contains events of a particular event category/condition. Each element corresponds to an event and is 
                                 represented by a list. The duration of each event in this array is the same.
            time_vector        : A double list, the time vector belongs to events contained in the event array.
            fs                 : An integer value represents the sampling frequency of events.
            aligment           : alignment strategy ("onset" or "offset") used to align events stored in event_array.
            ax                 : Matplotlib axis, where the plotting will be applied on
            y_label            : A string, y axis label of the panel
            colorbar_visibility: Boolean, it set the visibility of color bar for spectrogram values.

        Output
            return: A panel plotted on given axis
    """
    # get average spectrogram for selected event
    average_spectogram, t, f = get_average_spectogram(event_array, fs)

    # get time vector for spectrogram and plot the spectogram
    if(alignment=="onset"):
        time_vec_spectogram = t - 1
        cax = ax.pcolormesh(time_vec_spectogram, f, average_spectogram, shading='gouraud', vmin=-230, vmax=-150)
        ax.set_xlim([-1,(len(event_array[0])/fs)-1])
    else:
        time_vec_spectogram = (t - max(t)) + 1
        cax = ax.pcolormesh(time_vec_spectogram, f, average_spectogram, shading='gouraud', vmin=-230, vmax=-150)
        ax.set_xlim([1-(len(event_array[0])/fs),1])

    # plot a vertical line at x=0 to highlight the alingment
    ax.axvline(x=0, ymin=-1, ymax=1, ls='-', color="white")

    if(colorbar_visibility==True):
        # Add the colorbar to the new axis
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("bottom", size="5%", pad=0.5)
        cbar = plt.colorbar(cax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Power/Frequency (dB/Hz)', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=LABEL_SIZE)

    ax.set_ylim([0,100])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabel)

    # adjust the axis parameters
    set_axis(ax)
    
def plot_average_spectogram_for_event_category(accelerometer_events, time_vector, fs):
    
    plt  = get_figure_template()

    ax1  = plt.subplot2grid((75, 40), (0, 0)   , colspan=18, rowspan=10)
    ax2  = plt.subplot2grid((75, 40), (0, 20)  , colspan=18, rowspan=10)
    ax3  = plt.subplot2grid((75, 40), (15, 0)  , colspan=18, rowspan=10)
    ax4  = plt.subplot2grid((75, 40), (15, 20) , colspan=18, rowspan=10)

    get_average_spectogram_panel(accelerometer_events["tapping"]["all"]["onset"], time_vector, fs, "onset", ax=ax1,
                                 ylabel='Frequency [Hz]',colorbar_visibility=False)
    
    get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["all"]["onset"], time_vector, fs,"onset", ax=ax2, 
                                 ylabel='',  colorbar_visibility=False)

    get_average_spectogram_panel(accelerometer_events["tapping"]["all"]["offset"], time_vector, fs, "offset", ax=ax3,
                                 ylabel='Frequency [Hz]', colorbar_visibility=False)
    
    get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["all"]["offset"], time_vector, fs, "offset", ax=ax4, 
                                 ylabel='', colorbar_visibility=False)

    ax1.set_title("Tapping"              , fontsize=LABEL_SIZE_title, weight="bold")
    ax2.set_title("Involuntary Movements", fontsize=LABEL_SIZE_title, weight="bold")

def plot_average_spectogram_for_dyskinesia_severity(accelerometer_events, time_vector, fs):
    
    plt  = get_figure_template()
    ax1  = plt.subplot2grid((75, 40), (0, 0)   , colspan=15, rowspan=10)
    ax2  = plt.subplot2grid((75, 40), (0, 20)  , colspan=15, rowspan=10)

    # Tapping
    get_average_spectogram_panel(accelerometer_events["tapping"]["LID_none"]["onset"], 
                                 time_vector, fs, "onset", 
                                 ax=ax1, ylabel='Frequency [Hz]', colorbar_visibility=False)

    if(len(accelerometer_events["tapping"]["LID_mild"]["onset"])!=0):
        ax3  = plt.subplot2grid((75, 40), (15, 0)  , colspan=15, rowspan=10)
        ax3.set_title("Tapping [mild LID] ", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["tapping"]["LID_mild"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax3, ylabel='Frequency [Hz]', colorbar_visibility=False)

    if(len(accelerometer_events["tapping"]["LID_moderate"]["onset"])!=0):
        ax5  = plt.subplot2grid((75, 40), (30, 0)  , colspan=15, rowspan=10)
        ax5.set_title("Tapping [moderate LID] ", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["tapping"]["LID_moderate"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax5, ylabel='Frequency [Hz]', colorbar_visibility=False)

    if(len(accelerometer_events["tapping"]["LID_severe"]["onset"])!=0):
        ax7  = plt.subplot2grid((75, 40), (45, 0)  , colspan=15, rowspan=10)
        ax7.set_title("Tapping [severe LID] ", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["tapping"]["LID_severe"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax7, ylabel='Frequency [Hz]', colorbar_visibility=False)

    if(len(accelerometer_events["tapping"]["LID_extreme"]["onset"])!=0):
        ax9  = plt.subplot2grid((75, 40), (60, 0)  , colspan=15, rowspan=10)
        ax9.set_title("Tapping [extreme LID] ", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["tapping"]["LID_extreme"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax9, ylabel='Frequency [Hz]', colorbar_visibility=False)
    
    # involuntary_movement
    
    get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["LID_none"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax2, ylabel='Frequency [Hz]', colorbar_visibility=False)
    
    if(len(accelerometer_events["involuntary_movement"]["LID_mild"]["onset"])!=0):
        ax4  = plt.subplot2grid((75, 40), (15, 20) , colspan=15, rowspan=10)
        ax4.set_title("Involuntary Movements [mild LID]", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["LID_mild"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax4, ylabel='Frequency [Hz]', colorbar_visibility=False)
        

    if(len(accelerometer_events["involuntary_movement"]["LID_moderate"]["onset"])!=0):
        ax6  = plt.subplot2grid((75, 40), (30, 20) , colspan=15, rowspan=10)
        ax6.set_title("Involuntary Movements [moderate LID]", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["LID_moderate"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax6, ylabel='Frequency [Hz]', colorbar_visibility=False)

    if(len(accelerometer_events["involuntary_movement"]["LID_severe"]["onset"])!=0):
        ax8  = plt.subplot2grid((75, 40), (45, 20) , colspan=15, rowspan=10)
        ax8.set_title("Involuntary Movements [severe LID]", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["LID_severe"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax8, ylabel='Frequency [Hz]', colorbar_visibility=False)

    if(len(accelerometer_events["involuntary_movement"]["LID_extreme"]["onset"])!=0):
        ax10 = plt.subplot2grid((75, 40), (60, 20) , colspan=15, rowspan=10)
        ax10.set_title("Involuntary Movements [extreme LID]", fontsize=LABEL_SIZE, weight="bold")
        get_average_spectogram_panel(accelerometer_events["involuntary_movement"]["LID_extreme"]["onset"], 
                                     time_vector, fs, "onset", 
                                     ax=ax10, ylabel='Frequency [Hz]', colorbar_visibility=False)

    ax1.set_title("Tapping [no LID]"              , fontsize=LABEL_SIZE, weight="bold")
    ax2.set_title("Involuntary Movements [no LID]", fontsize=LABEL_SIZE, weight="bold")