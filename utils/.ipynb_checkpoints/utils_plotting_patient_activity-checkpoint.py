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

def plot_patient_activity(EVENTS_PATIENT, LFP_PATIENT, SUB):

    # check if the personal figure directory of the patient does exist, if not create
    directory = DATA_IO.path_figure + SUB
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # get all the indices
    indices                      = {}
    t_period                     = {}
    scores                       = {}

    indices["tapping"]           = utils_accelerometer.find_event_segments_indices(EVENTS_PATIENT.period_tap)
    indices["rest"]              = utils_accelerometer.find_event_segments_indices(EVENTS_PATIENT.period_rest)
    indices["free"]              = utils_accelerometer.find_event_segments_indices(EVENTS_PATIENT.period_free)
    t_period["tapping"]          = utils_accelerometer.find_timepoint_from_indices(EVENTS_PATIENT.times, indices["tapping"] )
    t_period["rest"]             = utils_accelerometer.find_timepoint_from_indices(EVENTS_PATIENT.times, indices["rest"] )
    t_period["free"]             = utils_accelerometer.find_timepoint_from_indices(EVENTS_PATIENT.times, indices["free"] )

    # plotting part

    plt        = utils_plotting.get_figure_template()
    
    ax_task    = plt.subplot2grid((80, 40), (0, 0), colspan=40, rowspan=2)
    
    ax_CDRS    = list(range(10))
    ax_CDRS[0] = plt.subplot2grid((80, 40), (3, 0), colspan=40, rowspan=2)
    ax_CDRS[1] = plt.subplot2grid((80, 40), (5, 0), colspan=40, rowspan=2)
    ax_CDRS[2] = plt.subplot2grid((80, 40), (7, 0), colspan=40, rowspan=2)  
    
    ax_c1      = plt.subplot2grid((80, 40), (10, 0), colspan=40, rowspan=3)
    ax_c2      = plt.subplot2grid((80, 40), (13, 0), colspan=40, rowspan=3)
    ax_c3      = plt.subplot2grid((80, 40), (16, 0), colspan=40, rowspan=3)
    ax_c4      = plt.subplot2grid((80, 40), (19, 0), colspan=40, rowspan=3)
    ax_c5      = plt.subplot2grid((80, 40), (22, 0), colspan=40, rowspan=3)
    ax_c6      = plt.subplot2grid((80, 40), (25, 0), colspan=40, rowspan=3)
    ax_c7      = plt.subplot2grid((80, 40), (28, 0), colspan=40, rowspan=3)
    ax_c8      = plt.subplot2grid((80, 40), (31, 0), colspan=40, rowspan=3)
    ax_c9      = plt.subplot2grid((80, 40), (34, 0), colspan=40, rowspan=3)
    ax_c10     = plt.subplot2grid((80, 40), (37, 0), colspan=40, rowspan=3)
    ax_c11     = plt.subplot2grid((80, 40), (40, 0), colspan=40, rowspan=3)
    ax_c12     = plt.subplot2grid((80, 40), (43, 0), colspan=40, rowspan=3)
    ax_c13     = plt.subplot2grid((80, 40), (46, 0), colspan=40, rowspan=3)
    ax_c14     = plt.subplot2grid((80, 40), (49, 0), colspan=40, rowspan=3)
    ax_c15     = plt.subplot2grid((80, 40), (52, 0), colspan=40, rowspan=3)
    ax_c16     = plt.subplot2grid((80, 40), (55, 0), colspan=40, rowspan=3)
    
    # task periods
    for period in t_period["tapping"]:
        ax_task.axvspan(period[0]/60, period[1]/60, color=utils_plotting.colors["tapping"], alpha=1)
    
    for period in t_period["rest"]:
        ax_task.axvspan(period[0]/60, period[1]/60, color=utils_plotting.colors["rest"], alpha=1)
    
    for period in t_period["free"]:
        ax_task.axvspan(period[0]/60, period[1]/60, color=utils_plotting.colors["free"], alpha=1)
    
    # CDRS scores
    plot_CDRS_evolution_panel(EVENTS_PATIENT.CDRS_right_hand_indexes, EVENTS_PATIENT.CDRS_right_hand_scores, "arm", ax_CDRS[0])
    plot_CDRS_evolution_panel(EVENTS_PATIENT.CDRS_left_hand_indexes , EVENTS_PATIENT.CDRS_left_hand_scores, "arm", ax_CDRS[1])
    plot_CDRS_evolution_panel(EVENTS_PATIENT.CDRS_total_indexes     , EVENTS_PATIENT.CDRS_total_scores, "total", ax_CDRS[2])
    
    ax_task.set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_task.set_xticklabels("")
    ax_task.set_yticklabels("")
    ax_task.set_ylabel("task", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_CDRS[0].set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_CDRS[0].set_xticklabels("")
    ax_CDRS[0].set_yticklabels("")
    ax_CDRS[0].set_ylabel("CDRS right arm", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_CDRS[1].set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_CDRS[1].set_xticklabels("")
    ax_CDRS[1].set_yticklabels("")
    ax_CDRS[1].set_ylabel("CDRS left arm", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_CDRS[2].set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_CDRS[2].set_xticklabels("")
    ax_CDRS[2].set_yticklabels("")
    ax_CDRS[2].set_ylabel("CDRS total", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    # LFP contacts
    # right hand voluntary movements
    ax_vol_r = sns.lineplot(x=EVENTS_PATIENT.times/60, y=EVENTS_PATIENT.right_voluntary_movements, ax=ax_vol_r, 
                           color=utils_plotting.colors["voluntary"]["moderate"])
    
    # left hand voluntary movements
    ax_vol_l = sns.lineplot(x=EVENTS_PATIENT.times/60, y=EVENTS_PATIENT.left_voluntary_movements, ax=ax_vol_l, 
                           color=utils_plotting.colors["voluntary"]["moderate"])
    
    # right hand voluntary movements
    ax_invol_r = sns.lineplot(x=EVENTS_PATIENT.times/60, y=EVENTS_PATIENT.right_involuntary_movements, ax=ax_invol_r, 
                              color=utils_plotting.colors["involuntary"]["moderate"])
    
    # left hand voluntary movements
    ax_invol_l = sns.lineplot(x=EVENTS_PATIENT.times/60, y=EVENTS_PATIENT.left_involuntary_movements, ax=ax_invol_l, 
                              color=utils_plotting.colors["involuntary"]["moderate"])
    ax_vol_r.set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_vol_r.set_xticklabels("")
    ax_vol_r.set_yticklabels("")
    ax_vol_r.set_ylabel("right \n tapping", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_vol_l.set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_vol_l.set_xticklabels("")
    ax_vol_l.set_yticklabels("")
    ax_vol_l.set_ylabel("left \n tapping", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_invol_r.set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
    ax_invol_r.set_xticklabels("")
    ax_invol_r.set_yticklabels("")
    ax_invol_r.set_ylabel("right \n involuntary", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    ax_invol_l.set_xlim([np.min(EVENTS_PATIENT.times)/60, np.max(EVENTS_PATIENT.times)/60])
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


def plot_LFP_contact_recordings(LFP_PATIENT, SUB, hemisphere):

    # check if the personal figure directory of the patient does exist, if not create
    directory = DATA_IO.path_figure + SUB
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # plotting part

    plt        = utils_plotting.get_figure_template()
    
    ax_c1      = plt.subplot2grid((80, 40), (10, 0), colspan=40, rowspan=3)
    ax_c2      = plt.subplot2grid((80, 40), (13, 0), colspan=40, rowspan=3)
    ax_c3      = plt.subplot2grid((80, 40), (16, 0), colspan=40, rowspan=3)
    ax_c4      = plt.subplot2grid((80, 40), (19, 0), colspan=40, rowspan=3)
    ax_c5      = plt.subplot2grid((80, 40), (22, 0), colspan=40, rowspan=3)
    ax_c6      = plt.subplot2grid((80, 40), (25, 0), colspan=40, rowspan=3)
    ax_c7      = plt.subplot2grid((80, 40), (28, 0), colspan=40, rowspan=3)
    ax_c8      = plt.subplot2grid((80, 40), (31, 0), colspan=40, rowspan=3)
    ax_c9      = plt.subplot2grid((80, 40), (34, 0), colspan=40, rowspan=3)
    ax_c10     = plt.subplot2grid((80, 40), (37, 0), colspan=40, rowspan=3)
    ax_c11     = plt.subplot2grid((80, 40), (40, 0), colspan=40, rowspan=3)
    ax_c12     = plt.subplot2grid((80, 40), (43, 0), colspan=40, rowspan=3)
    ax_c13     = plt.subplot2grid((80, 40), (46, 0), colspan=40, rowspan=3)
    ax_c14     = plt.subplot2grid((80, 40), (49, 0), colspan=40, rowspan=3)
    ax_c15     = plt.subplot2grid((80, 40), (52, 0), colspan=40, rowspan=3)
    ax_c16     = plt.subplot2grid((80, 40), (55, 0), colspan=40, rowspan=3)

    # LFP contacts
    t_LFP = LFP_PATIENT.times/60

    flag_contact_8 = False

    try:
        ax_c1 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['01'], ax=ax_c1, color="dimgray")
    except:
        pass

    try:
        ax_c2 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['02'], ax=ax_c2, color="dimgray")
    except:
        pass

    try:
        ax_c3 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['03'], ax=ax_c3, color="dimgray")
    except:
        pass

    try:
        ax_c4 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['04'], ax=ax_c4, color="dimgray")
    except:
        pass

    try:
        ax_c5 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['05'], ax=ax_c5, color="dimgray")
    except:
        pass

    try:
        ax_c6 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['06'], ax=ax_c6, color="dimgray")
    except:
        pass

    try:
        ax_c7 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['07'], ax=ax_c7, color="dimgray")
    except:
        pass

    try:
        ax_c8 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['08'], ax=ax_c8, color="dimgray")
    except:
        pass

    try:
        ax_c9  = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['09'], ax=ax_c9, color="dimgray")
    except:
        flag_contact_8 = True
        pass 
        
    try:
        ax_c10 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['10'], ax=ax_c10, color="dimgray")
    except:
        pass

    try:
        ax_c11 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['11'], ax=ax_c11, color="dimgray")
    except:
        pass
    
    try:
        ax_c12 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['12'], ax=ax_c12, color="dimgray")
    except:
        pass

    try:
        ax_c13 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['13'], ax=ax_c13, color="dimgray")
    except:
        pass

    try:
        ax_c14 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['14'], ax=ax_c14, color="dimgray")
    except:
        pass

    try:
        ax_c15 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['15'], ax=ax_c15, color="dimgray")
    except:
        pass
    
    try:
        ax_c16 = sns.lineplot(x=t_LFP, y=LFP_PATIENT.recordings[hemisphere]['16'], ax=ax_c16, color="dimgray")
    except:
        pass
    
    ax_c1.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c1.set_xticklabels("")
    ax_c1.set_yticklabels("")
    ax_c1.set_ylabel("C1", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    ax_c1.set_title("Patient " + SUB + " - " + hemisphere + " hemisphere", fontsize=utils_plotting.LABEL_SIZE_label)

    ax_c2.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c2.set_xticklabels("")
    ax_c2.set_yticklabels("")
    ax_c2.set_ylabel("C2", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c3.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c3.set_xticklabels("")
    ax_c3.set_yticklabels("")
    ax_c3.set_ylabel("C3", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c4.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c4.set_xticklabels("")
    ax_c4.set_yticklabels("")
    ax_c4.set_ylabel("C4", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c5.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c5.set_xticklabels("")
    ax_c5.set_yticklabels("")
    ax_c5.set_ylabel("C5", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c6.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c6.set_xticklabels("")
    ax_c6.set_yticklabels("")
    ax_c6.set_ylabel("C6", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c7.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c7.set_xticklabels("")
    ax_c7.set_yticklabels("")
    ax_c7.set_ylabel("C7", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c8.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c8.set_xticklabels("")
    ax_c8.set_yticklabels("")
    ax_c8.set_ylabel("C8", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    if(flag_contact_8 == False):
        ax_c9.set_xlim([np.min(t_LFP), np.max(t_LFP)])
        ax_c9.set_xticklabels("")
        ax_c9.set_yticklabels("")
        ax_c9.set_ylabel("C9", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    else:
        ax_c9.set_xlim([np.min(t_LFP), np.max(t_LFP)])
        ax_c9.set_yticklabels("")
        ax_c9.set_ylabel("C9", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c10.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c10.set_xticklabels("")
    ax_c10.set_yticklabels("")
    ax_c10.set_ylabel("C10", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c11.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c11.set_xticklabels("")
    ax_c11.set_yticklabels("")
    ax_c11.set_ylabel("C11", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c12.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c12.set_xticklabels("")
    ax_c12.set_yticklabels("")
    ax_c12.set_ylabel("C12", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c13.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c13.set_xticklabels("")
    ax_c13.set_yticklabels("")
    ax_c13.set_ylabel("C13", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c14.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c14.set_xticklabels("")
    ax_c14.set_yticklabels("")
    ax_c14.set_ylabel("C14", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    ax_c15.set_xlim([np.min(t_LFP), np.max(t_LFP)])
    ax_c15.set_xticklabels("")
    ax_c15.set_yticklabels("")
    ax_c15.set_ylabel("C15", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)

    if(flag_contact_8 == True):
        ax_c16.set_xlim([np.min(t_LFP), np.max(t_LFP)])
        ax_c16.set_xticklabels("")
        ax_c16.set_yticklabels("")
        ax_c16.set_ylabel("C16", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    else:
        ax_c16.set_xlim([np.min(t_LFP), np.max(t_LFP)])
        ax_c16.set_yticklabels("")
        ax_c16.set_ylabel("C16", fontsize=utils_plotting.LABEL_SIZE_label, rotation=0)
    
    utils_plotting.set_axis(ax_c1)
    utils_plotting.set_axis(ax_c2)
    utils_plotting.set_axis(ax_c3)
    utils_plotting.set_axis(ax_c4)
    utils_plotting.set_axis(ax_c5)
    utils_plotting.set_axis(ax_c6)
    utils_plotting.set_axis(ax_c7)
    utils_plotting.set_axis(ax_c8)
    utils_plotting.set_axis(ax_c9)
    utils_plotting.set_axis(ax_c10)
    utils_plotting.set_axis(ax_c11)
    utils_plotting.set_axis(ax_c12)
    utils_plotting.set_axis(ax_c13)
    utils_plotting.set_axis(ax_c14)
    utils_plotting.set_axis(ax_c15)
    utils_plotting.set_axis(ax_c16)


    plt.savefig(directory + "/patient_LFP_" + hemisphere + "_contacts.png", dpi=300)






