"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# inserting the lib folder to the compiler
import sys
sys.path.insert(0, './lib')

import utils_plotting

def plot_spectogram(spectogram, time_vector, title, normalization_type, cbar, file_path, figure_name):

    cm           = 1/2.54 # centimeters in inches
    fig, ax_spec = plt.subplots(figsize=(8*cm, 5*cm))
    
    if(normalization_type=="z-score"):
        c = ax_spec.pcolormesh(time_vector, np.linspace(4, 90, 87), spectogram, shading='gouraud', vmin=-5, vmax=5)
    elif(normalization_type=="percent"):
        c = ax_spec.pcolormesh(time_vector, np.linspace(4, 90, 87), spectogram, shading='gouraud', vmin=-50, vmax=100)

    if(cbar==True):
        colorbar = plt.colorbar(c, ax=ax_spec, label=normalization_type)
    ax_spec.set_xlabel('time (s)', fontsize=utils_plotting.LABEL_SIZE_label)
    ax_spec.set_ylabel('frequency (Hz)', fontsize=utils_plotting.LABEL_SIZE_label)
    ax_spec.set_title(title, fontsize=utils_plotting.LABEL_SIZE_label)
    utils_plotting.set_axis(ax_spec)

    full_path = file_path + figure_name

    # Check if the directory exists; if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"Created directory: {file_path}")

    plt.savefig(full_path + ".png", dpi=300)
    
def plot_patient_channels_cv(df_channel_cv, patient, figure_name):
    patient_left_channel_cv     = df_channel_cv[(df_channel_cv.patient==patient) & (df_channel_cv.hemisphere=="left")]
    patient_left_channel_cv.cv  = np.abs(patient_left_channel_cv.cv)
    patient_left_channel_cv     = patient_left_channel_cv[(patient_left_channel_cv.frequency <=35) | (patient_left_channel_cv.frequency >=60)]
    
    patient_right_channel_cv    = df_channel_cv[(df_channel_cv.patient==patient) & (df_channel_cv.hemisphere=="right")]
    patient_right_channel_cv.cv = np.abs(patient_right_channel_cv.cv)
    patient_right_channel_cv    = patient_right_channel_cv[(patient_right_channel_cv.frequency <=35) | (patient_right_channel_cv.frequency >=60)]
    
    # pivot the dataframe to get heatmap matrix
    heatmap_l_data = pd.pivot_table(patient_left_channel_cv, values='cv', index='frequency', columns='channel')
    heatmap_l_data = heatmap_l_data.iloc[::-1] # inverse the order of frequencies in index
    
    heatmap_r_data = pd.pivot_table(patient_right_channel_cv, values='cv', index='frequency', columns='channel')
    heatmap_r_data = heatmap_r_data.iloc[::-1] # inverse the order of frequencies in index
    
    # start plotting
    plt         = utils_plotting.get_figure_template()
    
    ax_left_g   = plt.subplot2grid((77, 66), (0, 0)  , colspan=30, rowspan=4)
    ax_left_bh  = plt.subplot2grid((77, 66), (4, 0)  , colspan=30, rowspan=4)
    ax_left_bl  = plt.subplot2grid((77, 66), (8, 0)  , colspan=30, rowspan=4)
    ax_left_a   = plt.subplot2grid((77, 66), (12, 0) , colspan=30, rowspan=4)
    ax_left_t   = plt.subplot2grid((77, 66), (16, 0) , colspan=30, rowspan=4)
    
    ax_right_g  = plt.subplot2grid((77, 66), (0, 36) , colspan=30, rowspan=4)
    ax_right_bh = plt.subplot2grid((77, 66), (4, 36) , colspan=30, rowspan=4)
    ax_right_bl = plt.subplot2grid((77, 66), (8, 36) , colspan=30, rowspan=4)
    ax_right_a  = plt.subplot2grid((77, 66), (12, 36), colspan=30, rowspan=4)
    ax_right_t  = plt.subplot2grid((77, 66), (16, 36), colspan=30, rowspan=4)
    
    # LEFT HEMISPHERE CHANNELS
    ax_left_g  = sns.heatmap(data=heatmap_l_data[heatmap_l_data.index >= 60], vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_left_g)
    ax_left_g.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_left_g.get_ylim()
    ax_left_g.set_yticks([y_min-4, y_max+2])
    ax_left_g.set_yticklabels(['60', '90'])
    
    ax_left_bh = sns.heatmap(data=heatmap_l_data[(heatmap_l_data.index >= 20) & (heatmap_l_data.index <= 35)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_left_bh)
    ax_left_bh.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_left_bh.get_ylim()
    ax_left_bh.set_yticks([y_min-2, y_max+2])
    ax_left_bh.set_yticklabels(['20', '35'])
    
    ax_left_bl = sns.heatmap(data=heatmap_l_data[(heatmap_l_data.index >= 12) & (heatmap_l_data.index <= 20)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_left_bl)
    ax_left_bl.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_left_bl.get_ylim()
    ax_left_bl.set_yticks([y_min-1.25, y_max+1])
    ax_left_bl.set_yticklabels(['12', '20'])
    
    ax_left_a = sns.heatmap(data=heatmap_l_data[(heatmap_l_data.index >= 8) & (heatmap_l_data.index <= 12)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_left_a)
    ax_left_a.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_left_a.get_ylim()
    ax_left_a.set_yticks([y_min-0.5, y_max+0.4])
    ax_left_a.set_yticklabels(['8', '12'])
    
    ax_left_t = sns.heatmap(data=heatmap_l_data[(heatmap_l_data.index >= 4) & (heatmap_l_data.index <= 8)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_left_t)
    ax_left_t.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_left_t.get_ylim()
    ax_left_t.set_yticks([y_min-0.5, y_max+0.4])
    ax_left_t.set_yticklabels(['4', '8'])
    ax_left_t.set_xticklabels(ax_left_t.get_xticklabels(), fontsize=utils_plotting.LABEL_SIZE, rotation=90)
    
    
    # RIGHT HEMISPHERE CHANNELS
    ax_right_g  = sns.heatmap(data=heatmap_r_data[heatmap_r_data.index >= 60], vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_right_g)
    ax_right_g.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_right_g.get_ylim()
    ax_right_g.set_yticks([y_min-4, y_max+2])
    ax_right_g.set_yticklabels(['60', '90'])
    
    ax_right_bh = sns.heatmap(data=heatmap_r_data[(heatmap_r_data.index >= 20) & (heatmap_r_data.index <= 35)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_right_bh)
    ax_right_bh.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_right_bh.get_ylim()
    ax_right_bh.set_yticks([y_min-2, y_max+2])
    ax_right_bh.set_yticklabels(['20', '35'])
    
    ax_right_bl = sns.heatmap(data=heatmap_r_data[(heatmap_r_data.index >= 12) & (heatmap_r_data.index <= 20)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_right_bl)
    ax_right_bl.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_right_bl.get_ylim()
    ax_right_bl.set_yticks([y_min-1.25, y_max+1])
    ax_right_bl.set_yticklabels(['12', '20'])
    
    ax_right_a = sns.heatmap(data=heatmap_r_data[(heatmap_r_data.index >= 8) & (heatmap_r_data.index <= 12)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_right_a)
    ax_right_a.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_right_a.get_ylim()
    ax_right_a.set_yticks([y_min-0.5, y_max+0.4])
    ax_right_a.set_yticklabels(['8', '12'])
    
    ax_right_t = sns.heatmap(data=heatmap_r_data[(heatmap_r_data.index >= 4) & (heatmap_r_data.index <= 8)], 
                             vmin=0, vmax=1, cmap="Reds", cbar=False, ax=ax_right_t)
    ax_right_t.set_ylabel("", fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    y_min, y_max = ax_right_t.get_ylim()
    ax_right_t.set_yticks([y_min-0.5, y_max+0.4])
    ax_right_t.set_yticklabels(['4', '8'])
    ax_right_t.set_xticklabels(ax_right_t.get_xticklabels(), fontsize=utils_plotting.LABEL_SIZE, rotation=90)
    
    ax_left_g.set_title("left hemisphere", fontsize=utils_plotting.LABEL_SIZE)
    ax_right_g.set_title("right hemisphere", fontsize=utils_plotting.LABEL_SIZE)
    
    utils_plotting.set_axis(ax_left_g)
    utils_plotting.set_axis(ax_left_bh)
    utils_plotting.set_axis(ax_left_bl)
    utils_plotting.set_axis(ax_left_a)
    utils_plotting.set_axis(ax_left_t)
    utils_plotting.set_axis(ax_right_g)
    utils_plotting.set_axis(ax_right_bh)
    utils_plotting.set_axis(ax_right_bl)
    utils_plotting.set_axis(ax_right_a)
    utils_plotting.set_axis(ax_right_t)

    plt.savefig(figure_name + ".png", dpi=300)
    plt.savefig(figure_name + ".svg", dpi=300)



    