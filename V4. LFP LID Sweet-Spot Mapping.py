# Import public packages and functions
import os
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
import pyvista as pv

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io, utils_spatial_analysis, utils_statistics, utils_plotting_spatial_mapping

from lib_data import DATA_IO

###############################################################################

# load LFP-PSD dataframe
LFP_PSD_controlateral = utils_io.load_LFP_event_PSD(event_category="tapping", event_laterality="controlateral") 

# concat all different dyskinesia severity dataframe into single dataframe
LFP_PSD = pd.DataFrame()
for severity in LFP_PSD_controlateral.keys():
    LFP_PSD = pd.concat([LFP_PSD, LFP_PSD_controlateral[severity]], ignore_index=True)

LFP_PSD["severity"] = LFP_PSD['dyskinesia_arm']

# Based on event onset time, assign no-LID events into noLID_noDOPA and noLID_DOPA groups
LFP_PSD.loc[(LFP_PSD['event_start_time'] < 30)  & (LFP_PSD['severity'] == "none"), 'severity'] = 'noLID_noDOPA'
LFP_PSD.loc[(LFP_PSD['event_start_time'] >= 30) & (LFP_PSD['severity'] == "none"), 'severity'] = 'noLID_DOPA'

# arrange the dataframe
LFP_PSD['hemisphere'] = LFP_PSD['LFP_hemisphere']
LFP_PSD['channel']    = LFP_PSD['LFP_channel']
LFP_PSD = LFP_PSD[['patient', 'hemisphere', 'channel', 
                   'pre_event_theta_mean', 'pre_event_alpha_mean', 
                   'pre_event_beta_low_mean','pre_event_beta_high_mean','pre_event_gamma_mean',
                   'event_theta_mean', 'event_alpha_mean', 
                   'event_beta_low_mean','event_beta_high_mean','event_gamma_mean', 
                   'post_event_theta_mean', 'post_event_alpha_mean',
                   'post_event_beta_low_mean','post_event_beta_high_mean','post_event_gamma_mean',
                   'severity']]

features = ['pre_event_theta_mean', 'pre_event_alpha_mean', 'pre_event_beta_low_mean',
            'pre_event_beta_high_mean','pre_event_gamma_mean',
            'event_theta_mean', 'event_alpha_mean', 'event_beta_low_mean',
            'event_beta_high_mean','event_gamma_mean', 
            'post_event_theta_mean', 'post_event_alpha_mean',
            'post_event_beta_low_mean','post_event_beta_high_mean','post_event_gamma_mean']

# load STN mesh
STN_mesh         = utils_io.load_STN_meshes() 
# load LFP channels MNI coordinates
MNI_LFP_channels = pd.read_pickle(DATA_IO.path_coordinates + "MNI_LFP_motor_channels.pkl") 
# create a 3D grid around STN
n_bins           = 4
STN_grid         = utils_spatial_analysis.create_3D_grid_around_anatomical_structure(STN_mesh["right_hemisphere"], n_bins=n_bins)
# find the position of each channel in these 3D grid across x,y,z axes
MNI_LFP_channels = utils_spatial_analysis.assign_recording_channels_to_grid_cells(MNI_LFP_channels, STN_grid)

# Merge information of STN grid cells into the LFP_PSD dataframe
# merge the data frames on 'patient', 'hemisphere', and 'channel'
LFP_PSD = pd.merge(LFP_PSD, 
                   MNI_LFP_channels[['patient', 'hemisphere', 'channel', 'grid_bin_x', 'grid_bin_y', 'grid_bin_z']],
                   on=['patient', 'hemisphere', 'channel'], 
                   how='left')
LFP_PSD = LFP_PSD.dropna(subset=['grid_bin_x', 'grid_bin_y', 'grid_bin_z'])
LFP_PSD["severity_numeric"] = LFP_PSD['severity'].map({'noLID_noDOPA':0, 'noLID_DOPA':1, 'mild':2, 'moderate':3})
LFP_PSD.reset_index(drop=True, inplace=True)

###############################################################################

df_R2_theta     = utils_statistics.measure_adjusted_r2_in_grid_cells_for_frequency_bands(pd.DataFrame(LFP_PSD), n_bins=n_bins,
                                                                                         frequency_band="theta", target_feature="severity_numeric")
df_R2_beta_low  = utils_statistics.measure_adjusted_r2_in_grid_cells_for_frequency_bands(pd.DataFrame(LFP_PSD), n_bins=n_bins,
                                                                                         frequency_band="beta_low", target_feature="severity_numeric")
df_R2_beta_high = utils_statistics.measure_adjusted_r2_in_grid_cells_for_frequency_bands(pd.DataFrame(LFP_PSD), n_bins=n_bins,
                                                                                         frequency_band="beta_high", target_feature="severity_numeric")
df_R2_gamma     = utils_statistics.measure_adjusted_r2_in_grid_cells_for_frequency_bands(pd.DataFrame(LFP_PSD), n_bins=n_bins,
                                                                                         frequency_band="gamma", target_feature="severity_numeric")

grid_bin_center_coordianates = utils_spatial_analysis.extract_grid_cell_centers(STN_grid, n_bins)
STN_dynamics                 = pd.DataFrame(grid_bin_center_coordianates)

STN_dynamics                 = STN_dynamics.merge(df_R2_theta, left_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], right_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], how='left')
STN_dynamics['R2_theta']     = STN_dynamics['adjusted_r2']
STN_dynamics                 = STN_dynamics.drop(columns=['adjusted_r2','frequency_band'])

STN_dynamics                 = STN_dynamics.merge(df_R2_beta_low, left_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], right_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], how='left')
STN_dynamics['R2_beta_low']  = STN_dynamics['adjusted_r2']
STN_dynamics                 = STN_dynamics.drop(columns=['adjusted_r2','frequency_band'])

STN_dynamics                 = STN_dynamics.merge(df_R2_beta_high, left_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], right_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], how='left')
STN_dynamics['R2_beta_high'] = STN_dynamics['adjusted_r2']
STN_dynamics                 = STN_dynamics.drop(columns=['adjusted_r2','frequency_band'])

STN_dynamics                 = STN_dynamics.merge(df_R2_gamma, left_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], right_on=['grid_bin_x', 'grid_bin_y','grid_bin_z'], how='left')
STN_dynamics['R2_gamma']     = STN_dynamics['adjusted_r2']
STN_dynamics                 = STN_dynamics.drop(columns=['adjusted_r2','frequency_band'])

STN_dynamics["x"]            = STN_dynamics["grid_bin_center_x"]
STN_dynamics["y"]            = STN_dynamics["grid_bin_center_y"]
STN_dynamics["z"]            = STN_dynamics["grid_bin_center_z"]

STN_dynamics.replace(np.nan, 0, inplace=True)

band              = "gamma"
stn_activity_mesh = utils_plotting_spatial_mapping.map_electrophysiological_activity_to_anatomical_surface(STN_dynamics, STN_mesh["right_hemisphere"], function='gaussian', epsilon=2)
plotter           = utils_plotting_spatial_mapping.plot_LFP_activity_distribution(STN_mesh["right_hemisphere"], stn_activity_mesh, feature=band, 
                                                                                  cmap="Reds", clim=[0.0, 0.3], file_path=DATA_IO.path_figure + "LFP_Maps/")

plotter.view_yz()

plotter.show()

