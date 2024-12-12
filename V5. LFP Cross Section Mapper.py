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
import matplotlib.colors as mcolors
import scikit_posthocs as sp
import matplotlib.cm as cm

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io, utils_misc, utils_spatial_analysis, utils_mutual_information

from lib_data import DATA_IO
from lib_Subcortical_Atlases import Distal_Atlas

###############################################################################

# load LFP-PSD dataframe
LFP_PSD_controlateral = utils_io.load_LFP_event_PSD(event_category="tapping", event_laterality="controlateral") 

LFP_PSD = pd.DataFrame()
for severity in LFP_PSD_controlateral.keys():
    LFP_PSD = pd.concat([LFP_PSD, LFP_PSD_controlateral[severity]], ignore_index=True)
    
LFP_PSD["severity"] = LFP_PSD['dyskinesia_arm']
LFP_PSD.loc[(LFP_PSD['event_start_time'] < 30)  & (LFP_PSD['severity'] == "none"), 'severity'] = 'noLID_noDOPA'
LFP_PSD.loc[(LFP_PSD['event_start_time'] >= 30) & (LFP_PSD['severity'] == "none"), 'severity'] = 'noLID_DOPA'

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

# load STN mesh
STN_mesh         = utils_io.load_STN_meshes() 
# load LFP channels MNI coordinates
MNI_LFP_channels = pd.read_pickle(DATA_IO.path_coordinates + "MNI_LFP_motor_channels.pkl") 
# create 3D grid around STN
STN_grid         = utils_spatial_analysis.create_3D_grid_around_anatomical_structure(STN_mesh["right_hemisphere"], n_bins=10)
# find the position of each channel in these 3D grid across x,y,z axes
MNI_LFP_channels = utils_spatial_analysis.assign_recording_channels_to_grid_cells(MNI_LFP_channels, STN_grid)

# merge information of STN grid cells into LFP_PSD dataframe
# merge the dataFrames on 'patient', 'hemisphere', and 'channel'
LFP_PSD = pd.merge(LFP_PSD, 
                   MNI_LFP_channels[['patient', 'hemisphere', 'channel', 'grid_bin_x', 'grid_bin_y', 'grid_bin_z']],
                   on=['patient', 'hemisphere', 'channel'], 
                   how='left')
LFP_PSD = LFP_PSD.dropna(subset=['grid_bin_x', 'grid_bin_y', 'grid_bin_z'])
LFP_PSD["severity_numeric"] = LFP_PSD['severity'].map({'noLID_noDOPA':0, 'noLID_DOPA':1, 'mild':2, 'moderate':3})
LFP_PSD.reset_index(drop=True, inplace=True)


# selected feature

neural_features = ['pre_event_theta_mean', 'pre_event_alpha_mean', 
                   'pre_event_beta_low_mean','pre_event_beta_high_mean','pre_event_gamma_mean',
                   'event_theta_mean', 'event_alpha_mean', 
                   'event_beta_low_mean','event_beta_high_mean','event_gamma_mean', 
                   'post_event_theta_mean', 'post_event_alpha_mean',
                   'post_event_beta_low_mean','post_event_beta_high_mean','post_event_gamma_mean']

MI = []
for feature in neural_features:
    MI_x       = utils_mutual_information.measure_mutual_information_along_axis(LFP_PSD, feature=feature, axis="x")
    MI_y       = utils_mutual_information.measure_mutual_information_along_axis(LFP_PSD, feature=feature, axis="y")
    MI_z       = utils_mutual_information.measure_mutual_information_along_axis(LFP_PSD, feature=feature, axis="z")
    MI_feature = pd.concat([MI_x, MI_y, MI_z], ignore_index=True)
    
    if(len(MI)==0):
        MI = pd.DataFrame(MI_feature)
    else:
        MI = pd.concat([MI, MI_feature], ignore_index=True)

# get the total MI scores across all features
MI = MI.groupby(["axis","cell_id"])[["MI"]].sum()
MI = MI.reset_index()

# Normalize MI values to the range [0, 1]
MI['normalized_MI'] = (MI['MI'] - MI['MI'].min()) / (MI['MI'].max() - MI['MI'].min()) 
# Map normalized MI to colors and convert to hex
MI['color'] = [cm.get_cmap('Reds')(value) for value in MI['normalized_MI']]
MI['color'] = MI['color'].apply(lambda rgba: f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}')


###############################################################################

def plot_specific_grid_rectangle(plotter, STN_mesh, grid, axis, cell_id, n_bin, color):
    
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    
    if(axis=="x"):
        cell_edges_in_axis = np.linspace(xmin, xmax, n_bin+1)
    elif(axis=="y"):
        cell_edges_in_axis = np.linspace(ymin, ymax, n_bin+1)
    elif(axis=="z"):
        cell_edges_in_axis = np.linspace(zmin, zmax, n_bin+1)
    
    cell_edge_low  = cell_edges_in_axis[cell_id]
    cell_edge_high = cell_edges_in_axis[cell_id+1]

    if(axis=="x"):
        filtered_mesh = STN_mesh.clip_box(bounds=(cell_edge_low, cell_edge_high, -float('inf'), float('inf'), -float('inf'), float('inf')), invert=False, crinkle=False)
        plotter.add_mesh(filtered_mesh, color=color, opacity=0.999)
    elif(axis=="y"):
        filtered_mesh = STN_mesh.clip_box(bounds=(-float('inf'), float('inf'), cell_edge_low, cell_edge_high, -float('inf'), float('inf')), invert=False, crinkle=False)
        plotter.add_mesh(filtered_mesh, color=color, opacity=0.999)
    elif(axis=="z"):
        filtered_mesh = STN_mesh.clip_box(bounds=(-float('inf'), float('inf'), -float('inf'), float('inf'), cell_edge_low, cell_edge_high), invert=False, crinkle=False)
        plotter.add_mesh(filtered_mesh, color=color, opacity=0.999)

        
        
plotter = pv.Plotter()

# Plot the cortex mesh with the corresponding scalars and alpha values
plotter.add_mesh(STN_mesh["right_hemisphere"], color='lightgray', opacity=0.05, specular=0, specular_power=1)

for index, row in MI[MI.axis=="y"].iterrows():
    plot_specific_grid_rectangle(plotter, STN_mesh=STN_mesh["right_hemisphere"], grid=STN_grid.copy(), axis=row.axis, cell_id=row.cell_id, n_bin=10, color=row.color)

    
plotter.background_color = "white"
plotter.add_axes(line_width=5, labels_off=False, color="black")

plotter.view_xz()
plotter.show(jupyter_backend='trame')