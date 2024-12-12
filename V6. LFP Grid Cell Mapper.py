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

import utils_io, utils_spatial_analysis, utils_statistics

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

STN_dynamics                 = pd.DataFrame(MNI_LFP_channels)

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

grid_bin_center_coordianates = utils_spatial_analysis.extract_grid_cell_centers(STN_grid, n_bins)

###############################################################################
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
#STN_dynamics                 = STN_dynamics.loc[~(STN_dynamics[['R2_theta', 'R2_beta_low', 'R2_beta_high', 'R2_gamma']] == 0).all(axis=1)]

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb

def colormap_to_hex(cmap, value):
    
    if not 0 <= value <= 0.5:
        raise ValueError("Value must be between 0 and 0.5.")

    try:
        colormap = plt.get_cmap(cmap)
    except ValueError:
        raise ValueError(f"'{cmap}' is not a valid colormap name.")
    
    normalized_value = value * 2
    rgba_color       = colormap(normalized_value)
    
    # Convert RGBA to HEX
    hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255),int(rgba_color[2] * 255),)
    return hex_color



def create_colormap(color, name='custom_colormap'):
    rgb_color = to_rgb(color)
    colors = [(1, 1, 1), rgb_color]  # From white to the target color
    return LinearSegmentedColormap.from_list(name, colors)

def value_to_hex(value, vmin, vmax, colormap):
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized_value = norm(value)
    
    # Get the RGB color from the colormap
    rgb_color = colormap(normalized_value)  # Returns (r, g, b, alpha)
    
    # Convert to hex
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_color[0] * 255),
        int(rgb_color[1] * 255),
        int(rgb_color[2] * 255)
    )

def plot_grid_cell(plotter, STN_mesh, grid, n_bins, x_id, y_id, z_id, color):
    
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    
    cell_edges_in_x_axis = np.linspace(xmin, xmax, n_bins+1)
    cell_edges_in_y_axis = np.linspace(ymin, ymax, n_bins+1)
    cell_edges_in_z_axis = np.linspace(zmin, zmax, n_bins+1)
    
    cell_x_edge_low      = cell_edges_in_x_axis[x_id]
    cell_x_edge_high     = cell_edges_in_x_axis[x_id+1]
    cell_y_edge_low      = cell_edges_in_y_axis[y_id]
    cell_y_edge_high     = cell_edges_in_y_axis[y_id+1]
    cell_z_edge_low      = cell_edges_in_z_axis[z_id]
    cell_z_edge_high     = cell_edges_in_z_axis[z_id+1]

    #section = STN_mesh.clip_box(bounds=(cell_x_edge_low, cell_x_edge_high, 
    #                                    cell_y_edge_low, cell_y_edge_high, 
    #                                    cell_z_edge_low, cell_z_edge_high), invert=False)
    section = pv.Box(bounds=(cell_x_edge_low, cell_x_edge_high, 
                                        cell_y_edge_low, cell_y_edge_high, 
                                        cell_z_edge_low, cell_z_edge_high))
    plotter.add_mesh(section, color=color, opacity=0.5, show_edges=False)





def plot_STN_cells(plotter, dataset, STN_mesh, STN_grid, n_bins, cmap):
    custom_cmap = create_colormap("#EC096F")
    
    for index, row in dataset.iterrows():
        plot_grid_cell(plotter, STN_mesh, STN_grid, n_bins, row.grid_bin_x, row.grid_bin_y, row.grid_bin_z, 
                       value_to_hex(row.adjusted_r2, 0, 0.5, custom_cmap))


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

plotter = pv.Plotter()
plotter.background_color = "white"

plotter.add_axes(line_width=5, labels_off=False, color="black")
# Plot the cortex mesh with the corresponding scalars and alpha values
plotter.add_mesh(STN_mesh["right_hemisphere"], color='lightgray', opacity=0.25, specular=0, specular_power=1)
plot_STN_cells(plotter, df_R2_theta, STN_mesh["right_hemisphere"], STN_grid, n_bins=n_bins, cmap="Reds")

for i,row in grid_bin_center_coordianates.iterrows():
    plotter.add_mesh(pv.Sphere(radius=0.1, center=(row.grid_bin_center_x, row.grid_bin_center_y, row.grid_bin_center_z)))
plotter.view_yz()
plotter.show(jupyter_backend='trame')

