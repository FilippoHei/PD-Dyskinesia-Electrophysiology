"""
Miscellaneous utilisation functions
"""

import os
import pandas as pd
import numpy as np
import pyvista as pv 
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


def nuclei_in_out(nucleus, MNI_LFP_channels):
    
    point_cloud = nucleus.points
    in_stn      = []
    
    for index, row in MNI_LFP_channels.iterrows():
        point         = np.asarray([row.x,row.y,row.z])
        hull          = ConvexHull(point_cloud)
        hull_delaunay = Delaunay(point_cloud)
        in_stn.append(hull_delaunay.find_simplex(point) >= 0)

    MNI_LFP_channels["in_stn"] = in_stn
    MNI_LFP_channels = MNI_LFP_channels[MNI_LFP_channels.in_stn==True]
    MNI_LFP_channels.reset_index(inplace=True)
    return MNI_LFP_channels
    
def measure_mean_psd_activity_for_LFP_channel(df_LFP_PSD, df_LFP_channel_coordinates, feature_set):
    
    LFP_dynamics_dict = {}

    # iterate across all features in feature set
    for feature in feature_set:
        
        channel_dynamics = df_LFP_channel_coordinates.copy()
        
        #iterate across all dyskinesia severity conditions
        for severity in list(df_LFP_PSD.keys()):
            
            dyskinesia_group = df_LFP_PSD[severity]
            values           = []
            
            # iterate across all LFP channels
            for index, row in channel_dynamics.iterrows():
                values.append(np.nanmedian(dyskinesia_group[(dyskinesia_group.patient == row.patient) & 
                                                          (dyskinesia_group.LFP_hemisphere == row.hemisphere) & 
                                                          (dyskinesia_group.LFP_channel == row.channel)][feature]))
            
            channel_dynamics[severity] = values
            
        # map left_hemisphere coordinates into right hemisphere
        channel_dynamics['x']  = channel_dynamics['x'].apply(lambda x: -x if x < 0 else x) 
        LFP_dynamics_dict[feature] = channel_dynamics

    return LFP_dynamics_dict
    

##########################################################################
##########################################################################
###########################################################################

def map_electrophysiological_activity_to_anatomical_surface(grid_activity, mesh, function='gaussian', epsilon=1.0):
    
    mesh_activity = mesh.copy()
    grid_points   = grid_activity[['x', 'y', 'z']].values
    grid_values   = {'theta': grid_activity['R2_theta'].values,'beta_low': grid_activity['R2_beta_low'].values,
                     'beta_high': grid_activity['R2_beta_high'].values, 'gamma': grid_activity['R2_gamma'].values}
    
    # build a KDTree for efficient nearest neighbor search
    grid_tree     = cKDTree(grid_points)
    mesh_points   = mesh_activity.points
    
    # for each point in the STN mesh, find the closest grid bin center point and set the R2 value of the closest bin center
    # to points on STN surface. Then apply RBF function with Gaussian kernel to map the R2 values on top of STN mesh surface
    distances, closest_indices = grid_tree.query(mesh_points)
    
    # apply RBF interpolation using the grid points and their values
    for key, values in grid_values.items():
        
        # fit the RBF with grid points and values
        rbf = Rbf(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], values, function=function, epsilon=epsilon)
        # interpolate the values for the mesh points
        mesh_activity.point_arrays[key] = rbf(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2])
    
    return mesh_activity


def plot_LFP_activity_distribution(stn_mesh, activity_mesh, feature, cmap="viridis", clim=(-50, 150), file_path=""):
    plotter = pv.Plotter()

    # Plot the stn mesh with the corresponding scalars and alpha values
    plotter.add_mesh(stn_mesh, color='white', opacity=1, specular=5, specular_power=50)

    # Define scalar values and transparency
    scalars = activity_mesh[feature]
    alpha   = np.where(scalars == 0, 0.0, 1)  # Set alpha to 0 for values equal to 0
    plotter.add_mesh(activity_mesh, scalars=feature, cmap=cmap, clim=clim, opacity=alpha)
    plotter.background_color = "white"
    plotter.add_axes(line_width=5, labels_off=True)
    plotter.camera_position = (-1.0, -1.0, 1.0)
   
    plotter.save_graphic(file_path + feature + ".svg")  
    
    plotter.show(jupyter_backend='trame')
    return plotter

def plot_cortical_activity_distribution(anatomical_structure_mesh, activity_mesh, feature, cmap="viridis", clim=(-50, 150), file_path=""):

    plotter = pv.Plotter()

    plotter.add_mesh(anatomical_structure_mesh, color='dimgray', opacity=0.025, specular=5, specular_power=50)
    
    # Define scalar values and transparency
    scalars = activity_mesh[feature]
    alpha   = np.where(scalars <= 0.01, 0.0, 1)  # Set alpha to 0 for values <=0.01
    
    plotter.add_mesh(activity_mesh, scalars=feature, cmap=cmap, clim=clim, opacity=alpha)
    plotter.background_color = "white"
    plotter.add_axes(line_width=5, labels_off=False)
    plotter.view_xy()
    plotter.save_graphic(file_path + feature + ".svg")  
    plotter.show(jupyter_backend='trame')

    return plotter

