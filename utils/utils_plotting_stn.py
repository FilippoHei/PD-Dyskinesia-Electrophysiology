"""
Miscellaneous utilisation functions
"""
import os
import pandas as pd
import numpy as np
import pyvista as pv 
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull, Delaunay

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
    
def map_LFP_activity_to_stn_by_radius(LFP_activity, stn_mesh, feature, severity, radius):
    
    # extract points from stn mesh
    stn_points = stn_mesh.copy().points

    # extract LFP points and values
    LFP_activity_filtered   = LFP_activity.copy()[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    LFP_channel_coordinates = LFP_activity_filtered[['x', 'y', 'z']].values
    LFP_feature_values      = LFP_activity_filtered[severity].values

    # create a mask for points within the radius
    mask = np.zeros(stn_points.shape[0], dtype=bool)

    for LFP_coord in LFP_channel_coordinates:
        # Calculate distance from the current LFP coordinate to all stn points
        distances = np.linalg.norm(stn_points - LFP_coord, axis=1)
        mask |= (distances < radius)

    # interpolate only for the points within the mask
    rbf = Rbf(LFP_channel_coordinates[:, 0], LFP_channel_coordinates[:, 1], LFP_channel_coordinates[:, 2], 
              LFP_feature_values, function='gaussian')
    
    # initialize the feature in the stn mesh with zeros
    stn_mesh[feature] = np.zeros(stn_points.shape[0])  # or any other default value

    # map LFP feature values onto the cortical surface for the points within the mask
    interpolated_values        = rbf(stn_points[mask, 0], stn_points[mask, 1], stn_points[mask, 2])
    stn_mesh[feature][mask] = interpolated_values

    # get min and max from the original LFP data
    min_val = LFP_feature_values.min()
    max_val = LFP_feature_values.max()

    # scale interpolated values to the original min and max
    if interpolated_values.max() != interpolated_values.min():  # Avoid division by zero
        stn_mesh[feature][mask] = min_val + (interpolated_values - interpolated_values.min()) * (max_val - min_val) / (interpolated_values.max() - interpolated_values.min())
    else:
        stn_mesh[feature][mask] = np.full(interpolated_values.shape, min_val)  # if all interpolated values are the same

    return stn_mesh

def map_LFP_activity_to_stn_by_radius_v2(LFP_activity, stn_mesh, feature, severity, radius):
    
    # Extract 3D points from the stn mesh (all points, not just surface)
    stn_points = stn_mesh.copy().points

    # Extract LFP points (3D coordinates and values for the given feature/severity)
    LFP_activity_filtered   = LFP_activity.copy()[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    LFP_channel_coordinates = LFP_activity_filtered[['x', 'y', 'z']].values
    LFP_feature_values      = LFP_activity_filtered[severity].values

    # Initialize mask to identify points within the radius
    mask = np.zeros(stn_points.shape[0], dtype=bool)

    # Vectorize the distance calculation: Calculate the distance of every STN point to each LFP coordinate
    for LFP_coord in LFP_channel_coordinates:
        distances = np.linalg.norm(stn_points - LFP_coord, axis=1)
        mask |= (distances < radius)  # Update the mask for points within the radius

    # Apply Radial Basis Function interpolation (for all points in the volume)
    rbf = Rbf(LFP_channel_coordinates[:, 0], LFP_channel_coordinates[:, 1], LFP_channel_coordinates[:, 2], 
              LFP_feature_values, function='gaussian')
    
    # Initialize the feature field within the STN volume with zeros (or another default value)
    stn_mesh[feature] = np.zeros(stn_points.shape[0])

    # Interpolate only for points within the mask (inside the specified radius)
    interpolated_values = rbf(stn_points[mask, 0], stn_points[mask, 1], stn_points[mask, 2])
    stn_mesh[feature][mask] = interpolated_values

    # Get min and max values from the original LFP data
    min_val = LFP_feature_values.min()
    max_val = LFP_feature_values.max()

    # Scale interpolated values back to the original LFP value range (optional, for consistency)
    if interpolated_values.max() != interpolated_values.min():  # Avoid division by zero
        stn_mesh[feature][mask] = min_val + (interpolated_values - interpolated_values.min()) * \
                                  (max_val - min_val) / (interpolated_values.max() - interpolated_values.min())
    else:
        stn_mesh[feature][mask] = np.full(interpolated_values.shape, min_val)  # if all values are the same

    return stn_mesh

    
def plot_LFP_activity_distribution(stn_mesh, activity_mesh, feature, cmap="viridis", clim=(-50, 150), file_path=""):
    plotter = pv.Plotter()

    # Plot the stn mesh with the corresponding scalars and alpha values
    plotter.add_mesh(stn_mesh, color='dimgray', opacity=0.025, specular=5, specular_power=50)

    # Define scalar values and transparency
    scalars = activity_mesh[feature]
    alpha   = np.where(scalars == 0, 0.0, 1)  # Set alpha to 0 for values equal to 0
    plotter.add_mesh(activity_mesh, scalars=feature, cmap=cmap, clim=clim, opacity=alpha)
    plotter.background_color = "white"
    plotter.add_axes(line_width=5, labels_off=True)
    plotter.view_yz()

    if not os.path.exists(file_path): os.makedirs(file_path)  
    plotter.save_graphic(file_path + feature + ".svg")  
    
    plotter.show(jupyter_backend='trame')