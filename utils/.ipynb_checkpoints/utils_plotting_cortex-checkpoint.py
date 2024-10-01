"""
Miscellaneous utilisation functions
"""
import os
import pandas as pd
import numpy as np
import pyvista as pv 
from scipy.interpolate import Rbf

def measure_mean_psd_activity_for_ECoG_channel(df_ECOG_PSD, df_ECOG_channel_coordinates, feature_set):
    
    ECOG_dynamics_dict = {}

    # iterate across all features in feature set
    for feature in feature_set:
        
        channel_dynamics = df_ECOG_channel_coordinates.copy()
        
        #iterate across all dyskinesia severity conditions
        for severity in list(df_ECOG_PSD.keys()):
            
            dyskinesia_group = df_ECOG_PSD[severity]
            values           = []
            
            # iterate across all ECoG channels
            for index, row in channel_dynamics.iterrows():
                values.append(np.nanmedian(dyskinesia_group[(dyskinesia_group.patient == row.patient) & 
                                                          (dyskinesia_group.ECoG_hemisphere == row.hemisphere) & 
                                                          (dyskinesia_group.ECoG_channel == row.channel)][feature]))
            
            channel_dynamics[severity] = values
            
        # map left_hemisphere coordinates into right hemisphere
        channel_dynamics['x']  = channel_dynamics['x'].apply(lambda x: -x if x < 0 else x) 
        ECOG_dynamics_dict[feature] = channel_dynamics

    return ECOG_dynamics_dict
    
def map_ecog_activity_to_cortex(ecog_activity, cortex_mesh, feature, severity):
    """
    Maps ECoG data onto a brain surface using interpolation.

    Parameters:
    - ecog_activity: DataFrame containing position of ECoG channels in MNI space ['x', 'y', 'z'] and corresponding
                     electrophysiological features observed in these channels in standard frequency bands
    - cortex_mesh  : PyVista PolyData object representing the cortical atlas mesh.

    Returns:
    - cortex_mesh  : PyVista PolyData object representing the cortical atlas mesh with interpolated feature values.
    """
    # extract points from cortex mesh
    cortex_points = cortex_mesh.points

    # extract ECoG points and values
    ecog_activity_filtered   = ecog_activity[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    ecog_channel_coordinates = ecog_activity_filtered[['x', 'y', 'z']].values
    ecog_feature_values      = ecog_activity_filtered[severity].values
    
    # interpolate linearly the feature values using RBF
    rbf = Rbf(ecog_channel_coordinates[:, 0], ecog_channel_coordinates[:, 1], ecog_channel_coordinates[:, 2], 
              ecog_feature_values, function='gaussian')

    # map the ECoG feature values onto the cortical surface
    cortex_mesh[feature] = rbf(cortex_points[:, 0], cortex_points[:, 1], cortex_points[:, 2])

    return cortex_mesh


def map_ecog_activity_to_cortex_by_radius(ecog_activity, cortex_mesh, feature, severity, radius):
    """
    Maps ECoG data onto a brain surface using interpolation,
    limited to areas within a specified radius from ECoG coordinates.

    Parameters:
    - ecog_activity: DataFrame containing position of ECoG channels in MNI space ['x', 'y', 'z'] and corresponding
                     electrophysiological features observed in these channels in standard frequency bands.
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh.
    - feature: The specific feature to map onto the cortex.
    - severity: The column indicating the severity or intensity of the feature values.
    - radius: The maximum distance from ECoG points for interpolation.

    Returns:
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh with interpolated feature values.
    """
    # Extract points from cortex mesh
    cortex_points = cortex_mesh.points

    # Extract ECoG points and values
    ecog_activity_filtered = ecog_activity[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    ecog_channel_coordinates = ecog_activity_filtered[['x', 'y', 'z']].values
    ecog_feature_values = ecog_activity_filtered[severity].values

    # Create a mask for points within the radius
    mask = np.zeros(cortex_points.shape[0], dtype=bool)

    for ecog_coord in ecog_channel_coordinates:
        # Calculate distance from the current ECoG coordinate to all cortex points
        distances = np.linalg.norm(cortex_points - ecog_coord, axis=1)
        mask |= (distances < radius)

    # Interpolate only for the points within the mask
    rbf = Rbf(ecog_channel_coordinates[:, 0], ecog_channel_coordinates[:, 1], ecog_channel_coordinates[:, 2], 
              ecog_feature_values, function='gaussian')
    
    # Map ECoG feature values onto the cortical surface
    cortex_mesh[feature] = np.zeros(cortex_points.shape[0])  # Initialize with zeros or any other default value
    cortex_mesh[feature][mask] = rbf(cortex_points[mask, 0], cortex_points[mask, 1], cortex_points[mask, 2])

    return cortex_mesh
    
def map_ecog_activity_to_cortex_by_radius_v2(ecog_activity, cortex_mesh, feature, severity, radius):
    """
    Maps ECoG data onto a brain surface using interpolation,
    limited to areas within a specified radius from ECoG coordinates,
    and scales interpolated values to match the original range.

    Parameters:
    - ecog_activity: DataFrame containing position of ECoG channels in MNI space ['x', 'y', 'z'] and corresponding
                     electrophysiological features observed in these channels in standard frequency bands.
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh.
    - feature: The specific feature to map onto the cortex.
    - severity: The column indicating the severity or intensity of the feature values.
    - radius: The maximum distance from ECoG points for interpolation.

    Returns:
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh with interpolated feature values.
    """
    # Extract points from cortex mesh
    cortex_points = cortex_mesh.copy().points

    # Extract ECoG points and values
    ecog_activity_filtered   = ecog_activity.copy()[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    ecog_channel_coordinates = ecog_activity_filtered[['x', 'y', 'z']].values
    ecog_feature_values      = ecog_activity_filtered[severity].values

    # Create a mask for points within the radius
    mask = np.zeros(cortex_points.shape[0], dtype=bool)

    for ecog_coord in ecog_channel_coordinates:
        # Calculate distance from the current ECoG coordinate to all cortex points
        distances = np.linalg.norm(cortex_points - ecog_coord, axis=1)
        mask |= (distances < radius)

    # Interpolate only for the points within the mask
    rbf = Rbf(ecog_channel_coordinates[:, 0], ecog_channel_coordinates[:, 1], ecog_channel_coordinates[:, 2], 
              ecog_feature_values, function='linear')
    
    # Initialize the feature in the cortex mesh with zeros
    cortex_mesh[feature] = np.zeros(cortex_points.shape[0])  # or any other default value

    # Map ECoG feature values onto the cortical surface for the points within the mask
    interpolated_values        = rbf(cortex_points[mask, 0], cortex_points[mask, 1], cortex_points[mask, 2])
    cortex_mesh[feature][mask] = interpolated_values

    # Get min and max from the original ECoG data
    min_val = ecog_feature_values.min()
    max_val = ecog_feature_values.max()

    # Scale interpolated values to the original min and max
    if interpolated_values.max() != interpolated_values.min():  # Avoid division by zero
        cortex_mesh[feature][mask] = min_val + (interpolated_values - interpolated_values.min()) * (max_val - min_val) / (interpolated_values.max() - interpolated_values.min())
    else:
        cortex_mesh[feature][mask] = np.full(interpolated_values.shape, min_val)  # If all interpolated values are the same

    return cortex_mesh
    
def plot_ECoG_activity_distribution(activity_mesh, feature, cmap="viridis", clim=(-50, 150)):
    
    plotter = pv.Plotter()
    plotter.add_mesh(activity_mesh, color="white", scalars=feature, cmap=cmap, clim=clim)
    # scalar_bar_args = {'title': feature, 'color': 'black'}
    # plotter.add_scalar_bar(**scalar_bar_args)
    plotter.background_color = "white"
    _ = plotter.add_axes(line_width=5, labels_off=True)
    plotter.view_xy()
    plotter.show(jupyter_backend='trame')

    
def plot_ECoG_activity_distribution_v2(cortex_mesh, activity_mesh, feature, cmap="viridis", clim=(-50, 150), file_path=""):
    plotter = pv.Plotter()

    # Plot the cortex mesh with the corresponding scalars and alpha values
    plotter.add_mesh(cortex_mesh, color='dimgray', opacity=0.025, specular=5, specular_power=50)
    
    # Define scalar values and transparency
    scalars = activity_mesh[feature]
    alpha   = np.where(scalars == 0, 0.0, 1)  # Set alpha to 0 for values equal to 0
    plotter.add_mesh(activity_mesh, scalars=feature, cmap=cmap, clim=clim, opacity=alpha)
    plotter.background_color = "white"
    plotter.add_axes(line_width=5, labels_off=True)
    plotter.view_xy()

    if not os.path.exists(file_path): os.makedirs(file_path)  
    plotter.save_graphic(file_path + feature + ".svg")  
    
    plotter.show(jupyter_backend='trame')