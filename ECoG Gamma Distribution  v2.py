import pyvista as pv 
import pyvista
from pyvista import demos
from pyvista import examples 
import pandas as pd
import numpy as np
import scipy.io
import sys
from scipy.interpolate import Rbf

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')
from lib_data import DATA_IO
import utils_io

# load dataframe of ECoG tapping events
ECOG_PSD_LID         = utils_io.load_ECoG_event_PSD(event_category="tapping")["controlateral"]["LID"]
ECOG_PSD_noLID       = utils_io.load_ECoG_event_PSD(event_category="tapping")["controlateral"]["noLID"]
ECOG_PSD_LID         = ECOG_PSD_LID[ECOG_PSD_LID.dyskinesia_arm=="mild"]
ECOG_PSD_LID.reset_index(inplace=True)

dyskinesia_group     = ECOG_PSD_LID

###############################################################################
MNI_ECoG_channels     = pd.read_pickle(DATA_IO.path_coordinates + "MNI_ECoG_channels.pkl")
ECoG_channel_dynamics = MNI_ECoG_channels.copy()
###############################################################################
feature = "event_beta_low_mean"
values  = []
for index, row in ECoG_channel_dynamics.iterrows():
    values.append(dyskinesia_group[(dyskinesia_group.patient == row.patient) &
                                   (dyskinesia_group.ECoG_hemisphere == row.hemisphere) & 
                                   (dyskinesia_group.ECoG_channel == row.channel)][feature].mean())
ECoG_channel_dynamics[feature] = values
ECoG_channel_dynamics['x'] = ECoG_channel_dynamics['x'].apply(lambda x: -x if x < 0 else x)
ECoG_channel_dynamics = ECoG_channel_dynamics[ECoG_channel_dynamics[feature].notna()]
###############################################################################
mesh = pv.read('cortex_right.vtk')  # Replace with your VTK file path
###############################################################################
points  = ECoG_channel_dynamics[['x', 'y', 'z']].values
dynamic = pv.PolyData(points)
dynamic[feature] = ECoG_channel_dynamics[feature].values
###############################################################################
def map_ecog_to_brain(ecog_df, brain_surface,feature):
    """
    Maps ECoG data onto a brain surface using interpolation.

    Parameters:
    - ecog_df: DataFrame containing ECoG data with columns ['x', 'y', 'z', 'event_gamma_mean'].
    - brain_surface: PyVista PolyData object representing the cortical atlas mesh.

    Returns:
    - brain_cloud: PyVista PolyData object with interpolated event_gamma_mean values.
    """
    # Step 1: Extract brain surface points
    brain_points = brain_surface.points

    # Step 2: Extract ECoG points and values
    ecog_points = ecog_df[['x', 'y', 'z']].values
    ecog_values = ecog_df[feature].values

    # Step 3: Interpolate the event_gamma_mean values using RBF
    rbf = Rbf(ecog_points[:, 0], ecog_points[:, 1], ecog_points[:, 2], ecog_values, function='linear')

    # Step 4: Map the ECoG values onto the brain surface points
    brain_surface[feature] = rbf(brain_points[:, 0], brain_points[:, 1], brain_points[:, 2])

    return brain_surface

brain_cloud_with_ecog = map_ecog_to_brain(ECoG_channel_dynamics, mesh, feature)
###############################################################################
plotter        = pv.Plotter()
#plotter.add_mesh(cortex_right, color="silver", opacity = 0.05, name="right cortex", style='surface', specular=1.0, specular_power=20, smooth_shading=True) 
#plotter.add_mesh(cortex_left, color="silver", opacity = 0.05, name="right left", style='surface', specular=1.0, specular_power=20, smooth_shading=True) 

plotter.add_mesh(brain_cloud_with_ecog, scalars=feature, cmap='viridis', clim=[-50, 150])
scalar_bar_args = {
    'title': feature,
    'color': 'black'  # Set the label color to black
}
plotter.add_scalar_bar(**scalar_bar_args)


plotter.background_color = "white"
_ = plotter.add_axes(line_width=5, labels_off=True)
plotter.view_xy()
plotter.show()
