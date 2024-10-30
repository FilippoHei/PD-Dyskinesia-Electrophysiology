import pandas as pd
import numpy as np
import pyvista as pv
import scipy.io
import sys

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')
from lib_data import DATA_IO

import utils_io, utils_plotting_stn

# load STN motor channels MNI coordinates

MNI_LFP_channels = pd.read_pickle(DATA_IO.path_coordinates + "MNI_LFP_motor_channels.pkl")

LFP_PSD          = utils_io.load_LFP_event_PSD(event_category="tapping", event_laterality="controlateral")
feature_set      = ['pre_event_theta_mean', 'pre_event_beta_low_mean', 'pre_event_beta_high_mean', 'pre_event_gamma_mean',
                    'event_theta_mean', 'event_beta_low_mean', 'event_beta_high_mean', 'event_gamma_mean',
                    'post_event_theta_mean', 'post_event_beta_low_mean', 'post_event_beta_high_mean', 'post_event_gamma_mean']
STN_dynamics     = utils_plotting_stn.measure_mean_psd_activity_for_LFP_channel(LFP_PSD, MNI_LFP_channels, feature_set)
STN_mesh         = utils_io.load_STN_meshes()

feature  = "gamma_mean"
radius   = 2

for severity in list(LFP_PSD.keys()):
    for segment in ["pre_event","event","post_event"]:
        stn_activity_mesh = utils_plotting_stn.map_LFP_activity_to_stn_by_radius_v2(STN_dynamics.copy(),
                                                                                 STN_mesh["right_hemisphere"].copy(),
                                                                                 feature=segment+"_"+feature, 
                                                                                 severity=severity, 
                                                                                 radius=radius)
        utils_plotting_stn.plot_LFP_activity_distribution(STN_mesh["right_hemisphere"], 
                                                          stn_activity_mesh,
                                                          feature=segment+"_"+feature, 
                                                          cmap="viridis", 
                                                          clim=[-50, 50],
                                                          file_path=DATA_IO.path_figure + "LFP_Maps/" + severity + "/")