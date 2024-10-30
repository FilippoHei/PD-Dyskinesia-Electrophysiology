import os
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from scipy import signal

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io, utils_misc, utils_cortical_atlas
from lib_data import DATA_IO

# load ECoG channel MNI coordinates
MNI_ECoG_channels       = pd.read_pickle(DATA_IO.path_coordinates + "MNI_ECoG_channels.pkl")

# load high resolution AAL3 cortex mesh
cortex_mesh             = utils_io.load_cortical_atlas_meshes()

# load AAL3 cortex parcellation objects
AAL3_image, AAL3_labels = utils_io.load_AAL3_files_for_cortical_parcellation()

# map ECoG channels on top of cortical surface
MNI_ECoG_channels       = utils_cortical_atlas.map_ECOG_channels_on_cortical_surface(MNI_ECoG_channels, cortex_mesh)

# parcellate mapped ECoG channels
MNI_ECoG_channels       = utils_cortical_atlas.parcellate_ECoG_channels_to_cortical_areas(AAL3_image, AAL3_labels, MNI_ECoG_channels)

color_cortical_regions = {}
color_cortical_regions['Motor cortex']  = "red"
color_cortical_regions['Sensory cortex'] = "blue"
color_cortical_regions['Prefrontal cortex'] = "green"
color_cortical_regions['Parietal cortex'] = "yellow"

import pyvista as pv 
plotter      = pv.Plotter()
plotter.add_mesh(cortex_mesh["right_hemisphere"], color='dimgray', opacity=0.1, specular=5, specular_power=50)
plotter.add_mesh(cortex_mesh["left_hemisphere"], color='dimgray', opacity=0.1, specular=5, specular_power=50)

for index, row in MNI_ECoG_channels.iterrows():
    plotter.add_mesh(pv.Sphere(radius=1, center=[row.x, row.y, row.z]), color=color_cortical_regions[row.AAL3_cortex], smooth_shading=True)
    
plotter.background_color = "white"
_ = plotter.add_axes(line_width=5, labels_off=True)
plotter.view_xy()
# Add a light source from the Z-axis
plotter.add_light(pv.Light(position=(0, 0, 1), color='white', intensity=0.5))
plotter.show()