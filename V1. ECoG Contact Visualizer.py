import pyvista as pv 

import pandas as pd
import numpy as np
import scipy.io
import sys

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')
import utils_io
from lib_data import DATA_IO

# load ECoG contact coordinates
MNI_coordinates      = pd.read_csv(DATA_IO.path_coordinates + "contact_coordinates.csv")
MNI_ECoG_coordinates = MNI_coordinates[MNI_coordinates.recording_type == "ecog"]


cortex_mesh  = utils_io.load_cortical_atlas_meshes()
cortex_right = cortex_mesh["right_hemisphere"]
cortex_left  = cortex_mesh["left_hemisphere"]


plotter      = pv.Plotter()
plotter.add_mesh(cortex_right, color='dimgray', opacity=0.025, specular=5, specular_power=50)
plotter.add_mesh(cortex_left, color='dimgray', opacity=0.025, specular=5, specular_power=50)

for index, contact in MNI_ECoG_coordinates.iterrows():
    plotter.add_mesh(pv.Sphere(radius=1, center=[contact.x, contact.y, contact.z]), color="darkorange", smooth_shading=True)

plotter.background_color = "white"
_ = plotter.add_axes(line_width=5, labels_off=True)
plotter.view_xy()
# Add a light source from the Z-axis
plotter.add_light(pv.Light(position=(0, 0, 1), color='white', intensity=0.5))
plotter.show()
