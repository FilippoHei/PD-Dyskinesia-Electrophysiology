import pyvista as pv 
import pyvista
from pyvista import demos
from pyvista import examples 
import pandas as pd
import numpy as np
import scipy.io
import sys
from scipy.interpolate import Rbf
import open3d as o3d
import pymesh

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')
from lib_data import DATA_IO
import utils_io


# load atlases
mesh = pv.read('cortex_righ.vtk')  # Replace with your VTK file path

# Set up the plotter
plotter = pv.Plotter()

# Add the mesh to the plotter
plotter.add_mesh(mesh, color='white', opacity=0.01, specular=1)

# Set background and camera
plotter.set_background('white')
plotter.camera_position = 'iso'

# Show the plot
plotter.show()