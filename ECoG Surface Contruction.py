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
cortical_atlas = scipy.io.loadmat('atlases/AAL3/CortexHiRes.mat')


cloud   = pv.PolyData(cortical_atlas["Vertices_rh"])
volume  = cloud.delaunay_3d(alpha=0.5, progress_bar=True)
volume.save('cortex_righ.vtk')

shell = volume.extract_geometry()
shell.plot()