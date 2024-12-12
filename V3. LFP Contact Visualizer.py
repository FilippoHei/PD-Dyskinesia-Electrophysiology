
# Import public packages and functions
import pandas as pd
import numpy as np
import pyvista as pv
import sys


import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io

from lib_data import DATA_IO

def is_point_inside_convex_hull(nucleus, point):
    from scipy.spatial import ConvexHull, Delaunay
    
    # Convert PyVista PolyData to numpy array of points
    point_cloud = nucleus.points
    point = np.asarray(point)
    hull = ConvexHull(point_cloud)
    hull_delaunay = Delaunay(point_cloud)
    is_inside = hull_delaunay.find_simplex(point) >= 0
    return is_inside

# load LFP channel MNI coordinates
MNI_LFP_channels = pd.read_pickle(DATA_IO.path_coordinates + "MNI_LFP_all_channels.pkl")

# load dataframe of LFP tapping events
LFP_PSD          = utils_io.load_LFP_event_PSD(event_category="tapping", event_laterality="controlateral")

STN_mesh         = utils_io.load_STN_meshes()


plotter = pv.Plotter()

# Plot the cortex mesh with the corresponding scalars and alpha values
plotter.add_mesh(STN_mesh["right_hemisphere"], color='dimgray', opacity=0.05, specular=10, specular_power=50)
plotter.add_mesh(STN_mesh["left_hemisphere"], color='dimgray', opacity=0.05, specular=10, specular_power=50)
plotter.background_color = "white"
plotter.add_axes(line_width=5, labels_off=True)

for index, contact in MNI_LFP_channels.iterrows():
    
    contact_coor = [contact.x, contact.y, contact.z]
    
    if(contact.hemisphere=="right"):
        is_inside = is_point_inside_convex_hull(STN_mesh["right_hemisphere"],contact_coor)
    else:
        is_inside = is_point_inside_convex_hull(STN_mesh["left_hemisphere"],contact_coor)
        contact_coor[0] = -1*contact_coor[0]
        
    if is_inside:
        plotter.add_mesh(pv.Sphere(radius=0.1, center=contact_coor), color="darkorange", smooth_shading=True)
    else:
        plotter.add_mesh(pv.Sphere(radius=0.1, center=contact_coor), color="white", smooth_shading=True)
        
    
plotter.view_xz()
plotter.add_light(pv.Light(position=(0, -1, 0), color='white', intensity=0.5))
plotter.show(jupyter_backend='trame')