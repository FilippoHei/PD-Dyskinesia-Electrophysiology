import pyvista as pv 
import pyvista
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import utils_MER

def get_gradient(mesh, position, color):

    z       = mesh.points[:, 2]
    z_norm  = (z - z.min()) / (z.max() - z.min())  # normalize 0→1
    r, g, b = to_rgb(color) # Convert color to RGB

    if position == "dorsal":
        scalars = z_norm                      # bottom→top gradient
        cmap = LinearSegmentedColormap.from_list("ventral", [(1,1,1), (r,g,b)])
    elif position == "ventral":
        scalars = 1 - z_norm                  # top→bottom gradient
        cmap = LinearSegmentedColormap.from_list("dorsal", [(1,1,1), (r,g,b)])
    else:  # "full"
        scalars = np.zeros_like(z_norm)       # all zeros → same color
        cmap = LinearSegmentedColormap.from_list("full", [(r,g,b), (r,g,b)])  # flat

    return scalars, cmap


def get_gradient_transparent(mesh, position, color, fade_power=3):

    min_opacity = 0.0
    max_opacity = 1.0
    z           = mesh.points[:, 2]
    z_norm      = (z - z.min()) / (z.max() - z.min())  # normalize 0→1

    if position == "dorsal":  
        opacity = z_norm ** fade_power  # slow ramp -> larger transparent area
    elif position == "ventral":  
        opacity = (1 - z_norm) ** fade_power
    else:  # "full"
        opacity = np.ones_like(z_norm)
    
    # Scale to min/max opacity
    opacity = min_opacity + (max_opacity - min_opacity) * opacity

    return to_rgb(color), opacity


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

# load basal ganglia nuclei meshes
plane           = "yz"


# color codes for basal ganglia nuclei
colors              = {}
colors["stn"]       = "lightgray"
colors["theta"]     = "#ea698bff"
colors["alpha"]     = "#b458c6ff"
colors["beta_low"]  = "#4ec1dfff"
colors["beta_high"] = "#1a659eff"
colors["gamma"]     = "#ff0a54ff"


# Define bands and their gradient positions
bands               = [("theta", "full"), ("alpha", "full"), ("beta_low", "ventral"), ("beta_high", "ventral"),("gamma", "dorsal")]
camera_positions    = {"xy": (0.0, 0.0, 1.0), "yz": (-0.6, -0.6, 1.0), "xz": (0.0, -1.0, 0.0)}



plotter = pv.Plotter(shape=(1, 5), border=False)

for i, (band, grad_position) in enumerate(bands):
    
    STN_meshes     = utils_MER.load_STN_meshes()
    STN_mesh       = STN_meshes["right"]["stn"]
    STN_SM_mesh    = STN_meshes["right"]["stn_SM"]
    color, opacity = get_gradient_transparent(STN_SM_mesh, position=grad_position, color=colors[band])
    
    plotter.subplot(0, i)
    
    # Base STN and SM meshes (semi-transparent)
    plotter.add_mesh(STN_mesh, color=colors["stn"], opacity=0.15)
    plotter.add_mesh(STN_SM_mesh, color=colors["stn"], opacity=0.01)
    
    # Gradient overlay
    plotter.add_mesh(STN_SM_mesh, color=color, opacity=opacity, show_scalar_bar=False)

    # Set camera for the chosen plane
    plotter.camera_position = camera_positions[plane]

plotter.set_background("white")
plotter.show_axes()
plotter.show()
