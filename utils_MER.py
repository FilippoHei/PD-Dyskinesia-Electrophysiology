import pandas as pd
import pickle
import pyvista as pv 
import numpy as np
import math
import nibabel as nib
from scipy.stats import spearmanr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


def load_STN_meshes():
    subthalamic_meshes                    = {}
    subthalamic_meshes["left"]            = {}
    subthalamic_meshes["right"]           = {}
    subthalamic_meshes["right"]["stn"]    = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/stn_right.vtk")
    subthalamic_meshes["left"]["stn"]     = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/stn_left.vtk")
    subthalamic_meshes["right"]["stn_SM"] = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/stn_SM_right.vtk")
    subthalamic_meshes["left"]["stn_SM"]  = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/stn_SM_left.vtk")
    return subthalamic_meshes

