"""
Power spectral utilisation functions
"""

import pandas as pd
import numpy as np
import pyvista as pv

def create_3D_grid_around_anatomical_structure(structure, n_bins):
    
    n_bins += 1
    
    # define the grid boundaries for given 3D anatomical structure (pyvista mesh)
    xmin, xmax, ymin, ymax, zmin, zmax = structure.bounds
    
    # calculate spacing based on bounds and the desired number of bins
    x_spacing = (xmax - xmin) / (n_bins - 1)
    y_spacing = (ymax - ymin) / (n_bins - 1)
    z_spacing = (zmax - zmin) / (n_bins - 1)
    
    # create a grid of points
    x_points = np.linspace(xmin, xmax, n_bins)
    y_points = np.linspace(ymin, ymax, n_bins)
    z_points = np.linspace(zmin, zmax, n_bins)
    
    # create the lines for each axis to show full grid lines
    lines = []
    
    # generate lines along the z-axis at each x-y slice
    for x in x_points:
        for y in y_points:
            points = np.array([[x, y, z] for z in z_points])
            line = pv.Line(points[0], points[-1], resolution=n_bins-1)
            lines.append(line)
    
    # generate lines along the x-axis at each y-z slice
    for y in y_points:
        for z in z_points:
            points = np.array([[x, y, z] for x in x_points])
            line = pv.Line(points[0], points[-1], resolution=n_bins-1)
            lines.append(line)
    
    # generate lines along the y-axis at each x-z slice
    for x in x_points:
        for z in z_points:
            points = np.array([[x, y, z] for y in y_points])
            line = pv.Line(points[0], points[-1], resolution=n_bins-1)
            lines.append(line)
    
    # create a single PolyData object from the lines
    grid = pv.PolyData()
    for line in lines:
        grid = grid.merge(line)

    return grid


def find_grid_cell_index_for_contact(point, grid):
    
    # extract bounds from the grid lines PolyData
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    
    # calculate the grid dimensions based on the lines created
    nx        = len(np.unique(grid.points[:, 0]))  # unique x-coordinates
    ny        = len(np.unique(grid.points[:, 1]))  # unique y-coordinates
    nz        = len(np.unique(grid.points[:, 2]))  # unique z-coordinates
    
    # calculate the cell size
    x_spacing = (xmax - xmin) / (nx - 1)
    y_spacing = (ymax - ymin) / (ny - 1)
    z_spacing = (zmax - zmin) / (nz - 1)

    # calculate the cell index along each axis
    x_index = int((point[0] - xmin) / x_spacing)
    y_index = int((point[1] - ymin) / y_spacing)
    z_index = int((point[2] - zmin) / z_spacing)
    
    # ensure the indices are within the grid bounds
    x_index = min(max(x_index, 0), nx - 1)
    y_index = min(max(y_index, 0), ny - 1)
    z_index = min(max(z_index, 0), nz - 1)
    
    return (x_index, y_index, z_index)

def assign_recording_channels_to_grid_cells(df_MNI_coordinates, grid):
    
    grid_bin_x = []
    grid_bin_y = []
    grid_bin_z = []
    
    for index, row in df_MNI_coordinates.iterrows():
        channel_coordinates = [row.x, row.y, row.z]
        cell_index          = find_grid_cell_index_for_contact(channel_coordinates, grid)
        grid_bin_x.append(cell_index[0])
        grid_bin_y.append(cell_index[1])
        grid_bin_z.append(cell_index[2])
    
    df_MNI_coordinates["grid_bin_x"] = grid_bin_x
    df_MNI_coordinates["grid_bin_y"] = grid_bin_y
    df_MNI_coordinates["grid_bin_z"] = grid_bin_z

    return df_MNI_coordinates

def extract_grid_cell_centers(grid, n_bins):
    
    # extract bounds from the grid lines PolyData
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    
    # compute bin widths
    dx = (xmax - xmin) / n_bins
    dy = (ymax - ymin) / n_bins
    dz = (zmax - zmin) / n_bins
    
    # compute centers along each axis
    x_centers = xmin + (np.arange(n_bins) + 0.5) * dx
    y_centers = ymin + (np.arange(n_bins) + 0.5) * dy
    z_centers = zmin + (np.arange(n_bins) + 0.5) * dz

    # generate 3D grid of centers and cell IDs
    x_grid, y_grid, z_grid = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # create dataframe
    data = {'grid_bin_center_x': x_grid.ravel(), 'grid_bin_center_y': y_grid.ravel(),'grid_bin_center_z': z_grid.ravel()}
    data = pd.DataFrame(data)

    # get the assign grid cell for each center point
    x_id = []
    y_id = []
    z_id = []

    for i, row in data.iterrows():
        point                     = [row.grid_bin_center_x, row.grid_bin_center_y, row.grid_bin_center_z]
        x_index, y_index, z_index = find_grid_cell_index_for_contact(point, grid)
        x_id.append(x_index)
        y_id.append(y_index)
        z_id.append(z_index)

    data["grid_bin_x"] = x_id
    data["grid_bin_y"] = y_id
    data["grid_bin_z"] = z_id
    
    return data
