"""
Utility functions for plotting wPLI results on a cortical mesh.
"""
import os
import pandas as pd
import numpy as np
import sys
import warnings
import pyvista as pv
import tkinter as tk
from pyvistaqt import BackgroundPlotter
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# project paths and helpers
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io
from lib_data import DATA_IO

def plot_wPLI_on_cortex(dataframe, filename, frequencyBand=None, onlyOneHemisphere=False, safeFile=True, fixedCmap=None, cmap="viridis", onlySignificant=False):
    '''
    Args:
        dataframe (pd.DataFrame): DataFrame containing wPLI results.
        filename (str): Base filename for saving screenshots. The actual filename will include the selected frequency band and hemisphere info.
        frequencyBand (str, optional): Frequency band to plot. Must be one of ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma', 'gamma_III']. If None, a GUI will prompt the user to select a band.
        onlyOneHemisphere (bool, optional): If True, plot only the right hemisphere and mirror left hemisphere coordinates to the right. Default is False.
        safeFile (bool, optional): If False, no screenshot will be saved. Default is True.
        fixedCmap (np.ndarray, optional): If provided, use these steps for the color bar limits. Default is None (which uses automatic limits).

    Returns:
        None. Displays an interactive cortical plot.
    '''
    # Calculate and average within the frequency bands
    bands = {
        "theta"     : (4, 8),
        "alpha"     : (8, 12),
        "beta_low"  : (12, 20),
        "beta_high" : (20, 35),
        "gamma"     : (60, 90),
        "gamma_III" : (80, 90)
    }
    
    if frequencyBand is not None:
        if not frequencyBand in list(bands.keys()):
            raise ValueError(f"Frequency band '{frequencyBand}' not recognized. Available bands: {list(bands.keys())}")

    band_results = []
    for (patient, ecog_ch), group in dataframe.groupby(["patient", "ECoG_channel"]):
        freqs = group.iloc[0]["freqs"]  # all rows have the same freqs
        # Get only the significant wPLI values if requested
        if onlySignificant:
            pValues = np.stack(group["p_values"].to_numpy())  # shape (n_LFP, n_freqs)
            sig_mask = pValues <= 0.05
            wplis = np.stack(group["wpli"].to_numpy())  # shape (n_LFP, n_freqs)
            wplis = np.where(sig_mask, wplis, np.nan)  # set non-significant values to NaN
        else:
            wplis = np.stack(group["wpli"].to_numpy())  # shape (n_LFP, n_freqs)
        mean_wpli = np.nanmean(wplis, axis=0)
        band_results.append(dict(patient=patient, ECoG_channel=ecog_ch, theta=mean_wpli[0], alpha=mean_wpli[1], beta_low=mean_wpli[2], beta_high=mean_wpli[3], gamma=mean_wpli[4], gamma_III=mean_wpli[5]))

    band_df = pd.DataFrame(band_results)

    # --- Step 2: Load ECoG channel coordinates ---
    MNI_coordinates      = pd.read_csv(DATA_IO.path_coordinates + "contact_coordinates.csv")
    MNI_ECoG_coordinates = MNI_coordinates[MNI_coordinates.recording_type == "ecog"]


    # insert x, y, z coordinates right after ECoG_channel
    col_idx = band_df.columns.get_loc("ECoG_channel")
    band_df.insert(col_idx + 1, "x", np.nan)
    band_df.insert(col_idx + 2, "y", np.nan)
    band_df.insert(col_idx + 3, "z", np.nan)

    # Merge band_df with coordinates
    for idx, row in band_df.iterrows():
        patient = row["patient"]
        bipolar = row["ECoG_channel"]

        # split the bipolar reference into two ints
        try:
            c1, c2 = bipolar.split("-")
            c1, c2 = int(c1), int(c2)
        except Exception:
            print(f"Skipping {bipolar} (invalid format)")
            continue

        # select coordinates for both contacts from this patient
        coords_patient = MNI_ECoG_coordinates[MNI_ECoG_coordinates["patient"] == int(patient)]
        c1_coords = coords_patient[coords_patient["contact"] == c1][["x", "y", "z"]].values
        c2_coords = coords_patient[coords_patient["contact"] == c2][["x", "y", "z"]].values

        if len(c1_coords) == 0 or len(c2_coords) == 0:
            print(f"Coordinates missing for patient {patient}, contacts {c1}-{c2}")
            continue

        # compute mean (Euclidean midpoint)
        mean_coords = np.mean(np.vstack([c1_coords, c2_coords]), axis=0)

        # Mirror coordinates to right hemisphere if required 
        if onlyOneHemisphere and mean_coords[0]<0:
            band_df.at[idx, "x"] = -mean_coords[0]
        else:
            band_df.at[idx, "x"] = mean_coords[0]
        band_df.at[idx, "y"] = mean_coords[1]
        band_df.at[idx, "z"] = mean_coords[2]

    # Pick which band to plot, either as argument or via user input
    if frequencyBand is not None:
        selected_band = frequencyBand
    else:
        selected_band = select_band_from_list(list(bands.keys()))
    print(f"Selected band: {selected_band}")
    
    # Extract coordinates and values for interpolation
    coords = band_df[["x", "y", "z"]].to_numpy()
    values = band_df[selected_band].to_numpy()
    
    # Omit missing values
    coords = coords[~np.isnan(values),:]
    values = values[~np.isnan(values)]

    # --- Step 3: Prepare PyVista cortical plot ---
    cortex_mesh  = utils_io.load_cortical_atlas_meshes()
    cortex_right = cortex_mesh["right_hemisphere"]
    cortex_left  = cortex_mesh["left_hemisphere"]

    # Merge both hemispheres into one for interpolation
    if onlyOneHemisphere:
        mesh_combined = cortex_right
        filepathSuffix = "oneHemisphere"
    else:
        mesh_combined = cortex_right.merge(cortex_left)
        filepathSuffix = "bothHemispheres"
    mesh_vertices = mesh_combined.points

    # Compute the distance of each cortical vertex to the nearest electrode and
    # only keep interpolated values within a radius. Outside, set to NaN.
    tree = cKDTree(coords)
    distances, _ = tree.query(mesh_vertices)

    # RBF (radial basis function) for interpolation onto mesh vertices (multiquadric kernel is smooth) 
    rbf = Rbf(coords[:,0], coords[:,1], coords[:,2], values, function="multiquadric", smooth=2)
    
    values_interp = rbf(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2])
    radius = 20 # threshold: max interpolation radius (mm)
    values_interp[distances > radius] = np.nan # only keep values within radius

    # --- Step 4: Add interpolated values as scalars to mesh ---
    mesh_combined["connectivity"] = values_interp
    mesh_smoothed = mesh_combined.smooth(n_iter=100, relaxation_factor=0.01)

    # --- Step 5: Plot ---    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    plotter = BackgroundPlotter(window_size=(screen_width, screen_height))
    
    if fixedCmap is None:
        plotter.add_mesh(
            mesh_smoothed, 
            scalars="connectivity", 
            cmap=cmap, 
            opacity=1.0, 
            smooth_shading=True,
        )
    else:
            plotter.add_mesh(
            mesh_smoothed, 
            scalars="connectivity", 
            cmap=cmap, 
            opacity=1.0, 
            smooth_shading=True, 
            clim=(min(fixedCmap), max(fixedCmap)),  # custom color limits
            scalar_bar_args={
                "title": "connectivity",
                "n_labels": len(fixedCmap),     # as of now BackgroundPlotter only supports a fixed number of labels
                "fmt": "%.2f"      # tick label format
            }
        )

    # Overlay electrodes as spheres
    for idx, row in band_df[["x", "y", "z"]].iterrows():
        sphere = pv.Sphere(radius=1.5, center=[row.x, row.y, row.z+5]) # offset z for visibility in the top view
        plotter.add_mesh(sphere, color="red", smooth_shading=True)

    plotter.background_color = "white"
    # Add title to the plot
    plotter.add_text(f"{selected_band} wPLI - Cortical Map", position='upper_left', font_size=16, color='black')
    
    plotter.view_vector((0, 0, 1))   # top view
    plotter.add_light(pv.Light(position=(0, 0, 1), color="white", intensity=0.6))
    if onlyOneHemisphere:
        plotter.camera.zoom(2.5)
    else:
        plotter.camera.zoom(1.5)
    plotter.render()  # Force render update before screenshot
    plotter.show()

    # --- Step 6: Save screenshot of the plot ---
    if safeFile:
        # If filename has an any folder paths or extensions, remove them
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        
        # Get the expression after the last underscore as folder name
        if "_" in filename:
            folder_name = filename.split("_")[-1]
            # Create folder if it doesn't exist
            os.makedirs(DATA_IO.path_figure + f"LFP-ECoG wPLI/{folder_name}/", exist_ok=True)
            folderPath = DATA_IO.path_figure + f"LFP-ECoG wPLI/{folder_name}/"
        else:
            os.makedirs(DATA_IO.path_figure + f"LFP-ECoG wPLI/", exist_ok=True)
            folderPath = DATA_IO.path_figure + f"LFP-ECoG wPLI/"
        
        if fixedCmap is not None:
            screenshot_filepath = folderPath + f"{filename}_{filepathSuffix}_{selected_band}_{min(fixedCmap):.2f}-{max(fixedCmap):.2f}.png"
        else:
            screenshot_filepath = folderPath + f"{filename}_{filepathSuffix}_{selected_band}.png"
        
        if onlySignificant:
            screenshot_filepath = screenshot_filepath.replace(".png", "_onlySign.png")
            
        print(f"Saving screenshot to {screenshot_filepath}")
        # Save screenshot with full screen size
        plotter.screenshot(screenshot_filepath)
    
    plotter.app_window.showMaximized()

    return


# Helper function for user input
def select_band_from_list(options):
    root = tk.Tk()
    root.title("Choose Frequency Band")
    
    var = tk.StringVar()

    # create listbox and insert items manually
    listbox = tk.Listbox(root, height=len(options))
    for opt in options:
        listbox.insert(tk.END, opt)
    listbox.pack(padx=20, pady=10)

    def confirm(event=None):
        selection = listbox.get(listbox.curselection())
        var.set(selection)
        root.quit()
    
    listbox.bind("<Double-1>", confirm)
    btn = tk.Button(root, text="Confirm", command=confirm)
    btn.pack(pady=5)

    root.mainloop()
    choice = var.get()
    root.destroy()
    return choice


def plot_interpDiff_wPLI_on_cortex(dataframeList, filename, frequencyBand=None, onlyOneHemisphere=False, safeFile=True, fixedCmap=None, cmap="viridis"):
    '''
    Sometimes not enough corresponding data points (i.e., channel pairs) are available in both conditions to compute a meaningful difference per channel pair.
    Thus this function interpolates each condition separately onto the cortical mesh and then computes the difference between the two interpolated maps.
    
    Args:
        dataframeList (list of pd.DataFrame): List containing two DataFrames with wPLI results to compare (e.g., [df_condition1, df_condition2]).
        filename (str): Base filename for saving screenshots. The actual filename will include the selected frequency band and hemisphere info.
        frequencyBand (str, optional): Frequency band to plot. Must be one of ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma', 'gamma_III']. If None, a GUI will prompt the user to select a band.
        onlyOneHemisphere (bool, optional): If True, plot only the right hemisphere and mirror left hemisphere coordinates to the right. Default is False.
        safeFile (bool, optional): If False, no screenshot will be saved. Default is True.
        fixedCmap (np.ndarray, optional): If provided, use these steps for the color bar limits. Default is None (which uses automatic limits).
        cmap (str, optional): Colormap to use. Default is "viridis".
    
    Returns:
        None. Displays an interactive cortical plot of the difference between the two conditions.
    '''
    if len(dataframeList) != 2:
        raise ValueError("dataframeList must contain exactly two DataFrames to compare.")
    df1, df2 = dataframeList
    
    # Calculate and average within the frequency bands
    bands = {
        "theta"     : (4, 8),
        "alpha"     : (8, 12),
        "beta_low"  : (12, 20),
        "beta_high" : (20, 35),
        "gamma"     : (60, 90),
        "gamma_III" : (80, 90)
    }
    
    if frequencyBand is not None:
        if not frequencyBand in list(bands.keys()):
            raise ValueError(f"Frequency band '{frequencyBand}' not recognized. Available bands: {list(bands.keys())}")
        
    # Pick which band to plot, either as argument or via user input
    if frequencyBand is not None:
        selected_band = frequencyBand
    else:
        selected_band = select_band_from_list(list(bands.keys()))
    print(f"Selected band: {selected_band}")
    
    interpolated_mesh = []
    coords_combined = []
    for dataframe in [df1, df2]:
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Each item in dataframeList must be a pandas DataFrame.")

        band_results = []
        for (patient, ecog_ch), group in dataframe.groupby(["patient", "ECoG_channel"]):
            freqs = group.iloc[0]["freqs"]  # all rows have the same freqs
            wplis = np.stack(group["wpli"].to_numpy())  # shape (n_LFP, n_freqs)
            mean_wpli = wplis.mean(axis=0)
            band_results.append(dict(patient=patient, ECoG_channel=ecog_ch, theta=mean_wpli[0], alpha=mean_wpli[1], beta_low=mean_wpli[2], beta_high=mean_wpli[3], gamma=mean_wpli[4], gamma_III=mean_wpli[5]))

        band_df = pd.DataFrame(band_results)

        # --- Step 2: Load ECoG channel coordinates ---
        MNI_coordinates      = pd.read_csv(DATA_IO.path_coordinates + "contact_coordinates.csv")
        MNI_ECoG_coordinates = MNI_coordinates[MNI_coordinates.recording_type == "ecog"]


        # insert x, y, z coordinates right after ECoG_channel
        col_idx = band_df.columns.get_loc("ECoG_channel")
        band_df.insert(col_idx + 1, "x", np.nan)
        band_df.insert(col_idx + 2, "y", np.nan)
        band_df.insert(col_idx + 3, "z", np.nan)

        # Merge band_df with coordinates
        for idx, row in band_df.iterrows():
            patient = row["patient"]
            bipolar = row["ECoG_channel"]

            # split the bipolar reference into two ints
            try:
                c1, c2 = bipolar.split("-")
                c1, c2 = int(c1), int(c2)
            except Exception:
                print(f"Skipping {bipolar} (invalid format)")
                continue

            # select coordinates for both contacts from this patient
            coords_patient = MNI_ECoG_coordinates[MNI_ECoG_coordinates["patient"] == int(patient)]
            c1_coords = coords_patient[coords_patient["contact"] == c1][["x", "y", "z"]].values
            c2_coords = coords_patient[coords_patient["contact"] == c2][["x", "y", "z"]].values

            if len(c1_coords) == 0 or len(c2_coords) == 0:
                print(f"Coordinates missing for patient {patient}, contacts {c1}-{c2}")
                continue

            # compute mean (Euclidean midpoint)
            mean_coords = np.mean(np.vstack([c1_coords, c2_coords]), axis=0)

            # Mirror coordinates to right hemisphere if required 
            if onlyOneHemisphere and mean_coords[0]<0:
                band_df.at[idx, "x"] = -mean_coords[0]
            else:
                band_df.at[idx, "x"] = mean_coords[0]
            band_df.at[idx, "y"] = mean_coords[1]
            band_df.at[idx, "z"] = mean_coords[2]
        
        # Extract coordinates and values for interpolation
        coords = band_df[["x", "y", "z"]].to_numpy()
        values = band_df[selected_band].to_numpy()
        
        # Omit missing values
        coords = coords[~np.isnan(values),:]
        values = values[~np.isnan(values)]
        
        # vertically concatenate coordinates for both conditions for later plotting
        coords_combined = np.vstack([coords_combined, coords]) if len(coords_combined)>0 else coords

        # --- Step 3: Prepare PyVista cortical plot ---
        cortex_mesh  = utils_io.load_cortical_atlas_meshes()
        cortex_right = cortex_mesh["right_hemisphere"]
        cortex_left  = cortex_mesh["left_hemisphere"]

        # Merge both hemispheres into one for interpolation
        if onlyOneHemisphere:
            mesh_combined = cortex_right
            filepathSuffix = "oneHemisphere"
        else:
            mesh_combined = cortex_right.merge(cortex_left)
            filepathSuffix = "bothHemispheres"
        mesh_vertices = mesh_combined.points

        # Compute the distance of each cortical vertex to the nearest electrode and
        # only keep interpolated values within a radius. Outside, set to NaN.
        tree = cKDTree(coords)
        distances, _ = tree.query(mesh_vertices)

        # RBF (radial basis function) for interpolation onto mesh vertices (multiquadric kernel is smooth) 
        rbf = Rbf(coords[:,0], coords[:,1], coords[:,2], values, function="multiquadric", smooth=2)
        
        values_interp = rbf(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2])
        radius = 20 # threshold: max interpolation radius (mm)
        values_interp[distances > radius] = np.nan # only keep values within radius
        interpolated_mesh.append(values_interp)

    # --- Step 4: Add interpolated values as scalars to mesh ---
    mesh_combined["connectivity"] = interpolated_mesh[1] - interpolated_mesh[0]  # difference between condition 2 and condition 1
    mesh_smoothed = mesh_combined.smooth(n_iter=100, relaxation_factor=0.01)

    # --- Step 5: Plot ---    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    plotter = BackgroundPlotter(window_size=(screen_width, screen_height))
    
    if fixedCmap is None:
        plotter.add_mesh(
            mesh_smoothed, 
            scalars="connectivity", 
            cmap=cmap, 
            opacity=1.0, 
            smooth_shading=True,
        )
    else:
            plotter.add_mesh(
            mesh_smoothed, 
            scalars="connectivity", 
            cmap=cmap, 
            opacity=1.0, 
            smooth_shading=True, 
            clim=(min(fixedCmap), max(fixedCmap)),  # custom color limits
            scalar_bar_args={
                "title": "connectivity",
                "n_labels": len(fixedCmap),     # as of now BackgroundPlotter only supports a fixed number of labels
                "fmt": "%.2f"      # tick label format
            }
        )

    # Overlay electrodes as spheres
    for idx in range(len(coords_combined)):
        sphere = pv.Sphere(radius=1.5, center=[coords_combined[idx, 0], coords_combined[idx, 1], coords_combined[idx, 2]+5]) # offset z for visibility in the top view
        plotter.add_mesh(sphere, color="red", smooth_shading=True)

    plotter.background_color = "white"
    # Add title to the plot
    plotter.add_text(f"{selected_band} wPLI - Cortical Map", position='upper_left', font_size=16, color='black')
    
    plotter.view_vector((0, 0, 1))   # top view
    plotter.add_light(pv.Light(position=(0, 0, 1), color="white", intensity=0.6))
    if onlyOneHemisphere:
        plotter.camera.zoom(2.5)
    else:
        plotter.camera.zoom(1.5)
    plotter.render()  # Force render update before screenshot
    plotter.show()

    # --- Step 6: Save screenshot of the plot ---
    if safeFile:
        # If filename has an any folder paths or extensions, remove them
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        
        # Get the expression after the last underscore as folder name
        if "_" in filename:
            folder_name = filename.split("_")[-1]
            # Create folder if it doesn't exist
            os.makedirs(DATA_IO.path_figure + f"LFP-ECoG wPLI/{folder_name}/", exist_ok=True)
            folderPath = DATA_IO.path_figure + f"LFP-ECoG wPLI/{folder_name}/"
        else:
            os.makedirs(DATA_IO.path_figure + f"LFP-ECoG wPLI/", exist_ok=True)
            folderPath = DATA_IO.path_figure + f"LFP-ECoG wPLI/"
        
        if fixedCmap is not None:
            screenshot_filepath = folderPath + f"{filename}_{filepathSuffix}_{selected_band}_{min(fixedCmap):.2f}-{max(fixedCmap):.2f}.png"
        else:
            screenshot_filepath = folderPath + f"{filename}_{filepathSuffix}_{selected_band}.png"
        print(f"Saving screenshot to {screenshot_filepath}")
        # Save screenshot with full screen size
        plotter.screenshot(screenshot_filepath)
    
    plotter.app_window.showMaximized()

    return