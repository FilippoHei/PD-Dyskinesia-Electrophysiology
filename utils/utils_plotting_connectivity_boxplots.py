"""
Utility functions for plotting wPLI results on a cortical mesh.
"""
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import warnings
from utils_plotting_wPLI import select_band_from_list
import seaborn as sns
from scipy.stats import ranksums

warnings.filterwarnings("ignore")

# project paths and helpers
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

from lib_data import DATA_IO

def plot_cortex_connectivity_boxplot(dataframe, filename, frequencyBand, regionSpecs='Motor cortex', 
                                     safeFile=True, condStrList=None, onlySignificant=False):
    '''
    This function plots boxplots of connectivity metrics (e.g., wPLI) for ECoG channels in a given cortical region.
    Args:
        df (pd.DataFrame): The input DataFrame containing connectivity metrics. 
            Can be a list of DataFrames for multiple conditions that will be plotted next to each other.
        filename (str): The base filename for saving the plot.
        frequencyBand (str): The frequency band to plot.
        regionSpecs (str): The cortical region specifications. Default is 'Motor cortex'. 
            If None, all regions are included. Can also be a list of regions.
        safeFile (bool): Whether to save the file safely (avoiding overwrites). Default is True.
        condStrList (list of str): List of condition names corresponding to each DataFrame in the dataframe list.
            If not provided, generic names will be used (Condition 1, Condition 2, ...).
        onlySignificant (bool): If True, only significant channels are plotted. Default is False.
    Returns:
        None. The plot is displayed and optionally saved to a file.
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
    
    # Convert single DataFrame to list for uniform processing
    if isinstance(dataframe, pd.DataFrame):
        dataframe = [dataframe]
    elif not isinstance(dataframe, list):
        raise ValueError("dataframe must be a pandas DataFrame or a list of DataFrames.")

    if frequencyBand is not None:
        if not frequencyBand in list(bands.keys()):
            raise ValueError(f"Frequency band '{frequencyBand}' not recognized. Available bands: {list(bands.keys())}")

    band_df = []
    for dataset in dataframe:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Each item in dataframe must be a pandas DataFrame.")
        band_results = []
        for (patient, ecog_ch), group in dataset.groupby(["patient", "ECoG_channel"]):
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

        band_df.append(pd.DataFrame(band_results))
        

    # --- Step 2: Load ECoG channel MNI coordinates ---
    MNI_ECoG_coordinates       = pd.read_pickle(DATA_IO.path_coordinates + "MNI_ECoG_channels.pkl")
    
    # Check if the regionSpecs is valid
    if regionSpecs is not None:
        valid_regions = MNI_ECoG_coordinates['AAL3_cortex'].unique()
        # check if all provided regions are valid (if regionSpecs is a list)
        if isinstance(regionSpecs, list):
            for region in regionSpecs:
                if region not in valid_regions:
                    raise ValueError(f"regionSpecs '{region}' is not valid. Choose from: {valid_regions}")
            MNI_ECoG_coordinates = MNI_ECoG_coordinates[MNI_ECoG_coordinates['AAL3_cortex'].isin(regionSpecs)]
            
        else:
            if regionSpecs not in valid_regions:
                raise ValueError(f"regionSpecs '{regionSpecs}' is not valid. Choose from: {valid_regions}")
            MNI_ECoG_coordinates = MNI_ECoG_coordinates[MNI_ECoG_coordinates['AAL3_cortex'] == regionSpecs]
        
        # Filter band_df to keep only matching combinations of patient and the respective ECoG_channel
        valid_combinations = set(zip(MNI_ECoG_coordinates["patient"], MNI_ECoG_coordinates["channel"]))
        for idx in range(len(band_df)):
            band_df[idx] = band_df[idx][
                band_df[idx].apply(lambda row: (row["patient"], row["ECoG_channel"]) in valid_combinations, axis=1)
            ]
            # insert x, y, z coordinates right after ECoG_channel
            col_idx = band_df[idx].columns.get_loc("ECoG_channel")
            band_df[idx].insert(col_idx + 1, "x", np.nan)
            band_df[idx].insert(col_idx + 2, "y", np.nan)
            band_df[idx].insert(col_idx + 3, "z", np.nan)
            band_df[idx].insert(col_idx + 4, "AAL3_parcellation", "")
            band_df[idx].insert(col_idx + 5, "AAL3_cortex", "")

            # Merge band_df with coordinates
            for row_idx, row in band_df[idx].iterrows():
                patient = row["patient"]
                bipolar = row["ECoG_channel"]

                # select coordinates for channel from this patient
                coords_patient = MNI_ECoG_coordinates[(MNI_ECoG_coordinates["patient"] == patient) & (MNI_ECoG_coordinates["channel"] == bipolar)]
                if coords_patient.shape[0] == 0:
                    print(f"Warning: No coordinates found for patient {patient}, channel {bipolar}. Skipping.")
                    continue
                band_df[idx].at[row_idx, "x"] = coords_patient["x"].values[0]
                band_df[idx].at[row_idx, "y"] = coords_patient["y"].values[0]
                band_df[idx].at[row_idx, "z"] = coords_patient["z"].values[0]
                band_df[idx].at[row_idx, "AAL3_parcellation"] = coords_patient["AAL3_parcellation"].values[0]
                band_df[idx].at[row_idx, "AAL3_cortex"] = coords_patient["AAL3_cortex"].values[0]
    elif regionSpecs is None:
        print(f"No regionSpecs provided, using all available regions.")

    # Pick which band to plot, either as argument or via user input
    if frequencyBand is not None:
        selected_band = frequencyBand
    else:
        selected_band = select_band_from_list(list(bands.keys()))
    print(f"Selected band: {selected_band}")

    valuesBoxplot = []
    for idx in range(len(band_df)):
        # Extract values for boxplots
        values = band_df[idx][selected_band].to_numpy()
        # Omit missing values
        values = values[~np.isnan(values)]
        valuesBoxplot.append(values)

    # --- Step 3: Plot boxplots ---
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=valuesBoxplot)

    if condStrList is None or (condStrList is not None and len(condStrList) != len(valuesBoxplot)):
        condStrList = [f"Condition {i+1}" for i in range(len(valuesBoxplot))]
    ax.set_xticklabels(condStrList) 

    if regionSpecs is None:
        title_str = 'AllRegions'
    else:
        title_str = ' & '.join(regionSpecs) if isinstance(regionSpecs, list) else regionSpecs
    plt.title(f"wPLI of {' '.join(word.capitalize() for word in selected_band.replace('_', ' ').split())} Connectivity over {title_str}")
    plt.ylabel("wPLI")
    plt.grid(True)
    y_max = ax.get_ylim()[1]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]        
    
    # Stats for each boxplot
    # Calculate p-values for each comparison between boxplots as rank-sum test
    if len(valuesBoxplot) > 1:
        print("P-values for pairwise rank-sum test:")
        p_values = np.ones((len(valuesBoxplot), len(valuesBoxplot)))
        offset_counter = 1
        for i in range(len(valuesBoxplot)):
            for j in range(i+1, len(valuesBoxplot)):
                stat, p = ranksums(valuesBoxplot[i], valuesBoxplot[j])
                p_values[i, j] = p
                p_values[j, i] = p
                print(f"     {condStrList[i]} vs {condStrList[j]}: {p:.4f}")
                # Overlay significance stars on the boxplot
                # *** for p < 0.001, ** for p < 0.01, * for p < 0.05, n.s. otherwise
                if p <= 0.001:
                    star = '***'
                elif p <= 0.01:
                    star = '**'
                elif p <= 0.05:
                    star = '*'
                else:
                    star = 'n.s.'
                # Plot horizontal line with star above the boxplots
                if j-i==1:
                    offset = 0
                else:
                    offset = offset_counter
                    offset_counter += 1
                # Adjust y position based on the number of comparisons to avoid overlap
                # Instead of absolute offsets use relative offsets based on y_max and y_range
                ax.plot([i+0.1, j-0.1], [y_max + 0.02*y_range*offset, y_max + 0.02*y_range*offset], color='black')
                ax.text((i+j)/2, y_max + 0.03*y_range*offset, star, fontsize=8, color='black', ha='center')
        
        # Update y-axis limit to accommodate stars
        ax.set_ylim(top=y_max + 0.03*y_range*offset_counter)
    else:
        p_values = None
    
    # --- Step 4: Save the plot ---
    if filename is not None and safeFile:
        # If filename has an any folder paths or extensions, remove them
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        
        # Get the expression after the last underscore as folder name
        if "_" in filename:
            folder_name = filename.split("_")[-1]
            # Create folder if it doesn't exist
            os.makedirs(DATA_IO.path_figure + f"LFP-ECoG wPLI/{folder_name}/Boxplots/", exist_ok=True)
            folderPath = DATA_IO.path_figure + f"LFP-ECoG wPLI/{folder_name}/Boxplots/"
        else:
            os.makedirs(DATA_IO.path_figure + f"LFP-ECoG wPLI/Boxplots/", exist_ok=True)
            folderPath = DATA_IO.path_figure + f"LFP-ECoG wPLI/Boxplots/"
        if regionSpecs is None:
            regionSpecs_str = 'AllRegions'
        else:
            regionSpecs_str = '&'.join(region.replace(' ', '') for region in regionSpecs) if isinstance(regionSpecs, list) else regionSpecs.replace(' ', '')
        filepath = folderPath + f"{filename}_{regionSpecs_str}_{selected_band}.svg"        
        if onlySignificant:
            filepath = filepath.replace(".svg", "_onlySign.svg")
        
        # Saving files
        plt.savefig(filepath, dpi=300)
        
        num_elements = [len(v) for v in valuesBoxplot]
        medians = [np.median(v) for v in valuesBoxplot]
        mins = [np.min(v) for v in valuesBoxplot]
        maxs = [np.max(v) for v in valuesBoxplot]
        upper_quartiles = [np.percentile(v, 75) for v in valuesBoxplot]
        lower_quartiles = [np.percentile(v, 25) for v in valuesBoxplot]
        upper_whiskers = [min(maxs[i], upper_quartiles[i] + 1.5 * (upper_quartiles[i] - lower_quartiles[i])) for i in range(len(valuesBoxplot))]
        lower_whiskers = [max(mins[i], lower_quartiles[i] - 1.5 * (upper_quartiles[i] - lower_quartiles[i])) for i in range(len(valuesBoxplot))]
        means = [np.mean(v) for v in valuesBoxplot]
        stds = [np.std(v) for v in valuesBoxplot]
        
        with open(filepath.replace(".svg", "_meta.txt"), "w") as f:
            f.write(f"Conditions: {condStrList}\n")
            f.write(f"Number of elements per boxplot: {num_elements}\n")
            f.write("Medians: [" + ", ".join([f"{m:.6e}" for m in medians]) + "]\n")
            f.write("Mins: [" + ", ".join([f"{m:.6e}" for m in mins]) + "]\n")
            f.write("Maxs: [" + ", ".join([f"{m:.6e}" for m in maxs]) + "]\n")
            f.write("Lower quartiles: [" + ", ".join([f"{m:.6e}" for m in lower_quartiles]) + "]\n")
            f.write("Upper quartiles: [" + ", ".join([f"{m:.6e}" for m in upper_quartiles]) + "]\n")
            f.write("Lower whiskers: [" + ", ".join([f"{m:.6e}" for m in lower_whiskers]) + "]\n")
            f.write("Upper whiskers: [" + ", ".join([f"{m:.6e}" for m in upper_whiskers]) + "]\n")
            f.write("Means: [" + ", ".join([f"{m:.6e}" for m in means]) + "]\n")
            f.write("Standard deviations: [" + ", ".join([f"{m:.6e}" for m in stds]) + "]\n")
            if p_values is not None:
                f.write("P-values for pairwise rank-sum test (not corrected):\n")
                for i in range(len(condStrList)):
                    for j in range(i+1, len(condStrList)):
                        f.write(f"     {condStrList[i]} vs {condStrList[j]}: {p_values[i, j]:.6e}\n")
        print(f"Plot saved to '{filepath}' with metadata as '_meta.txt'.")

    return plt.show()


def plot_cortex_connectivity_dyskResolution_boxplot(df, filename, frequencyBand, regionSpecs='Motor cortex', onlyMovingHand=True, safeFile=True):
    '''
    This function plots boxplots of connectivity metrics (e.g., wPLI) for ECoG channels in a given cortical region, with a focus on dyskinesia resolution.
    Args:
        df (pd.DataFrame): The input DataFrame containing connectivity metrics.
        filename (str): The base filename for saving the plot.
        frequencyBand (str): The frequency band to plot.
        regionSpecs (str): The cortical region specifications. Default is 'Motor cortex'. If None, all regions are included.
    '''
