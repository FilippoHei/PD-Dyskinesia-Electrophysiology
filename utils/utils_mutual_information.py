"""
Mutual information utilisation functions
"""

import pandas as pd
import numpy as np
import random
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

def measure_spectrogram_mi(df_spectrogram, n_iterations):

    n_rows = np.shape(df_spectrogram.iloc[0].spectrogram)[0]
    n_cols = np.shape(df_spectrogram.iloc[0].spectrogram)[1]
    
    matrix_MI            = np.zeros([n_rows, n_cols])
    matrix_MI_pvalues    = np.ones([n_rows, n_cols])
    matrix_bootstrap_max = np.zeros([n_rows, n_cols])

    for freq_i in range(n_rows):
    
        frequency_value = freq_i + 4
    
        if((frequency_value<=35) | (frequency_value>=60)): # frequency range of interest
            
            print(str(frequency_value) + " Hz mutual information measurement started")
            
            for time_j in range(n_cols):
                
                spectogram_i_j = [matrix[freq_i, time_j] for matrix in df_spectrogram.spectrogram.values]
                severity       = df_spectrogram.dyskinesia_group.values
            
                # get the not nan indexed together from both array (if there are any)
                mask           = ~np.isnan(spectogram_i_j) & ~np.isnan(severity) 
                # mask the arrays to remove nan values
                spectogram_i_j = np.array(spectogram_i_j)[mask]
                severity       = np.array(severity)[mask]
    
                # measure the mutual information between spectrogram values and severity
                MI, pvalue, bootstrap_max           = bootstrap_mi(spectogram_i_j, severity, n_iterations=n_iterations)
    
                matrix_MI[freq_i,time_j]            = MI
                matrix_MI_pvalues[freq_i,time_j]    = pvalue
                matrix_bootstrap_max[freq_i,time_j] = bootstrap_max

    return matrix_MI, matrix_MI_pvalues, matrix_bootstrap_max
    
def measure_mutual_information_along_axis(dataset, feature, axis):
    
    dataset_MI = pd.DataFrame(columns = ["feature", "axis", "cell_id", "MI"])
    
    if(axis=="x"): grid_feature = "grid_bin_x"
    elif(axis=="y"): grid_feature = "grid_bin_y"
    elif(axis=="z"): grid_feature = "grid_bin_z"
        
    for cell_id in dataset[grid_feature].unique():
        mi             = mutual_info_score(dataset[dataset[grid_feature]==cell_id][feature], dataset[dataset[grid_feature]==cell_id]["severity_numeric"])
        row            = {}
        row["feature"] = feature
        row["axis"]    = axis
        row["cell_id"] = int(cell_id)
        row["MI"]      = mi
        dataset_MI.loc[len(dataset_MI)] = row
    dataset_MI = dataset_MI.sort_values(by='cell_id')
        
    return dataset_MI

def bootstrap_mi(x, y, n_iterations=1000):
    
    # calculate the observed mutual information
    observed_mi = mutual_info_classif(x.reshape(-1, 1), y)
    
    # initialize an array to store the bootstrap MI values
    bootstrap_mi_values = []
    
    # bootstrap sampling
    for _ in range(n_iterations):
        y_shuffled = random.sample(list(y), len(y))
        bootstrap_mi_values.append(mutual_info_classif(x.reshape(-1, 1), y_shuffled))
    
    bootstrap_mi_values = np.array(bootstrap_mi_values)
    
    p_value       = np.mean(bootstrap_mi_values >= observed_mi)
    bootstrap_max = np.max(bootstrap_mi_values)
    return observed_mi, p_value, bootstrap_max

def discretize_array(data, bin_size):
    
    max_val  = np.max(data)
    min_val  = np.min(data)
    
    if (min_val % bin_size) != 0:
        min_val = min_val - (min_val % bin_size)
    
    # Adjust max_val up to the nearest higher multiple of bin_size if needed
    if (max_val - min_val) % bin_size != 0:
        max_val = min_val + (np.ceil((max_val - min_val) / bin_size) * bin_size)
    
    bins        = np.arange(min_val, max_val + bin_size, bin_size)
    binned_data = pd.cut(data, bins=bins, labels=False, right=False)
    
    return binned_data