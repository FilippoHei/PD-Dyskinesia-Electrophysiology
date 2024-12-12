"""
Miscellaneous utilisation functions
"""

import pandas as pd
import numpy as np

from scipy import stats
import statsmodels

def permutation_corr(x,y):  # explore all possible pairings by permuting `x`
    dof = len(x)-2  # len(x) == len(y)
    rs  = stats.spearmanr(x, y).statistic  # ignore pvalue
    transformed = rs * np.sqrt(dof / ((rs+1.0)*(1.0-rs)))
    return transformed

def spearsman_correlation_with_multiple_comparison_correction(dataset, group_variable, target_variable, features, correction_type):

    assert correction_type in ['bonferroni', 'fdr_bh ', 'holm'], f'incorrect correction method ({correction_type})'
    
    # create an empty array for measured linear correlations
    corr_spearman = pd.DataFrame(columns=["group", "feature", "coefficient", "pvalue"])

    # for each group, measure the correlation for each feature in the feature list.
    for group in dataset[group_variable].unique():
        for feature in features:
            print(group + " : " + feature)
            x    = list(dataset[dataset[group_variable] == group][target_variable])
            y    = list(dataset[dataset[group_variable] == group][feature])
            coef = stats.spearmanr(x, y).statistic
            ref  = stats.permutation_test((x,y), permutation_corr, alternative='two-sided', permutation_type='pairings')
            corr_spearman.loc[len(corr_spearman)] = {"group":group, "feature":feature, "coefficient":coef,  "pvalue":ref.pvalue}
    
    # Do multiple comparison corrections with selected method
    corrected_p                       = statsmodels.stats.multitest.multipletests(corr_spearman.pvalue, alpha=0.05, method=correction_type)[1]
    corr_spearman["pvalue_corrected"] = corrected_p
    return  corr_spearman


def bootstrap_test(dataset, x, y, n_iterations=500):

    # calculate the observed Spearman's correlation
    observed_corr, _      = stats.spearmanr(dataset[x], dataset[y])
    
    # initialize a list to store the permutation correlations
    permuted_correlations = []

    for _ in range(n_iterations):
        
        # permute the y variable
        permuted_y = np.random.permutation(dataset[y].values)
        
        # calculate the Spearman correlation with the permuted y
        permuted_corr, _ = stats.spearmanr(dataset[x], permuted_y)
        permuted_correlations.append(permuted_corr)

    # compute the p-value as the proportion of correlations that are as extreme as the observed one
    p_value = np.sum(np.abs(permuted_correlations) >= np.abs(observed_corr)) / n_iterations
    
    return observed_corr, p_value
    
def spearsman_correlation_along_spatial_axes(dataset, features, target_feature, axis, n_bins, patient_grouping=False):
    
    if(axis=="x"):
        grid_feature = "grid_bin_x"
    elif(axis=="y"):
        grid_feature = "grid_bin_y"
    elif(axis=="z"):
        grid_feature = "grid_bin_z"
    
    dataset_corr = pd.DataFrame(columns=["feature_1", "feature_2", "axis", "grid_bin", "correlation", "pvalue"])
    
    for bin in dataset[grid_feature].unique():
    
        if(patient_grouping==True):
            
            dataset_bin         = dataset[dataset[grid_feature]==bin]
            dataset_bin_patient = pd.DataFrame(dataset_bin.groupby(["patient",target_feature])[features].mean())
            dataset_bin_patient.reset_index(inplace=True)
        
            for feature in features:
                corr, p            = bootstrap_test(dataset_bin_patient, feature, target_feature)
                row                = {}
                row["feature_1"]   = feature
                row["feature_2"]   = target_feature
                row["axis"]        = axis
                row["grid_bin"]    = bin
                row["correlation"] = corr
                row["pvalue"]      = p
                dataset_corr.loc[len(dataset_corr)]  = row
        else:
            
            dataset_bin = dataset[dataset[grid_feature]==bin]
            
            for feature in features:
                corr, p            = bootstrap_test(dataset_bin, feature, target_feature)
                row                = {}
                row["feature_1"]   = feature
                row["feature_2"]   = target_feature
                row["axis"]        = axis
                row["grid_bin"]    = bin
                row["correlation"] = corr
                row["pvalue"]      = p
                dataset_corr.loc[len(dataset_corr)]  = row

    return dataset_corr
