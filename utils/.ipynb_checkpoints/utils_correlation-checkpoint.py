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
