"""
Mutual information utilisation functions
"""

import pandas as pd
import numpy as np

def panzeri_treves_correction(df, feature_col, target_col):
    # Calculate the empirical probability distribution of the feature
    feature_prob = df[feature_col].value_counts(normalize=True)
    target_prob  = df[target_col].value_counts(normalize=True)

    # Calculate the bias
    bias = 0.0
    for feature_val in feature_prob.index:
        for target_val in target_prob.index:
            p_xy = len(df[(df[feature_col] == feature_val) & (df[target_col] == target_val)]) / len(df)
            if p_xy > 0:
                bias += p_xy * np.log2(p_xy)

    # Calculate the corrected mutual information
    mi_corrected = mi_raw - bias
    return mi_corrected