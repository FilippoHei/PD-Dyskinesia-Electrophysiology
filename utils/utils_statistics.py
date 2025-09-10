"""
Statistic utilisation functions
"""
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

sys.path.insert(0, './lib')

import utils_spectrogram
    
def point_wise_LME_spectrogram_within_severity(dataframe, feature_spectrogram, n_downsampling):
    
    X          = np.array(dataframe[feature_spectrogram].to_list())
    X          = X.squeeze(axis=1)
    not_nan_i  = ~np.isnan(X).any(axis=(1, 2))
    X          = X[not_nan_i]

    # get patients and spectrogram data
    data       = np.array(X)
    data       = utils_spectrogram.downsample_time_axis_for_spectrogram(data, target_time_points=n_downsampling)
    patients   = dataframe.patient.to_list()
    patients   = np.array(patients)[not_nan_i]

    # Store p-values
    n_subjects = data.shape[0]
    n_freq     = data.shape[1]
    n_time     = data.shape[2]
    p_values   = np.zeros((n_freq, n_time))
    z_scores   = np.zeros((n_freq, n_time))
    
    for f_i in range(n_freq):
        for t_i in range(n_time):
            
            y  = data[:, f_i, t_i] # get all the points for the corresponding frequency x timepoint combination
            df = pd.DataFrame({'voxel': y, 'patient': patients})
            
            try:
                # Fit LME
                model              = mixedlm("voxel ~ 1", df, groups=df["patient"])
                result             = model.fit(reml=False)
                p_values[f_i, t_i] = result.pvalues['Intercept']  # p-value for deviation from 0
                z_scores[f_i, t_i] = result.tvalues['severity']
            except Exception as e:
                p_values[f_i, t_i] = 1
                z_scores[f_i, t_i] = 0
    
        print(str(90-(f_i)) + " Hz is completed...")

    # just in case any p value is np.nan
    p_values[np.isnan(p_values)] = 1

    return p_values

def point_wise_LME_spectrogram_between_severity(dataframe, feature_spectrogram, n_downsampling):
    
    X          = np.array(dataframe[feature_spectrogram].to_list())
    X          = X.squeeze(axis=1)
    not_nan_i  = ~np.isnan(X).any(axis=(1, 2))
    X          = X[not_nan_i]

    # get patients and spectrogram data
    data       = np.array(X)
    data       = utils_spectrogram.downsample_time_axis_for_spectrogram(data, target_time_points=n_downsampling)
    patients   = np.array(dataframe.patient.to_list())[not_nan_i]
    severity   = np.array(dataframe.severity.to_list())[not_nan_i]

    # Store p-values
    n_subjects = data.shape[0]
    n_freq     = data.shape[1]
    n_time     = data.shape[2]
    p_values   = np.zeros((n_freq, n_time))
    z_scores   = np.zeros((n_freq, n_time)) 
    
    for f_i in range(n_freq):
        for t_i in range(n_time):
            
            y  = data[:, f_i, t_i] # get all the points for the corresponding frequency x timepoint combination
            df = pd.DataFrame({'voxel': y, 'patient': patients,'severity': severity})
            
            try:
                # Fit LME
                model              = mixedlm("voxel ~ severity", df, groups=df["patient"])
                result             = model.fit(reml=False)
                p_values[f_i, t_i] = result.pvalues['severity'] 
                z_scores[f_i, t_i] = result.tvalues['severity']
            except Exception as e:
                p_values[f_i, t_i] = 1
                z_scores[f_i, t_i] = 0
    
        print(str(90-(f_i)) + " Hz is completed...")

    # just in case any p value is np.nan
    p_values[np.isnan(p_values)] = 1

    return p_values, z_scores
    

def set_up_mixedlm_with_interaction(dataset, response_variable, independent_variable, block_variable, random_effect, random_intercept, random_slope, REML_state):
    if((random_intercept==True) & (random_slope==False)):
        model = smf.mixedlm(f"" +response_variable + " ~ " + independent_variable + " * " +  block_variable, dataset, 
                            groups=dataset[random_effect], re_formula="~1").fit(reml=REML_state)
    elif((random_intercept==True) & (random_slope==True)):
        model = smf.mixedlm(f"" +response_variable +  "~ " + independent_variable + " * " +  block_variable, dataset, 
                            groups=dataset[random_effect], re_formula="~1+"+independent_variable).fit(reml=REML_state)
    return model
    
def run_LMM_model_with_interaction(dataset, response_variables, independent_variable, block_variable, 
                                   random_effect, random_intercept, random_slope):

    groups   = sorted(dataset[independent_variable].unique()) 
    segments = sorted(dataset[block_variable].unique()) 
    
    print("Linear Mixed Effect Model with Interaction Started")
    print("------------------------------------------------------------------")
    print("--> independent variable : " + str(independent_variable) + " [" +  ", ".join(groups) + "] ")
    print("--> block variable       : " + str(block_variable) + " [" +  ", ".join(segments) + "] ")
    print("--> interaction          : " + str(independent_variable) + " * " + str(str(block_variable)))
    print("--> random effect        : " + str(random_effect))
    print("--> random intercept     : " + str(random_intercept))
    print("--> random slope         : " + str(random_slope))
    print("------------------------------------------------------------------")
    
    reference_severity = groups[0]
    reference_segment  = segments[0]
    results            = pd.DataFrame(columns=["feature", "reference_severity", "reference_segment", "comparison_severity", "comparison_segment", 
                                               "model", "coefficient","p_value"])

    for feature in response_variables:
        print("--> response variable    : " + feature)
        model = set_up_mixedlm_with_interaction(dataset=dataset, response_variable=feature, 
                                                independent_variable=independent_variable, block_variable=block_variable, random_effect=random_effect, 
                                                random_intercept=random_intercept, random_slope=random_slope, REML_state=True)
        
        df            = pd.DataFrame(model.pvalues).reset_index()
        df["term"]    = df["index"]
        df["p_value"] = df[0]
        df            = df[["term","p_value"]]
        
        for index, row in df.iterrows():
            if(":" in row.term):
                new_row                        = {} 
                new_row["feature"]             = feature
                new_row["reference_severity"]  = reference_severity
                new_row["reference_segment"]   = reference_segment
                new_row["comparison_severity"] = row.term.split(":")[0].split(".")[1][0:-1]
                new_row["comparison_segment"]  = row.term.split(":")[1].split(".")[1][0:-1]
                new_row["model"]               = model
                new_row["coefficient"]         = model.params[index]
                new_row["p_value"]             = row.p_value
                results.loc[len(results)]      = new_row 

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    return results
    

def set_up_mixedlm(dataset, response_variable, independent_variable, random_effect, random_intercept, random_slope, constant_variance, REML_state):
    
    formula              = f"{response_variable} ~ C({independent_variable})"

    if(random_intercept and not random_slope):
        re_formula = "~1"
    elif(random_intercept and random_slope):
        re_formula = f"~1 + C({independent_variable})"

    if(constant_variance):
        model = smf.mixedlm(formula, dataset, groups=dataset[random_effect], re_formula=re_formula).fit(reml=True)
    else:
        vc_formula = {random_effect: f"0 + C({independent_variable})"}
        vc_formula = {random_effect: f"1 + C({independent_variable})"}
        vc_formula = {random_effect: f"1"}
        model = smf.mixedlm(formula, dataset, groups=dataset[random_effect], re_formula=re_formula, vc_formula=vc_formula).fit(reml=True)
        
    return model

def run_LMM_model(dataset, response_variables, independent_variable, random_effect, random_intercept, constant_variance, random_slope):

    df_LMM_results         = pd.DataFrame(columns=["feature", "group_1", "group_2", "coefficient", "model", "converged", "p_value"]) 
    groups                 = sorted(dataset[independent_variable].unique())                      # the distinct group categories
    reference_group        = sorted(dataset[independent_variable].unique())[0]                   # the reference group based on alphabetical order
    no_of_remaining_groups = len(dataset[independent_variable].unique()) - 1                     # the number of groups that can be compared with the reference group
    
    print("Linear Mixed Effect Model Started")
    print("------------------------------------------------------------------")
    print("--> independent variable : " + str(independent_variable))
    print("--> groups               : " + ", ".join(groups))
    print("--> random effect        : " + str(random_effect))
    print("--> random intercept     : " + str(random_intercept))
    print("--> random slope         : " + str(random_slope))
    print("------------------------------------------------------------------")
    
    for feature in response_variables:
    
        print("--> response variable    : " + feature)
        
        model   = set_up_mixedlm(dataset=dataset, response_variable=feature, independent_variable=independent_variable, 
                                 random_effect=random_effect, random_intercept=random_intercept, random_slope=random_slope, 
                                 constant_variance=constant_variance, REML_state=True)
        pvalues = model.pvalues # get uncorrected p-values from the LMM model
        coeffs  = model.params  # get coefficients from the LMM model
        
        for group_i in range(no_of_remaining_groups):
            row                = {}
            row["feature"]     = feature
            row["group_1"]     = reference_group
            row["group_2"]     = groups[group_i+1]
            row["coefficient"] = coeffs[group_i+1]
            pvalue             = pvalues[group_i+1]
            row["p_value"]     = pvalue
            row["model"]       = model
            row["converged"]   = model.converged
    
            if(np.isnan(pvalue) == True): # then there is a big possibility of having collinearity between groups
    
                print("    -> issue: " + reference_group + " vs " + row["group_2"])
                
                # singularity refers to a case where one or more variables in the dataset are perfectly correlated with others. 
                # This results in a design matrix that is not invertible, which can lead to errors when fitting linear mixed effect models.
                singularity_issue = check_singularity_issue(dataset=dataset, independent_variable=independent_variable, 
                                                            response_variable=feature, drop_first=False)
    
                # Variance Inflation Factor (VIF) is used to detect multicollinearity in our LMM model by quantifying how much the variance of 
                # a regression coefficient is inflated due to multicollinearity among the predictors/independent variables. 
                # When all groups are included, their sum sometimes can be perfectly correlated with the intercept (or constant), leading to an "infinitely large" VIF.
                vif_scores = measure_variance_inflation_factor(dataset=dataset, independent_variable="grouping_2", 
                                                               response_variable="post_event_gamma_mean", drop_first=False)
                if(singularity_issue==True):
                    print("        -> warning: some eigenvalues close to zero, indicating a potential singularity issue in covariance matrix.")
                if(np.isinf(vif_scores.vif).any()==True): # check if vif scores go to inf which implies a high degree of multicollinearity
                    print("        -> warning: infinite VIF values, indicating a high degree of multicollinearity leading failed estimation of model parameters")

                print("        -> switching from REML to ML Estimation")

                # try again with fitting LMM with Maximum likelihood estimation instead of Restricted Maximum Likelihood Estimation
                model   = set_up_mixedlm(dataset=dataset, response_variable=feature, independent_variable=independent_variable, 
                                         random_effect=random_effect, random_intercept=random_intercept, random_slope=random_slope, 
                                         constant_variance=constant_variance, REML_state=False)
                pvalues = model.pvalues # get uncorrected p-values from the LMM model
                coeffs  = model.params  # get coefficients from the LMM model
                row["coefficient"] = coeffs[group_i+1]
                pvalue             = pvalues[group_i+1]
                row["p_value"]     = pvalue
                row["model"]       = model
                row["converged"]   = model.converged

                if(np.isnan(pvalue) == False):
                    print("        -> issue resolved: the result for given two groups was added to the output!")
                    df_LMM_results.loc[len(df_LMM_results)] = row
                else:
                    print("        -> issue unsolved: the result for given two groups was not add to the output!")
            else:
                df_LMM_results.loc[len(df_LMM_results)] = row

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    
    return df_LMM_results

def apply_multiple_correction(p_values, correction_method):
    return multipletests(p_values, alpha=0.05, method=correction_method)[1]

def interpret_vif(vif):
    if vif < 5:
        return 'low'
    elif 5 <= vif < 10:
        return 'moderate'
    else:
        return 'high'
        
def measure_variance_inflation_factor(dataset, independent_variable, response_variable, drop_first):

    # convert the random effect column to dummy variables (one-hot encoding)
    df_encoded = pd.get_dummies(dataset[[independent_variable,response_variable]], columns=[independent_variable], drop_first=drop_first)
    
    # define the independent variables (the one-hot encoded columns) and the dependent variable
    X = df_encoded.drop(columns=[response_variable]).astype(int)
    y = df_encoded[response_variable]
    
    # adding a constant term helps in understanding the multicollinearity of the model while accounting for the baseline effect
    X = sm.add_constant(X)
    
    # calculate VIF for each variable
    vif_data                   = pd.DataFrame()
    vif_data['feature']        = X.columns
    vif_data['vif']            = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data['interpretation'] = vif_data['vif'].apply(interpret_vif)

    return vif_data


def check_singularity_issue(dataset, independent_variable, response_variable, drop_first):

    # convert the random effect column to dummy variables (one-hot encoding)
    df_encoded = pd.get_dummies(dataset[[independent_variable,response_variable]], columns=[independent_variable], drop_first=drop_first)
    
    # define the independent variables (the one-hot encoded columns) and the dependent variable
    X = df_encoded.drop(columns=[response_variable]).astype(int)
    y = df_encoded[response_variable]
    
    # adding a constant term helps in understanding the multicollinearity of the model while accounting for the baseline effect
    X = sm.add_constant(X)
    
    # check for singularity by examining the rank of the design matrix
    rank        = np.linalg.matrix_rank(X)   
    # check eigenvalues of the correlation matrix
    eigenvalues = np.linalg.eigvals(X.T @ X)
    
    # Check if any eigenvalues are close to zero (indicative of singularity)
    if np.any(np.isclose(eigenvalues, 0)):
        return True
    else:
        return False

############################################################################################
############################################################################################
# REGRESSION MODELS ########################################################################
############################################################################################
############################################################################################

def train_OLS(dataset, predictors, target):
    X       = dataset[predictors].values
    y       = dataset[target].values 
    X       = sm.add_constant(X)
    model   = sm.OLS(y, X).fit()
    return model.rsquared_adj
    
def measure_adjusted_r2_in_grid_bins_along_axis(dataset, predictors, n_bins, axis):

    if(axis=="x"):
        grid_feature = "grid_bin_x"
    elif(axis=="y"):
        grid_feature = "grid_bin_y"
    elif(axis=="z"):
        grid_feature = "grid_bin_z"
    
    df_R_square = pd.DataFrame(columns=["axis", "grid_bin","adjusted_r2"])
    
    for bin in range(n_bins):
        
        dataset_bin     = dataset[dataset[grid_feature]==bin]
        row             = {}
        row["axis"]     = axis
        row["grid_bin"] = bin
        
        if(len(dataset_bin)>0):
            adjusted_r2        = train_OLS(dataset_bin, predictors, target="severity_numeric")
            row["adjusted_r2"] = adjusted_r2
        else:
            row["adjusted_r2"] = 0
    
        df_R_square.loc[len(df_R_square)]  = row

    return df_R_square

def measure_adjusted_r2_in_grid_cells_for_frequency_bands(dataset, n_bins, frequency_band, target_feature):
    if(frequency_band=="theta"):
        predictors = ['pre_event_theta_mean', 'event_theta_mean', 'post_event_theta_mean']
    elif(frequency_band=="beta_low"):
        predictors = ['pre_event_beta_low_mean', 'event_beta_low_mean', 'post_event_beta_low_mean']
    elif(frequency_band=="beta_high"):
        predictors = ['pre_event_beta_high_mean', 'event_beta_high_mean', 'post_event_beta_high_mean']
    elif(frequency_band=="gamma"):
        predictors = ['pre_event_gamma_mean', 'event_gamma_mean', 'post_event_gamma_mean']
        
    df_cell_R_square = pd.DataFrame(columns=["frequency_band","grid_bin_x","grid_bin_y","grid_bin_z","adjusted_r2"])
    
    for x in range(n_bins):
        for y in range(n_bins):
            for z in range(n_bins):
    
                cell_dynamics = dataset[(dataset.grid_bin_x==x) & (dataset.grid_bin_y==y) & (dataset.grid_bin_z ==z)]
                
                if(len(cell_dynamics)!=0): # if we have contacts in the selected grid cell (x,y,z, axes combined)
                    
                    cell_adjusted_r2      = train_OLS(cell_dynamics, predictors, target=target_feature)
                    cell_adjusted_r2      = max(cell_adjusted_r2, 0) # <0, than make it 0
                        
                    row                   = {}
                    row["frequency_band"] = frequency_band
                    row["grid_bin_x"]     = x
                    row["grid_bin_y"]     = y
                    row["grid_bin_z"]     = z
                    row["adjusted_r2"]    = cell_adjusted_r2
                    df_cell_R_square.loc[len(df_cell_R_square)] = row

    return df_cell_R_square
            



