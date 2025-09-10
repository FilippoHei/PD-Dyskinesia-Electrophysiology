import pandas as pd
import numpy as np
import sys
import os
import math
import pingouin as pg

from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from itertools import product

from sklearn import manifold
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

############################################################################################
############################################################################################
# PCA MODEL ################################################################################
############################################################################################
############################################################################################

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from scipy.stats import ttest_ind
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import numpy as np

def get_kruskal_of_principal_component(pca_space, label_feature, principal_axis):
    groups      = pca_space[label_feature].unique()
    data_groups = [pca_space[pca_space[label_feature] == g][principal_axis].values for g in groups]
    stat, p_val = kruskal(*data_groups)
    return p_val
    
def get_ttest_of_principal_component(pca_space, label_feature, principal_axis):

    groups = pca_space[label_feature].unique()
    if(len(groups) != 2): raise ValueError(f"Expected exactly two groups, found {len(groups)}: {groups}")

    g1, g2 = groups
    x1     = pca_space[pca_space[label_feature]==g1][principal_axis].values
    x2     = pca_space[pca_space[label_feature]==g2][principal_axis].values
    t_stat, p_val = ttest_ind(x1, x2, equal_var=False)
    
    return p_val
    
def get_js_distance_of_principal_component(pca_space, label_feature, principal_axis):
    # Extract the two groups
    groups     = pca_space[label_feature].unique()
    if(len(groups) != 2): raise ValueError(f"Expected exactly two groups, found {len(groups)}: {groups}")
    
    g1, g2     = groups[0], groups[1]
    x1         = pca_space[pca_space[label_feature] == g1][principal_axis].values
    x2         = pca_space[pca_space[label_feature] == g2][principal_axis].values

    
    xmin, xmax = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    xs         = np.linspace(xmin, xmax, 200)
    pdf1       = gaussian_kde(x1)(xs)
    pdf2       = gaussian_kde(x2)(xs)

    # Normalize to probability distributions
    pdf1      /= pdf1.sum()
    pdf2      /= pdf2.sum()

    # Compute Jensen-Shannon distance
    js_dist    = jensenshannon(pdf1, pdf2)
    return js_dist

def get_cohens_d_of_princial_component(pca_space, label_feature, principal_axis):
    cohen_d = np.abs(pg.compute_effsize(pca_space[pca_space[label_feature]==pca_space.severity.unique()[0]][principal_axis], 
                                        pca_space[pca_space[label_feature]==pca_space.severity.unique()[1]][principal_axis], 
                                        eftype='cohen'))
    return cohen_d

def angle_between_PCA_spaces(PCA1_components, PCA2_components):
    from scipy.linalg import subspace_angles
    angles    = subspace_angles(PCA1_components, PCA2_components)
    PC1_angle = angles[0]
    PC2_angle = angles[1]
    return radians_to_degrees(PC1_angle), radians_to_degrees(PC2_angle)

def radians_to_degrees(radian_value):
    return radian_value * 180 / math.pi

def get_PCA_results(dataframe, label_feature, features_pre_event, features_event, features_post_event):
    
    pre_data_pca  , pre_comp  , pre_explained_variance  , pre_pca_contributions   = build_PCA_model(dataframe, features=features_pre_event, label=label_feature, n_components=3)
    event_data_pca, event_comp, event_explained_variance, event_pca_contributions = build_PCA_model(dataframe, features=features_event, label=label_feature, n_components=3)
    post_data_pca , post_comp , post_explained_variance , post_pca_contributions  = build_PCA_model(dataframe, features=features_post_event, label=label_feature, n_components=3)

    pre_pca_contributions["segment"]   = "pre"
    event_pca_contributions["segment"] = "event"
    post_pca_contributions["segment"]  = "post"
    
    results                            = dict()
    results["pre"]                     = dict()
    results["event"]                   = dict()
    results["post"]                    = dict()
    
    results["pre"]["pca"]              = pre_data_pca
    results["pre"]["components"]       = pre_comp
    results["pre"]["ev"]               = pre_explained_variance
    results["pre"]["loadings"]         = pre_pca_contributions
    results["pre"]["t_test_PC1"]       = get_ttest_of_principal_component(pre_data_pca, "severity", principal_axis="PC1")
    results["pre"]["t_test_PC2"]       = get_ttest_of_principal_component(pre_data_pca, "severity", principal_axis="PC2")
    results["pre"]["cohens_d_PC2"]     = get_cohens_d_of_princial_component(pre_data_pca, "severity", principal_axis="PC2")
    results["pre"]["cohens_d_PC1"]     = get_cohens_d_of_princial_component(pre_data_pca, "severity", principal_axis="PC1")
    results["pre"]["js_PC1"]           = get_js_distance_of_principal_component(pre_data_pca, "severity", principal_axis="PC1")
    results["pre"]["js_PC2"]           = get_js_distance_of_principal_component(pre_data_pca, "severity", principal_axis="PC2")
    
    results["event"]["pca"]            = event_data_pca
    results["event"]["components"]     = event_comp
    results["event"]["ev"]             = event_explained_variance
    results["event"]["loadings"]       = event_pca_contributions
    results["event"]["t_test_PC1"]     = get_ttest_of_principal_component(event_data_pca, "severity", principal_axis="PC1")
    results["event"]["t_test_PC2"]     = get_ttest_of_principal_component(event_data_pca, "severity", principal_axis="PC2")
    results["event"]["cohens_d_PC1"]   = get_cohens_d_of_princial_component(event_data_pca, "severity", principal_axis="PC1")
    results["event"]["cohens_d_PC2"]   = get_cohens_d_of_princial_component(event_data_pca, "severity", principal_axis="PC2")
    results["event"]["js_PC1"]         = get_js_distance_of_principal_component(event_data_pca, "severity", principal_axis="PC1")
    results["event"]["js_PC2"]         = get_js_distance_of_principal_component(event_data_pca, "severity", principal_axis="PC2")
    
    results["post"]["pca"]             = post_data_pca
    results["post"]["components"]      = post_comp
    results["post"]["ev"]              = post_explained_variance
    results["post"]["loadings"]        = post_pca_contributions
    results["post"]["t_test_PC1"]      = get_ttest_of_principal_component(post_data_pca, "severity", principal_axis="PC1")
    results["post"]["t_test_PC2"]      = get_ttest_of_principal_component(post_data_pca, "severity", principal_axis="PC2")
    results["post"]["cohens_d_PC1"]    = get_cohens_d_of_princial_component(post_data_pca, "severity", principal_axis="PC1")
    results["post"]["cohens_d_PC2"]    = get_cohens_d_of_princial_component(post_data_pca, "severity", principal_axis="PC2")
    results["post"]["js_PC1"]          = get_js_distance_of_principal_component(post_data_pca, "severity", principal_axis="PC1")
    results["post"]["js_PC2"]          = get_js_distance_of_principal_component(post_data_pca, "severity", principal_axis="PC2")

    df_distances                       = pd.DataFrame()
    df_distances["segment"]            = ["pre","pre","event","event","post","post"]
    df_distances["pc"]                 = ["pc1","pc2","pc1","pc2","pc1","pc2"]
    df_distances["p_value"]            = [results["pre"]["t_test_PC1"]  , results["pre"]["t_test_PC2"], 
                                          results["event"]["t_test_PC1"], results["event"]["t_test_PC2"],
                                          results["post"]["t_test_PC1"] , results["post"]["t_test_PC2"]]
    df_distances["cohens_d"]           = [results["pre"]["cohens_d_PC1"]  , results["pre"]["cohens_d_PC2"], 
                                          results["event"]["cohens_d_PC1"], results["event"]["cohens_d_PC2"],
                                          results["post"]["cohens_d_PC1"] , results["post"]["cohens_d_PC2"]]

    df_distances["p_value_corrected"]  = df_distances.groupby("segment")["p_value"].transform(lambda x: multipletests(x, method="holm")[1])
    df_distances.loc[df_distances["p_value_corrected"] > 0.05, "cohens_d"] = 0.05
    results["distances"]               = df_distances

    loadings                           = pd.concat([pre_pca_contributions, event_pca_contributions, post_pca_contributions], ignore_index=True)
    loadings["band"]                   = ["theta", "alpha", "beta_low", "beta_high", "gamma"] * 3   
    results["loadings"]                = loadings

    return results

def build_PCA_model(dataframe, features, label, n_components=5):

    X                    = dataframe[features]
    X_scaled             = StandardScaler().fit_transform(X)
    X                    = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    Y                    = dataframe[label]

    # apply PCA
    pca                  = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    data_pca = pd.DataFrame()
    for i in range(n_components):
        data_pca["PC"+str(i+1)] = principal_components[:,i]
    data_pca["severity"] = dataframe[label].to_list()

    # get feature contributions and total explained variance for each principal axis
    feature_contributions           = pca.components_.T * np.sqrt(pca.explained_variance_)
    explained_variance              = pd.DataFrame()
    explained_variance["component"] = data_pca.columns[0:-1].tolist()
    explained_variance["score"]     = np.transpose(pca.explained_variance_ratio_)

    pca_contributions = pd.DataFrame(feature_contributions, columns=data_pca.columns[0:-1].tolist(), index=X.columns).abs()
    pca_components    = pca.components_[:2].T #only first two components we are interested in

    return data_pca, pca_components, explained_variance, pca_contributions


def ch_score(dataset_pca, group_label, group_mapping):

    pca_features      = list(set(dataset_pca.columns).difference(set([group_label])))
    X                 = dataset_pca[pca_features]
    Y                 = dataset_pca[group_label]
    
    groups            = list(group_mapping.keys())
    calinski_harabasz = np.zeros([len(groups), len(groups)])
    
    for i in range(len(groups)):
        for j in range(len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            if(i!=j):
                X_temp = X[(Y==group1) | (Y==group2)].values
                labels = Y[(Y==group1) | (Y==group2)].map(group_mapping)
                calinski_harabasz[i][j] = calinski_harabasz_score(X_temp, labels)
            else:
                calinski_harabasz[i][j] = 0
    calinski_harabasz = pd.DataFrame(data=calinski_harabasz, columns=groups, index=groups)
    
    return calinski_harabasz
    

def manifold_lle(X, y, colors_group, parameter_grid):

    print("-------------------------------------------------------------------------------------") 
    print("-------------------------------- MANIFOLD GENERATION --------------------------------") 
    print("-------------------------------------------------------------------------------------") 
    print(" MANIFOLDING TECHNIQUE : Locally-Linear Embeddings")

    results_lle = pd.DataFrame(columns=["method", "n_components", "n_neighbors", "neighbors_algorithm", "silhouette", "ch_score", 
                                        "js_D1", "js_D2", "js_D3", "js_avg_pairwise", "transformed_data"])
    
    # create all combinations of parameters / grid
    param_combinations = list(product(*parameter_grid.values()))

    # iterate through each combination
    for combination in param_combinations:

        # parameter combination
        param_dict                   = dict(zip(parameter_grid.keys(), combination))

        # contruct the manifold
        lle_standard                 = manifold.LocallyLinearEmbedding(**param_dict)
        data_transformed             = lle_standard.fit_transform(X.copy())                     # reconstruct in manifold space
        data_transformed             = pd.DataFrame(data_transformed, columns=["D1","D2","D3"]) # turn datapoints into dataframe
        data_transformed["group"]    = y.copy()
        data_transformed["color"]    = data_transformed["group"].map(colors_group)

        # measure the cluster separation quality
        manifold_metrics             = cluster_separation_metrics(X=data_transformed[["D1","D2","D3"]], y=data_transformed.group)

        row                          = dict()
        row["method"]                = param_dict["method"]
        row["n_components"]          = param_dict["n_components"]
        row["n_neighbors"]           = param_dict["n_neighbors"]
        row["neighbors_algorithm"]   = param_dict["neighbors_algorithm"]
        row["silhouette"]            = manifold_metrics["silhouette"]
        row["ch_score"]              = manifold_metrics["ch_score"]
        row["js_D1"]                 = manifold_metrics["js_D1"]
        row["js_D2"]                 = manifold_metrics["js_D2"]
        row["js_D3"]                 = manifold_metrics["js_D3"]

        js_total                     = manifold_metrics["js_D1"] + manifold_metrics["js_D2"] + manifold_metrics["js_D3"]
        lower_triangle               = js_total.where(np.tril(np.ones(js_total.shape), k=-1).astype(bool))
        js_avg_pairwise              = lower_triangle.stack().mean()
        row["js_avg_pairwise"]       = js_avg_pairwise
        
        row["transformed_data"]      = data_transformed

        results_lle.loc[len(results_lle)] = row
        print(f" >> manifold construction with parameters: {param_dict}")
       
    return results_lle

def jensen_shannon_distance_with_KDE(list1, list2, grid_points=400):
    
    # convert the lists to numpy arrays
    list1  = np.array(list1)
    list2  = np.array(list2)
    
    # perform Kernel Density Estimation for both lists
    kde1   = gaussian_kde(list1)
    kde2   = gaussian_kde(list2)
    
    # define the grid for evaluation
    x_grid = np.linspace(min(min(list1), min(list2)), max(max(list1), max(list2)), grid_points)
    
    # compute the density values for both KDEs on the grid
    p      = kde1(x_grid)
    q      = kde2(x_grid)
    
    # normalize the distributions to sum to 1 to get probability distributions which is required JSD
    p     /= p.sum()
    q     /= q.sum()
    jsd    = jensenshannon(p, q)
    
    return jsd
    
def cluster_separation_metrics(X, y):
    
    metrics               = dict()
    metrics["ch_score"]   = calinski_harabasz_score(X=X, labels=y)
    metrics["silhouette"] = silhouette_score(X, y, metric='euclidean')

    for dimension in ["D1","D2","D3"]:
        distances = pd.DataFrame(index=y.unique(), columns=y.unique())
        for group1 in y.unique():
            for group2 in y.unique():
                distance = jensen_shannon_distance_with_KDE(X[y==group1][dimension],
                                                            X[y==group2][dimension], grid_points=500)
                distances.loc[group1, group2] = distance
        metrics["js_"+dimension] = distances
    return metrics  