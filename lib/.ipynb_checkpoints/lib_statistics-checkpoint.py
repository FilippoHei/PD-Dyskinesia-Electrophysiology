import pingouin as pg
import pandas as pd
import numpy as np
import scikit_posthocs as sp

class NON_PARAMETRIC_TEST:
    
    def kruskall_wallis(dataset, group_variable, feature_set, subset_variable=""):

        if(subset_variable==""):
            
            # store feature and corresponding corrected p-values in dataframe
            test_results  = pd.DataFrame(columns=["feature", "pvalue"])
            p_uncorrected = []
            
            for feature in feature_set:
                
                kw    = pg.kruskal(data=dataset, dv=feature, between=group_variable)
                p     = kw.iloc[0]["p-unc"]
                p_uncorrected.append(p)
            
            # correct p-values with Holm-Bonferroni correction
            p_corrected = pg.multicomp(pvals=p_uncorrected, method='holm')[1]
            
            for i in range(len(feature_set)):
                row                                 = {}
                row["feature"]                      = feature_set[i]
                row["pvalue"]                       = p_corrected[i]
                test_results.loc[len(test_results)] = row
        else:
            
            # store feature and corresponding corrected p-values in dataframe
            test_results  = pd.DataFrame(columns=[subset_variable, "feature", "pvalue"])
            p_uncorrected = []
            
            for subset in dataset[subset_variable].unique():

                dataset_subset = dataset[dataset[subset_variable]==subset]
                
                for feature in feature_set:
                    
                    kw    = pg.kruskal(data=dataset_subset, dv=feature, between=group_variable)
                    p     = kw.iloc[0]["p-unc"]
                    p_uncorrected.append(p)
                
                # correct p-values with Holm-Bonferroni correction
                p_corrected = pg.multicomp(pvals=p_uncorrected, method='holm')[1]
                
                
                for i in range(len(feature_set)):
                    row                                 = {}
                    row[subset_variable]                = subset
                    row["feature"]                      = feature_set[i]
                    row["pvalue"]                       = p_corrected[i]
                    test_results.loc[len(test_results)] = row
    
        return test_results


    def dunn_test(dataset, group_variable, feature_set, subset_variable=""):
        
        if(subset_variable==""):
        
            test_results = pd.DataFrame(columns=["feature", "group1", "group2", "pvalue"])
            
            for feature in feature_set:
                
                res = sp.posthoc_dunn(dataset, group_col=group_variable, val_col=feature,  p_adjust='holm')
                
                for group1 in res.columns:
                    for group2 in res.columns:
                        
                        if(group1!=group2):
                            row                                 = {}
                            row["feature"]                      = feature
                            row["group1"]                       = group1
                            row["group2"]                       = group2
                            row["pvalue"]                       = res[group1][group2]
                            test_results.loc[len(test_results)] = row
        else:
            
            test_results = pd.DataFrame(columns=[subset_variable, "feature", "group1", "group2", "pvalue"])
            
            for subset in dataset[subset_variable].unique():
                
                dataset_subset = dataset[dataset[subset_variable]==subset]
        
                for feature in feature_set:
                    
                    res = sp.posthoc_dunn(dataset_subset, group_col=group_variable, val_col=feature,  p_adjust='holm')
            
                    for group1 in res.columns:
                        for group2 in res.columns:
                            
                            if(group1!=group2):
                                row                                 = {}
                                row[subset_variable]                = subset
                                row["feature"]                      = feature
                                row["group1"]                       = group1
                                row["group2"]                       = group2
                                row["pvalue"]                       = res[group1][group2]
                                test_results.loc[len(test_results)] = row
        return test_results
