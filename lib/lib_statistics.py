import pingouin as pg
import pandas as pd
import numpy as np




class NON_PARAMETRIC_TEST:
    
    def kruskall_wallis(dataset, group_variable, feature_set, subset_variable=""):

        if(subset_variable==""):
            
            # store feature and corresponding corrected p-values in dataframe
            kw_results = pd.DataFrame(columns=["feature", "pvalue"])
            
            p_uncorrected = []
            for feature in feature_set:
                kw    = pg.kruskal(data=dataset, dv=feature, between=group_variable)
                p     = kw.iloc[0]["p-unc"]
                p_uncorrected.append(p)
            
            # correct p-values with Holm-Bonferroni correction
            p_corrected = pg.multicomp(pvals=p_uncorrected, method='holm')[1]
            
            for i in range(len(feature_set)):
                row                             = {}
                row["feature"]                  = feature_set[i]
                row["pvalue"]                   = p_corrected[i]
                kw_results.loc[len(kw_results)] = row
        else:
            # store feature and corresponding corrected p-values in dataframe
            kw_results = pd.DataFrame(columns=[subset_variable, "feature", "pvalue"])
            
            p_uncorrected = []
            for subset in dataset[subset_variable].unique():
                for feature in feature_set:
                    kw    = pg.kruskal(data=dataset, dv=feature, between=group_variable)
                    p     = kw.iloc[0]["p-unc"]
                    p_uncorrected.append(p)
                
                # correct p-values with Holm-Bonferroni correction
                p_corrected = pg.multicomp(pvals=p_uncorrected, method='holm')[1]
                
                
                for i in range(len(feature_set)):
                    row                             = {}
                    row[subset_variable]            = subset
                    row["feature"]                  = feature_set[i]
                    row["pvalue"]                   = p_corrected[i]
                    kw_results.loc[len(kw_results)] = row
    
        return kw_results



