"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from decimal import Decimal
from matplotlib.colors import BoundaryNorm

LABEL_SIZE       = 5
LABEL_SIZE_label = 6 
LABEL_SIZE_title = 7 

# color dataframe
colors                            = {}

colors["task_tapping"]            = "#d6573a"
colors["task_rest"]               = "#73a4a8"
colors["task_free"]               = "#646198"

colors["tapping"]                 = {}
colors["tapping"]["none"]         = "#93B93C"
colors["tapping"]["mild"]         = "#EF8A06"
colors["tapping"]["moderate"]     = "#DC2F02"
colors["tapping"]["severe"]       = "#9D0208"
colors["tapping"]["extreme"]      = "#370617"

colors["no_LID"]                  = "#93B93C"
colors["no_LID_no_DOPA"]          = "#386641"
colors["no_LID_DOPA"]             = "#A7C957"

def get_figure_template():
    
    plt.rc('font', serif="Neue Haas Grotesk Text Pro")
    fig = plt.figure(edgecolor='none')
    fig.tight_layout()
    fig.patch.set_visible(False)
    cm = 1/2.54  # centimeters in inches
    plt.subplots(figsize=(18.5*cm, 21*cm))
    plt.axis('off') 
    return plt

def get_polar_figure_template():
    
    plt.rc('font', serif="Neue Haas Grotesk Text Pro")
    fig = plt.figure(edgecolor='none')
    fig.tight_layout()
    fig.patch.set_visible(False)
    cm = 1/2.54  # centimeters in inches
    plt.subplots(figsize=(18.5*cm, 21*cm), subplot_kw=dict(polar=True))
    plt.axis('off') 
    return plt

def set_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=LABEL_SIZE)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE)
    ax.yaxis.offsetText.set_fontsize(LABEL_SIZE)

def count_plot(data, feat_x, hue, axis, color_palette, order, hue_order, legend=False):
    if(hue==""):
        if(order==""):
            axis = sns.countplot(data=data, x=feat_x, orient="h", palette=color_palette, hue_order=hue_order, legend=legend, ax=axis)
        else:
            axis = sns.countplot(data=data, x=feat_x, orient="h", palette=color_palette, order=order, hue_order=hue_order, legend=legend, ax=axis)
    else:
        if(order==""):
            axis = sns.countplot(data=data, x=feat_x, hue=hue, orient="h", palette=color_palette, hue_order=hue_order, legend=legend, ax=axis)
        else:
            axis = sns.countplot(data=data, x=feat_x, hue=hue, orient="h", palette=color_palette, order=order, hue_order=hue_order, legend=legend, ax=axis)

    axis.set_xlabel("")
    set_axis(axis)
    
    axis.legend(loc="lower center", fontsize=LABEL_SIZE, bbox_to_anchor=(0.5, -0.6), title=None, ncol=2, frameon=False)    
    
    return axis

def boxplot(data, feat_x, feat_y, hue, axis, color_palette, order, hue_order, orient, legend=False, show_whiskers=True):
    if(hue==""):
        if(order==""):
            if(show_whiskers==True):
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, palette=color_palette, showfliers=False, 
                                   hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
            else:
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, palette=color_palette, showfliers=False,
                                   whiskerprops={'linewidth': 0}, flierprops={'marker': 'o'}, capprops={'linewidth': 0},
                                   hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
        else:
            if(show_whiskers==True):
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, palette=color_palette, showfliers=False, 
                                   order=order, hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
            else:
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, palette=color_palette, showfliers=False, 
                                   whiskerprops={'linewidth': 0}, flierprops={'marker': 'o'}, capprops={'linewidth': 0},
                                   order=order, hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
    else:
        if(order==""):
            if(show_whiskers==True):
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, hue=hue, palette=color_palette, showfliers=False, 
                                   hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
            else:
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, hue=hue, palette=color_palette, showfliers=False, 
                                   whiskerprops={'linewidth': 0}, flierprops={'marker': 'o'}, capprops={'linewidth': 0},
                                   hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
        else:
            if(show_whiskers==True):
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, hue=hue, palette=color_palette, showfliers=False, 
                                   order=order, hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
            else:
                axis = sns.boxplot(data=data, x=feat_x, y=feat_y, hue=hue, palette=color_palette, showfliers=False, 
                                   whiskerprops={'linewidth': 0}, flierprops={'marker': 'o'}, capprops={'linewidth': 0},
                                   order=order, hue_order=hue_order, orient=orient, width=0.5, linewidth=0.5, legend=legend, ax=axis)
    
    axis.set_xlabel("")
    set_axis(axis)
    
    axis.legend(loc="lower center", fontsize=LABEL_SIZE, bbox_to_anchor=(0.5, -0.6), title=None, ncol=2, frameon=False)
    
    return axis

def heatmap_significance(dataset, order, axis):
    
    stat_heatmap = dataset.pivot(index="group1", columns="group2", values="pvalue")
    stat_heatmap = stat_heatmap[order]
    stat_heatmap = stat_heatmap.reindex(order)
    stat_heatmap.replace(np.nan, 1, inplace=True)

    bounds  = [0, 0.001, 0.01, 0.05, 1]
    heat_c  = ['seagreen', 'mediumseagreen', 'lightgreen', 'white']
    heat_n  = BoundaryNorm(bounds, ncolors=len(heat_c))
    val_min = 0
    val_max = 0.05

    annot   = pd.DataFrame(columns=order, index=order)
    for group1 in order:
        for group2 in order:
            pval = float(stat_heatmap.loc[group1,group2])
            if((pval > 0.01) & (pval <= 0.05)):
                pval = 2
            elif((pval > 0.001) & (pval <= 0.01)):
                pval = 3
            elif(pval <= 0.001):
                pval = 4
            else:
                pval = 1
            pval = format(pval, '.2f')
            annot.at[group1,group2] = pval

    for group in order:
        annot.loc[annot[group] == "2.00", group] = "*"
        annot.loc[annot[group] == "3.00", group] = "**"
        annot.loc[annot[group] == "4.00", group] = "***"
        
    axis = sns.heatmap(stat_heatmap, vmin=val_min, vmax=val_max, cmap=heat_c, norm = heat_n, 
                       annot=annot, annot_kws={"fontsize":LABEL_SIZE}, cbar=False, fmt="", ax=axis)
        
    axis.set_yticks([])
    axis.set_xlabel('')
    axis.set_ylabel('')
    axis.set_facecolor("white")
    axis.set_xticklabels(order, rotation = 90)
    set_axis(axis)
    return axis

def plot_coefficient_heatmap(df, features, alpha, ax):
    df['group_combination'] = df['group_1'] + ' vs ' + df['group_2']
    df['significant'] = df['p_value_corrected'] <= alpha
    
    # Pivot the DataFrame for the heatmap and significance mask
    heatmap_data = df.pivot(index='feature', columns='group_combination', values='coefficient')
    heatmap_data = heatmap_data.reindex(index=features)
    
    heatmap_data = heatmap_data[['noLID_noDOPA vs noLID_DOPA', 'noLID_noDOPA vs mild', 'noLID_noDOPA vs moderate', 
                                  'noLID_DOPA vs mild', 'noLID_DOPA vs moderate', 'mild vs moderate']]
    
    heatmap_significance = df.pivot(index='feature', columns='group_combination', values='significant')
    heatmap_significance = heatmap_significance.reindex(index=features)
    heatmap_significance = heatmap_significance[['noLID_noDOPA vs noLID_DOPA', 'noLID_noDOPA vs mild', 
                                                 'noLID_noDOPA vs moderate', 'noLID_DOPA vs mild', 
                                                 'noLID_DOPA vs moderate', 'mild vs moderate']]
    
    # Plot heatmap on the provided axis
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Coefficient'},
                annot_kws={"fontsize":LABEL_SIZE},linewidths=0, linecolor='black', mask=~heatmap_significance, 
                vmin=-75, vmax=75, ax=ax)
    
    
    # Add vertical lines
    ax.axvline(x=3, color='black', linestyle='-', linewidth=2)  # Vertical line at x=3
    ax.axvline(x=5, color='black', linestyle='-', linewidth=2)  # Vertical line at x=5
    
    # Set x-axis ticks
    ax.set_xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels(['noLID-DOPA', 'mild', 'moderate', 'mild', 'moderate', 'moderate'], rotation=0, fontsize=LABEL_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=LABEL_SIZE)
    # Create a twin axis for the top ticks
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # Ensure the limits match
    ax2.set_xticks(ticks=[1.5, 4, 5.5])
    ax2.set_xticklabels(['noLID-noDOPA', 'noLID-DOPA', 'mild'], rotation=0, fontsize=LABEL_SIZE) 
    

    # Remove spines
    ax2.spines.left.set_visible(False)
    ax2.spines.bottom.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    # Remove the axes' borders (if still visible)
    ax.xaxis.set_ticks_position('none')  # Hide the x-axis ticks
    ax.yaxis.set_ticks_position('none')  # Hide the y-axis ticks
    ax2.xaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks

    ax.set_xlabel("")  # Optionally clear the x-label
    ax.set_ylabel("")  # Optionally clear the y-label
    return ax

def plot_coefficient_heatmap(df, features, alpha, ax):
    df['group_combination'] = df['group_1'] + ' vs ' + df['group_2']
    df['significant'] = df['p_value_corrected'] <= alpha
    
    # Pivot the DataFrame for the heatmap and significance mask
    heatmap_data = df.pivot(index='feature', columns='group_combination', values='coefficient')
    heatmap_data = heatmap_data.reindex(index=features)
    heatmap_data = heatmap_data[['noLID_noDOPA vs noLID_DOPA', 'noLID_noDOPA vs mild', 'noLID_noDOPA vs moderate', 
                                  'noLID_DOPA vs mild', 'noLID_DOPA vs moderate', 'mild vs moderate']]
    
    heatmap_significance = df.pivot(index='feature', columns='group_combination', values='significant')
    heatmap_significance = heatmap_significance.reindex(index=features)
    heatmap_significance = heatmap_significance[['noLID_noDOPA vs noLID_DOPA', 'noLID_noDOPA vs mild', 
                                                 'noLID_noDOPA vs moderate', 'noLID_DOPA vs mild', 
                                                 'noLID_DOPA vs moderate', 'mild vs moderate']]
    
    # Plot heatmap with color mask applied
    sns.heatmap(heatmap_data, annot=False, cmap="coolwarm", cbar_kws={'label': 'Coefficient'},
                linewidths=0, linecolor='black', mask=~heatmap_significance,  # Mask applies only to color
                vmin=-75, vmax=75, ax=ax)

    # Overlay annotations explicitly
    for i in range(heatmap_data.shape[0]):  # Loop over rows
        for j in range(heatmap_data.shape[1]):  # Loop over columns
            value = heatmap_data.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f"{value:.1f}", 
                    ha='center', va='center', fontsize=LABEL_SIZE, color='black')

    # Add vertical lines
    ax.axvline(x=3, color='black', linestyle='-', linewidth=2)  # Vertical line at x=3
    ax.axvline(x=5, color='black', linestyle='-', linewidth=2)  # Vertical line at x=5
    
    # Set x-axis ticks
    ax.set_xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels(['noLID-DOPA', 'mild', 'moderate', 'mild', 'moderate', 'moderate'], rotation=0, fontsize=LABEL_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=LABEL_SIZE)
    
    # Create a twin axis for the top ticks
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # Ensure the limits match
    ax2.set_xticks(ticks=[1.5, 4, 5.5])
    ax2.set_xticklabels(['noLID-noDOPA', 'noLID-DOPA', 'mild'], rotation=0, fontsize=LABEL_SIZE) 
    
    # Remove spines
    ax2.spines.left.set_visible(False)
    ax2.spines.bottom.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    # Remove the axes' borders (if still visible)
    ax.xaxis.set_ticks_position('none')  # Hide the x-axis ticks
    ax.yaxis.set_ticks_position('none')  # Hide the y-axis ticks
    ax2.xaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks

    ax.set_xlabel("")  # Optionally clear the x-label
    ax.set_ylabel("")  # Optionally clear the y-label
    return ax