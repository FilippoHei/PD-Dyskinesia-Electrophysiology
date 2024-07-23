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

LABEL_SIZE       = 5
LABEL_SIZE_label = 6 
LABEL_SIZE_title = 7 

# color dataframe
colors                            = {}

colors["tapping"]                 = "#d6573a"
colors["rest"]                    = "#73a4a8"
colors["free"]                    = "#646198"

colors["tapping_movement"]        = "#ef6351"


colors["voluntary"]               = {}
colors["voluntary"]["none"]       = "#09686B"
colors["voluntary"]["mild"]       = "#EF8A06"
colors["voluntary"]["moderate"]   = "#DC2F02"
colors["voluntary"]["severe"]     = "#9D0208"
colors["voluntary"]["extreme"]    = "#370617"

colors["involuntary"]             = {}
colors["involuntary"]["mild"]     = "#00AACC"
colors["involuntary"]["moderate"] = "#006AA3"
colors["involuntary"]["severe"]   = "#023579"
colors["involuntary"]["extreme"]  = "#03045E"

colors["no_LID"]                  = "#93B93C"
colors["no_LID_no_DOPA"]          = "#386641"
colors["no_LID_DOPA"]             = "#A7C957"

def get_figure_template():
    
    plt.rc('font', serif="Neue Haas Grotesk Text Pro")
    fig = plt.figure()
    fig.tight_layout()
    cm = 1/2.54  # centimeters in inches
    plt.subplots(figsize=(18.5*cm, 21*cm))
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