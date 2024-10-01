"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_plotting, utils_accelerometer

from lib_data import DATA_IO

def plot_tapping_period(SUB_ACCELEROMETER, tapping_period_indices, tapping_period_no):
    t_start_index  = tapping_period_indices[tapping_period_no][0]
    t_finish_index = tapping_period_indices[tapping_period_no][1]
    
    t_period = np.linspace(SUB_ACCELEROMETER.times[t_start_index]/60, SUB_ACCELEROMETER.times[t_finish_index]/60, t_finish_index-t_start_index)
    period_R = SUB_ACCELEROMETER.ACC_R[t_start_index:t_finish_index]
    period_L = SUB_ACCELEROMETER.ACC_L[t_start_index:t_finish_index]
    plt.plot(t_period, period_R, "r", label="right hand")
    plt.plot(t_period, period_L, "b", label="left hand")
    plt.ylim([0, 5e-6])
    plt.legend(loc="upper right")
    plt.xlabel("time [minutes]")
    plt.title("Patient " + SUB_ACCELEROMETER.SUB + " - Tapping Period No: " + str(tapping_period_no+1))
    plt.show()
    
