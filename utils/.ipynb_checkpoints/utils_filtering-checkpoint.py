"""
filtering utilization functions
"""

import pandas as pd
import mne
from scipy.signal import butter, lfilter, freqz

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y    = lfilter(b, a, data)
    return y
    
def bandpass_filter(data, fs, l_freq, h_freq):
    return mne.filter.filter_data(data=data ,sfreq=fs, l_freq=l_freq, h_freq=h_freq, method='fir', fir_window='hamming', verbose=False)