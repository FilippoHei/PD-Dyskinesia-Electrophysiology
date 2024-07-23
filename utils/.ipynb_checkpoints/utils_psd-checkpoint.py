"""
Power spectral utilisation functions
"""

import pandas as pd
from scipy import signal
import numpy as np

"""
def measure_normalized_psd(recording, fs):
    
    wlength     = fs        
    noverlap    = wlength/2

    if(len(recording)< fs):
        recording = np.pad(recording, (0, fs-len(recording)), mode='constant')

    # replace np.nan values with 0, if any
    recording = np.array(recording)
    recording[np.isnan(recording)] = 0

    freq, psd   = signal.welch(recording, fs=fs, window='hamming', nperseg=wlength, noverlap=noverlap, nfft=wlength*2)
    total_power = np.trapz(psd[((freq>=5) & (freq<=45)) | ((freq>=55) & (freq<=95))]) # the total power between 5-45 and 55-95 Hz
    psd_norm    = psd / total_power # normalize psd based on the total power

    # focus only Hz<=100
    psd_norm    = psd_norm[(freq>=4) & (freq<=100)]
    freq        = freq[(freq>=4) & (freq<=100)]

    return freq, psd_norm
"""

def measure_normalized_psd(recording, fs):
    
    wlength     = fs/2 
    noverlap    = 0

    if(len(recording)< fs):
        recording = np.pad(recording, (0, fs-len(recording)), mode='constant')

    # replace np.nan values with 0, if any
    recording = np.array(recording)
    recording[np.isnan(recording)] = 0

    freq, psd   = signal.welch(recording, fs=fs, window='hamming', nperseg=wlength, noverlap=noverlap, nfft=fs)
    total_power = np.trapz(psd[((freq>=5) & (freq<=45)) | ((freq>=55) & (freq<=95))]) # the total power between 5-45 and 55-95 Hz
    psd_norm    = psd / total_power # normalize psd based on the total power

    # focus only Hz<=100
    psd_norm    = psd_norm[(freq>=4) & (freq<=100)]
    freq        = freq[(freq>=4) & (freq<=100)]

    return freq, psd_norm
