"""
Power spectral utilisation functions
"""

import pandas as pd
from scipy import signal
import numpy as np
import sys
import pickle

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io

from lib_ECoG import ECoG
from lib_LFP import LFP
from lib_data import DATA_IO

def measure_absolute_coherence(x, y, fs):
    
    wlength  = int(fs/4) # 250 ms window length
    noverlap = wlength/2 # 50% overlap
    f, Cxy   = signal.coherence(x, y, nperseg=wlength, noverlap=noverlap, nfft=fs, fs=fs)
    Cxy      = Cxy[(f>=4) & (f<=100)]
    f        = f[(f>=4) & (f<=100)]
    
    return f, Cxy

def extract_baseline_coherence_between_ECOG_LFP_channels():

    fs                          = 2048
    print("ECOG...")
    BASELINE_ECOG               = utils_io.load_baseline_recordings(recording_type="ECoG")
    print("LFP...")
    BASELINE_LFP                = utils_io.load_baseline_recordings(recording_type="LFP")
    baseline_ECOG_LFP_coherence = pd.DataFrame(columns=["patient", "ECOG_hemisphere", "LFP_hemisphere", "ECOG_channel", "LFP_channel", "coherence"])
    
    for patient in BASELINE_ECOG.keys():
        
        for h_ECOG in ["right", "left"]:
            for h_LFP in ["right", "left"]:
            
                channels_ECOG = BASELINE_ECOG[patient][h_ECOG].keys()
                channels_LFP  = BASELINE_LFP[patient][h_LFP].keys()
                
                for c_ECOG in channels_ECOG:
                    for c_LFP in channels_LFP:
        
                        
                        # measure the baseline coherence between two channels 
                        ECOG_recording = BASELINE_ECOG[patient][h_ECOG][c_ECOG]
                        LFP_recording  = BASELINE_LFP[patient][h_LFP][c_LFP]
                        
                        # in case of nan values, replace them with 0
                        ECOG_recording[np.isnan(ECOG_recording)] = 0
                        LFP_recording[np.isnan(LFP_recording)] = 0
        
                        f, Cxy = measure_absolute_coherence(ECOG_recording, LFP_recording, fs=fs)
        
                        # save to the dataframe
                        row                    = {}
                        row["patient"]         = patient
                        row["ECOG_hemisphere"] = h_ECOG
                        row["LFP_hemisphere"]  = h_LFP
                        row["ECOG_channel"]    = c_ECOG
                        row["LFP_channel"]     = c_LFP
                        row["coherence"]       = Cxy

                        if(~all(row["coherence"])): #if coherence array is not completerly np.nan
                            baseline_ECOG_LFP_coherence.loc[len(baseline_ECOG_LFP_coherence)] = row
                        
    return baseline_ECOG_LFP_coherence
                