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

from lib_ECoG import ECoG
from lib_LFP import LFP
from lib_data import DATA_IO

def measure_normalized_psd(recording, fs):
    
    wlength     = int(fs/4) # 250 ms window length
    noverlap    = wlength/2 # 50% overlap

    # replace np.nan values with 0, if any
    recording   = np.array(recording)
    recording[np.isnan(recording)] = 0

    try:
        # scenario 1
        
        freq, psd   = signal.welch(recording, fs=fs, window='hamming', nperseg=wlength, noverlap=noverlap, nfft=fs)
        total_power = np.trapz(psd[((freq>=5) & (freq<=45)) | ((freq>=55) & (freq<=95))]) # the total power between 5-45 and 55-95 Hz
        psd_norm    = psd / total_power # normalize psd based on the total power

        # focus only Hz<=100
        psd_norm    = psd_norm[(freq>=4) & (freq<=100)]
        freq        = freq[(freq>=4) & (freq<=100)]
    except:
        return [], []

    return freq, psd_norm


def measure_absolute_psd(recording, fs):
    
    wlength     = int(fs/4) # 250 ms window length
    noverlap    = wlength/2 # 50% overlap

    # replace np.nan values with 0, if any
    recording   = np.array(recording)
    recording[np.isnan(recording)] = 0

    try:
        freq, psd   = signal.welch(recording, fs=fs, window='hamming', nperseg=wlength, noverlap=noverlap, nfft=fs)
        # focus only Hz<=100
        psd    = psd[(freq>=4) & (freq<=100)]
        freq   = freq[(freq>=4) & (freq<=100)]
    except:
        return [], []

    return freq, psd

def psd_change_from_baseline_activity(psd_baseline, psd_event):
    
    psd_baseline = np.array(psd_baseline)
    psd_event = np.array(psd_event)

    # Calculate the percentage change
    percentage_change = (psd_event - psd_baseline) / psd_baseline * 100

    return percentage_change

def extract_normalized_psd_of_events(SUB, patient_dataset, patient_baseline, recording_type):

    # emptpy dataframe to save the results of spectral features
    if(recording_type=="LFP"):
        df_power_spectrum = pd.DataFrame(columns=['patient', 'event_no', 'event_category', 'event_laterality',
                                                  'event_start_time', 'duration', 'LFP_hemisphere', 'LFP_channel',
                                                  'pre_event_psd', 'event_psd', 'post_event_psd', 'dyskinesia_arm', 'dyskinesia_total',
                                                  # pre-event features
                                                  "pre_event_beta_peak", "pre_event_beta_peak_frequency", 
                                                  "pre_event_gamma_peak", "pre_event_gamma_peak_frequency", 
                                                  "pre_event_theta_mean", "pre_event_alpha_mean", "pre_event_beta_low_mean", 
                                                  "pre_event_beta_high_mean", "pre_event_gamma_mean",
                                                  # event features
                                                  "event_beta_peak", "event_beta_peak_frequency", 
                                                  "event_gamma_peak", "event_gamma_peak_frequency", 
                                                  "event_theta_mean", "event_alpha_mean", "event_beta_low_mean", 
                                                  "event_beta_high_mean", "event_gamma_mean",
                                                  # post-event features
                                                  "post_event_beta_peak", "post_event_beta_peak_frequency", 
                                                  "post_event_gamma_peak", "post_event_gamma_peak_frequency", 
                                                  "post_event_theta_mean", "post_event_alpha_mean", "post_event_beta_low_mean", 
                                                  "post_event_beta_high_mean", "post_event_gamma_mean"])
    else:
        df_power_spectrum = pd.DataFrame(columns=['patient', 'event_no', 'event_category', 'event_laterality',
                                                  'event_start_time', 'duration', 'ECoG_hemisphere', 'ECoG_channel',
                                                  'pre_event_psd', 'event_psd', 'post_event_psd', 'dyskinesia_arm', 'dyskinesia_total',
                                                  # pre-event features
                                                  "pre_event_beta_peak", "pre_event_beta_peak_frequency", 
                                                  "pre_event_gamma_peak", "pre_event_gamma_peak_frequency", 
                                                  "pre_event_theta_mean", "pre_event_alpha_mean", "pre_event_beta_low_mean", 
                                                  "pre_event_beta_high_mean", "pre_event_gamma_mean",
                                                  # event features
                                                  "event_beta_peak", "event_beta_peak_frequency", 
                                                  "event_gamma_peak", "event_gamma_peak_frequency", 
                                                  "event_theta_mean", "event_alpha_mean", "event_beta_low_mean", 
                                                  "event_beta_high_mean", "event_gamma_mean",
                                                  # post-event features
                                                  "post_event_beta_peak", "post_event_beta_peak_frequency", 
                                                  "post_event_gamma_peak", "post_event_gamma_peak_frequency", 
                                                  "post_event_theta_mean", "post_event_alpha_mean", "post_event_beta_low_mean", 
                                                  "post_event_beta_high_mean", "post_event_gamma_mean"])
    
    # iterate in the event dataframe
    for index, row in patient_dataset.iterrows():
    
        # get the time array between 0-5 mins
        if(recording_type=="LFP"):
            hemisphere   = row["LFP_hemisphere"]
            channel      = row["LFP_channel"]
        else:
            hemisphere   = row["ECoG_hemisphere"]
            channel      = row["ECoG_channel"]
        
        # get baseline recording for selected hemisphere and channel
        baseline_rec = patient_baseline[SUB][hemisphere][channel]
        
        # get baseline PSD from baseline recording
        freq, psd_base  = measure_absolute_psd(baseline_rec, fs=2048)
    
        # measure the PSD in pre-event, event and post-event segments
        freq, psd_pre   = measure_absolute_psd(row["pre_event_recording"], fs=2048)
        freq, psd_event = measure_absolute_psd(row["event_recording"], fs=2048)
        freq, psd_post  = measure_absolute_psd(row["post_event_recording"], fs=2048)
    
        # normalized the PSD of three segments based on baseline PSD
        psd_pre_event_norm  = psd_change_from_baseline_activity(psd_base, psd_pre)
        psd_event_norm      = psd_change_from_baseline_activity(psd_base, psd_event)
        psd_post_event_norm = psd_change_from_baseline_activity(psd_base, psd_post)

        psd_row                     = {} 
        psd_row['patient']          = row['patient'] 
        psd_row['event_no']         = row['event_no']
        psd_row['event_category']   = row['event_category']
        psd_row['event_laterality'] = row['event_laterality']
        psd_row['event_start_time'] = row['event_start_time']
        psd_row['duration']         = row['duration']

        if(recording_type=="LFP"):
            psd_row['LFP_hemisphere']   = row['LFP_hemisphere']
            psd_row['LFP_channel']      = row['LFP_channel']
        else:
            psd_row['ECoG_hemisphere']  = row['ECoG_hemisphere']
            psd_row['ECoG_channel']     = row['ECoG_channel']
            
        psd_row['dyskinesia_arm']   = row['dyskinesia_arm']
        psd_row['dyskinesia_total'] = row['dyskinesia_total']
        psd_row['pre_event_psd']    = psd_pre_event_norm
        psd_row['event_psd']        = psd_event_norm
        psd_row['post_event_psd']   = psd_post_event_norm

        peaks_pre_event                           = find_peaks_in_frequency_bands(freq, psd_pre_event_norm)
        psd_row['pre_event_beta_peak']            = peaks_pre_event['beta_peak']
        psd_row['pre_event_beta_peak_frequency']  = peaks_pre_event['beta_peak_frequency']
        psd_row['pre_event_gamma_peak']           = peaks_pre_event['gamma_peak']
        psd_row['pre_event_gamma_peak_frequency'] = peaks_pre_event['gamma_peak_frequency']
        psd_row['pre_event_theta_mean']           = np.mean(psd_pre_event_norm[(freq>=4) & (freq<=8)]) 
        psd_row['pre_event_alpha_mean']           = np.mean(psd_pre_event_norm[(freq>=8) & (freq<=12)]) 
        psd_row['pre_event_beta_low_mean']        = np.mean(psd_pre_event_norm[(freq>=12) & (freq<=20)]) 
        psd_row['pre_event_beta_high_mean']       = np.mean(psd_pre_event_norm[(freq>=20) & (freq<=35)]) 
        psd_row['pre_event_gamma_mean']           = np.mean(psd_pre_event_norm[(freq>=60) & (freq<=90)]) 

        peaks_event                           = find_peaks_in_frequency_bands(freq, psd_event_norm)
        psd_row['event_beta_peak']            = peaks_event['beta_peak']
        psd_row['event_beta_peak_frequency']  = peaks_event['beta_peak_frequency']
        psd_row['event_gamma_peak']           = peaks_event['gamma_peak']
        psd_row['event_gamma_peak_frequency'] = peaks_event['gamma_peak_frequency']
        psd_row['event_theta_mean']           = np.mean(psd_event_norm[(freq>=4) & (freq<=8)]) 
        psd_row['event_alpha_mean']           = np.mean(psd_event_norm[(freq>=8) & (freq<=12)]) 
        psd_row['event_beta_low_mean']        = np.mean(psd_event_norm[(freq>=12) & (freq<=20)]) 
        psd_row['event_beta_high_mean']       = np.mean(psd_event_norm[(freq>=20) & (freq<=35)]) 
        psd_row['event_gamma_mean']           = np.mean(psd_event_norm[(freq>=60) & (freq<=90)]) 

        peaks_post_event                           = find_peaks_in_frequency_bands(freq, psd_post_event_norm)
        psd_row['post_event_beta_peak']            = peaks_post_event['beta_peak']
        psd_row['post_event_beta_peak_frequency']  = peaks_post_event['beta_peak_frequency']
        psd_row['post_event_gamma_peak']           = peaks_post_event['gamma_peak']
        psd_row['post_event_gamma_peak_frequency'] = peaks_post_event['gamma_peak_frequency']
        psd_row['post_event_theta_mean']           = np.mean(psd_post_event_norm[(freq>=4) & (freq<=8)]) 
        psd_row['post_event_alpha_mean']           = np.mean(psd_post_event_norm[(freq>=8) & (freq<=12)]) 
        psd_row['post_event_beta_low_mean']        = np.mean(psd_post_event_norm[(freq>=12) & (freq<=20)]) 
        psd_row['post_event_beta_high_mean']       = np.mean(psd_post_event_norm[(freq>=20) & (freq<=35)]) 
        psd_row['post_event_gamma_mean']           = np.mean(psd_post_event_norm[(freq>=60) & (freq<=90)]) 

        df_power_spectrum.loc[len(df_power_spectrum)] = psd_row

    return freq, df_power_spectrum

def find_peaks_in_frequency_bands(freq, psd):
    
    peaks                = {}
    
    # gamma band (60-90)
    freq_gamma           = freq[(freq>=60) & (freq<=90)]
    psd_gamma            = psd[(freq>=60) & (freq<=90)]
    
    try:
        index_gamma          = find_most_prominent_peak_index(psd_gamma)
        peak_gamma           = psd_gamma[index_gamma]
        peak_frequency_gamma = freq_gamma[index_gamma]
        peaks["gamma_peak"]           = peak_gamma
        peaks["gamma_peak_frequency"] = peak_frequency_gamma
    except:
        peaks["gamma_peak"]           = np.nan
        peaks["gamma_peak_frequency"] = np.nan
    
    # beta band (12-35)
    freq_beta            = freq[(freq>=12) & (freq<=35)]
    psd_beta             = psd[(freq>=12) & (freq<=35)] 
    
    try:
        index_beta                    = find_most_prominent_peak_index(-psd_beta)
        peak_beta                     = psd_beta[index_beta]
        peak_frequency_beta           = freq_beta[index_beta]
        peaks["beta_peak"]            = peak_beta
        peaks["beta_peak_frequency"]  = peak_frequency_beta
    except:
        peaks["beta_peak"]            = np.nan
        peaks["beta_peak_frequency"]  = np.nan

    return peaks

def find_most_prominent_peak_index(data): 
    
    # Find all peaks in the inverted data
    peaks, _ = find_peaks(data)
    
    # Calculate prominences of the peaks
    prominences = peak_prominences(data, peaks)[0]
    
    # Find the index of the most prominent peak
    most_prominent_peak_idx = np.argmax(prominences)
    
    # Get the index of the most prominent peak in the original data
    most_prominent_peak = peaks[most_prominent_peak_idx]
    
    return most_prominent_peak

def normalize_patient_ephysiology_event_psd_to_baseline(df_events, recording_type, event_mode, PATH):

    df_LID   = pd.DataFrame()
    df_noLID = pd.DataFrame()

    for SUB in df_events.patient.unique():

        # get the electrophysiological recordings of events of selected patient
        if(recording_type=="ECoG"):
            REC_SUB                                 = ECoG(PATH, SUB)
            patient_all, patient_noLID, patient_LID = REC_SUB.get_patient_events(df_events, SUB=SUB, event_mode=event_mode)
        else:
            REC_SUB                                 = LFP(PATH, SUB)
            patient_all, patient_noLID, patient_LID = REC_SUB.get_patient_events(df_events, SUB=SUB, event_mode=event_mode)
    
        # get baseline electrophysiological recordings
        patient_baseline  = REC_SUB.get_baseline_recording(t_min=0, t_max=5)  

        try:
            freq, psd_LID  = extract_normalized_psd_of_events(SUB, patient_LID, patient_baseline, recording_type)
            if(len(df_LID) == 0):
                df_LID = psd_LID
            else:
                df_LID = pd.concat([df_LID, psd_LID])
        except:
            pass
    
        try:
            freq, psd_noLID = extract_normalized_psd_of_events(SUB, patient_noLID, patient_baseline, recording_type)
            if(len(df_noLID) == 0):
                df_noLID = psd_noLID
            else:
                df_noLID = pd.concat([df_noLID, psd_noLID])
        except:
            pass

    # remove all the infinity psd arrays
    df_LID   = df_LID[~df_LID['event_psd'].apply(lambda x: np.any(np.isinf(x)))]
    df_LID   = df_LID[~df_LID['pre_event_psd'].apply(lambda x: np.any(np.isinf(x)))]
    df_LID   = df_LID[~df_LID['post_event_psd'].apply(lambda x: np.any(np.isinf(x)))]
    df_noLID = df_noLID[~df_noLID['event_psd'].apply(lambda x: np.any(np.isinf(x)))]
    df_noLID = df_noLID[~df_noLID['pre_event_psd'].apply(lambda x: np.any(np.isinf(x)))]
    df_noLID = df_noLID[~df_noLID['post_event_psd'].apply(lambda x: np.any(np.isinf(x)))]
    
    df_LID.reset_index(drop=True, inplace=True)
    df_noLID.reset_index(drop=True, inplace=True)
    return df_noLID, df_LID

def get_patient_baseline_LFP_channel_PSD(SUB, baseline):
    df_channel_psd = pd.DataFrame(columns=["patient", "LFP_hemisphere", "LFP_channel", "baseline_psd"])
    
    for hemisphere in baseline[SUB].keys():
        for channel in baseline[SUB][hemisphere].keys():
            
            channel_baseline_recording           = baseline[SUB][hemisphere][channel]
            freq, channel_baseline_recording_psd = measure_absolute_psd(channel_baseline_recording, fs=2048)
            
            row                   = {} 
            row["patient"]        = SUB
            row["LFP_hemisphere"] = hemisphere
            row["LFP_channel"]    = channel
            row["baseline_psd"]   = channel_baseline_recording_psd
            
            df_channel_psd.loc[len(df_channel_psd)] = row 
    return df_channel_psd
    
def measure_baseline_psd(baseline_recordings):
    for key, value in baseline_recordings.items():
        if isinstance(value, dict):  # If the value is a dictionary, recurse into it
            measure_baseline_psd(value)
        else:
            freq, psd_base  = measure_absolute_psd(baseline_recordings[key], fs=2048)
            baseline_recordings[key] = psd_base

###################################################################################################################################

"""
def extract_spectral_features(frequency, psd, mode):
    
    if(mode=="peak"):
        peaks, _       = signal.find_peaks(psd, width=2)
    else: # find valleys
        data           = np.array(psd)

        # if there are positive peaks which higher than valley, it can obscure the negative peak estimation
        # hence we replace all positive values with zeros and get the absolute of array values.
        data[data>=0]  = 0
        data           = np.abs(data)
        peaks, _       = signal.find_peaks(data, width=2)
    
    # Find the peak with the greatest value
    if peaks.size > 0:
        peak_index     = peaks[np.argmax(np.abs(psd[peaks]))]
        peak_frequency = frequency[peak_index]
        peak_value     = psd[peak_index]
    else:
        peak_frequency = np.nan
        peak_value     = np.nan

    features                   = {} 
    features["peak_frequency"] = peak_frequency
    features["peak_value"]     = peak_value
    features["mean_value"]     = np.nanmean(psd)
        
    return features
"""
def extract_spectral_features(frequency, psd):
    
    data     = np.array(psd)
    data     = np.abs(data)
    peaks, _ = signal.find_peaks(data, width=2)
    
    # Find the peak with the greatest value
    if peaks.size > 0:
        peak_index     = peaks[np.argmax(np.abs(psd[peaks]))]
        peak_frequency = frequency[peak_index]
        peak_value     = psd[peak_index]
    else:
        peak_frequency = np.nan
        peak_value     = np.nan

    features                   = {} 
    features["peak_frequency"] = peak_frequency
    features["peak_value"]     = peak_value
    features["mean_value"]     = np.nanmean(psd)
        
    return features

def extract_spectral_features_for_event_segment(segment_psd, segment_name):
    
    frequency      = np.linspace(4,100,97) #fixed
    LFP_features   = {}

    theta     = extract_spectral_features(frequency=frequency[(frequency>=4)  & (frequency<=8)] , psd=segment_psd[(frequency>=4)  & (frequency<=8)])
    alpha     = extract_spectral_features(frequency=frequency[(frequency>=8)  & (frequency<=12)], psd=segment_psd[(frequency>=8)  & (frequency<=12)])
    beta_low  = extract_spectral_features(frequency=frequency[(frequency>=12) & (frequency<=20)], psd=segment_psd[(frequency>=12) & (frequency<=20)])
    beta_high = extract_spectral_features(frequency=frequency[(frequency>=20) & (frequency<=35)], psd=segment_psd[(frequency>=20) & (frequency<=35)])
    gamma     = extract_spectral_features(frequency=frequency[(frequency>=60) & (frequency<=90)], psd=segment_psd[(frequency>=60) & (frequency<=90)])
    
    LFP_features[segment_name + "_theta_peak"]               = theta["peak_value"]
    LFP_features[segment_name + "_theta_peak_frequency"]     = theta["peak_frequency"]
    LFP_features[segment_name + "_theta_mean"]               = theta["mean_value"]
    LFP_features[segment_name + "_alpha_peak"]               = alpha["peak_value"]
    LFP_features[segment_name + "_alpha_peak_frequency"]     = alpha["peak_frequency"]
    LFP_features[segment_name + "_alpha_mean"]               = alpha["mean_value"]
    LFP_features[segment_name + "_beta_low_peak"]            = beta_low["peak_value"]
    LFP_features[segment_name + "_beta_low_peak_frequency"]  = beta_low["peak_frequency"]
    LFP_features[segment_name + "_beta_low_mean"]            = beta_low["mean_value"]
    LFP_features[segment_name + "_beta_high_peak"]           = beta_high["peak_value"]
    LFP_features[segment_name + "_beta_high_peak_frequency"] = beta_high["peak_frequency"]
    LFP_features[segment_name + "_beta_high_mean"]           = beta_high["mean_value"]
    LFP_features[segment_name + "_gamma_peak"]             = gamma["peak_value"]
    LFP_features[segment_name + "_gamma_peak_frequency"]   = gamma["peak_frequency"]
    LFP_features[segment_name + "_gamma_mean"]             = gamma["mean_value"]

    return LFP_features

def extract_PSD_features_for_events(dataset):
    # create new features for the dataframe
    for feature in ['event_theta_peak', 'event_theta_peak_frequency', 'event_theta_mean', 
                    'event_alpha_peak', 'event_alpha_peak_frequency', 'event_alpha_mean', 
                    'event_beta_low_peak', 'event_beta_low_peak_frequency', 'event_beta_low_mean', 
                    'event_beta_high_peak', 'event_beta_high_peak_frequency', 'event_beta_high_mean', 
                    'event_gamma_peak', 'event_gamma_peak_frequency', 'event_gamma_mean', 
                    'gamma_peak_onset_ratio', 'gamma_peak_offset_ratio', 'gamma_baseline_onset_ratio', 'gamma_baseline_offset_ratio', 
                    'beta_low_peak_onset_ratio', 'beta_low_peak_offset_ratio', 'beta_low_baseline_onset_ratio', 'beta_low_baseline_offset_ratio', 
                    'beta_high_peak_onset_ratio', 'beta_high_peak_offset_ratio', 'beta_high_baseline_onset_ratio', 'beta_high_baseline_offset_ratio', 
                    'alpha_baseline_onset_ratio', 'alpha_baseline_offset_ratio', 'theta_baseline_onset_ratio', 'theta_baseline_offset_ratio']:
        dataset[feature] = np.nan
    
    # add new features to the dataframe
    for index,row in dataset.iterrows():
        
        pre_event_LFP_features  = extract_spectral_features_for_event_segment(row.pre_event_psd.copy(), segment_name="pre_event")
        event_LFP_features      = extract_spectral_features_for_event_segment(row.event_psd.copy(), segment_name="event")
        post_event_LFP_features = extract_spectral_features_for_event_segment(row.post_event_psd.copy(), segment_name="post_event")
        
        features = {}
        features.update(pre_event_LFP_features)
        features.update(event_LFP_features)
        features.update(post_event_LFP_features)
    
        dataset.at[index, 'event_theta_peak']               = event_LFP_features['event_theta_peak']
        dataset.at[index, 'event_theta_peak_frequency']     = event_LFP_features['event_theta_peak_frequency']
        dataset.at[index, 'event_theta_mean']               = event_LFP_features['event_theta_mean']
        dataset.at[index, 'event_alpha_peak']               = event_LFP_features['event_alpha_peak']
        dataset.at[index, 'event_alpha_peak_frequency']     = event_LFP_features['event_alpha_peak_frequency']
        dataset.at[index, 'event_alpha_mean']               = event_LFP_features['event_alpha_mean']
        dataset.at[index, 'event_beta_low_peak']            = event_LFP_features['event_beta_low_peak']
        dataset.at[index, 'event_beta_low_peak_frequency']  = event_LFP_features['event_beta_low_peak_frequency']
        dataset.at[index, 'event_beta_low_mean']            = event_LFP_features['event_beta_low_mean']
        dataset.at[index, 'event_beta_high_peak']           = event_LFP_features['event_beta_high_peak']
        dataset.at[index, 'event_beta_high_peak_frequency'] = event_LFP_features['event_beta_high_peak_frequency']
        dataset.at[index, 'event_beta_high_mean']           = event_LFP_features['event_beta_high_mean']
        dataset.at[index, 'event_gamma_peak']               = event_LFP_features['event_gamma_peak']
        dataset.at[index, 'event_gamma_peak_frequency']     = event_LFP_features['event_gamma_peak_frequency']
        dataset.at[index, 'event_gamma_mean']               = event_LFP_features['event_gamma_mean']
        
        
        dataset.at[index, "gamma_peak_onset_ratio"]         = (features['pre_event_gamma_peak'] - features['event_gamma_peak']) / features['event_gamma_peak'] * 100
        dataset.at[index, "gamma_peak_offset_ratio"]        = (features['post_event_gamma_peak'] - features['event_gamma_peak']) / features['event_gamma_peak'] * 100
        dataset.at[index, "gamma_baseline_onset_ratio"]     = (features['pre_event_gamma_mean'] - features['event_gamma_mean']) / features['event_gamma_mean'] * 100
        dataset.at[index, "gamma_baseline_offset_ratio"]    = (features['post_event_gamma_mean'] - features['event_gamma_mean']) / features['event_gamma_mean'] * 100
        dataset.at[index, "beta_high_peak_onset_ratio"]       = (features['pre_event_beta_high_peak'] - features['event_beta_high_peak']) / features['event_beta_high_peak'] * 100
        dataset.at[index, "beta_high_peak_offset_ratio"]      = (features['post_event_beta_high_peak'] - features['event_beta_high_peak']) / features['event_beta_high_peak'] * 100
        dataset.at[index, "beta_high_baseline_onset_ratio"]   = (features['pre_event_beta_high_mean'] - features['event_beta_high_mean']) / features['event_beta_high_mean'] * 100
        dataset.at[index, "beta_high_baseline_offset_ratio"]  = (features['post_event_beta_high_mean'] - features['event_beta_high_mean']) / features['event_beta_high_mean'] * 100
        dataset.at[index, "beta_low_peak_onset_ratio"]        = (features['pre_event_beta_low_peak'] - features['event_beta_low_peak']) / features['event_beta_low_peak'] * 100
        dataset.at[index, "beta_low_peak_offset_ratio"]       = (features['post_event_beta_low_peak'] - features['event_beta_low_peak']) / features['event_beta_low_peak'] * 100
        dataset.at[index, "beta_low_baseline_onset_ratio"]    = (features['pre_event_beta_low_mean'] - features['event_beta_low_mean']) / features['event_beta_low_mean'] * 100
        dataset.at[index, "beta_low_baseline_offset_ratio"]   = (features['post_event_beta_low_mean'] - features['event_beta_low_mean']) / features['event_beta_low_mean'] * 100
        dataset.at[index, "alpha_baseline_onset_ratio"]       = (features['pre_event_alpha_mean'] - features['event_alpha_mean']) / features['event_alpha_mean'] * 100
        dataset.at[index, "alpha_baseline_offset_ratio"]      = (features['post_event_alpha_mean'] - features['event_alpha_mean']) / features['event_alpha_mean'] * 100
        dataset.at[index, "theta_baseline_onset_ratio"]       = (features['pre_event_theta_mean'] - features['event_theta_mean']) / features['event_theta_mean'] * 100
        dataset.at[index, "theta_baseline_offset_ratio"]      = (features['post_event_theta_mean'] - features['event_theta_mean']) / features['event_theta_mean'] * 100

    return dataset