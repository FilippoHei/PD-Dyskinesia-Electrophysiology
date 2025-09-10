"""
Power spectral utilisation functions
"""

import pandas as pd
from scipy import signal
import numpy as np
import pickle
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
    
    psd_baseline      = np.array(psd_baseline)
    psd_event         = np.array(psd_event)

    # Calculate the percentage change
    percentage_change = (psd_event - psd_baseline) / psd_baseline * 100

    return percentage_change

def extract_normalized_psd_of_events(SUB, patient_dataset, patient_baseline, recording_type):

    # emptpy dataframe to save the results of spectral features
    if(recording_type=="LFP"):
        df_power_spectrum = pd.DataFrame(columns=['patient', 'event_no', 'event_category', 'event_laterality',
                                                  'event_start_time', 'duration', 'LFP_hemisphere', 'LFP_channel',
                                                  'pre_event_psd', 'event_psd', 'post_event_psd', 'dyskinesia_arm', 'dyskinesia_total'])
    else:
        df_power_spectrum = pd.DataFrame(columns=['patient', 'event_no', 'event_category', 'event_laterality',
                                                  'event_start_time', 'duration', 'ECoG_hemisphere', 'ECoG_channel',
                                                  'pre_event_psd', 'event_psd', 'post_event_psd', 'dyskinesia_arm', 'dyskinesia_total'])
    
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
        baseline_rec     = patient_baseline[SUB][hemisphere][channel]
        
        # get baseline PSD from baseline recording
        freq, psd_base   = measure_absolute_psd(baseline_rec, fs=2048)
    
        # measure the PSD in pre-event, event and post-event segments
        freq, psd_pre    = measure_absolute_psd(row["pre_event_recording"], fs=2048)
        freq, psd_event  = measure_absolute_psd(row["event_recording"], fs=2048)
        freq, psd_post   = measure_absolute_psd(row["post_event_recording"], fs=2048)
    
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
        df_power_spectrum.loc[len(df_power_spectrum)] = psd_row

    return freq, df_power_spectrum

def normalize_patient_ephysiology_event_psd_to_baseline(df_events, recording_type, event_mode, PATH):

    df_LID   = pd.DataFrame()
    df_noLID = pd.DataFrame()

    for SUB in df_events.patient.unique():

        print("Patient - " + SUB)
        
        if(recording_type=="ECoG"):
            # get the electrophysiological recordings of events of the selected patient
            patient_all, patient_noLID, patient_LID = ECoG.get_patient_events(df_events, SUB=SUB, event_mode=event_mode)
            # get baseline electrophysiological recordings
            patient_baseline  = ECoG.load_baseline_recording(SUB)  
        else:
            # get the electrophysiological recordings of events of the selected patient
            patient_all, patient_noLID, patient_LID = LFP.get_patient_events(df_events, SUB=SUB, event_mode=event_mode)
            # get baseline electrophysiological recordings
            patient_baseline  = LFP.load_baseline_recording(SUB)  

        try:
            freq, psd_LID  = extract_normalized_psd_of_events(SUB, patient_LID, patient_baseline, recording_type)
            psd_LID        = extract_PSD_features_for_events(psd_LID)
            if(len(df_LID) == 0):
                df_LID = psd_LID
            else:
                df_LID = pd.concat([df_LID, psd_LID])
        except:
            pass
    
        try:
            freq, psd_noLID = extract_normalized_psd_of_events(SUB, patient_noLID, patient_baseline, recording_type)
            psd_noLID       = extract_PSD_features_for_events(psd_noLID)
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

def measure_maximum_deviation(array):
    positive_max = max([x for x in array if x > 0], default=0)
    negative_min = min([x for x in array if x < 0], default=0)
    return positive_max if positive_max >= abs(negative_min) else negative_min
    
def extract_spectral_features(frequency, psd):
    
    features                    = {} 
    features["deviation_value"] = measure_maximum_deviation(psd)
    features["mean_value"]      = np.nanmean(psd)
        
    return features

def extract_spectral_features_for_event_segment(segment_psd, segment_name):
    
    frequency      = np.linspace(4,100,97) #fixed
    LFP_features   = {}

    theta     = extract_spectral_features(frequency=frequency[(frequency>=4)  & (frequency<=8)] , psd=segment_psd[(frequency>=4)  & (frequency<=8)])
    alpha     = extract_spectral_features(frequency=frequency[(frequency>=8)  & (frequency<=12)], psd=segment_psd[(frequency>=8)  & (frequency<=12)])
    beta      = extract_spectral_features(frequency=frequency[(frequency>=12) & (frequency<=35)], psd=segment_psd[(frequency>=12) & (frequency<=35)])
    beta_low  = extract_spectral_features(frequency=frequency[(frequency>=12) & (frequency<=20)], psd=segment_psd[(frequency>=12) & (frequency<=20)])
    beta_high = extract_spectral_features(frequency=frequency[(frequency>=20) & (frequency<=35)], psd=segment_psd[(frequency>=20) & (frequency<=35)])
    gamma     = extract_spectral_features(frequency=frequency[(frequency>=60) & (frequency<=90)], psd=segment_psd[(frequency>=60) & (frequency<=90)])
    gamma_I   = extract_spectral_features(frequency=frequency[(frequency>=60) & (frequency<=70)], psd=segment_psd[(frequency>=60) & (frequency<=70)])
    gamma_II  = extract_spectral_features(frequency=frequency[(frequency>=70) & (frequency<=80)], psd=segment_psd[(frequency>=70) & (frequency<=80)])
    gamma_III = extract_spectral_features(frequency=frequency[(frequency>=80) & (frequency<=90)], psd=segment_psd[(frequency>=80) & (frequency<=90)])
    
    LFP_features[segment_name + "_theta_deviation"]     = theta["deviation_value"]
    LFP_features[segment_name + "_theta_mean"]          = theta["mean_value"]
    LFP_features[segment_name + "_alpha_deviation"]     = alpha["deviation_value"]
    LFP_features[segment_name + "_alpha_mean"]          = alpha["mean_value"]
    LFP_features[segment_name + "_beta_deviation"]      = beta["deviation_value"]
    LFP_features[segment_name + "_beta_mean"]           = beta["mean_value"]
    LFP_features[segment_name + "_beta_low_deviation"]  = beta_low["deviation_value"]
    LFP_features[segment_name + "_beta_low_mean"]       = beta_low["mean_value"]
    LFP_features[segment_name + "_beta_high_deviation"] = beta_high["deviation_value"]
    LFP_features[segment_name + "_beta_high_mean"]      = beta_high["mean_value"]
    LFP_features[segment_name + "_gamma_deviation"]     = gamma["deviation_value"]
    LFP_features[segment_name + "_gamma_mean"]          = gamma["mean_value"]
    LFP_features[segment_name + "_gamma_I_deviation"]   = gamma_I["deviation_value"]
    LFP_features[segment_name + "_gamma_I_mean"]        = gamma_I["mean_value"]
    LFP_features[segment_name + "_gamma_II_deviation"]  = gamma_II["deviation_value"]
    LFP_features[segment_name + "_gamma_II_mean"]       = gamma_II["mean_value"]
    LFP_features[segment_name + "_gamma_III_deviation"] = gamma_III["deviation_value"]
    LFP_features[segment_name + "_gamma_III_mean"]      = gamma_III["mean_value"]

    return LFP_features

def extract_PSD_features_for_events(dataset):
    # create new features for the dataframe
    for feature in ['pre_event_theta_deviation', 'pre_event_theta_mean', 
                    'pre_event_alpha_deviation', 'pre_event_alpha_mean',
                    'pre_event_beta_deviation','pre_event_beta_mean',
                    'pre_event_beta_low_deviation', 'pre_event_beta_low_mean', 
                    'pre_event_beta_high_deviation', 'pre_event_beta_high_mean', 
                    'pre_event_gamma_deviation', 'pre_event_gamma_mean',
                    'pre_event_gamma_I_deviation', 'pre_event_gamma_I_mean',
                    'pre_event_gamma_II_deviation', 'pre_event_gamma_II_mean',
                    'pre_event_gamma_III_deviation', 'pre_event_gamma_III_mean',
                    'event_theta_deviation', 'event_theta_mean', 
                    'event_alpha_deviation', 'event_alpha_mean',
                    'event_beta_deviation', 'event_beta_mean', 
                    'event_beta_low_deviation', 'event_beta_low_mean', 
                    'event_beta_high_deviation', 'event_beta_high_mean', 
                    'event_gamma_deviation', 'event_gamma_mean',
                    'event_gamma_I_deviation', 'event_I_gamma_mean',
                    'event_gamma_II_deviation', 'event_II_gamma_mean',
                    'event_gamma_III_deviation', 'event_III_gamma_mean',
                    'post_event_theta_deviation', 'post_event_theta_mean', 
                    'post_event_alpha_deviation', 'post_event_alpha_mean',
                    'post_event_beta_deviation', 'post_event_beta_mean',
                    'post_event_beta_low_deviation', 'post_event_beta_low_mean', 
                    'post_event_beta_high_deviation', 'post_event_beta_high_mean', 
                    'post_event_gamma_deviation', 'post_event_gamma_mean',
                    'post_event_I_gamma_deviation', 'post_event_I_gamma_mean',
                    'post_event_II_gamma_deviation', 'post_event_II_gamma_mean',
                    'post_event_III_gamma_deviation', 'post_event_III_gamma_mean']:
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

        dataset.at[index, 'pre_event_theta_deviation']      = pre_event_LFP_features['pre_event_theta_deviation']
        dataset.at[index, 'pre_event_theta_mean']           = pre_event_LFP_features['pre_event_theta_mean']
        dataset.at[index, 'pre_event_alpha_deviation']      = pre_event_LFP_features['pre_event_alpha_deviation']
        dataset.at[index, 'pre_event_alpha_mean']           = pre_event_LFP_features['pre_event_alpha_mean']
        dataset.at[index, 'pre_event_beta_deviation']       = pre_event_LFP_features['pre_event_beta_deviation']
        dataset.at[index, 'pre_event_beta_mean']            = pre_event_LFP_features['pre_event_beta_mean']
        dataset.at[index, 'pre_event_beta_low_deviation']   = pre_event_LFP_features['pre_event_beta_low_deviation']
        dataset.at[index, 'pre_event_beta_low_mean']        = pre_event_LFP_features['pre_event_beta_low_mean']
        dataset.at[index, 'pre_event_beta_high_deviation']  = pre_event_LFP_features['pre_event_beta_high_deviation']
        dataset.at[index, 'pre_event_beta_high_mean']       = pre_event_LFP_features['pre_event_beta_high_mean']
        dataset.at[index, 'pre_event_gamma_deviation']      = pre_event_LFP_features['pre_event_gamma_deviation']
        dataset.at[index, 'pre_event_gamma_mean']           = pre_event_LFP_features['pre_event_gamma_mean']
        dataset.at[index, 'pre_event_gamma_I_deviation']    = pre_event_LFP_features['pre_event_gamma_I_deviation']
        dataset.at[index, 'pre_event_gamma_I_mean']         = pre_event_LFP_features['pre_event_gamma_I_mean']
        dataset.at[index, 'pre_event_gamma_II_deviation']   = pre_event_LFP_features['pre_event_gamma_II_deviation']
        dataset.at[index, 'pre_event_gamma_II_mean']        = pre_event_LFP_features['pre_event_gamma_II_mean']
        dataset.at[index, 'pre_event_gamma_III_deviation']  = pre_event_LFP_features['pre_event_gamma_III_deviation']
        dataset.at[index, 'pre_event_gamma_III_mean']       = pre_event_LFP_features['pre_event_gamma_III_mean']
        
        dataset.at[index, 'event_theta_deviation']          = event_LFP_features['event_theta_deviation']
        dataset.at[index, 'event_theta_mean']               = event_LFP_features['event_theta_mean']
        dataset.at[index, 'event_alpha_deviation']          = event_LFP_features['event_alpha_deviation']
        dataset.at[index, 'event_alpha_mean']               = event_LFP_features['event_alpha_mean']
        dataset.at[index, 'event_beta_deviation']           = event_LFP_features['event_beta_deviation']
        dataset.at[index, 'event_beta_mean']                = event_LFP_features['event_beta_mean']
        dataset.at[index, 'event_beta_low_deviation']       = event_LFP_features['event_beta_low_deviation']
        dataset.at[index, 'event_beta_low_mean']            = event_LFP_features['event_beta_low_mean']
        dataset.at[index, 'event_beta_high_deviation']      = event_LFP_features['event_beta_high_deviation']
        dataset.at[index, 'event_beta_high_mean']           = event_LFP_features['event_beta_high_mean']
        dataset.at[index, 'event_gamma_deviation']          = event_LFP_features['event_gamma_deviation']
        dataset.at[index, 'event_gamma_mean']               = event_LFP_features['event_gamma_mean']
        dataset.at[index, 'event_gamma_I_deviation']        = event_LFP_features['event_gamma_I_deviation']
        dataset.at[index, 'event_gamma_I_mean']             = event_LFP_features['event_gamma_I_mean']
        dataset.at[index, 'event_gamma_II_deviation']       = event_LFP_features['event_gamma_II_deviation']
        dataset.at[index, 'event_gamma_II_mean']            = event_LFP_features['event_gamma_II_mean']
        dataset.at[index, 'event_gamma_III_deviation']      = event_LFP_features['event_gamma_III_deviation']
        dataset.at[index, 'event_gamma_III_mean']           = event_LFP_features['event_gamma_III_mean']

        dataset.at[index, 'post_event_theta_deviation']     = post_event_LFP_features['post_event_theta_deviation']
        dataset.at[index, 'post_event_theta_mean']          = post_event_LFP_features['post_event_theta_mean']
        dataset.at[index, 'post_event_alpha_deviation']     = post_event_LFP_features['post_event_alpha_deviation']
        dataset.at[index, 'post_event_alpha_mean']          = post_event_LFP_features['post_event_alpha_mean']
        dataset.at[index, 'post_event_beta_deviation']      = post_event_LFP_features['post_event_beta_deviation']
        dataset.at[index, 'post_event_beta_mean']           = post_event_LFP_features['post_event_beta_mean']
        dataset.at[index, 'post_event_beta_low_deviation']  = post_event_LFP_features['post_event_beta_low_deviation']
        dataset.at[index, 'post_event_beta_low_mean']       = post_event_LFP_features['post_event_beta_low_mean']
        dataset.at[index, 'post_event_beta_high_deviation'] = post_event_LFP_features['post_event_beta_high_deviation']
        dataset.at[index, 'post_event_beta_high_mean']      = post_event_LFP_features['post_event_beta_high_mean']
        dataset.at[index, 'post_event_gamma_deviation']     = post_event_LFP_features['post_event_gamma_deviation']
        dataset.at[index, 'post_event_gamma_mean']          = post_event_LFP_features['post_event_gamma_mean']
        dataset.at[index, 'post_event_gamma_I_deviation']   = post_event_LFP_features['post_event_gamma_I_deviation']
        dataset.at[index, 'post_event_gamma_I_mean']        = post_event_LFP_features['post_event_gamma_I_mean']
        dataset.at[index, 'post_event_gamma_II_deviation']  = post_event_LFP_features['post_event_gamma_II_deviation']
        dataset.at[index, 'post_event_gamma_II_mean']       = post_event_LFP_features['post_event_gamma_II_mean']
        dataset.at[index, 'post_event_gamma_III_deviation'] = post_event_LFP_features['post_event_gamma_III_deviation']
        dataset.at[index, 'post_event_gamma_III_mean']      = post_event_LFP_features['post_event_gamma_III_mean']

    return dataset

def select_ECoG_PSD_based_on_cortical_parcellation(df_PSD, data_MNI, cortical_region):
    
    df_PSD_cortical_region = {}
    
    for severity in df_PSD.keys():
        merged_df   = pd.merge(df_PSD[severity], data_MNI, left_on=['patient', 'ECoG_hemisphere'], right_on=['patient', 'hemisphere'])
        filtered_df = merged_df[merged_df['AAL3_cortex'] == cortical_region]
        filtered_df.reset_index(inplace=True, drop=True)
        df_PSD_cortical_region[severity] = filtered_df

    return df_PSD_cortical_region