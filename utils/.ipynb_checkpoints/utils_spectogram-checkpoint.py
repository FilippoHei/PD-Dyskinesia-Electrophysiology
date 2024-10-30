"""
Utilisation function for plotting
"""

import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet

# inserting the lib folder to the compiler
import sys
sys.path.insert(0, './lib')


def measure_TFR_for_patient_LFP_baseline(patient, baseline, fs):

    # create an empty dataframe
    df_channel_baseline_spectogram = pd.DataFrame(columns=["patient", "LFP_hemisphere", "LFP_channel", "baseline_frequency_mean", "baseline_frequency_std"])
        
    for hemisphere in baseline[patient].keys():
        for channel in baseline[patient][hemisphere].keys():

            # for the selected hemisphere and the channel for patient, get the baseline recordings.
            channel_baseline          = baseline[patient][hemisphere][channel]

            # Since the baseline recordings correspond to 5 minutes, we will divide the baseline into 4 seconds segments with
            # 50% overlap. For each segment, we will estimate the MWT and measure the average  spectrogram for 4 seconds for baseline (N x fs*4).
            # Furthermore, we also take the average of all columns to represent baseline MWT as (M X 1) vector. Then we will use either this vector
            # mean and std values of each frequency in average baseline MWT. So, we will save both, the baseline MWT column, mean, and std values for each
            # frequency.

            segment_length            = (fs*4)
            segment_overlap           = segment_length / 2

            # get the start indices of each segment in baseline recording
            start_indices             = np.arange(0, len(channel_baseline) - segment_length + 1, segment_overlap).astype(int)
            
            # get baseline segments
            channel_baseline_segments = np.array([channel_baseline[i:i+segment_length] for i in start_indices])

            # reshape the data
            channel_baseline_segments = np.array(channel_baseline_segments).reshape(np.shape(channel_baseline_segments)[0], 1, 
                                                                                    np.shape(channel_baseline_segments)[1])
            # measure the spectogram for each segment
            baseline_spectogram       = tfr_array_morlet(channel_baseline_segments, sfreq=fs, n_cycles=5, freqs=np.linspace(4,90,87), output='power', verbose=False) 

            # measure the average spectrogram across all segments (N x fs*4)
            baseline_spectogram       = np.nanmean(baseline_spectogram, axis=0)[0]

            # get the mean value for each frequency (from 4-90Hz)
            baseline_frequency_mean   = np.nanmean(baseline_spectogram, axis=1)

            # get the std value for each frequency (from 4-90Hz)
            baseline_frequency_std    = np.std(baseline_spectogram, axis=1)
                
            row                       = {} 
            row["patient"]            = patient
            row["LFP_hemisphere"]     = hemisphere
            row["LFP_channel"]        = channel
            row["frequency_mean"]     = baseline_frequency_mean
            row["frequency_std"]      = baseline_frequency_std
            
            df_channel_baseline_spectogram.loc[len(df_channel_baseline_spectogram)] = row 
    
    return df_channel_baseline_spectogram


def measure_TFR_for_patient_LFP_channel_baseline(baseline, patient, hemisphere, channel, fs):

    # for the selected hemisphere and the channel for patient, get the baseline recordings.
    channel_baseline          = baseline[patient][hemisphere][channel]

    # since the baseline recordings corresponds to 5 minutes, we will divide the baseline into 4 seconds segments with
    # 50% overalap. And for each segment, we will estimate the MWT and measure and average  spectogram for 4 seconds for baseline (N x fs*4).
    # Furthermore, we also take the average of all columns to represent baseline MWT as (M X 1) vector. Then we will use either this vector
    # mean and std values of each frequency in average baseline MWT. So, we will save both, baseline MWT column, mean and std values for each
    # frequency.

    segment_length            = (fs*4)
    segment_overlap           = segment_length / 2

    # get the start indices of each segments in baseline recording
    start_indices             = np.arange(0, len(channel_baseline) - segment_length + 1, segment_overlap).astype(int)
            
    # get baseline segments
    channel_baseline_segments = np.array([channel_baseline[i:i+segment_length] for i in start_indices])

    # reshape the data
    channel_baseline_segments = np.array(channel_baseline_segments).reshape(np.shape(channel_baseline_segments)[0], 1, 
                                                                                    np.shape(channel_baseline_segments)[1])
    # measure the spectogram for each segment
    baseline_spectogram       = tfr_array_morlet(channel_baseline_segments, sfreq=fs, n_cycles=5, freqs=np.linspace(4,90,87), output='power', verbose=False) 

    # measure the average spectogram across all segments (N x fs*4)
    baseline_spectogram       = np.nanmean(baseline_spectogram, axis=0)[0]

    # get the mean value for each frequency (from 4-90Hz)
    baseline_frequency_mean   = np.nanmean(baseline_spectogram, axis=1)

    # get the std value for each frequency (from 4-90Hz)
    baseline_frequency_std    = np.std(baseline_spectogram, axis=1)

    # reshape mean and std arrays
    baseline_frequency_mean   = baseline_frequency_mean.reshape(len(baseline_frequency_mean), 1)
    baseline_frequency_std    = baseline_frequency_std.reshape(len(baseline_frequency_std), 1)
    
    return baseline_frequency_mean, baseline_frequency_std

def measure_TFR_for_patient_LFP_channel_events(dataset, patient, hemisphere, channel, fs):

    # get the correspondings events of patient with particular hemisphere and LFP channel
    dataset_patient_channel_events = dataset[(dataset.patient==patient) & (dataset.LFP_hemisphere==hemisphere) & (dataset.LFP_channel==channel)].copy()
    dataset_patient_channel_events.reset_index(drop=True, inplace=True)
    
    event_spectograms = []
    
    for index, row in dataset_patient_channel_events.iterrows():
        
        event_recording  = np.array(row.event_recording_onset_alingned)
        event_recording  = event_recording.reshape(1, 1, len(event_recording))
    
        # get morlet spectogram of the event
        event_spectogram = tfr_array_morlet(event_recording, sfreq=fs, n_cycles=5, freqs=np.linspace(4,90,87), output='power', verbose=False)[0]
        # pad the spectogram with np.nan until it reaches to 4 seconds
        event_spectogram = np.pad(event_spectogram, ((0, 0), (0, 0), (0, fs*4 - event_recording.shape[2])), constant_values=np.nan)
    
        event_spectograms.append(event_spectogram[0])

    return event_spectograms


def get_normalized_event_spectogram_by_channel(dataset_events, dataset_baseline, patient, hemisphere, channel, fs):
    
    # get event spectograms in the channel
    channel_event_spectograms         = measure_TFR_for_patient_LFP_channel_events(dataset = dataset_events,
                                                                                   patient = patient,
                                                                                   hemisphere = hemisphere,
                                                                                   channel = channel,
                                                                                   fs = fs)
    
    # get event baseline spectogram of the the channel
    freq_mean, freq_std               = measure_TFR_for_patient_LFP_channel_baseline(baseline = dataset_baseline,
                                                                                     patient = patient,
                                                                                     hemisphere = hemisphere,
                                                                                     channel = channel,
                                                                                     fs = fs)
    
    return channel_event_spectograms, freq_mean, freq_std

def get_LFP_channel_coefficient_of_variation(dataset_events, dataset_baseline, fs, stn_area):

    df_channel_cv = pd.DataFrame()
    
    for patient in dataset_events.patient.unique():
        print("Patient " + patient + " - STN " + stn_area + " area: LFP channel cv measurement started...")
        for hemisphere in dataset_events[dataset_events.patient==patient].LFP_hemisphere.unique():
            for channel in dataset_events[(dataset_events.patient==patient)&(dataset_events.LFP_hemisphere==hemisphere)].LFP_channel.unique():
                
                event_spectograms, baseline_mean, baseline_std = get_normalized_event_spectogram_by_channel(dataset_events = dataset_events,
                                                                                                            dataset_baseline = dataset_baseline, 
                                                                                                            patient=patient, 
                                                                                                            hemisphere=hemisphere, 
                                                                                                            channel=channel, 
                                                                                                            fs=fs)
                
                
                # get average spectogram and then normalize to baseline frequency mean
                event_spectogram_avg  = np.nanmean(event_spectograms, axis=0)
                event_spectogram_norm = (event_spectogram_avg - baseline_mean) / baseline_mean

                # instead of measuring cv between -2 to 2 second around event onset, measure the cv for between -1 to 1 for more robust estimation
                event_spectogram_norm = event_spectogram_norm[:, fs:3*fs]
                
                # get the mean value for each frequency (from 4-90Hz)
                spectogram_freq_mean  = np.nanmean(event_spectogram_norm, axis=1)
                
                # get the std value for each frequency (from 4-90Hz)
                spectogram_freq_std   = np.std(event_spectogram_norm, axis=1)
                
                # reshape mean and std arrays
                channel_cv            = spectogram_freq_std / spectogram_freq_mean
                channel_cv            = channel_cv.reshape(len(channel_cv),1)
                
                df_patient_channel_cv               = pd.DataFrame()
                df_patient_channel_cv["frequency"]  = np.linspace(4,90,87)
                df_patient_channel_cv["patient"]    = patient
                df_patient_channel_cv["hemisphere"] = hemisphere
                df_patient_channel_cv["channel"]    = channel
                df_patient_channel_cv["cv"]         = channel_cv
                df_patient_channel_cv["stn_area"]   = stn_area

                if(len(df_channel_cv)==0):
                    df_channel_cv = df_patient_channel_cv
                else:
                    df_channel_cv = pd.concat([df_channel_cv, df_patient_channel_cv], ignore_index=True)
                print("---> " + hemisphere + " hemisphere - " + channel + " channel is completed.")

    return df_channel_cv