import pandas as pd
from scipy import signal
import numpy as np
import pickle
import sys
import pickle

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')

import tensorpac
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude

def normalize_erpac(event_erpac, baseline_erpac):
    try:
        baseline               = baseline_erpac.copy()[:, None]
        normalized_event_erpac = ((event_erpac - baseline) / baseline) * 100
        return normalized_event_erpac
    except:
        print(">> ERPAC normalization resulted with error!")
        return np.nan

def normalized_channel_erpac_by_baseline(event_dataset, baseline_dataset):
    normalized_erpacs = []
    for index, row in event_dataset.iterrows():
        
        try:
            channel_event_erpac    = row["erpac"]
            channel_baseline_erpac = baseline_dataset[(baseline_dataset.patient==row.patient) & 
                                                      (baseline_dataset.hemisphere==row.hemisphere) & 
                                                      (baseline_dataset.channel==row.channel)].erpac.values[0]
            normalized_erpacs.append(normalize_erpac(event_erpac=channel_event_erpac, baseline_erpac=channel_baseline_erpac))
        except:
            normalized_erpacs.append(np.nan)
            #print(f"Patient {row.patient} - Hemisphere {row.hemisphere} - Channel {row.channel} doesn't have valid baseline ERPAC!")
    
    event_dataset["erpac_normalized"] = normalized_erpacs
    event_dataset                     = event_dataset[event_dataset.erpac_normalized.notna()]
    return event_dataset

def trim_erpac_array(row, global_t):

    # pad the erpac array
    t           = np.array(row['t_vector'])
    erpac       = np.array(row['erpac_mean'])

    return erpac[(t>=global_t[0]) & (t<=global_t[-1])]
    
def get_mean_erpac_line(dataset, baseline, phase_band, amplitude_band, alignment, normalize, fs):

    baseline_couple                = baseline[(baseline.phase_band==phase_band) & (baseline.amplitude_band==amplitude_band)]
    frequency_couple               = dataset[(dataset.phase_band==phase_band) & (dataset.amplitude_band==amplitude_band) & (dataset.alignment==alignment)]
    frequency_couple['erpac']      = frequency_couple['erpac'].apply(lambda x: np.squeeze(np.array(x), axis=1))
    if(normalize==True):
        frequency_couple               = normalized_channel_erpac_by_baseline(frequency_couple, baseline_couple)
        frequency_couple['erpac_mean'] = frequency_couple['erpac_normalized'].apply(lambda x: np.nanmean(np.array(x), axis=0))
    else:
        frequency_couple['erpac_mean'] = frequency_couple['erpac'].apply(lambda x: np.nanmean(np.array(x), axis=0))
    frequency_couple.reset_index(drop=True, inplace=True)
    
    # Get all unique t_vectors (since same across channels per patient)
    unique_t_vectors = frequency_couple.groupby('patient')['t_vector'].first().to_list()
    t_min            = -1
    t_max            = 2
    
    for t_vec in unique_t_vectors:
        if (t_vec[0]  > t_min) : t_min = t_vec[0]
        if (t_vec[-1] < t_max) : t_max = t_vec[-1]
            
    global_t = np.arange(t_min, t_max + 1/fs, 1/fs) 
    
    # Apply padding
    frequency_couple['erpac_trimmed'] = frequency_couple.apply(trim_erpac_array, axis=1, global_t=global_t)
    
    erpac_values = np.array(frequency_couple['erpac_trimmed'].to_list())
    mean_erpac   = np.nanmean(erpac_values, axis=0)
    se_erpac     = np.nanstd(erpac_values, axis=0, ddof=1) / np.sqrt(erpac_values.shape[0]) 
    return global_t, erpac_values, mean_erpac, se_erpac

def get_mean_erpac_matrix(dataset, baseline, phase_band, amplitude_band, alignment, normalize, fs, t_start, t_end):

    baseline_couple                = baseline[(baseline.phase_band==phase_band) & (baseline.amplitude_band==amplitude_band)]
    frequency_couple               = dataset[(dataset.phase_band==phase_band) & (dataset.amplitude_band==amplitude_band) & (dataset.alignment==alignment)]
    frequency_couple['erpac']      = frequency_couple['erpac'].apply(lambda x: np.squeeze(np.array(x), axis=1))
    if(normalize==True):
        frequency_couple               = normalized_channel_erpac_by_baseline(frequency_couple, baseline_couple)

    common_t       = np.arange(t_start, t_end + 1/fs, 1/fs)
    n_time         = len(common_t)
    aligned_erpacs = []
    
    for i, row in frequency_couple.iterrows():
        
        t_vec         = row.t_vector
        if(normalize==True) : erpac = row.erpac_normalized
        else                : erpac = row.erpac
            
        mask          = (t_vec >= t_start) & (t_vec <= t_end)
        trimmed_t     = t_vec[mask]
        trimmed_erpac = erpac[:, mask]
    
        # find how many samples to pad before and after trimmed data
        pad_before    = int(np.round((trimmed_t[0] - t_start) * fs)) if trimmed_t.size > 0 else full_length
        pad_after     = int(np.round((t_end - trimmed_t[-1]) * fs)) if trimmed_t.size > 0 else full_length
        
        # pad trimmed_erpac with NaNs
        padded_erpac  = np.pad(trimmed_erpac, pad_width=((0, 0), (pad_before, pad_after)), mode='constant', constant_values=np.nan)
        aligned_erpacs.append(padded_erpac)
    mean_erpac = np.nanmean(aligned_erpacs, axis=0)
    
    return common_t, frequency_couple.erpac_object.to_list()[0].f_amp, mean_erpac, aligned_erpacs


def pad_erpac_array(row, global_t):

    # pad the erpac array
    t           = np.array(row['t_vector'])
    erpac       = np.array(row['erpac'])
    padded      = np.full_like(global_t, np.nan, dtype=float)
    indices     = np.searchsorted(global_t, t)

    # Make sure indices are within range (should be if t within t_min,t_max)
    valid_mask  = (indices >= 0) & (indices < len(global_t))
    indices     = indices[valid_mask]
    erpac_valid = erpac[valid_mask]

    # Place erpac values in padded array
    padded[indices] = erpac_valid
    return padded

    
def create_aligned_recordings(df, fs=2048):
    
    # onset alignment (pre_event + event)
    min_pre_len              = df['pre_event_recording'].apply(len).min()
    min_event_len            = df['event_recording'].apply(len).min()
    onset_aligned_recordings = []
    
    for _, row in df.iterrows():
        pre          = row['pre_event_recording'][-min_pre_len:]   # last N samples from pre-event
        event        = row['event_recording'][:min_event_len]   # first M samples from event
        merged_onset = pre + event
        onset_aligned_recordings.append(merged_onset)
        
    pre_time          = np.linspace(-min_pre_len/fs, -1/fs, min_pre_len)
    event_time        = np.linspace(0, (min_event_len - 1)/fs, min_event_len)
    onset_time_vector = np.concatenate([pre_time, event_time])
    
    # offset alignment (event + post_event)
    min_post_len              = df['post_event_recording'].apply(len).min()
    offset_aligned_recordings = []
    
    for _, row in df.iterrows():
        event         = row['event_recording'][-min_event_len:]     # last M samples from event (align to event end)
        post          = row['post_event_recording'][:min_post_len]  # first K samples from post-event
        merged_offset = event + post
        offset_aligned_recordings.append(merged_offset)
        
    event_time_offset  = np.linspace(-(min_event_len - 1)/fs, 0, min_event_len)
    post_time          = np.linspace(1/fs, min_post_len/fs, min_post_len)
    offset_time_vector = np.concatenate([event_time_offset, post_time])
    
    # add to DataFrame
    df                             = df.copy()
    df['onset_aligned_recording']  = onset_aligned_recordings
    df['offset_aligned_recording'] = offset_aligned_recordings
    
    return df, onset_time_vector, offset_time_vector


def channel_wise_ERPAC(data_matrix, patient, hemisphere, channel, severity, fs):

    df_erpac                       = pd.DataFrame(columns=["patient", "hemisphere", "channel", "severity", "phase_band", "amplitude_band", "erpac", "erpac_object"])
    
    # define band descriptions for ERPAC
    band_phase                     = dict()
    band_phase["theta"]            = [4 , 8 ]
    band_phase["alpha"]            = [8 , 12]
    band_phase["beta_low"]         = [12, 20]
    band_phase["beta_high"]        = [20, 35]
    band_phase["gamma"]            = [60, 90]

    # define band descriptions for ERPAC
    band_amplitude                 = dict()
    band_amplitude["theta"]        = [4 , 8 , 1, 1]
    band_amplitude["alpha"]        = [8 , 12, 1, 1]
    band_amplitude["beta_low"]     = [12, 20, 1, 1]
    band_amplitude["beta_high"]    = [20, 35, 1, 1]
    band_amplitude["gamma"]        = [60, 90, 1, 1]
    
    pac_pairs                      = [("theta", "alpha"), ("theta", "beta_low"), ("theta", "beta_high"), ("theta", "gamma"), ("alpha", "beta_low"),
                                      ("alpha", "beta_high"), ("alpha", "gamma"), ("beta_low", "beta_high"), ("beta_low", "gamma"), ("beta_high", "gamma")]
    
    # loop through frequency pairs for ERPAC estimation within the channel
    for (phase_band, amplitude_band) in pac_pairs:
    
        try:
            # measure erpac
            erpac_object                = EventRelatedPac(f_pha=band_phase[phase_band], f_amp=band_amplitude[amplitude_band])
            erpac                       = erpac_object.filterfit(fs, data_matrix, method='gc', smooth=100)
            row                         = dict()
            row["patient"]              = patient
            row["hemisphere"]           = hemisphere
            row["channel"]              = channel
            row["severity"]             = severity
            row["phase_band"]           = phase_band
            row["amplitude_band"]       = amplitude_band
            row["erpac"]                = erpac
            row["erpac_object"]         = erpac_object
            df_erpac.loc[len(df_erpac)] = row
            
        except:
            pass
            
    return df_erpac

def get_ERPAC_for_dyskinesia_severity(dataset, recording_type, severity, fs):

    grouped = dataset.groupby(["patient", "hemisphere","channel"])
    
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("             EVENT-RELATED PHASE AMPLITUDE COUPLING ANALYSIS            ")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print(f"recording : {recording_type}")
    print(f"severity  : {severity}")
    
    df_erpac_severity = []
    
    for (patient, hemisphere, channel), channel_df in grouped:
        
        print(f">>> patient : {patient} - hemisphere : {hemisphere} - channel : {channel}"  )
    
        # create aligned recordings
        channel_df, t_onset, t_offset = create_aligned_recordings(channel_df, fs=fs)
        onset                         = np.array(channel_df.onset_aligned_recording.tolist())
        offset                        = np.array(channel_df.offset_aligned_recording.tolist())
    
        # onset ERPAC
        onset_erpac                   = channel_wise_ERPAC(onset, patient, hemisphere, channel, severity, fs)
        onset_erpac["t_vector"]       = [t_onset.copy() for _ in range(len(onset_erpac))]
        onset_erpac["alignment"]      = "onset"
        print (">>> >>> onset aligned recordings: ERPAC estimation completed for the channel!")
    
        # offset ERPAC
        offset_erpac                  = channel_wise_ERPAC(offset, patient, hemisphere, channel, severity, fs)
        offset_erpac["t_vector"]      = [t_offset.copy() for _ in range(len(offset_erpac))]
        offset_erpac["alignment"]     = "offset"
        print (">>> >>> offset aligned recordings: ERPAC estimation completed for the channel!")
        
        df_erpac_channel              = pd.concat([onset_erpac, offset_erpac], ignore_index=True)
        df_erpac_severity.append(df_erpac_channel)
    
    df_erpac_severity = pd.concat(df_erpac_severity, ignore_index=True)
    return df_erpac_severity

    
