"""
Miscellaneous utilisation functions
"""

import pandas as pd
import numpy as np
import ast
import os
from os import listdir
from scipy.interpolate import interp1d

# converting string representation of a numeric array into numeric representation
def convert_to_array(string):
    try:
        # Ensure that the string is properly formatted
        if isinstance(string, str) and string.startswith('[') and string.endswith(']'):
            # Use ast.literal_eval to safely evaluate the string as a Python literal expression
            return ast.literal_eval(string)
        else:
            raise ValueError("String is not properly formatted as a list.")
    except (ValueError, SyntaxError) as e:
        # Handle cases where the string is not a valid literal expression
        print(f"Error parsing string: {string}\nException: {e}")
        return None

# get the SUB codes which stated in the PATH
def get_SUB_list(PATH):    
    SUB_list = []
    for dir in os.listdir(PATH):
        if("sub" in dir):
            SUB_list.append(dir[4:])
            
    return SUB_list

# get file names with specific format from given PATH
def get_files_with_specific_format(PATH, suffix=".csv" ):
    filenames = listdir(PATH)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# combine two python dictionary structure key to key measure by appending the values
def combine_dictionaries(dict1, dict2):
    
    combined_dict = {}

    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            combined_dict[key] = combine_dictionaries(dict1[key], dict2[key])
        elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
            combined_dict[key] = dict1[key] + dict2[key]
        else:
            raise ValueError("The structure of the dictionaries is not consistent.")

    return combined_dict

def combine_dataframes(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

def interpolate_signal(signal, target_duration, fs):
    
    original_length     = len(signal)
    original_duration   = original_length / fs
    original_time       = np.linspace(0, original_duration, original_length)
    
    if(original_duration<target_duration):
        # Ensure the target time vector ends at the same maximum time as the original
        target_time         = np.linspace(0, original_duration, int(target_duration * fs))
        
        # Interpolate using the original and target time vectors
        interpolator        = interp1d(original_time, signal, kind='cubic', fill_value="extrapolate")
        interpolated_signal = interpolator(target_time)
        
        return interpolated_signal
    else:
        return signal

def interpolate_2d_array(array_2d, target_length, fs):
   
    interpolated_array = []
    
    for signal in array_2d:
        
        original_length   = len(signal)

        # Only interpolate when the original duration of the signal is shorter than the target duration
        if(original_length<target_time):
            
            original_duration    = original_length / fs
            original_time        = np.linspace(0, original_duration, original_length)
            
            target_time          = np.linspace(0, original_duration, int(target_length * fs))
            
            interpolator         = interp1d(original_time, signal, kind='linear', fill_value="extrapolate")
            interpolated_signal  = interpolator(target_time)
            
            interpolated_array.append(interpolated_signal)
            
        else:
            interpolated_array.append(signal)
    
    # Convert the list of interpolated signals back into a 2D array
    interpolated_array = np.array(interpolated_array)
    
    return interpolated_array

def create_spatial_data_for_PSD_features(df_PSD, MNI_coordinates, data_type, feature):
    
    df_feature               = []
    
    for severity in df_PSD.keys():

        if(data_type=="lfp"):
            dynamic               = df_PSD[severity][["patient","LFP_hemisphere","LFP_channel",feature]]
            dynamic["hemisphere"] = dynamic.LFP_hemisphere
            dynamic["channel"]    = dynamic.LFP_channel
            dynamic["feature"]    = feature
            dynamic["value"]      = dynamic[feature]
            dynamic               = dynamic[["patient","hemisphere","channel","feature"]]
            dynamic               = pd.merge(dynamic, MNI_coordinates, on=['patient', 'hemisphere', 'channel'], how='inner')
        else:
            dynamic = df_PSD[severity][["patient","ECoG_hemisphere","ECoG_channel",feature]]
            dynamic["hemisphere"] = dynamic.ECoG_hemisphere
            dynamic["channel"]    = dynamic.ECoG_channel
            dynamic["feature"]    = feature
            dynamic["value"]      = dynamic[feature]
            dynamic               = dynamic[["patient","hemisphere","channel","feature","value"]]
            dynamic               = pd.merge(dynamic, MNI_coordinates, on=['patient', 'hemisphere', 'channel'], how='inner')
    
        dynamic               = dynamic[["patient","feature","value","x","y","z","AAL3_cortex"]]
        dynamic["severity"]   = severity
    
        if(len(df_feature)==0):
            df_feature = dynamic
        else:
            df_feature = pd.concat([df_feature, dynamic], ignore_index=True)

    return df_feature

def get_onset_and_offset_aligned_recordings(dataframe, fs):
    
    rec_onset_aligned  = []
    rec_offset_aligned = []
    
    for index, row in dataframe.iterrows():
    
        recording_onset  = []
        recording_offset = []
        
        if(len(row.event_recording)<fs*2):
            recording_onset.extend(row.pre_event_recording)
            recording_onset.extend(row.event_recording)
            recording_onset.extend(row.post_event_recording[0:fs*2 - len(row.event_recording)])
    
            recording_offset.extend(row.pre_event_recording[-(fs*2 - len(row.event_recording)):])
            recording_offset.extend(row.event_recording)
            recording_offset.extend(row.post_event_recording)
        else:
            recording_onset.extend(row.pre_event_recording)
            recording_onset.extend(row.event_recording[0:fs*2])
    
            recording_offset.extend(row.event_recording[-fs*2:]) # get 2 second section from the end of event 
            recording_offset.extend(row.post_event_recording)

        # if there are missing segments (due to the artifact removal), dont put that event to the dataframe
        if(len(recording_onset)==fs*4):
            rec_onset_aligned.append(recording_onset)
        else:
            rec_onset_aligned.append(np.nan)
            
        if(len(recording_offset)==fs*4):
            rec_offset_aligned.append(recording_offset)
        else:
            rec_offset_aligned.append(np.nan)

    dataframe["recording_onset_aligned"]  = rec_onset_aligned
    dataframe["recording_offset_aligned"] = rec_offset_aligned
    
    return dataframe

def spectrogram_downsampling_with_mean(spectrogram, fs, time_interval_in_second):
    sample_interval   = 1 / fs  # in seconds)
    downsample_factor = int(time_interval_in_second/sample_interval)
    
    # Reshape the matrix: split the time axis into groups of 20 and take the mean along the new axis
    downsampled_spectrogram = spectrogram[:, :spectrogram.shape[1] // downsample_factor * downsample_factor].reshape(
        spectrogram.shape[0], -1, downsample_factor).mean(axis=2)
    return downsampled_spectrogram
