"""
Utilisation function for accelerometer recordings
"""

import numpy as np
import pandas as pd

dyskinesia_severity             = {}
dyskinesia_severity["no"]       = 0
dyskinesia_severity["mild"]     = 1
dyskinesia_severity["moderate"] = 2
dyskinesia_severity["severe"]   = 3
dyskinesia_severity["extreme"]  = 4

def create_event_segment_dictionary(dataset, kinematics, fs, segment):
    
    acc_events = {}

    # loop in event categories
    for event_category in dataset.event_category.unique().tolist():
        acc_events[event_category] = {}

        # loop in dyskinesia severity
        for severity in ["no","mild","moderate","severe"]:
            events = kinematics.extract_event_segment(dataset, event_category=event_category, dyskinesia_score=dyskinesia_severity[severity],segment=segment)
            acc_events[event_category]["LID_"+severity] = events
                
    return acc_events

def create_accelerometer_event_dictionary(dataset, kinematic, fs, dyskinesia_strategy, t_observation):
    
    acc_events = {}

    # loop in event categories (voluntary taps vs involuntary remaining movements)
    for event_category in dataset.event_category.unique().tolist():
        
        acc_events[event_category] = {}

        # loop in dyskinesia severity
        for severity in ["","none","mild","moderate","severe","extreme"]:

            # if particular severity is not selected, consider all events
            if(severity != ""):
                
                acc_events[event_category][severity] = {}

                # loop in alignment strategies
                for alignment in ["onset", "offset"]:

                    # get aligned event arrays for the selected combination
                    events = kinematic.extract_accelerometer_events(dataset, event_category=event_category, dyskinesia_score=severity, 
                                                                    alignment=alignment, dyskinesia_strategy=dyskinesia_strategy, t_observation=t_observation)
                    acc_events[event_category][severity][alignment] = events
            else:
                
                acc_events[event_category]["all"] = {}

                # loop in alignment strategies
                for alignment in ["onset", "offset"]:

                    # get aligned event arrays for the selected combination
                    events      = kinematic.extract_accelerometer_events(dataset, event_category=event_category, alignment=alignment, 
                                                                         dyskinesia_strategy=dyskinesia_strategy, t_observation=t_observation)

                    acc_events[event_category]["all"][alignment] = events
                
    return acc_events

def pad_aligned_events(data, fs, padding_for="onset"):

    # if event data array is empty
    if(len(data)):
        # Find the maximum length among all arrays
        max_length  = max(len(arr) for arr in data) 

        # if the events aligned for their onset, pad the end
        if(padding_for=="onset"):
            data_padded = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=1e-8) for arr in data]
    
        # if the events aligned for their offset, pad the beginning
        elif(padding_for=="offset"):
            data_padded = [np.pad(arr, (max_length - len(arr), 0), mode='constant', constant_values=1e-8) for arr in data]

        assert len(data) == len(data_padded), f'The size of the padded data does not match with original data'
    else:
        data_padded = []
    return data_padded

def get_event_time_vector(data, fs, alignment):

    # get event length
    data_length = len(data)

    # if the events aligned for their onset, pad the end
    if(alignment=="onset"):
        return np.linspace(-1, data_length/fs-1, data_length)

    # if the events aligned for their offset, pad the beginning
    elif(alignment=="offset"):
        return np.linspace(-data_length/fs + 1, 1, data_length)

# two functions utilized for plotting of patient experimental periods, CDRS scores and movements
def find_event_segments_indices(array):
    sections = []
    start = None

    for i in range(len(array)):
        if array[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                sections.append((start, i - 1))
                start = None

    if start is not None:
        sections.append((start, len(array) - 1))

    return sections

def find_timepoint_from_indices(data_array, indices):
    pairs = []
    for start, end in indices:
        pair = (data_array[start], data_array[end])
        pairs.append(pair)
    return pairs

# For given patient event history, it gets the extracts interval (start and finish time) and dyskinesia severity in the selected body part
def get_CDRS_evaluation_intervals(EVENTS_HISTORY, body_part):

    times           = EVENTS_HISTORY.CDRS_dataframe.dopa_time.to_list()
    scores          = EVENTS_HISTORY.CDRS_dataframe[body_part].to_list()
    
    previous_score  = scores[0]
    previous_time   = times[0]
    intervals_time  = []
    intervals_score = []
    
    for i in range(len(scores)):
        if(scores[i] != previous_score):
            intervals_time.append((previous_time, (times[i] + times[i-1])/2))
            intervals_score.append(previous_score)
            previous_score = scores[i]
            previous_time  = (times[i] + times[i-1])/2
    
    intervals_time.append((previous_time, times[-1]))
    intervals_score.append(scores[-1])

    # recordings length and last evaluation time don't always match. After the last CDRS evaluation, if the recording continues, the last evaluation will be kept until the end of the recording period.
    intervals_time.append((times[-1], np.max(EVENTS_HISTORY.times)/60))
    intervals_score.append(scores[-1])
    
    return intervals_time, intervals_score