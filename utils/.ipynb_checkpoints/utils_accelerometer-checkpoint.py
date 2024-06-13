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

def create_accelerometer_event_dictionary(dataset, kinematic, fs, t_observation):
    
    acc_events = {}

    # loop in event categories (voluntary taps vs involuntary remaining movements)
    for event_category in dataset.event_category.unique().tolist():
        
        acc_events[event_category] = {}

        # loop in dyskinesia severity
        for severity in ["","none","mild","moderate","severe","extreme"]:

            # if particular severity is not selected, consider all events
            if(severity != ""):
                
                acc_events[event_category]["LID_"+severity] = {}

                # loop in alignment strategies
                for alignment in ["onset", "offset"]:

                    # get aligned event arrays for the selected combination
                    events = kinematic.extract_accelerometer_events(dataset, event_category=event_category, dyskinesia_score=severity, 
                                                                    alignment=alignment, t_observation=t_observation)
                    acc_events[event_category]["LID_"+severity][alignment] = events
            else:
                
                acc_events[event_category]["all"] = {}

                # loop in alignment strategies
                for alignment in ["onset", "offset"]:

                    # get aligned event arrays for the selected combination
                    events      = kinematic.extract_accelerometer_events(dataset, event_category=event_category, alignment=alignment, t_observation=t_observation)

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