"""
Utilisation function for accelerometer recordings
"""

import numpy as np
import pandas as pd


def create_accelerometer_event_dictionary(dataset, kinematic):
    acc_events = {}
    for event_category in dataset.event_category.unique().tolist():
        acc_events[event_category] = {}
        for dys_score in [-1,0,1,2,3]:
            if(dys_score != -1):
                acc_events[event_category]["dys"+str(dys_score)] = {}
                for alignment in ["onset", "offset"]:
                    acc_events[event_category]["dys"+str(dys_score)][alignment] = kinematic.extract_accelerometer_events(dataset, 
                                                                                                                         event_category=event_category,
                                                                                                                         dyskinesia_score=dys_score, 
                                                                                                                         alignment=alignment)
            else:
                acc_events[event_category]["all"] = {}
                for alignment in ["onset", "offset"]:
                    acc_events[event_category]["all"][alignment] = kinematic.extract_accelerometer_events(dataset, event_category=event_category, alignment=alignment)
                
    return acc_events