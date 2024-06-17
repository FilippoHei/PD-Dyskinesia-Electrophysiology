import os
import pandas as pd
import numpy as np
import sys
import json
from utils.utils_fileManagement import load_class_pickle, mergedData

from lib_data import DATA_IO
import utils_accelerometer

class ACCELEROMETER:

    def __init__(self, PATH, SUB):
        
        #assert DAT_SOURCE in ['acc_left', 'acc_right'], f'Please pass accelerometer DAT_SOURCE ({DAT_SOURCE})'

        # use DATA_IO to load data structure
        data_IO_r           = DATA_IO(PATH, SUB, 'acc_right')
        data_IO_l           = DATA_IO(PATH, SUB, 'acc_left')
        self.__dat_r        = data_IO_r.get_data()
        self.__dat_l        = data_IO_l.get_data()

        # populate class fields
        self.fs             = self.__dat_r.fs
        self.times          = self.__dat_r.data[:,self.__dat_r.colnames.index('dopa_time')]
        
        # load right accelerometer data
        self.ACC_R_X = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_X')]
        self.ACC_R_Y = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_Y')]
        self.ACC_R_Z = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_Z')]

        # load left accelerometer data
        self.ACC_L_X = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_X')]
        self.ACC_L_Y = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_Y')]
        self.ACC_L_Z = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_Z')]

    def extract_event_segment(self, event_dataset, laterality="", event="", event_category="", dyskinesia_score="", segment="event"):
        
        fs             = self.fs
        
        if(laterality!=""):
            # check if the selected laterality is valid
            assert laterality in ["right", "left", "bilateral"], f'Please choose laterality as "right", "left", "bilateral"'
            if(laterality=="right"):
                dyskinesia_body_part = "CDRS_right_hand"
            elif(laterality=="left"):
                dyskinesia_body_part = "CDRS_left_hand"
            elif(laterality=="bilateral"):
                dyskinesia_body_part = "CDRS_total_hands"    
        elif(laterality==""): # if laterality was not given, check the total hand scores
            dyskinesia_body_part = "CDRS_total_hands"
        
        if(event!=""):
            # check if the selected event type is valid
            assert event in event_dataset.event.unique().tolist(), f'Please enter valid event as "move", "tap"'

        if(event_category!=""): 
            # check if the event category is valid
            assert event_category in event_dataset.event_category.unique().tolist(), f'Please enter valid event category, not ({event_category})'
            
        # check if segment is valid
        assert segment in ["pre", "event", "post"], f'Please choose segment as "pre", "event", or "post"'
       
        #################################################################################################################################
        dataset = event_dataset[event_dataset['laterality']==laterality] if laterality != "" else event_dataset   # select laterality
        dataset = dataset[dataset['event']==event] if event != "" else dataset                                    # select event type
        dataset = dataset[dataset['event_category']==event_category] if event_category != "" else dataset         # select event category
        dataset = dataset[dataset[dyskinesia_body_part]==dyskinesia_score] if dyskinesia_score != "" else dataset # select dyskinesia score
        #################################################################################################################################
        
        # create empty arrays for storing accelerometer data for selected event category
        acc_svm = []

        # iterate across dataframe rows
        for _, row in dataset.iterrows():

            # get pre-event activity
            if(segment=="pre"):
                start_index    = row['event_start_index'] - fs    # 1 sec before event onset
                finish_index   = row['event_start_index']         # event onset
                
            # get event activity
            elif(segment=="event"):
                start_index    = row['event_start_index']         # event onset
                finish_index   = row['event_finish_index']        # event offset

            # get event activity
            elif(segment=="post"):
                start_index    = row['event_finish_index']        # event offset
                finish_index   = row['event_finish_index'] + fs   # 1 sec after event offset
    
            # get laterality information
            laterality     = row['laterality']
            
            if(laterality == "right"):
                acc_data_x = self.ACC_R_X[start_index:finish_index].tolist()
                acc_data_y = self.ACC_R_Y[start_index:finish_index].tolist()
                acc_data_z = self.ACC_R_Z[start_index:finish_index].tolist()
            else:
                acc_data_x = self.ACC_L_X[start_index:finish_index].tolist()
                acc_data_y = self.ACC_L_Y[start_index:finish_index].tolist()
                acc_data_z = self.ACC_L_Z[start_index:finish_index].tolist()
            
            # measure the signal vector magnitude
            svm  = np.sqrt(np.array(acc_data_x)**2 + np.array(acc_data_y)**2 + np.array(acc_data_z)**2)
            acc_svm.append(svm)
        
        return acc_svm
                                   
    def extract_accelerometer_events(self, event_dataset, laterality="", event="", event_category="", dyskinesia_score="", alignment="onset", t_observation=4):

        fs = self.fs
        
        if(laterality!=""):
            # check if the selected laterality is valid
            assert laterality in ["right", "left", "bilateral"], f'Please choose laterality as "right", "left", "bilateral"'
            if(laterality=="right"):
                dyskinesia_body_part = "CDRS_right_hand"
            elif(laterality=="left"):
                dyskinesia_body_part = "CDRS_left_hand"
            elif(laterality=="bilateral"):
                dyskinesia_body_part = "CDRS_total_hands"
                
        elif(laterality==""): # if laterality was not given, check the total hand scores
            dyskinesia_body_part = "CDRS_total_hands"
            
        if(event!=""):
            # check if the selected event type is valid
            assert event in event_dataset.event.unique().tolist(), f'Please enter valid event as "move", "tap"'

        if(event_category!=""): 
            # check if the event category is valid
            assert event_category in event_dataset.event_category.unique().tolist(), f'Please enter valid event category, not ({event_category})'
            
        # check if the event alignment strategy is valid
        assert alignment in ["onset", "offset"], f'Please choose alignment as "onset", "offset"'
       
        #################################################################################################################################
        dataset = event_dataset[event_dataset['laterality']==laterality] if laterality != "" else event_dataset   # select laterality
        dataset = dataset[dataset['event']==event] if event != "" else dataset                                    # select event type
        dataset = dataset[dataset['event_category']==event_category] if event_category != "" else dataset               # select event category
        dataset = dataset[dataset[dyskinesia_body_part]==dyskinesia_score] if dyskinesia_score != "" else dataset # select dyskinesia score
        #################################################################################################################################
        
        # create empty arrays for storing accelerometer data (signal vector magnitude of three axis) for selected event category
        acc_svm = []
    
        # iterate across dataframe rows
        for _, row in dataset.iterrows():
    
            # If events aligned based on their onset
            if(alignment=="onset"):
                start_index    = row['event_start_index'] - fs           # 1 sec before event onset
                finish_index   = start_index + (t_observation * fs)      # t_observation sec later
                
            # If events aligned based on their offset
            else:
                finish_index   = row['event_finish_index'] + fs          # 1 sec after event offset
                start_index    = finish_index - (t_observation * fs)     # t_observation sec before
    
            # get laterality information
            laterality     = row['laterality']
            
            if(laterality == "right"):
                acc_data_x = self.ACC_R_X[start_index:finish_index].tolist()
                acc_data_y = self.ACC_R_Y[start_index:finish_index].tolist()
                acc_data_z = self.ACC_R_Z[start_index:finish_index].tolist()
            else:
                acc_data_x = self.ACC_L_X[start_index:finish_index].tolist()
                acc_data_y = self.ACC_L_Y[start_index:finish_index].tolist()
                acc_data_z = self.ACC_L_Z[start_index:finish_index].tolist()

            # measure the signal vector magnitude
            svm  = np.sqrt(np.array(acc_data_x)**2 + np.array(acc_data_y)**2 + np.array(acc_data_z)**2)
            acc_svm.append(svm)
            
        return acc_svm

    def create_event_segment_dictionary(self, dataset, fs, segment):
    
        acc_events = {}
    
        # loop in event categories
        for event_category in dataset.event_category.unique().tolist():
            acc_events[event_category] = {}
    
            # loop in dyskinesia severity
            for severity in ["none","mild","moderate","severe","extreme"]:
                events = self.extract_event_segment(dataset, event_category=event_category, dyskinesia_score=severity,segment=segment)
                acc_events[event_category]["LID_"+severity] = events
                    
        return acc_events

    def __measure_temporal_metrics(self, dataframe, data, patient, fs, event_category, dyskinesia_severity, segment):
        
        data                             = np.array(data)
        
        metrics                          = {}
        metrics["patient"]               = patient
        metrics["event_category"]        = event_category
        metrics["dyskinesia_severity"]   = dyskinesia_severity
        metrics["event_segment"]         = segment
        
        metrics["mean"]                  = np.mean(data)
        metrics["std"]                   = np.std(data)
        metrics["RMS"]                   = np.sqrt(np.mean(data**2))
        metrics["range"]                 = np.ptp(data)
        metrics["median"]                = np.median(data)
        metrics["iqr"]                   = np.percentile(data, 75) - np.percentile(data, 25)
        metrics["peak"]                  = np.max(data)
        metrics["peak_location"]         = np.argmax(abs(data))/fs                           # temporal location of peak (pos or neg) in seconds
        metrics["mean_crossing_rate"]    = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
        metrics["signal_energy"]         = np.sum(data**2)
        metrics["signal_magnitude_area"] = np.sum(np.abs(data))                              # signal magnitude area: sum of the absolute values
        metrics["crest_factor"]          = np.max(np.abs(data)) / np.sqrt(np.mean(data**2))  # ratio of the peak value to the RMS value.
        metrics["impulse_factor"]        = np.max(np.abs(data)) / np.mean(np.abs(data))      # ratio of the peak value to the mean value.
        metrics["shape_factor"]          = np.sqrt(np.mean(data**2)) / np.mean(np.abs(data)) # ratio of the RMS value to the mean value.
        
        # clearance factor: Ratio of the peak value to the RMS value of the mean absolute deviation
        metrics["clearance_factor"]      = np.max(np.abs(data)) / np.mean(np.sqrt(np.abs(data - np.mean(data)))) 
        
        dataframe.loc[len(dataframe)]    = metrics
        
        return dataframe
        
    def extract_temporal_metrics_from_event_segments(self, patient, event_dataframe):
        
        df_metrics = pd.DataFrame(columns=["patient","event_category","dyskinesia_severity","event_segment",
                                           'mean', 'std', 'RMS', 'range', 'median', 'iqr', 'peak', "peak_location",
                                           'mean_crossing_rate', 'signal_energy', 'signal_magnitude_area', 
                                           'crest_factor', 'impulse_factor', 'shape_factor', 'clearance_factor'])
    
        for segment in ["pre", "event", "post"]:
            
            event_segment_dict = self.create_event_segment_dictionary(event_dataframe, self.fs, segment=segment)
        
            for event_category in event_segment_dict.keys():
                for dyskinesia_severity in ["LID_none", "LID_mild", "LID_moderate", "LID_severe", "LID_extreme"]:
                    event_segment_array = event_segment_dict[event_category][dyskinesia_severity]
                    if(len(event_segment_array)!=0):
                        for i in range(len(event_segment_array)):
                            even_segment = event_segment_array[i]
                            df_metrics   = self.__measure_temporal_metrics(df_metrics, even_segment, patient, self.fs, event_category, dyskinesia_severity, segment)

        metric_list = ['mean', 'std', 'RMS', 'range', 'median', 'iqr', 'peak', "peak_location", 'mean_crossing_rate', 
                       'signal_energy', 'signal_magnitude_area',  'crest_factor', 'impulse_factor', 'shape_factor', 'clearance_factor']
        
        return df_metrics, metric_list







    