import os
import pandas as pd
import numpy as np
import sys
import json
from utils.utils_fileManagement import load_class_pickle, mergedData

sys.path.insert(0, './utils/')

#import utils scripts
import utils_filtering, utils_accelerometer

from lib_data import DATA_IO


class ACCELEROMETER:

    def __init__(self, PATH, SUB):
        
        print("ACCELEROMETER: SUB-" + SUB)
        print("... loading started")

        # use DATA_IO to load data structure
        data_IO_r     = DATA_IO(PATH, SUB, 'acc_right')
        data_IO_l     = DATA_IO(PATH, SUB, 'acc_left')
        
        self.SUB      = SUB
        self.__dat_r  = data_IO_r.get_data()
        self.__dat_l  = data_IO_l.get_data()

        # populate class fields
        self.fs       = self.__dat_r.fs
        self.times    = self.__dat_r.data[:,self.__dat_r.colnames.index('dopa_time')]
        
        # load right accelerometer data
        self.ACC_R_X  = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_X')]
        self.ACC_R_Y  = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_Y')]
        self.ACC_R_Z  = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_Z')]

        # load left accelerometer data
        self.ACC_L_X  = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_X')]
        self.ACC_L_Y  = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_Y')]
        self.ACC_L_Z  = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_Z')]

        if(SUB=="019"):
            self.ACC_R_X  = utils_filtering.bandpass_filter(self.ACC_R_X.astype(float), fs=512, l_freq=1, h_freq=48).astype(float)
            self.ACC_R_Y  = utils_filtering.bandpass_filter(self.ACC_R_Y.astype(float), fs=512, l_freq=1, h_freq=48).astype(float)
            self.ACC_R_Z  = utils_filtering.bandpass_filter(self.ACC_R_Z.astype(float), fs=512, l_freq=1, h_freq=48).astype(float)
            self.ACC_L_X  = utils_filtering.bandpass_filter(self.ACC_L_X.astype(float), fs=512, l_freq=1, h_freq=48).astype(float)
            self.ACC_L_Y  = utils_filtering.bandpass_filter(self.ACC_L_Y.astype(float), fs=512, l_freq=1, h_freq=48).astype(float)
            self.ACC_L_Z  = utils_filtering.bandpass_filter(self.ACC_L_Z.astype(float), fs=512, l_freq=1, h_freq=48).astype(float)

        self.ACC_R    = np.sqrt(np.array(self.ACC_R_X.tolist())**2 + np.array(self.ACC_R_Y.tolist())**2 + np.array(self.ACC_R_Z.tolist())**2)
        self.ACC_L    = np.sqrt(np.array(self.ACC_L_X.tolist())**2 + np.array(self.ACC_L_Y.tolist())**2 + np.array(self.ACC_L_Z.tolist())**2)
                                   
    def extract_accelerometer_events(self, dataset):
    
        dataset_accelerometer = pd.DataFrame(columns=['patient', 'laterality', 'event_no', 'event_category', 'event_start_index', 'event_finish_index', 'event_start_time',
                                                  'event_finish_time', 'duration', 'CDRS_right_hand', 'CDRS_left_hand', 'CDRS_total', 'dyskinesia_arm', 'dyskinesia_total',
                                                  'event_onset_aligned', 'event_offset_aligned'])
        for index in range(len(dataset)):

            row                       = {} 
            row['patient']            = dataset.iloc[index]['patient']
            row['laterality']         = dataset.iloc[index]['laterality']
            row['event_no']           = dataset.iloc[index]['event_no']
            row['event_category']     = dataset.iloc[index]['event_category']
            row['event_start_index']  = dataset.iloc[index]['event_start_index']
            row['event_finish_index'] = dataset.iloc[index]['event_finish_index']
            row['event_start_time']   = dataset.iloc[index]['event_start_time']
            row['event_finish_time']  = dataset.iloc[index]['event_finish_time']
            row['duration']           = dataset.iloc[index]['duration']
            row['CDRS_right_hand']    = dataset.iloc[index]['CDRS_right_hand']
            row['CDRS_left_hand']     = dataset.iloc[index]['CDRS_left_hand']
            row['CDRS_total']         = dataset.iloc[index]['CDRS_total']
            row['dyskinesia_arm']     = dataset.iloc[index]['dyskinesia_arm']
            row['dyskinesia_total']   = dataset.iloc[index]['dyskinesia_total']
            
            laterality = row['laterality']
            i_start    = row['event_start_index']
            i_finish   = row['event_finish_index']
            
            if(laterality=="right"):
                onset_rec  = self.ACC_R[i_start  - self.fs*2 : i_start  + self.fs*2].tolist()
                offset_rec = self.ACC_R[i_finish - self.fs*2 : i_finish + self.fs*2].tolist()     
            else:
                onset_rec  = self.ACC_L[i_start  - self.fs*2 : i_start  + self.fs*2].tolist()
                offset_rec = self.ACC_L[i_finish - self.fs*2 : i_finish + self.fs*2].tolist()   

            if(len(onset_rec) == len(offset_rec) == self.fs*4): #if segments lengths of these recordings were equal to = fs*4
                row['event_onset_aligned']                            = onset_rec
                row['event_offset_aligned']                           = offset_rec
                dataset_accelerometer.loc[len(dataset_accelerometer)] = row 

        return dataset_accelerometer

    def extract_temporal_metrics_from_event_segments(self, patient, event_dataframe):
            
            df_metrics = pd.DataFrame(columns=["patient","event_category","dyskinesia_severity","event_segment", "duration", 'peak', "peak_location"])
        
            for segment in ["pre", "event", "post"]:
                
                event_segment_dict = self.create_event_segment_dictionary(event_dataframe, self.fs, segment=segment)
            
                for event_category in event_segment_dict.keys():
                    for dyskinesia_severity in ["LID_none", "LID_mild", "LID_moderate", "LID_severe", "LID_extreme"]:
                        event_segment_array = event_segment_dict[event_category][dyskinesia_severity]
                        if(len(event_segment_array)!=0):
                            for i in range(len(event_segment_array)):
                                even_segment = event_segment_array[i]
                                df_metrics   = self.__measure_temporal_metrics(df_metrics, even_segment, patient, self.fs, event_category, dyskinesia_severity, segment)
    
            metric_list = ['duration', 'mean', 'std', 'RMS', 'range', 'median', 'iqr', 'peak', "peak_location", 'mean_crossing_rate', 
                           'signal_energy', 'signal_magnitude_area',  'crest_factor', 'impulse_factor', 'shape_factor', 'clearance_factor']
            
            return df_metrics, metric_list







    