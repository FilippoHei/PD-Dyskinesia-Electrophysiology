import pingouin as pg
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import sys

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

#import utils scripts
import utils_misc, utils_psd

from lib_data import DATA_IO

class LFP:
    
    def __init__(self, PATH, SUB):
        
        self.__PATH         = PATH
        self.__SUB          = SUB

        # use DATA_IO to load data structure
        data_IO_r           = DATA_IO(PATH, SUB, 'lfp_right')
        data_IO_l           = DATA_IO(PATH, SUB, 'lfp_left')
        self.__dat_r        = data_IO_r.get_data()
        self.__dat_l        = data_IO_l.get_data()

        # populate class fields
        self.fs             = self.__dat_r.fs
        self.times          = self.__dat_r.times

        # load right LFP data
        self.LFP_R_channel_32 = self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_03')] - self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_02')]
        self.LFP_R_channel_43 = self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_04')] - self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_03')]
        self.LFP_R_channel_54 = self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_05')] - self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_03')]
        self.LFP_R_channel_65 = self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_06')] - self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_05')]
        self.LFP_R_channel_76 = self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_07')] - self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_06')]
        self.LFP_R_channel_87 = self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_08')] - self.__dat_r.data[:,self.__dat_r.colnames.index('LFP_R_07')]

        # load LEFT LFP data 
        self.LFP_L_channel_32 = self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_03')] - self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_02')]
        self.LFP_L_channel_43 = self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_04')] - self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_03')]
        self.LFP_L_channel_54 = self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_05')] - self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_03')]
        self.LFP_L_channel_65 = self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_06')] - self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_05')]
        self.LFP_L_channel_76 = self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_07')] - self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_06')]
        self.LFP_L_channel_87 = self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_08')] - self.__dat_l.data[:,self.__dat_l.colnames.index('LFP_L_07')]
        
    def extract_LFP_events(self, dataset, t_observation=4):
        """
        Description
            This static method calculates the average power spectral density (PSD) and its corresponding error for a dataset of local field potential (LFP) recordings. 
            The PSD is normalized and represented as a percentage. The method can compute either the standard deviation or standard error as the error metric.

        Input
            dataset       : A dataframe contains events regarding: "patient","laterality","event_type", "event_no", "event_start_index", "event_finish_index", 
                            "event_start_time", "event_finish_time", "duration", "task" columns. It can be empty or filled previously.
            t_observation : Numeric value represents how long the 

        Output
            event_indices: A list containing tupples (start_index, finish_index) of index information for events
        """
        
        fs        = self.fs
        alignment = "onset" # LFP activity is only aligned to event onsets

        # create empty dataframe to populate
        events = pd.DataFrame(columns=["event_category", "event_laterality", "LFP_hemisphere", "LFP_channel", "recording", 
                                       "event_start_time", 'CDRS_right_hand', 'CDRS_left_hand', 'CDRS_total_right', 'CDRS_total_left'])
    
        for index, event in dataset.iterrows():

            row                     = {}
            row["event_category"]   = event['event_category']
            row["event_laterality"] = event['laterality']
            row["event_start_time"] = event['event_start_time']
            row["CDRS_right_hand"]  = event['CDRS_right_hand']
            row["CDRS_left_hand"]   = event['CDRS_left_hand']
            row["CDRS_total_right"] = event['CDRS_total_right']
            row["CDRS_total_left"]  = event['CDRS_total_left']
            #row["event_no"]         = event['event_no']
            #row["CDRS_face"]        = event['CDRS_face']
            #row["CDRS_neck"]        = event['CDRS_neck']
            #row["CDRS_trunk"]       = event['CDRS_trunk']
            #row["CDRS_right_leg"]   = event['CDRS_right_leg']
            #row["CDRS_left_leg"]    = event['CDRS_left_leg']
            #row["CDRS_total_hands"] = event['CDRS_total_hands']
            #row["CDRS_total"]       = event['CDRS_total']
              
            for channel in ["32","43","54","65","76","87"]:

                for hemisphere in ["right","left"]:
                        
                    # events aligned based on their onset
                    start_index    = event['event_start_index'] - fs         # 1 sec before event onset
                    finish_index   = start_index + (t_observation * fs)      # t_observation sec later

                    if(hemisphere=="right"):
                        if(channel=="32"):
                            recording = self.LFP_R_channel_32[start_index:finish_index]
                        elif(channel=="43"):
                            recording = self.LFP_R_channel_43[start_index:finish_index]
                        elif(channel=="54"):
                            recording = self.LFP_R_channel_54[start_index:finish_index]
                        elif(channel=="65"):
                            recording = self.LFP_R_channel_65[start_index:finish_index]
                        elif(channel=="76"):
                            recording = self.LFP_R_channel_76[start_index:finish_index]
                        elif(channel=="87"):
                            recording = self.LFP_R_channel_87[start_index:finish_index]
                    else:
                        if(channel=="32"):
                            recording = self.LFP_L_channel_32[start_index:finish_index]
                        elif(channel=="43"):
                            recording = self.LFP_L_channel_43[start_index:finish_index]
                        elif(channel=="54"):
                            recording = self.LFP_L_channel_54[start_index:finish_index]
                        elif(channel=="65"):
                            recording = self.LFP_L_channel_65[start_index:finish_index]
                        elif(channel=="76"):
                            recording = self.LFP_L_channel_76[start_index:finish_index]
                        elif(channel=="87"):
                            recording = self.LFP_L_channel_87[start_index:finish_index]

                    row["LFP_hemisphere"] = hemisphere
                    row["LFP_channel"]    = channel
                    row["recording"]      = recording.tolist()
                    row["alignment"]      = alignment

                    recording = np.array(recording, dtype=float)

                    # if LFP recording does not contain any np.nan value, then add to the dataframe
                    if(np.isnan(recording).any()==False):
                        events.loc[len(events)] = row
                    
        return events

    def extract_LFP_events_segments(self, dataset):
        """
        Description
            This static method calculates the average power spectral density (PSD) and its corresponding error for a dataset of local field potential (LFP) recordings. 
            The PSD is normalized and represented as a percentage. The method can compute either the standard deviation or standard error as the error metric.

        Input
            dataset       : A dataframe contains events regarding: "patient","laterality","event_type", "event_no", "event_start_index", "event_finish_index", 
                            "event_start_time", "event_finish_time", "duration", "task" columns. It can be empty or filled previously.
            t_observation : Numeric value represents how long the 

        Output
            event_indices: A list containing tupples (start_index, finish_index) of index information for events
        """
        
        fs        = self.fs
        alignment = "onset" # LFP activity is only aligned to event onsets

        # create empty dataframe to populate
        events = pd.DataFrame(columns=["event_category", "event_laterality", "LFP_hemisphere", "LFP_channel", "pre_event_recording", "event_recording", "post_event_recording",
                                       "event_start_time", 'CDRS_right_hand', 'CDRS_left_hand', 'CDRS_total_right', 'CDRS_total_left'])
    
        for index, event in dataset.iterrows():

            row                     = {}
            row["event_category"]   = event['event_category']
            row["event_laterality"] = event['laterality']
            row["event_start_time"] = event['event_start_time']
            row["CDRS_right_hand"]  = event['CDRS_right_hand']
            row["CDRS_left_hand"]   = event['CDRS_left_hand']
            row["CDRS_total_right"] = event['CDRS_total_right']
            row["CDRS_total_left"]  = event['CDRS_total_left']
            #row["event_no"]         = event['event_no']
            #row["CDRS_face"]        = event['CDRS_face']
            #row["CDRS_neck"]        = event['CDRS_neck']
            #row["CDRS_trunk"]       = event['CDRS_trunk']
            #row["CDRS_right_leg"]   = event['CDRS_right_leg']
            #row["CDRS_left_leg"]    = event['CDRS_left_leg']
            #row["CDRS_total_hands"] = event['CDRS_total_hands']
            #row["CDRS_total"]       = event['CDRS_total']
              
            for channel in ["32","43","54","65","76","87"]:

                for hemisphere in ["right","left"]:
                        
                    # events aligned based on their onset
                    start_index_pre    = event['event_start_index'] - fs  # pre-event start index: 1 sec before event onset
                    finish_index_pre   = event['event_start_index']       # pre-event finish index: event onset

                    start_index_event  = event['event_start_index']       # event start index
                    finish_index_event = event['event_finish_index']      # event finish index

                    start_index_post   = event['event_finish_index']      # post-event start index: event offset
                    finish_index_post  = event['event_finish_index'] + fs # post-event finish index: 1 sec after event offset

                    if(hemisphere=="right"):
                        if(channel=="32"):
                            recording_pre   = self.LFP_R_channel_32[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_R_channel_32[start_index_event:finish_index_event]
                            recording_post  = self.LFP_R_channel_32[start_index_post:finish_index_post]
                        elif(channel=="43"):
                            recording_pre   = self.LFP_R_channel_43[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_R_channel_43[start_index_event:finish_index_event]
                            recording_post  = self.LFP_R_channel_43[start_index_post:finish_index_post]
                        elif(channel=="54"):
                            recording_pre   = self.LFP_R_channel_54[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_R_channel_54[start_index_event:finish_index_event]
                            recording_post  = self.LFP_R_channel_54[start_index_post:finish_index_post]
                        elif(channel=="65"):
                            recording_pre   = self.LFP_R_channel_65[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_R_channel_65[start_index_event:finish_index_event]
                            recording_post  = self.LFP_R_channel_65[start_index_post:finish_index_post]
                        elif(channel=="76"):
                            recording_pre   = self.LFP_R_channel_76[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_R_channel_76[start_index_event:finish_index_event]
                            recording_post  = self.LFP_R_channel_76[start_index_post:finish_index_post]
                        elif(channel=="87"):
                            recording_pre   = self.LFP_R_channel_87[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_R_channel_87[start_index_event:finish_index_event]
                            recording_post  = self.LFP_R_channel_87[start_index_post:finish_index_post]
                    else:
                        if(channel=="32"):
                            recording_pre   = self.LFP_L_channel_32[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_L_channel_32[start_index_event:finish_index_event]
                            recording_post  = self.LFP_L_channel_32[start_index_post:finish_index_post]
                        elif(channel=="43"):
                            recording_pre   = self.LFP_L_channel_43[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_L_channel_43[start_index_event:finish_index_event]
                            recording_post  = self.LFP_L_channel_43[start_index_post:finish_index_post]
                        elif(channel=="54"):
                            recording_pre   = self.LFP_L_channel_54[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_L_channel_54[start_index_event:finish_index_event]
                            recording_post  = self.LFP_L_channel_54[start_index_post:finish_index_post]
                        elif(channel=="65"):
                            recording_pre   = self.LFP_L_channel_65[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_L_channel_65[start_index_event:finish_index_event]
                            recording_post  = self.LFP_L_channel_65[start_index_post:finish_index_post]
                        elif(channel=="76"):
                            recording_pre   = self.LFP_L_channel_76[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_L_channel_76[start_index_event:finish_index_event]
                            recording_post  = self.LFP_L_channel_76[start_index_post:finish_index_post]
                        elif(channel=="87"):
                            recording_pre   = self.LFP_L_channel_87[start_index_pre:finish_index_pre]
                            recording_event = self.LFP_L_channel_87[start_index_event:finish_index_event]
                            recording_post  = self.LFP_L_channel_87[start_index_post:finish_index_post]

                    # check how much of pre and post-segments of the event is empty due to the artifact removal
                    # perc_empty_pre_rec  = (np.isnan(recording_pre).sum()/len(recording_pre)) * 100   # check how much of pre-event recording is empty (due to artifact removal)
                    # perc_empty_post_rec = (np.isnan(recording_post).sum()/len(recording_post)) * 100 # check how much of post-event recording is empty (due to artifact removal)

                    # if more than half of pre and post segments are empty dont add them to the dataframe work, instead put np.nan
                    # if(perc_empty_pre_rec <= 20): 
                    #     row["pre_event_recording"]  = np.nan
                    # else:
                    #    recording_pre[np.isnan(recording_pre)] = 0 # before adding to dataframe replace nan with 0.
                    #    row["pre_event_recording"]  = recording_pre.tolist()

                    # if(perc_empty_post_rec <= 50): 
                    #     row["post_event_recording"]  = np.nan
                    # else:
                    #    recording_post[np.isnan(recording_post)] = 0 # before adding to dataframe replace nan with 0.
                    #    row["post_event_recording"] = recording_post.tolist()

                    # recording_pre[np.isnan(recording_pre)] = 0 # before adding to dataframe replace nan with 0.
                    # recording_post[np.isnan(recording_post)] = 0 # before adding to dataframe replace nan with 0.

                    row["LFP_hemisphere"]       = hemisphere
                    row["LFP_channel"]          = channel
                    row["pre_event_recording"]  = recording_pre.tolist()
                    row["event_recording"]      = recording_event.tolist()
                    row["post_event_recording"] = recording_post.tolist()

                    # convert segment arrays into float type
                    recording_pre   = np.array(recording_pre, dtype=float)
                    recording_event = np.array(recording_event, dtype=float)
                    recording_post  = np.array(recording_post, dtype=float)
                    
                    # if LFP event recording does not contain any np.nan value, then add to the dataframe
                    if(np.isnan(recording_event).any()==False):
                        events.loc[len(events)] = row
                    
                    
        return events
        
    @staticmethod
    def select_LFP_recordings(dataset, hemisphere="", channel="32", event_category="tapping", dyskinesia_severity="all"):

        if(hemisphere!=""): # if particular hemisphere is selected
            
            if(hemisphere=="right"):
                ipsilateral_hand    = "right"
                ipsilateral_scale   = "CDRS_right_hand"
                controlateral_hand  = "left"
                controlateral_scale = "CDRS_left_hand"
            else:
                ipsilateral_hand    = "left"
                ipsilateral_scale   = "CDRS_left_hand"
                controlateral_hand  = "right"
                controlateral_scale = "CDRS_right_hand"
        
            LFP_ipsilateral_events   = dataset[(dataset.LFP_hemisphere==hemisphere) & 
                                               (dataset.LFP_channel==channel) & 
                                               (dataset.event_laterality==ipsilateral_hand) &
                                               (dataset.event_category==event_category)]
        
            LFP_controlateral_events = dataset[(dataset.LFP_hemisphere==hemisphere) & 
                                               (dataset.LFP_channel==channel) & 
                                               (dataset.event_laterality==controlateral_hand) &
                                               (dataset.event_category==event_category)]
            
            # in case of particular dyskinesia severity is selected 
            if(dyskinesia_severity!="all"):
                LFP_ipsilateral_events   = LFP_ipsilateral_events[LFP_ipsilateral_events[ipsilateral_scale]==dyskinesia_severity]
                LFP_controlateral_events = LFP_controlateral_events[LFP_controlateral_events[controlateral_scale]==dyskinesia_severity]

        else: # if a particular hemisphere is not selected
            LFP_ipsilateral_events = dataset[(dataset.event_laterality == dataset.LFP_hemisphere) &
                                             (dataset.LFP_channel==channel) &
                                             (dataset.event_category==event_category)]

            LFP_controlateral_events = dataset[(dataset.event_laterality != dataset.LFP_hemisphere) &
                                               (dataset.LFP_channel==channel) &
                                               (dataset.event_category==event_category)]

            # in case of particular dyskinesia severity is selected 
            if(dyskinesia_severity!="all"):
                LFP_ipsilateral_events = LFP_ipsilateral_events[((LFP_ipsilateral_events.event_laterality == "right") 
                                                                 & (LFP_ipsilateral_events.CDRS_right_hand == dyskinesia_severity)) | 
                                                                ((LFP_ipsilateral_events.event_laterality == "left") 
                                                                 & (LFP_ipsilateral_events.CDRS_left_hand == dyskinesia_severity))]
                
                LFP_controlateral_events = LFP_controlateral_events[((LFP_controlateral_events.event_laterality == "right") 
                                                                     & (LFP_controlateral_events.CDRS_right_hand == dyskinesia_severity)) | 
                                                                    ((LFP_controlateral_events.event_laterality == "left") 
                                                                     & (LFP_controlateral_events.CDRS_left_hand == dyskinesia_severity))]
            
        #LFP_ipsilateral_events['recording']   = LFP_ipsilateral_events['recording'].apply(utils_misc.convert_to_array)
        #LFP_controlateral_events['recording'] = LFP_controlateral_events['recording'].apply(utils_misc.convert_to_array)
        
        return LFP_ipsilateral_events, LFP_controlateral_events

    @staticmethod
    def extract_average_psd_for_LFP(dataset, segment, error_bar):

        # create an empty array
        psd_array = []

        # pass through all the rows in the dataframe containing LFP recordings.
        for index, row in dataset.iterrows():

            # measure the normalized/relative PSD for give sampling frequency (constant)
            if(segment=="pre_event"):
                freq, psd = utils_psd.measure_normalized_psd(row.pre_event_recording, fs=2048) 
            elif(segment=="event"):
                freq, psd = utils_psd.measure_normalized_psd(row.event_recording, fs=2048)
            elif(segment=="post_event"):
                freq, psd = utils_psd.measure_normalized_psd(row.post_event_recording, fs=2048)
                
            psd_array.append(psd)

        # measure the mean power spectrum for all selected events and multiply by 100 to represent the percentage.
        mean_psd = np.mean(psd_array, axis=0) * 100 

        # based on the selected error bar, find either the standard deviation or standard error around the average PSD for each frequency
        if(error_bar=="sd"):
            error       = np.std(psd_array, axis=0) * 100
            error_label = "standard deviation"
        elif(error_bar=="se"):
            error       = 2 * np.std(psd_array, axis=0) / np.sqrt(len(psd_array)) * 100
            error_label = "standard error"

        return freq, mean_psd, error

    @staticmethod
    def measure_LFP_power_spectra_with_laterality(dataset, hemisphere, channel, event_category, segment, severity):

        # get the ipsilateral events for the selected hemisphere for the given channel, event_category, and dyskinesia severity aspects.
        LFP_ipsilateral, LFP_controlateral = LFP.select_LFP_recordings(dataset, hemisphere=hemisphere, channel=channel, 
                                                                       event_category=event_category, dyskinesia_severity=severity)
        if(len(LFP_ipsilateral)!=0):
            freq_ipsilateral, mean_ipsilateral, error_ipsilateral = LFP.extract_average_psd_for_LFP(LFP_ipsilateral, segment, error_bar="se")
        else:
            freq_ipsilateral = []; mean_ipsilateral = []; error_ipsilateral = []
            
        if(len(LFP_controlateral)!=0):
            freq_controlateral, mean_controlateral, error_controlateral = LFP.extract_average_psd_for_LFP(LFP_controlateral, segment, error_bar="se")
        else:
            freq_controlateral = []; mean_controlateral = []; error_controlateral = []
    
        return freq_ipsilateral, mean_ipsilateral, error_ipsilateral, freq_controlateral, mean_controlateral, error_controlateral
