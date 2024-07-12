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

        print("LFP Recording: SUB-" + SUB)
        
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

        # rereferencing for channels 
        self.bipolar_channels_8_contacts = ["2-1", "3-1", "4-1",
                                            "3-2", "4-3", "4-2",
                                            "5-2", "6-3", "7-4",
                                            "6-5", "7-6", "7-5",
                                            "8-5", "8-6", "8-7"]
        
        self.bipolar_channels_16_contacts = ["2-1"  , "3-2"  , "3-1",
                                             "4-1"  , "5-2"  , "6-3",
                                             "5-4"  , "6-5"  , "6-4",
                                             "7-4"  , "8-5"  , "9-6",
                                             "8-7"  , "9-8"  , "9-7",
                                             "10-7" , "11-8" , "12-9",
                                             "11-10", "12-11", "12-10",
                                             "13-10", "14-11", "15-12",
                                             "14-13", "15-14", "15-13"]

        self.bipolar_channels = self.bipolar_channels_16_contacts

        # load right LFP data
        self.recordings          = {}
        self.recordings["right"] = {}
        self.recordings["left"]  = {}
        
        for channel_pair in self.bipolar_channels:
            
            channel_1   = channel_pair.split("-")[0]
            channel_2   = channel_pair.split("-")[1]
            r_channel_1 = next((self.__dat_r.colnames[i] for i, element in enumerate(self.__dat_r.colnames) if channel_1 in element), None)
            r_channel_2 = next((self.__dat_r.colnames[i] for i, element in enumerate(self.__dat_r.colnames) if channel_2 in element), None)
            l_channel_1 = next((self.__dat_l.colnames[i] for i, element in enumerate(self.__dat_l.colnames) if channel_1 in element), None)
            l_channel_2 = next((self.__dat_l.colnames[i] for i, element in enumerate(self.__dat_l.colnames) if channel_2 in element), None)
            
            try:
                self.recordings["right"][channel_pair] = self.__dat_r.data[:,self.__dat_r.colnames.index(r_channel_1)] - self.__dat_r.data[:,self.__dat_r.colnames.index(r_channel_2)]
            except Exception as error:
                print("... SUB - " + self.__SUB + " : R" +  channel_pair + " channel was not found!")

            try:
                self.recordings["left"][channel_pair] = self.__dat_l.data[:,self.__dat_l.colnames.index(l_channel_1)] - self.__dat_l.data[:,self.__dat_l.colnames.index(l_channel_2)]
            except Exception as error:
                print("... SUB - " + self.__SUB + " : L" +  channel_pair + " channel was not found!")
        
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
              
            for channel in self.bipolar_channels:

                for hemisphere in ["right","left"]:
                        
                    # events aligned based on their onset
                    start_index    = event['event_start_index'] - fs         # 1 sec before event onset
                    finish_index   = start_index + (t_observation * fs)      # t_observation sec later

                    """
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
                    """
                    recording             = self.recordings[hemisphere][channel][start_index:finish_index]

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
        events = pd.DataFrame(columns=["patient", "event_no", "event_category", "event_laterality", "event_start_time", "LFP_hemisphere", "LFP_channel", 
                                       "pre_event_recording", "event_recording", "post_event_recording",
                                       "CDRS_face", "CDRS_neck", "CDRS_trunk", "CDRS_right_leg", "CDRS_left_leg",
                                       "CDRS_right_hand", "CDRS_left_hand", "CDRS_total_right", "CDRS_total_left", "CDRS_total"])
    
        for index, event in dataset.iterrows():

            row                     = {}
            row["patient"]          = self.__SUB
            row["event_category"]   = event['event_category']
            row["event_laterality"] = event['laterality']
            row["event_start_time"] = event['event_start_time']
            row["CDRS_right_hand"]  = event['CDRS_right_hand']
            row["CDRS_left_hand"]   = event['CDRS_left_hand']
            row["CDRS_total_right"] = event['CDRS_total_right']
            row["CDRS_total_left"]  = event['CDRS_total_left']
            row["event_no"]         = event['event_no']
            row["CDRS_face"]        = event['CDRS_face']
            row["CDRS_neck"]        = event['CDRS_neck']
            row["CDRS_trunk"]       = event['CDRS_trunk']
            row["CDRS_right_leg"]   = event['CDRS_right_leg']
            row["CDRS_left_leg"]    = event['CDRS_left_leg']
            row["CDRS_total_hands"] = event['CDRS_total_hands']
            row["CDRS_total"]       = event['CDRS_total']
              
            for channel in self.bipolar_channels: 

                for hemisphere in ["right","left"]:
                        
                    # events aligned based on their onset
                    start_index_pre    = event['event_start_index'] - fs  # pre-event start index: 1 sec before event onset
                    finish_index_pre   = event['event_start_index']       # pre-event finish index: event onset

                    start_index_event  = event['event_start_index']       # event start index
                    finish_index_event = event['event_finish_index']      # event finish index

                    start_index_post   = event['event_finish_index']      # post-event start index: event offset
                    finish_index_post  = event['event_finish_index'] + fs # post-event finish index: 1 sec after event offset

                    # only check the channel where we have LFP recordings 
                    if(channel in self.recordings[hemisphere].keys()):
                        
                        recording_pre               = self.recordings[hemisphere][channel][start_index_pre:finish_index_pre]
                        recording_event             = self.recordings[hemisphere][channel][start_index_event:finish_index_event]
                        recording_post              = self.recordings[hemisphere][channel][start_index_post:finish_index_post]

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
    def select_LFP_recordings_for_hand_activity(dataset, hemisphere="", dyskinesia_severity="all"):

        bilateral_scale = "CDRS_bilateral_hand"
        
        if(hemisphere!=""): # if particular hemisphere is selected

            if(hemisphere=="right"):
                ipsilateral_hand    = "right"
                ipsilateral_scale   = "CDRS_right_hand"
                controlateral_hand  = "left"
                controlateral_scale = "CDRS_left_hand"
                
            elif(hemisphere=="left"):
                ipsilateral_hand    = "left"
                ipsilateral_scale   = "CDRS_left_hand"
                controlateral_hand  = "right"
                controlateral_scale = "CDRS_right_hand"
        
            LFP_ipsilateral_events   = dataset[(dataset.LFP_hemisphere==hemisphere) & (dataset.event_laterality==ipsilateral_hand)]
            LFP_controlateral_events = dataset[(dataset.LFP_hemisphere==hemisphere) & (dataset.event_laterality==controlateral_hand)]
            LFP_bilateral_events     = dataset[(dataset.event_laterality=="bilateral")]

            # in case of particular dyskinesia severity is selected 
            if(dyskinesia_severity!="all"):
                LFP_ipsilateral_events   = LFP_ipsilateral_events[LFP_ipsilateral_events[ipsilateral_scale]==dyskinesia_severity]
                LFP_controlateral_events = LFP_controlateral_events[LFP_controlateral_events[controlateral_scale]==dyskinesia_severity]
                LFP_bilateral_events     = LFP_bilateral_events[LFP_bilateral_events[bilateral_scale]==dyskinesia_severity]

        else: # if a particular hemisphere is not selected
            LFP_ipsilateral_events   = dataset[(dataset.event_laterality == dataset.LFP_hemisphere)]
            LFP_controlateral_events = dataset[(dataset.event_laterality != dataset.LFP_hemisphere)]
            LFP_bilateral_events     = dataset[(dataset.event_laterality =="bilateral")]

            # in case of particular dyskinesia severity is selected 
            if(dyskinesia_severity!="all"):
                LFP_ipsilateral_events   = LFP_ipsilateral_events[((LFP_ipsilateral_events.event_laterality == "right") 
                                                                  & (LFP_ipsilateral_events.CDRS_right_hand == dyskinesia_severity)) | 
                                                                  ((LFP_ipsilateral_events.event_laterality == "left") 
                                                                  & (LFP_ipsilateral_events.CDRS_left_hand == dyskinesia_severity))]
                
                LFP_controlateral_events = LFP_controlateral_events[((LFP_controlateral_events.event_laterality == "right") 
                                                                     & (LFP_controlateral_events.CDRS_left_hand == dyskinesia_severity)) | 
                                                                    ((LFP_controlateral_events.event_laterality == "left") 
                                                                     & (LFP_controlateral_events.CDRS_right_hand == dyskinesia_severity))]
                
                LFP_bilateral_events     = LFP_bilateral_events[LFP_bilateral_events[bilateral_scale]==dyskinesia_severity]
            
        return LFP_ipsilateral_events, LFP_controlateral_events, LFP_bilateral_events

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
    def measure_LFP_power_spectra(dataset, hemisphere, segment, severity):

        PSD_components                  = {} 
        PSD_components["ipsilateral"]   = {} 
        PSD_components["controlateral"] = {} 
        PSD_components["bilateral"]     = {} 
        
        # get the ipsilateral events for the selected hemisphere for the selected hemisphere and dyskinesia severity aspects.
        LFP_ipsilateral, LFP_controlateral, LFP_bilateral = LFP.select_LFP_recordings_for_hand_activity(dataset, hemisphere=hemisphere, dyskinesia_severity=severity)

        # check if there are ipsilateral events
        if(len(LFP_ipsilateral)!=0):
            freq_ipsilateral, mean_ipsilateral, error_ipsilateral = LFP.extract_average_psd_for_LFP(LFP_ipsilateral, segment, error_bar="se")
        else:
            freq_ipsilateral = []; mean_ipsilateral = []; error_ipsilateral = []
            
        PSD_components["ipsilateral"]["frequency"] = freq_ipsilateral
        PSD_components["ipsilateral"]["mean_psd"]  = mean_ipsilateral
        PSD_components["ipsilateral"]["error_psd"] = error_ipsilateral
        
        # check if there are controlateral events
        if(len(LFP_controlateral)!=0):
            freq_controlateral, mean_controlateral, error_controlateral = LFP.extract_average_psd_for_LFP(LFP_controlateral, segment, error_bar="se")
        else:
            freq_controlateral = []; mean_controlateral = []; error_controlateral = []

        PSD_components["controlateral"]["frequency"] = freq_controlateral
        PSD_components["controlateral"]["mean_psd"]  = mean_controlateral
        PSD_components["controlateral"]["error_psd"] = error_controlateral

        # check if there are bilateral events
        if(len(LFP_bilateral)!=0):
            freq_bilateral, mean_bilateral, error_bilateral = LFP.extract_average_psd_for_LFP(LFP_bilateral, segment, error_bar="se")
        else:
            freq_bilateral = []; mean_bilateral = []; error_bilateral = []
            
        PSD_components["bilateral"]["frequency"] = freq_bilateral
        PSD_components["bilateral"]["mean_psd"]  = mean_bilateral
        PSD_components["bilateral"]["error_psd"] = error_bilateral
    
        return PSD_components