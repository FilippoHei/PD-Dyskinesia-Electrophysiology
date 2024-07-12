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

class ECoG:
    
    def __init__(self, PATH, SUB):

        print("ECoG Recording: SUB-" + SUB)
        
        self.__PATH = PATH
        self.__SUB  = SUB

        # use DATA_IO to load data structure
        try:
            data_IO = DATA_IO(PATH, SUB, 'ecog_right')
        except: 
            print("---> left hemisphere ECoG recording found...")
            self.hemisphere = "left"

        try:
            data_IO = DATA_IO(PATH, SUB, 'ecog_left')
        except: 
            print("---> right hemisphere ECoG recording found...")
            self.hemisphere = "right"
            
        self.__dat            = data_IO.get_data()

        # populate class fields
        self.fs               = self.__dat.fs
        self.times            = self.__dat.times

        # rereferencing for channels 
        self.bipolar_channels = ["2-1", "3-2", "4-3", "5-4", "6-5", "7-6", "8-7", "9-8", "10-9", "11-10", "12-11"]

        # load right LFP data
        self.recordings       = {}
        
        for channel_pair in self.bipolar_channels:
            
            channel_1      = channel_pair.split("-")[0]
            channel_2      = channel_pair.split("-")[1]
            channel_1_name = next((self.__dat.colnames[i] for i, element in enumerate(self.__dat.colnames) if channel_1 in element), None)
            channel_2_name = next((self.__dat.colnames[i] for i, element in enumerate(self.__dat.colnames) if channel_2 in element), None)
            
            try:
                self.recordings[channel_pair] = self.__dat.data[:,self.__dat.colnames.index(channel_1_name)] - self.__dat.data[:,self.__dat.colnames.index(channel_2_name)]
            except Exception as error:
                print("... SUB - " + self.__SUB + " : " +  channel_pair + " channel was not found!")
        
    def extract_ECoG_events_segments(self, dataset):
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
        alignment = "onset" # ECoG activity is only aligned to event onsets

        # create empty dataframe to populate
        events = pd.DataFrame(columns=["event_no", "event_category", "event_laterality", "event_start_time", "ECoG_hemisphere", "ECoG_channel", 
                                       "pre_event_recording", "event_recording", "post_event_recording",  
                                       "CDRS_face", "CDRS_neck", "CDRS_trunk", "CDRS_right_leg", "CDRS_left_leg",
                                       "CDRS_right_hand", "CDRS_left_hand", "CDRS_total_right", "CDRS_total_left", "CDRS_total"])
    
        for index, event in dataset.iterrows():

            row                     = {}
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
      
                # events aligned based on their onset
                start_index_pre    = event['event_start_index'] - fs  # pre-event start index: 1 sec before event onset
                finish_index_pre   = event['event_start_index']       # pre-event finish index: event onset

                start_index_event  = event['event_start_index']       # event start index
                finish_index_event = event['event_finish_index']      # event finish index

                start_index_post   = event['event_finish_index']      # post-event start index: event offset
                finish_index_post  = event['event_finish_index'] + fs # post-event finish index: 1 sec after event offset

                # only check the channel where we have LFP recordings 
                if(channel in self.recordings.keys()):
                        
                    recording_pre               = self.recordings[channel][start_index_pre:finish_index_pre]
                    recording_event             = self.recordings[channel][start_index_event:finish_index_event]
                    recording_post              = self.recordings[channel][start_index_post:finish_index_post]

                    row["ECoG_hemisphere"]      = self.hemisphere
                    row["ECoG_channel"]         = channel
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
