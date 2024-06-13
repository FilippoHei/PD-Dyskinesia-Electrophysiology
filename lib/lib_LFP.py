import pingouin as pg
import pandas as pd
import numpy as np
import scikit_posthocs as sp

from lib_data import DATA_IO

class LFP:
    
    def __init__(self, PATH, SUB, DAT_SOURCE):

        assert DAT_SOURCE in ['lfp_left', 'lfp_right'], f'Please pass LFP DAT_SOURCE ({DAT_SOURCE})'
        
        self.__PATH         = PATH
        self.__SUB          = SUB
        self.__DAT_SOURCE   = DAT_SOURCE
        
        # use DATA_IO to load data structure
        data_IO             = DATA_IO(PATH, SUB, DAT_SOURCE)
        self.__dat          = data_IO.get_data()

        self.fs             = self.__dat.fs
        self.times          = self.__dat.times

        if(DAT_SOURCE == 'lfp_left'): #right hemisphere LFP recordings
            self.hemisphere     = "left" 
            self.LFP_channel_32 = self.__dat.data[:,self.__dat.colnames.index('LFP_L_03')] - self.__dat.data[:,self.__dat.colnames.index('LFP_L_02')]
            self.LFP_channel_43 = self.__dat.data[:,self.__dat.colnames.index('LFP_L_04')] - self.__dat.data[:,self.__dat.colnames.index('LFP_L_03')]
            self.LFP_channel_54 = self.__dat.data[:,self.__dat.colnames.index('LFP_L_05')] - self.__dat.data[:,self.__dat.colnames.index('LFP_L_03')]
            self.LFP_channel_65 = self.__dat.data[:,self.__dat.colnames.index('LFP_L_06')] - self.__dat.data[:,self.__dat.colnames.index('LFP_L_05')]
            self.LFP_channel_76 = self.__dat.data[:,self.__dat.colnames.index('LFP_L_07')] - self.__dat.data[:,self.__dat.colnames.index('LFP_L_06')]
            self.LFP_channel_87 = self.__dat.data[:,self.__dat.colnames.index('LFP_L_08')] - self.__dat.data[:,self.__dat.colnames.index('LFP_L_07')]
        else:
            self.hemisphere     = "right" 
            self.LFP_channel_32 = self.__dat.data[:,self.__dat.colnames.index('LFP_R_03')] - self.__dat.data[:,self.__dat.colnames.index('LFP_R_02')]
            self.LFP_channel_43 = self.__dat.data[:,self.__dat.colnames.index('LFP_R_04')] - self.__dat.data[:,self.__dat.colnames.index('LFP_R_03')]
            self.LFP_channel_54 = self.__dat.data[:,self.__dat.colnames.index('LFP_R_05')] - self.__dat.data[:,self.__dat.colnames.index('LFP_R_03')]
            self.LFP_channel_65 = self.__dat.data[:,self.__dat.colnames.index('LFP_R_06')] - self.__dat.data[:,self.__dat.colnames.index('LFP_R_05')]
            self.LFP_channel_76 = self.__dat.data[:,self.__dat.colnames.index('LFP_R_07')] - self.__dat.data[:,self.__dat.colnames.index('LFP_R_06')]
            self.LFP_channel_87 = self.__dat.data[:,self.__dat.colnames.index('LFP_R_08')] - self.__dat.data[:,self.__dat.colnames.index('LFP_R_07')]


    def extract_LFP_segments(self, event_dataset, hemisphere="", event_type="", event_category="", dyskinesia_score="", alignment="onset", t_observation=4):
    
        fs             = self.fs
            
        if(hemisphere!=""):
            # check if the selected hemisphere is valid
            assert hemisphere in ["right", "left", "bilateral"], f'Please choose hemisphere as "right", "left", "bilateral"'
                
        if(event_type!=""):
            # check if the selected event type is valid
            assert event_category in event_dataset.event.unique().tolist(), f'Please enter valid event type as "move", "tap"'
    
        if(event_category!=""): 
            # check if the event category is valid
            assert event_category in event_dataset.event_category.unique().tolist(), f'Please enter valid event category, not ({event_category})'
                
        # check if event alignment strategy is valid
        assert alignment in ["onset", "offset"], f'Please choose alignment as "onset", "offset"'
           
        #################################################################################################################################
        dataset = event_dataset[event_dataset['hemisphere']==hemisphere] if hemisphere != "" else event_dataset # select hemisphere
        dataset = dataset[dataset['event']==event_type] if event_type != "" else dataset                        # select event type
        dataset = dataset[dataset['event_category']==event_category] if event_category != "" else dataset       # select event category
        dataset = dataset[dataset['dyskinesia_score']==dyskinesia_score] if dyskinesia_score != "" else dataset # select dyskinesia score
        #################################################################################################################################
            
        # create empty arrays for storing LFP data for selected event category
        LFP_32 = []
        LFP_43 = []
        LFP_54 = []
        LFP_65 = []
        LFP_76 = []
        LFP_87 = []
        
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
                
            LFP_data_32 = self.LFP_channel_32[start_index:finish_index].tolist()
            LFP_data_43 = self.LFP_channel_43[start_index:finish_index].tolist()
            LFP_data_54 = self.LFP_channel_54[start_index:finish_index].tolist()
            LFP_data_65 = self.LFP_channel_65[start_index:finish_index].tolist()
            LFP_data_76 = self.LFP_channel_76[start_index:finish_index].tolist()
            LFP_data_87 = self.LFP_channel_87[start_index:finish_index].tolist()

                
            LFP_32.append(LFP_data_32)
            LFP_43.append(LFP_data_43)
            LFP_54.append(LFP_data_54)
            LFP_65.append(LFP_data_65)
            LFP_76.append(LFP_data_76)
            LFP_87.append(LFP_data_87)
    
        # store selected events in 3 axis of LFP data into a dictionary file
        LFP_events               = {}
        LFP_events["channel_32"] = LFP_32
        LFP_events["channel_43"] = LFP_43
        LFP_events["channel_54"] = LFP_54
        LFP_events["channel_65"] = LFP_65
        LFP_events["channel_76"] = LFP_76
        LFP_events["channel_87"] = LFP_87
            
        return LFP_events
