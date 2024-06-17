import pingouin as pg
import pandas as pd
import numpy as np
import scikit_posthocs as sp

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
        
        fs     = self.fs
        #events = pd.DataFrame(columns=["event_no", "event_category", "event_laterality", "LFP_hemisphere", "channel", "recording", 
        #                               "alignment", 'CDRS_face', 'CDRS_neck', 'CDRS_trunk', 'CDRS_right_hand', 'CDRS_right_leg', 'CDRS_left_hand',
        #                               'CDRS_left_leg', 'CDRS_total_hands','CDRS_total_right', 'CDRS_total_left', 'CDRS_total'])

        events = pd.DataFrame(columns=["event_category", "event_laterality", "LFP_hemisphere", "channel", "recording", 
                                       "alignment", 'CDRS_right_hand', 'CDRS_left_hand'])
    
        for index, event in dataset.iterrows():
            
            row                     = {}
            #row["event_no"]         = event['event_no']
            row["event_category"]   = event['event_category']
            row["event_laterality"] = event['laterality']
            #row["CDRS_face"]        = event['CDRS_face']
            #row["CDRS_neck"]        = event['CDRS_neck']
            #row["CDRS_trunk"]       = event['CDRS_trunk']
            row["CDRS_right_hand"]  = event['CDRS_right_hand']
            #row["CDRS_right_leg"]   = event['CDRS_right_leg']
            row["CDRS_left_hand"]   = event['CDRS_left_hand']
            #row["CDRS_left_leg"]    = event['CDRS_left_leg']
            #row["CDRS_total_hands"] = event['CDRS_total_hands']
            #row["CDRS_total_right"] = event['CDRS_total_right']
            #row["CDRS_total_left"]  = event['CDRS_total_left']
            #row["CDRS_total"]       = event['CDRS_total']
              
            for channel in ["32","43","54","65","76","87"]:

                for hemisphere in ["right","left"]:

                    for alignment in ["onset"]:
                    #for alignment in ["onset","offset"]:
                        
                        # If events aligned based on their onset
                        if(alignment=="onset"):
                            start_index    = event['event_start_index'] - fs         # 1 sec before event onset
                            finish_index   = start_index + (t_observation * fs)      # t_observation sec later
                                        
                        # If events aligned based on their offset
                        else:
                            finish_index   = event['event_finish_index'] + fs        # 1 sec after event offset
                            start_index    = finish_index - (t_observation * fs)     # t_observation sec before

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
                        row["channel"]        = channel
                        row["recording"]      = recording.tolist()
                        row["alignment"]      = alignment
                
                        events.loc[len(events)] = row
                    
        return events
