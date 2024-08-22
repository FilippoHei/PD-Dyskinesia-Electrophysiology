import pingouin as pg
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import sys
from scipy import signal

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

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
        self.__contacts       = ["01", "02", "03", "04", "05", "06"]
        
        if(SUB=="016"):
            self.bipolar_channels = ["04-02", "05-04", "06-05"] #for patient 16
        else:
            self.bipolar_channels = ["02-01", "03-02", "04-03", "05-04", "06-05", "07-06", "08-07", "09-08", "10-09", "11-10", "12-11"]

        # load right LFP data
        self.recordings                  = {}
        self.recordings[self.hemisphere] = {}
        self.__get_channel_recordings()

    def __get_channel_recordings(self):
        
        for channel_pair in self.bipolar_channels:
            
            channel_1      = channel_pair.split("-")[0]
            channel_2      = channel_pair.split("-")[1]
            channel_1_name = next((self.__dat.colnames[i] for i, element in enumerate(self.__dat.colnames) if channel_1 in element), None)
            channel_2_name = next((self.__dat.colnames[i] for i, element in enumerate(self.__dat.colnames) if channel_2 in element), None)
            
            try:
                self.recordings[self.hemisphere][channel_pair] = self.__dat.data[:,self.__dat.colnames.index(channel_1_name)] - self.__dat.data[:,self.__dat.colnames.index(channel_2_name)]
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
        
        fs     = self.fs

        # create empty dataframe to populate
        events = pd.DataFrame(columns=["patient", "event_no", "event_category", "event_laterality", "event_start_time", "duration",
                                       "ECoG_hemisphere", "ECoG_channel", "pre_event_recording", "event_recording", "post_event_recording",
                                       "CDRS_right_hand", "CDRS_left_hand", "CDRS_total", 'dyskinesia_arm', 'dyskinesia_total'])
    
        for index, event in dataset.iterrows():

            row                     = {}
            row["patient"]          = self.__SUB
            row["event_no"]         = event['event_no']
            row["event_category"]   = event['event_category']
            row["event_laterality"] = event['laterality']
            row["event_start_time"] = event['event_start_time']
            row["duration"]         = event['duration']
            row["CDRS_right_hand"]  = event['CDRS_right_hand']
            row["CDRS_left_hand"]   = event['CDRS_left_hand']
            row["CDRS_total"]       = event['CDRS_total']
            row["dyskinesia_arm"]   = event['dyskinesia_arm']
            row["dyskinesia_total"] = event['dyskinesia_total']
              
            for channel in self.bipolar_channels: 
      
                # events aligned based on their onset
                start_index_pre    = event['event_start_index'] - fs * 2  # pre-event start index: 2 sec before event onset
                finish_index_pre   = event['event_start_index']           # pre-event finish index: event onset

                start_index_event  = event['event_start_index']           # event start index
                finish_index_event = event['event_finish_index']          # event finish index

                start_index_post   = event['event_finish_index']          # post-event start index: event offset
                finish_index_post  = event['event_finish_index'] + fs * 2 # post-event finish index: 2 sec after event offset

                # only check the channel where we have ECoG recordings 
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

    def get_baseline_recording(self, t_min=0, t_max=5):

        # create empty dictionary
        baseline_recordings             = {}
        baseline_recordings[self.__SUB] = {}
    
        # get baseline time array from t_min to t_max minutes
        baseline_t      = (self.times/60>=t_min) & (self.times/60<=t_max)

        # iterate between hemispheres
        for hemisphere in ["right", "left"]:
            
            baseline_recordings[self.__SUB][hemisphere] = {}

            # iterate between LFP channels of selected hemisphere
            for channel in self.bipolar_channels:

                try:
                    # get baseline recording for the selected hemisphere and channel and add to dictionary
                    baseline_recordings[self.__SUB][hemisphere][channel] = self.recordings[hemisphere][channel][baseline_t].astype(float)
                except:
                    print("... SUB - " + self.__SUB + " : " + hemisphere + "_" + channel + " channel was not found!")

        return baseline_recordings

    @staticmethod
    def define_onset_aligned_recordings(dataset, fs, pad=False):

        # if pre_event_recording contains nan values, it will change the onset position of the event. 
        # So remove all the rows where pre_event_recording contains np.nan
        dataset = dataset[~dataset['pre_event_recording'].apply(lambda x: any(pd.isna(i) for i in x))]
    
        event_recording_onset_aligned = []
        
        for index, row in dataset.iterrows():
            
            rec = []
            rec.extend(row.pre_event_recording.copy())
        
            if(len(row.event_recording) < fs*2):
                rec.extend(row.event_recording.copy())
                if(pad==True):
                    rec = np.pad(rec, (0, max(0, fs*4 - len(rec))), mode='constant', constant_values=np.nan)
            else:
                rec.extend(row.event_recording[0:fs*2].copy())
        
            event_recording_onset_aligned.append(rec)
            
        dataset["event_recording_onset_alingned"] = event_recording_onset_aligned
    
        return dataset
        
    @staticmethod
    def get_patient_events(dataset, SUB, event_mode):
    
        dataset_patient       = dataset[dataset.patient == SUB]                                                        # select patient
    
        if(event_mode == "controlateral"): # oly controlateral events
            dataset_patient       = dataset_patient[dataset_patient.event_laterality != dataset_patient.ECoG_hemisphere]    
        elif(event_mode == "ipsilateral"): # oly ipsilateral events
            dataset_patient       = dataset_patient[dataset_patient.event_laterality == dataset_patient.ECoG_hemisphere] 
    
        # if the laterality (ipsi vs contro) of event doesnt matter, do nothing
        
        dataset_patient       = dataset_patient[dataset_patient.event_start_time >= 5]                                 # tappings after 5 minutes
        dataset_patient_noLID = dataset_patient[dataset_patient.dyskinesia_arm == "none"]                              # noLID events
        dataset_patient_LID   = dataset_patient[dataset_patient.dyskinesia_arm != "none"]                              # events
        
        dataset_patient_noLID.reset_index(drop=True, inplace=True)
        dataset_patient_LID.reset_index(drop=True, inplace=True)
        
        return dataset_patient, dataset_patient_noLID, dataset_patient_LID