import os
import pandas as pd
import numpy as np
import sys
import json
from utils.utils_fileManagement import load_class_pickle, mergedData

from lib_data import DATA_IO
import utils_accelerometer

class EVENTS:

    event_severity             = {}
    event_severity["none"]     = 0
    event_severity["mild"]     = 1
    event_severity["moderate"] = 2
    event_severity["severe"]   = 3
    event_severity["extreme"]  = 4
    
    def __init__(self, PATH, SUB, DAT_SOURCE):

        # setting environmental variables
        self.__PATH         = PATH
        self.__SUB          = SUB
        self.__DAT_SOURCE   = DAT_SOURCE
        
        print("EVENT HISTORY: SUB-" + SUB)
        print("... loading started")

        # use DATA_IO to load data structure
        data_IO             = DATA_IO(PATH, SUB, DAT_SOURCE)
        self.__dat          = data_IO.get_data()

        # populate class fields
        self.fs             = self.__dat.fs
        self.times          = self.__dat.data[:,self.__dat.colnames.index('dopa_time')]
        self.task           = self.__dat.data[:,self.__dat.colnames.index('task')]
        self.left_tap       = self.__dat.data[:,self.__dat.colnames.index('left_tap')]
        self.right_tap      = self.__dat.data[:,self.__dat.colnames.index('right_tap')]
        self.left_move      = self.__dat.data[:,self.__dat.colnames.index('left_move')]
        self.right_move     = self.__dat.data[:,self.__dat.colnames.index('right_move')]
        self.no_move        = self.__dat.data[:,self.__dat.colnames.index('no_move')]
        self.period_rest    = (self.task == "rest").astype(int)
        self.period_free    = (self.task == "free").astype(int)
        self.period_tap     = (self.task == "tap").astype(int)
        
        print("... task periods were defined")
        self.__define_events()
        print("... events were categorized")
        self.get_dyskinesia_scores()
        print("... dyskinesia evaluation was collected")
        print("... event loading completed")
        print("--------------------------------------------------------------------")

    def __operator_event_difference(self, array_A, array_B):
        """
        Description
            This method finds events that occurred in array_A but not in array_B. The two arrays were expected to have the same length

        Input
            array_A: A binary list represents the existence of event=1 and absence=0.
            array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return [1 if event_A == 1 and event_B == 0 else 0 for event_A, event_B in zip(array_A, array_B)]
    
    def __operator_event_intersection(self, array_A, array_B):
        """
        Description
            This method finds events that occurred both in array_A and array_B (and operator). The two arrays were expected to have the same length

        Input
            array_A: A binary list represents the existence of event=1 and absence=0.
            array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return ([a & b for a, b in zip(array_A, array_B)])
    
    def __operator_event_union(self, array_A, array_B):
        """
        Description
            This method finds events in array_A or array_B (or operator). The two arrays were expected to have the same length

        Input
            array_A: A binary list represents the existence of event=1 and absence=0.
            array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return ([a | b for a, b in zip(array_A, array_B)])

    def __define_events(self):
        """
        Description
            This method defines the types of events that were detected during the recording session. The definition of each event is defined based on the following criteria:
            VOLUNTARY TAPPING: the tapping event observed in tap field + not observed in the move field + observed during the tapping period
            
        Output
            The definitions of events are added as a class field that can be accessible.
        """
        self.left_voluntary_movements    = self.__operator_event_intersection(self.__operator_event_difference(self.left_tap, self.left_move), self.period_tap)
        self.left_involuntary_movements  = self.__operator_event_intersection(self.__operator_event_difference(self.left_move, self.left_tap), self.period_rest)
        self.right_voluntary_movements   = self.__operator_event_intersection(self.__operator_event_difference(self.right_tap, self.right_move), self.period_tap)
        self.right_involuntary_movements = self.__operator_event_intersection(self.__operator_event_difference(self.right_move, self.right_tap), self.period_rest)

        # remove possible bilateral voluntary movements
        self.left_voluntary_movements    = self.__operator_event_difference(self.left_voluntary_movements, self.right_voluntary_movements)
        self.right_voluntary_movements   = self.__operator_event_difference(self.right_voluntary_movements, self.left_voluntary_movements)


    def __get_CDRS_evaluation_intervals(self, body_part):

        """
        Description
            The method extracts the indexes of dyskinesia evaluations and corresponding score for the selected body part.
        """
        times           = self.CDRS_dataframe.dopa_time.to_list()
        scores          = self.CDRS_dataframe[body_part].to_list()
        
        previous_score  = scores[0]
        previous_time   = times[0]
        intervals_time  = []
        intervals_score = []

        intervals_time.append((np.min(self.times)/60, times[0]))
        intervals_score.append(scores[0])
        
        for i in range(len(scores)):
            if(scores[i] != previous_score):
                intervals_time.append((previous_time, (times[i] + times[i-1])/2))
                intervals_score.append(previous_score)
                previous_score = scores[i]
                previous_time  = (times[i] + times[i-1])/2
        
        intervals_time.append((previous_time, times[-1]))
        intervals_score.append(scores[-1])
    
        # recordings length and last evaluation time don't always match. After the last CDRS evaluation, if the recording continues, 
        # The last evaluation will be kept until the end of the recording period.
        intervals_time.append((times[-1], np.max(self.times)/60))
        intervals_score.append(scores[-1])
        
        return intervals_time, intervals_score
    
    def get_dyskinesia_scores(self):
        """
        Description
            This method reads the Excel file containing the CDRS scores (right, left, total) of dyskinesia events and their corresponding timestamps included in different 
            sheets named with "sub-xxx" notation. This CDRS file is expected to be located under the "PATH\data" directory with the name CDRS.xlsx. The sheet of this Excel 
            belonging to the selected patients will be saved into a dataframe structure. To get the timestamp of dyskinesia scores (the same length as self.times field), 
            we first get the registration time of dyskinesia evaluation and corresponding dyskinesia score and fill an empty array with this score until the next 
            evaluation is made.
            
        Output
            The definitions of dyskinesia scores in the right, left, and bilateral side were added as a field. It also returns a Python dictionary with three fields:
            - key: "CDRS_right", value: an integer array
            - key: "CDRS_left", value: an integer array
            - key: "CDRS_total", value: an integer array
        """

        # check if the CDRS scores exist in the directory
        PATH_CDRS = self.__PATH +"\\data\\"
        assert os.path.exists(PATH_CDRS +"\\CDRS.xlsx"), f'CDRS.xlsx does not exist in directory: ({PATH_CDRS}) '

        # read the Excell sheet for the given SUB into the dataframe
        CDRS = pd.read_excel(PATH_CDRS +"CDRS.xlsx", sheet_name="sub-"+self.__SUB)

        CDRS = CDRS[['dopa_time','CDRS_face','CDRS_neck','CDRS_trunk',
                     'CDRS_upper_right','CDRS_upper_left','CDRS_lower_right','CDRS_lower_left',
                     'CDRS_total_right', 'CDRS_total_left', 'CDRS_total']]
        
        CDRS.dropna(inplace=True)
        self.CDRS_dataframe = CDRS

        self.CDRS_right_hand_indexes, self.CDRS_right_hand_scores  = self.__get_CDRS_evaluation_intervals("CDRS_upper_right")
        self.CDRS_left_hand_indexes, self.CDRS_left_hand_scores    = self.__get_CDRS_evaluation_intervals("CDRS_upper_left")
        self.CDRS_total_indexes, self.CDRS_total_scores            = self.__get_CDRS_evaluation_intervals("CDRS_total")



    def __get_event_indices(self, array):
        """
        Description
            This method finds the indices of the beginning and end of the event in the array. Basically, one of the events (move or tapping) array will be provided as a parameter
            to the function.

        Input
            array: A binary list represents the existence=1 and absence=0 of a particular event (move/tapping).

        Output
            event_indices: A list containing tuples (start_index, finish_index) of index information for events
        """
        event_indices     = []
        event_started     = False
        event_start_index = None
        
        for i, num in enumerate(array):
            if num == 1:
                if not event_started:
                    event_start_index = i
                    event_started = True
            elif event_started:
                event_indices.append((event_start_index, i - 1))
                event_started = False
        
        # If an event is ongoing at the end of the array
        if event_started:
            event_indices.append((event_start_index, len(array) - 1))
        
        return event_indices

    def __populate_dataframe(self, dataset, patient, laterality, event_indices, event_category):

        """
        Description
            This method adds a dataframe that contains all events detected for the given side of the patients. Initially, it acquires all the events contained 
            in _tap and _move fields. The information regarding the task (tapping, free, rest), the category (voluntary tapping, involuntary tapping, involuntary movement),
            event start and finish timestamps, etc are stored in this dataframe.

        Input
            dataset: A dataframe contains "patient","laterality","event_type", "event_no", "event_start_index", "event_finish_index", "event_start_time", 
                            "event_finish_time", "duration", "task" columns. It can be empty or filled previously.
            patient: A string representing the patient code
            laterality: A string representing the laterality of the limb (right/left/bilateral) information
            event_indices: A list containing tuples (start_index, finish_index) of index information for events
            event_type: A string represents the type of event that is considered
        
        Output
            dataset: The more populated version of the given dataframe structure
        """
        
        counter    = 1
        
        for event in event_indices:
            event_start_i = event[0]
            event_end_i   = event[1]

            if(event_start_i!=event_end_i):
    
                # add event to the dataset
                dataset.loc[len(dataset)] = {"patient"            : patient, 
                                             "laterality"         : laterality, 
                                             "event_no": "p_" + patient + "_" + laterality + "_" + event_category + str(counter),
                                             "event_category"     : event_category,       
                                             "event_start_index"  : event_start_i, 
                                             "event_finish_index" : event_end_i, 
                                             "event_start_time"   : self.times[event_start_i]/60, 
                                             "event_finish_time"  : self.times[event_end_i]/60, 
                                             "duration"           : self.times[event_end_i] - self.times[event_start_i]}
                counter += 1

    def __get_event_CDRS_score(self, events, scores, event_start_time):
        for i in range(len(events)):
            start, end = events[i]
            if start <= event_start_time < end:
                return scores[i]

    def __get_dyskinesia_label_arm_strategy(self, arm_score, total_score):
        # old one
        if(total_score==0):
            return "none"
        else:
            if(arm_score==1):
                return "mild"
            elif(arm_score>=2):
                return "moderate"

    def __get_dyskinesia_label_total_strategy(self, arm_score, total_score):
        if(total_score==0):
            return "none"
        else:
            if(arm_score>0):
                if(total_score<=4):
                    return "mild"
                elif(total_score>4) :
                    return "moderate"
               
    def get_event_dataframe(self, tap_indices):

        """
        Description
            This method populates an event dataframe for a given patient, laterality, and event type.

        Input
            dataset: A dataframe contains "patient","laterality","event_type", "event_no", "event_start_index", "event_finish_index", "event_start_time", 
                                                 "event_finish_time", "duration", "task" columns
            patient: A string representing the patient code
            laterality: A string representing the laterality information
            event_indices: A list containing tuples (start_index, finish_index) of index information for events
            event_type: A string represents the type of event that is considered
        
        Output
            dataset: The more populated version of the given dataframe structure
        """
        # get start and finish indices of different event types in different hands
        right_tap_indices = tap_indices["right"]
        left_tap_indices  = tap_indices["left"]
        
        # create an empty event dataframe
        events = pd.DataFrame(columns=["patient", "laterality", "event_no", "event_category", "event_start_index", 
                                       "event_finish_index", "event_start_time", "event_finish_time", "duration", 
                                       "CDRS_right_hand", "CDRS_left_hand", "CDRS_total",
                                       "dyskinesia_arm", "dyskinesia_total"])   

        # populate the empty dataframe
        self.__populate_dataframe(events, self.__SUB, "right", right_tap_indices, "tapping")
        self.__populate_dataframe(events, self.__SUB, "left" , left_tap_indices , "tapping")					

        
        # for each index check the laterality and get the corresponding dyskinesia score on the event onset
        for index in events.index:
            events.loc[index, ['CDRS_right_hand']] = self.__get_event_CDRS_score(self.CDRS_right_hand_indexes, self.CDRS_right_hand_scores, events.iloc[index].event_start_time)
            events.loc[index, ['CDRS_left_hand']]  = self.__get_event_CDRS_score(self.CDRS_left_hand_indexes, self.CDRS_left_hand_scores, events.iloc[index].event_start_time)
            events.loc[index, ['CDRS_total']]      = self.__get_event_CDRS_score(self.CDRS_total_indexes, self.CDRS_total_scores, events.iloc[index].event_start_time)

            if(events.loc[index, 'laterality']=="right"):
                events.loc[index, ['dyskinesia_arm']] = self.__get_dyskinesia_label_arm_strategy(events.loc[index, 'CDRS_right_hand'], events.loc[index, 'CDRS_total'])
                events.loc[index, ['dyskinesia_total']] = self.__get_dyskinesia_label_total_strategy(events.loc[index, 'CDRS_right_hand'], events.loc[index, 'CDRS_total'])
            else:
                events.loc[index, ['dyskinesia_arm']] = self.__get_dyskinesia_label_arm_strategy(events.loc[index, 'CDRS_left_hand'], events.loc[index, 'CDRS_total'])
                events.loc[index, ['dyskinesia_total']] = self.__get_dyskinesia_label_total_strategy(events.loc[index, 'CDRS_left_hand'], events.loc[index, 'CDRS_total'])
                 
        return events
    

        




    