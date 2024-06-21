import os
import pandas as pd
import numpy as np
import sys
import json
from utils.utils_fileManagement import load_class_pickle, mergedData

from lib_data import DATA_IO
import utils_accelerometer

class EVENTS:
        
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
        self.bilateral_move = np.array([a & b for a, b in zip(self.left_move.astype(int).tolist(), self.right_move.astype(int).tolist())])
        self.bilateral_tap  = np.array([a & b for a, b in zip(self.left_tap.astype(int).tolist(), self.right_tap.astype(int).tolist())])
        self.no_move        = self.__dat.data[:,self.__dat.colnames.index('no_move')]
        self.period_rest    = (self.task == "rest").astype(int)
        self.period_free    = (self.task == "free").astype(int)
        self.period_tap     = (self.task == "tap").astype(int)
        
        print("... task periods were defined")
        self.__define_events()
        print("... events were categorized")
        self.get_dyskinesia_scores()
        print("... dyskinesia evaluation was collected")
        print("... loading completed")

    def __operator_event_difference(self, array_A, array_B):
        """
        Description
            This method finds events that occurred in array_A but not in array_B. The two arrays were expected to have the same length

        Input
            array_A: A binary list represents the existence of event=1 and absence=0.
            array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            return: A binary list with the same length as provided arrays.
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
            return: A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return ([a & b for a, b in zip(array_A, array_B)])
    
    def __operator_event_union(self, array_A, array_B):
        """
        Description
            This method finds events that occurred either in array_A or array_B (or operator). The two arrays were expected to have the same length

        Input
            array_A: A binary list represents the existence of event=1 and absence=0.
            array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            return: A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return ([a | b for a, b in zip(array_A, array_B)])

    def __define_events(self):
        """
        Description
            This method defines the types of events that were detected during the recording session. The definition of each event is defined based on the following criteria:
            VOLUNTARY TAPPING: the tapping event observed in tap field + not observed in the move field + observed during the tapping period
            
        Output
            return: The definitions of events are added as a class field that can be accessible.
        """
        self.left_voluntary_tap          = self.__operator_event_intersection(self.__operator_event_difference(self.left_tap, self.left_move), self.period_tap)
        self.left_involuntary_movements  = self.__operator_event_difference(self.left_move, self.left_tap)
        
        self.right_voluntary_tap         = self.__operator_event_intersection(self.__operator_event_difference(self.right_tap, self.right_move), self.period_tap)
        self.right_involuntary_movements = self.__operator_event_difference(self.right_move, self.right_tap)


    def get_dyskinesia_scores(self):
        """
        Description
            This method reads the Excel file containing the CDRS scores (right, left, total) of dyskinesia events and their corresponding timestamps included in different 
            sheets named with "sub-xxx" notation. This CDRS file is expected to be located under the "PATH\data" directory with the name CDRS.xlsx. The sheet of this Excel 
            belonging to the selected patients will be saved into a dataframe structure. To get the timestamp of dyskinesia scores (the same length as self.times field), 
            we first get the registration time of dyskinesia evaluation and corresponding dyskinesia score and fill an empty array with this score until the next 
            evaluation is made.
            
        Output
            return: The definitions of dyskinesia scores in the right, left, and bilateral side were added as a field. It also returns a 
                     Python dictionary with three fields:
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

        self.__CDRS_dataframe = CDRS

        
        t_CDRS      = CDRS.dopa_time.to_numpy()             # get the timestamp of evaluation time (an array in minutes) 
        
        # Find the indices where t_CDRS <= t
        # self.times / 60: to represent recording times in minutes instead of seconds
        CDRS_times       = np.searchsorted(CDRS.dopa_time, self.times / 60) # find the indices of evaluation times in the time array
        CDRS_times       = CDRS_times-1
        
        CDRS_face        = CDRS_times.copy()
        CDRS_neck        = CDRS_times.copy()
        CDRS_trunk       = CDRS_times.copy()
        CDRS_upper_right = CDRS_times.copy()
        CDRS_upper_left  = CDRS_times.copy()
        CDRS_lower_right = CDRS_times.copy()
        CDRS_lower_left  = CDRS_times.copy()
        CDRS_total_right = CDRS_times.copy()
        CDRS_total_left  = CDRS_times.copy()
        CDRS_total       = CDRS_times.copy()

        for index in CDRS.index:
            CDRS_face[CDRS_face==index]               = CDRS.iloc[index].CDRS_face
            CDRS_neck[CDRS_neck==index]               = CDRS.iloc[index].CDRS_neck
            CDRS_trunk[CDRS_trunk==index]             = CDRS.iloc[index].CDRS_trunk
            CDRS_upper_right[CDRS_upper_right==index] = CDRS.iloc[index].CDRS_upper_right
            CDRS_upper_left[CDRS_upper_left==index]   = CDRS.iloc[index].CDRS_upper_left
            CDRS_lower_right[CDRS_lower_right==index] = CDRS.iloc[index].CDRS_lower_right
            CDRS_lower_left[CDRS_lower_left==index]   = CDRS.iloc[index].CDRS_lower_left
            CDRS_total_right[CDRS_total_right==index] = CDRS.iloc[index].CDRS_total_right
            CDRS_total_left[CDRS_total_left==index]   = CDRS.iloc[index].CDRS_total_left
            CDRS_total[CDRS_total==index]             = CDRS.iloc[index].CDRS_total
            
        # Take values at found indices
        self.CDRS_face        = CDRS_face
        self.CDRS_neck        = CDRS_neck
        self.CDRS_trunk       = CDRS_trunk
        self.CDRS_upper_right = CDRS_upper_right 
        self.CDRS_upper_left  = CDRS_upper_left 
        self.CDRS_lower_right = CDRS_lower_right 
        self.CDRS_lower_left  = CDRS_lower_left 
        self.CDRS_total_right = CDRS_total_right 
        self.CDRS_total_left  = CDRS_total_left 
        self.CDRS_total       = CDRS_total 

    def __get_event_indices(self, array):
        """
        Description
            This method finds the indices of the beginning and end of the event in the array. Basically, one of the events (move or tapping) array will be provided as a parameter
            to the function.

        Input
            array: A binary list represents the existence=1 and absence=0 of a particular event (move/tapping).

        Output
            event_indices: A list containing tupples (start_index, finish_index) of index information for events
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

    def __populate_dataframe(self, dataset, patient, laterality, event_indices, event_type):

        """
        Description
            This method add a dataframe that contains all events detected for the given side of the patients. Initially, it acquires all the events contained 
            in _tap and _move fields. The information regarding the task (tapping, free, rest), the category (voluntary tapping, involuntary tapping, involuntary movement),
            event start and finish timestamps, etc are stored in this dataframe.

        Input
            :param dataset: A dataframe contains "patient","laterality","event_type", "event_no", "event_start_index", "event_finish_index", "event_start_time", 
                            "event_finish_time", "duration", "task" columns. It can be empty or filled previously.
            :param patient: A string representing the patient code
            :param laterality: A string representing the laterality of the limb (right/left/bilateral) information
            :param event_indices: A list containing tuples (start_index, finish_index) of index information for events
            :param event_type: A string represents the type of event that is considered
        
        Output
            :return dataset: The more populated version of the given dataframe structure
        """
        
        counter    = 1
        
        for event in event_indices:
            event_start_i = event[0]
            event_end_i   = event[1]

            if(event_start_i!=event_end_i):
    
                # get the corresponding tasks that were patient conducting during the event
                event_tasks   = self.task[event_start_i:event_end_i].tolist()
                # assing event to specific task [rest, tapping, free] based on the majority of duration passed for a given event
                event_task    = max(set(event_tasks), key=event_tasks.count) 
                # add event to the dataset
                dataset.loc[len(dataset)] = {"patient": patient, 
                                             "laterality": laterality, 
                                             "event": event_type, 
                                             "task": event_task,
                                             "event_no": "p_" + patient + "_" + laterality + "_" + event_type + str(counter),
                                             "event_start_index" : event_start_i, 
                                             "event_finish_index" : event_end_i, 
                                             "event_start_time" : self.times[event_start_i]/60, 
                                             "event_finish_time" : self.times[event_end_i]/60, 
                                             "duration": self.times[event_end_i] - self.times[event_start_i]}
                counter += 1

    def get_event_dataframe(self):

        """
        Description
            This method populates an event dataframe for a given patient, laterality, and event type.

        Input
            :param dataset: A dataframe contains "patient","laterality","event_type", "event_no", "event_start_index", "event_finish_index", "event_start_time", 
                                                 "event_finish_time", "duration", "task" columns
            :param patient: A string representing the patient code
            :param laterality: A string representing the laterality information
            :param event_indices: A list containing tuples (start_index, finish_index) of index information for events
            :param event_type: A string represents the type of event that is considered
        
        Output
            :return dataset: The more populated version of the given dataframe structure
        """
        # get start and finish indices of different event types in different hands
        left_moves        = self.__get_event_indices(self.left_move.tolist())
        left_tapping      = self.__get_event_indices(self.left_tap.tolist())
        right_moves       = self.__get_event_indices(self.right_move.tolist())
        right_tapping     = self.__get_event_indices(self.right_tap.tolist())
        bilateral_moves   = self.__get_event_indices(self.bilateral_move.tolist())
        bilateral_tapping = self.__get_event_indices(self.bilateral_tap.tolist())

        # create an empty event dataframe
        events = pd.DataFrame(columns=["patient", "laterality", "event", "task", "event_no", "event_start_index", "event_finish_index", 
                                       "event_start_time", "event_finish_time", "duration"])   

        # populate the empty dataframe
        self.__populate_dataframe(events, self.__SUB, "left"     , left_moves       , "move")
        self.__populate_dataframe(events, self.__SUB, "left"     , left_tapping     , "tap")
        self.__populate_dataframe(events, self.__SUB, "right"    , right_moves      , "move")
        self.__populate_dataframe(events, self.__SUB, "right"    , right_tapping    , "tap")
        self.__populate_dataframe(events, self.__SUB, "bilateral", bilateral_moves  , "move" )
        self.__populate_dataframe(events, self.__SUB, "bilateral", bilateral_tapping, "tap" )								

        # define dyskinesia_score column
        events["CDRS_face"]        = 0
        events["CDRS_neck"]        = 0
        events["CDRS_trunk"]       = 0
        events["CDRS_upper_right"] = 0
        events["CDRS_lower_right"] = 0
        events["CDRS_upper_left"]  = 0
        events["CDRS_lower_left"]  = 0
        events["CDRS_total_right"] = 0
        events["CDRS_total_left"]  = 0
        events["CDRS_total"]       = 0
        
        # for each index check the laterality and get the corresponding dyskinesia score on the event onset
        for index in events.index:
            events.loc[index, ['CDRS_face']]        = self.CDRS_face[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_neck']]        = self.CDRS_neck[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_trunk']]       = self.CDRS_trunk[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_upper_right']] = self.CDRS_upper_right[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_lower_right']] = self.CDRS_lower_right[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_upper_left']]  = self.CDRS_upper_left[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_lower_left']]  = self.CDRS_lower_left[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_total_right']] = self.CDRS_total_right[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_total_left']]  = self.CDRS_total_left[events.iloc[index].event_start_index]
            events.loc[index, ['CDRS_total']]       = self.CDRS_total[events.iloc[index].event_start_index]
            
        # define if the movement is voluntary or not as a column
        events["is_voluntary"]  = False
        events.loc[(events.event == "tap") & (events.task == "tap"), 'is_voluntary'] = True
        
        # add event categories to the dataframe
        events = self.__define_event_categories(events)

        # combine the score of two hands
        events['CDRS_total_hands'] = events['CDRS_upper_right'] + events['CDRS_upper_left']
        
        # map numerical values to severity equivalents
        events['CDRS_face']        = events['CDRS_face'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_neck']        = events['CDRS_neck'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_trunk']       = events['CDRS_trunk'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_right_hand']  = events['CDRS_upper_right'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_right_leg']   = events['CDRS_lower_right'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_left_hand']   = events['CDRS_upper_left'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_left_leg']    = events['CDRS_lower_left'].map({0:'none', 1:'mild', 2:'moderate', 3:'severe', 4:'extreme'})
        events['CDRS_total_hands'] = pd.cut(events['CDRS_total_hands'], bins=[-1, 0, 2, 4, 6, 8], labels=['none', 'mild', 'moderate', 'severe', 'extreme'])
        events['CDRS_total_right'] = pd.cut(events['CDRS_total_right'], bins=[-1, 0, 2, 4, 6, 8], labels=['none', 'mild', 'moderate', 'severe', 'extreme'])
        events['CDRS_total_left']  = pd.cut(events['CDRS_total_left'], bins=[-1, 0, 2, 4, 6, 8], labels=['none', 'mild', 'moderate', 'severe', 'extreme'])
        events['CDRS_total']       = pd.cut(events['CDRS_total'], bins=[-1, 0, 7, 14, 21, 28], labels=['none', 'mild', 'moderate', 'severe', 'extreme'])
        
        events = events[['patient', 'laterality', 'event_no', 'event', 'task', 'is_voluntary',
                         'event_category', 'event_start_index', 'event_finish_index',
                         'event_start_time', 'event_finish_time', 'duration', 'CDRS_face',
                         'CDRS_neck', 'CDRS_trunk', 'CDRS_right_hand',
                         'CDRS_right_leg', 'CDRS_left_hand', 'CDRS_left_leg',
                         'CDRS_total_hands', 'CDRS_total_right', 'CDRS_total_left', 'CDRS_total']]
        return events
    
    def get_CDRS_dataframe(self):
        return self.__CDRS_dataframe

    def __define_event_categories(self, dataset):
    
        def categorize_event(row):
            if (row['event'] == 'tap' and row['task'] == 'tap') : return 'tapping'
            #elif (row['event'] == 'tap' and row['task'] != 'tap'): return 'involuntary_tapping'
            else : return 'involuntary_movement'
            
            
    
        # Apply the function to create the new column 'event_category'
        dataset['event_category'] = dataset.apply(categorize_event, axis=1)
        dataset                   = dataset[['patient', 'laterality', 'event_no', 'event', 'task', 'is_voluntary', 'event_category', 'event_start_index', 
                                             'event_finish_index', 'event_start_time', 'event_finish_time', 'duration', 'CDRS_face', 'CDRS_neck', 'CDRS_trunk',
                                             'CDRS_upper_right','CDRS_lower_right','CDRS_upper_left','CDRS_lower_left','CDRS_total_right','CDRS_total_left','CDRS_total']]
        return dataset
        




    