"""
Miscellaneous utilisation functions
"""

import pandas as pd
import numpy as np
import sys
import os
from os import listdir
import pickle

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_misc

from lib_data import DATA_IO
from lib_LFP import LFP 
from lib_ECoG import ECoG

def load_baseline_recordings(recording_type):

    baseline_recordings = {}
    SUB_LIST            = utils_misc.get_SUB_list(DATA_IO.path_data) # get the SUB id list which we have a recording of them

    for SUB in SUB_LIST:
        try:
            with open(DATA_IO.path_events + "baseline_recordings/" + recording_type + "/" + SUB + ".pkl", 'rb') as handle:
                baseline_recordings[SUB] = {}
                SUB_baseline             = pickle.load(handle)
                
                baseline_recordings[SUB]["right"] = list(SUB_baseline.values())[0]["right"]
                baseline_recordings[SUB]["left"]  = list(SUB_baseline.values())[0]["left"]
        except:
            print("Patient " + SUB + ": does not have baseline recordings")
    return baseline_recordings


def load_LFP_events(event_category, stn_areas, fs):
    
    event_dictionary = {}
    
    for stn_area in stn_areas:
        
        event_dictionary[stn_area] = {}
        # import raw event recordings
        EVENTS                     = pd.read_pickle(DATA_IO.path_events + "LFP_"+event_category+"_EVENTS_"+stn_area+"_AREA.pkl")
        EVENTS                     = LFP.define_onset_aligned_recordings(EVENTS, fs, pad=False)
        EVENTS                     = EVENTS[EVENTS.patient!="019"]
        
        # get controlateral and ipsilateral events
        EVENTS_CONTROLATERAL       = EVENTS[EVENTS.event_laterality != EVENTS.LFP_hemisphere]
        EVENTS_IPSILATERAL         = EVENTS[EVENTS.event_laterality == EVENTS.LFP_hemisphere]
        
        EVENTS_CONTROLATERAL_noLID = EVENTS_CONTROLATERAL[EVENTS_CONTROLATERAL.dyskinesia_arm=="none"]
        EVENTS_CONTROLATERAL_LID   = EVENTS_CONTROLATERAL[EVENTS_CONTROLATERAL.dyskinesia_arm!="none"]
        EVENTS_IPSILATERAL_noLID   = EVENTS_IPSILATERAL[EVENTS_IPSILATERAL.dyskinesia_arm=="none"]
        EVENTS_IPSILATERAL_LID     = EVENTS_IPSILATERAL[EVENTS_IPSILATERAL.dyskinesia_arm!="none"]
    
        event_dictionary[stn_area]["controlateral"]          = {}
        event_dictionary[stn_area]["controlateral"]["noLID"] = EVENTS_CONTROLATERAL_noLID
        event_dictionary[stn_area]["controlateral"]["LID"]   = EVENTS_CONTROLATERAL_LID
        event_dictionary[stn_area]["ipsilateral"]            = {}
        event_dictionary[stn_area]["ipsilateral"]["noLID"]   = EVENTS_CONTROLATERAL_noLID
        event_dictionary[stn_area]["ipsilateral"]["LID"]     = EVENTS_CONTROLATERAL_LID

    return event_dictionary

def load_ECoG_events(event_category, fs):
    
    event_dictionary = {}
    
    # import raw event recordings
    EVENTS                     = pd.read_pickle(DATA_IO.path_events + "ECoG_EVENTS.pkl")
    EVENTS                     = ECoG.define_onset_aligned_recordings(EVENTS, fs, pad=False)
    EVENTS                     = EVENTS[EVENTS.patient!="019"]
        
    # get controlateral and ipsilateral events
    EVENTS_CONTROLATERAL       = EVENTS[EVENTS.event_laterality != EVENTS.ECoG_hemisphere]
    EVENTS_IPSILATERAL         = EVENTS[EVENTS.event_laterality == EVENTS.ECoG_hemisphere]
        
    EVENTS_CONTROLATERAL_noLID = EVENTS_CONTROLATERAL[EVENTS_CONTROLATERAL.dyskinesia_arm=="none"]
    EVENTS_CONTROLATERAL_LID   = EVENTS_CONTROLATERAL[EVENTS_CONTROLATERAL.dyskinesia_arm!="none"]
    EVENTS_IPSILATERAL_noLID   = EVENTS_IPSILATERAL[EVENTS_IPSILATERAL.dyskinesia_arm=="none"]
    EVENTS_IPSILATERAL_LID     = EVENTS_IPSILATERAL[EVENTS_IPSILATERAL.dyskinesia_arm!="none"]
    
    event_dictionary["controlateral"]          = {}
    event_dictionary["controlateral"]["noLID"] = EVENTS_CONTROLATERAL_noLID
    event_dictionary["controlateral"]["LID"]   = EVENTS_CONTROLATERAL_LID
    event_dictionary["ipsilateral"]            = {}
    event_dictionary["ipsilateral"]["noLID"]   = EVENTS_CONTROLATERAL_noLID
    event_dictionary["ipsilateral"]["LID"]     = EVENTS_CONTROLATERAL_LID

    return event_dictionary

def load_ECoG_event_PSD(event_category):
    
    event_dictionary                           = {}    
    event_dictionary["controlateral"]          = {}
    event_dictionary["controlateral"]["noLID"] = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/CONTROLATERAL_noLID.pkl")
    event_dictionary["controlateral"]["LID"]   = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/CONTROLATERAL_LID.pkl")
    event_dictionary["ipsilateral"]            = {}
    event_dictionary["ipsilateral"]["noLID"]   = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/IPSILATERAL_noLID.pkl")
    event_dictionary["ipsilateral"]["LID"]     = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/IPSILATERAL_LID.pkl")

    # select only particular event category
    event_dictionary["controlateral"]["noLID"] = event_dictionary["controlateral"]["noLID"][event_dictionary["controlateral"]["noLID"].event_category==event_category]
    event_dictionary["controlateral"]["LID"]   = event_dictionary["controlateral"]["LID"][event_dictionary["controlateral"]["LID"].event_category==event_category]
    event_dictionary["ipsilateral"]["noLID"]   = event_dictionary["ipsilateral"]["noLID"][event_dictionary["ipsilateral"]["noLID"].event_category==event_category]
    event_dictionary["ipsilateral"]["LID"]     = event_dictionary["ipsilateral"]["LID"][event_dictionary["ipsilateral"]["LID"].event_category==event_category] 

    return event_dictionary


    
    