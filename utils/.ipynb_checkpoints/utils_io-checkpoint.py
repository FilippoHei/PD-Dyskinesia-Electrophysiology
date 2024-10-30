"""
Miscellaneous utilisation functions
"""

import pandas as pd
import numpy as np
import sys
import os
from os import listdir
import pickle
import pyvista as pv 
import nibabel as nib

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
        except:
            pass
        try:
            baseline_recordings[SUB]["right"] = list(SUB_baseline.values())[0]["right"]
        except:
            print("Patient " + SUB + ": does not have right hemisphere baseline recordings")
        try:
            baseline_recordings[SUB]["left"]  = list(SUB_baseline.values())[0]["left"]
        except:
            print("Patient " + SUB + ": does not have left hemisphere baseline recordings")
            
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
    event_dictionary["controlateral"]["noLID"] = EVENTS_CONTROLATERAL_noLID[EVENTS_CONTROLATERAL_noLID.event_category==event_category]
    event_dictionary["controlateral"]["LID"]   = EVENTS_CONTROLATERAL_LID[EVENTS_CONTROLATERAL_LID.event_category==event_category]
    event_dictionary["ipsilateral"]            = {}
    event_dictionary["ipsilateral"]["noLID"]   = EVENTS_IPSILATERAL_noLID[EVENTS_IPSILATERAL_noLID.event_category==event_category]
    event_dictionary["ipsilateral"]["LID"]     = EVENTS_IPSILATERAL_LID[EVENTS_IPSILATERAL_LID.event_category==event_category]

    return event_dictionary

def load_ECoG_event_PSD(event_category, event_laterality):
    
    event_dictionary = {} 
    
    if(event_laterality == "controlateral"):   
        data_noLID = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/CONTROLATERAL_noLID.pkl")
        data_LID   = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/CONTROLATERAL_LID.pkl")
    else:
        data_noLID = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/IPSILATERAL_noLID.pkl")
        data_LID   = pd.read_pickle(DATA_IO.path_events + "psd/ECoG/IPSILATERAL_LID.pkl")
        
    event_dictionary["noLID_noDOPA"] = data_noLID[(data_noLID.event_start_time <= 30) & (data_noLID.event_category==event_category)]
    event_dictionary["noLID_DOPA"]   = data_noLID[(data_noLID.event_start_time > 30) & (data_noLID.event_category==event_category)]
    event_dictionary["mild"]         = data_LID[(data_LID.dyskinesia_arm=="mild") & (data_LID.event_category==event_category)]
    event_dictionary["moderate"]     = data_LID[(data_LID.dyskinesia_arm=="moderate") & (data_LID.event_category==event_category)]

    event_dictionary["noLID_noDOPA"].reset_index(inplace=True)
    event_dictionary["noLID_DOPA"].reset_index(inplace=True)
    event_dictionary["mild"].reset_index(inplace=True)
    event_dictionary["moderate"].reset_index(inplace=True)

    return event_dictionary

def load_LFP_event_PSD(event_category, event_laterality):
    
    event_dictionary = {} 
    
    if(event_laterality == "controlateral"):   
        data_noLID = pd.read_pickle(DATA_IO.path_events + "psd/LFP/CONTROLATERAL_MOTOR_noLID.pkl")
        data_LID   = pd.read_pickle(DATA_IO.path_events + "psd/LFP/CONTROLATERAL_MOTOR_LID.pkl")
        data_noLID = data_noLID[data_noLID.patient!="019"]
        data_LID   = data_LID[data_LID.patient!="019"]
    else:
        data_noLID = pd.read_pickle(DATA_IO.path_events + "psd/LFP/IPSILATERAL_MOTOR_noLID.pkl")
        data_LID   = pd.read_pickle(DATA_IO.path_events + "psd/LFP/IPSILATERAL_MOTOR_LID.pkl")
        data_noLID = data_noLID[data_noLID.patient!="019"]
        data_LID   = data_LID[data_LID.patient!="019"]
        
    event_dictionary["noLID_noDOPA"] = data_noLID[(data_noLID.event_start_time <= 30) & (data_noLID.event_category==event_category)]
    event_dictionary["noLID_DOPA"]   = data_noLID[(data_noLID.event_start_time > 30) & (data_noLID.event_category==event_category)]
    event_dictionary["mild"]         = data_LID[(data_LID.dyskinesia_arm=="mild") & (data_LID.event_category==event_category)]
    event_dictionary["moderate"]     = data_LID[(data_LID.dyskinesia_arm=="moderate") & (data_LID.event_category==event_category)]

    event_dictionary["noLID_noDOPA"].reset_index(inplace=True)
    event_dictionary["noLID_DOPA"].reset_index(inplace=True)
    event_dictionary["mild"].reset_index(inplace=True)
    event_dictionary["moderate"].reset_index(inplace=True)

    return event_dictionary

def load_cortical_atlas_meshes():
    meshes                     = {} 
    meshes["right_hemisphere"] = pv.read(DATA_IO.path_atlas_cortical + 'cortex_right.vtk')
    meshes["left_hemisphere"]  = pv.read(DATA_IO.path_atlas_cortical + 'cortex_left.vtk')
    return meshes

def load_STN_meshes():
    meshes                     = {} 
    meshes["right_hemisphere"] = pv.read(DATA_IO.path_atlas_subthalamic + 'stn_right.vtk')
    meshes["left_hemisphere"]  = pv.read(DATA_IO.path_atlas_subthalamic + 'stn_left.vtk')
    return meshes


def load_AAL3_files_for_cortical_parcellation():
    AAL3_image  = nib.load(DATA_IO.path_atlas_cortical + 'AAL3.nii')
    AAL3_labels = pd.read_csv(DATA_IO.path_atlas_cortical + "AAL3_labels.csv")
    return AAL3_image, AAL3_labels

    
    