import os
import pandas as pd
import numpy as np
import sys
import json
from utils.utils_fileManagement import load_class_pickle, mergedData

class DATA_IO:

    # definition of static fields
    path_events           = "../events/"
    path_coordinates      = "../coordinates/"
    path_figure           = "../figures/"
    path_data             = "../data/"
        
    def __init__(self, PATH, SUB, DAT_SOURCE):
        self.__PATH        = PATH
        self.__SUB         = SUB
        self.__DAT_SOURCE  = DAT_SOURCE
        self.__data        = self.__check__and_load_file()

    def __check__and_load_file(self):
        """
        This method provides the data structure for simultaneously collected electrophysiological and kinematic data (LFP, ECoG, or accelerometer).
        For the given subject, the directory denoted with PATH will be checked to load the selected data source.

        :return: The data source file contains the fs, times, task, left_tap, right_tap, left_move, and right_move
        """
            
        PATH       = self.__PATH
        SUB        = self.__SUB
        DAT_SOURCE = self.__DAT_SOURCE
            
        assert os.path.exists(PATH), f'PATH does not exist ({PATH})'
        assert DAT_SOURCE in ['lfp_left', 'lfp_right', 'ecog_left','ecog_right', 'acc_left', 'acc_right'], f'incorrect DAT_SOURCE ({DAT_SOURCE})'
        
        folder = os.path.join(PATH, 'data', f'sub-{SUB}')
        fname = f'{SUB}_mergedData_v4.0_{DAT_SOURCE}.P'
        
        assert fname in os.listdir(folder), (f'FILE {fname} not in {folder}')
        data = load_class_pickle(os.path.join(folder, fname))
        return data

    def get_data(self):
        """
        This method provides the data structure for simultaneously collected electrophysiological and kinematic data (LFP, ECoG, or accelerometer).
        For the given subject, the directory denoted with PATH will be checked to load the selected data source.

        :return: the stored data structure
        """
        return self.__data



