"""
Miscellaneous utilisation functions
"""

import pandas as pd
import ast
import os
from os import listdir

# converting string representation of a numeric array into numeric representation
def convert_to_array(string):
    try:
        # Ensure that the string is properly formatted
        if isinstance(string, str) and string.startswith('[') and string.endswith(']'):
            # Use ast.literal_eval to safely evaluate the string as a Python literal expression
            return ast.literal_eval(string)
        else:
            raise ValueError("String is not properly formatted as a list.")
    except (ValueError, SyntaxError) as e:
        # Handle cases where the string is not a valid literal expression
        print(f"Error parsing string: {string}\nException: {e}")
        return None

# get the SUB codes which stated in the PATH
def get_SUB_list(PATH):    
    SUB_list = []
    for dir in os.listdir(PATH):
        if("sub" in dir):
            SUB_list.append(dir[4:])
            
    return SUB_list

# get file names with specific format from given PATH
def get_files_with_specific_format(PATH, suffix=".csv" ):
    filenames = listdir(PATH)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# combine two python dictionary structure key to key measure by appending the values
def combine_dictionaries(dict1, dict2):
    
    combined_dict = {}

    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            combined_dict[key] = combine_dictionaries(dict1[key], dict2[key])
        elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
            combined_dict[key] = dict1[key] + dict2[key]
        else:
            raise ValueError("The structure of the dictionaries is not consistent.")

    return combined_dict

def combine_dataframes(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)