"""
Miscellaneous utilisation functions
"""

import pandas as pd
import ast

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