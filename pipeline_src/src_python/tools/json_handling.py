# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:03:11 2025

@author: jctourtellotte
"""

# --- Outside Modules --- #
import numpy as np
import json
from pathlib import Path

# --- Included Modules --- # 





class NumpyArrayEncoder(json.JSONEncoder):
    """ Custom encoder for all common numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
# path_to_experiments_folder - folder containing the json for each experiment
# aggregates information into a dictionary of dictionaries with entries keys
# set by experiment name
def load_json_experiments_data(path_to_experiments_folder:str):
    json_directory = Path(path_to_experiments_folder)
    json_files = sorted(json_directory.glob('*.json'))

    if not json_files:
        print(f"No *.json files found in {json_directory}")
        raise ValueError(f"No *.json files found in {json_directory}")
        # will this print even if nothing catches the raised exception?
        # if yes, then can delete the preceding print
    else:
        print(f"Found {len(json_files)} JSON files. Loading them...")
        all_experiments_data = {}
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    # Load the list of dictionaries from the current file
                    experiment_data = json.load(f)
                    
                    # Use the filename (without .json) as the key
                    experiment_name = file_path.stem 
                    all_experiments_data[experiment_name] = experiment_data
                    print(f" -> Loaded {file_path.name}")
    
            except Exception as e:
                print(f"Could not load or process {file_path.name}. Error: {e}")
                
        return all_experiments_data



def load_dict_list_from_json(dict_path = None):
        if dict_path is None:
    # Set default values
            print("No data dictionary file path was provided. Without such a document, the program is unable to proceed.")
        else:
            try:
                with open(dict_path, 'r') as dict_file:
                    new_dict = json.load(dict_file)
                    return new_dict
            except FileNotFoundError:
                print(f"Error: JSON configuration file not found at '{dict_path}'.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{dict_path}'. Check the file format.")
