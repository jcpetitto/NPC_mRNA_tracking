# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:54:43 2025

@author: jctourtellotte
"""

# --- Outside Modules --- #
import sys
import os
# --- Included Modules --- # 
# import pipeline configuration
import config
# configurations determined by detecting attributes 
#   of the system running the pipeline (ie. run torch on GPU, MPS, or CPU)
from tools.utility_functions import config_from_file
from tools.directory_to_dict import confirm_strains, process_strains

from tools.output_handling import load_json_experiments_data, NumpyArrayEncoder


class ImagingPipeline:
    
    def __init__(self, config_path):
# setting default options
# !!!: update when pipeline complete
        self._cfg = {}
        self._img_processors = {}
        self._experiments = {}
        # device config loaded via config module
        self.set_device(config.DEVICE)
 #       self.set_dfloat_torch(config.dfloat_torch)
 #       self.set_dfloat_np(config.dfloat_np)
        self.load_config_file(config_path)
        self.load_experimental_data()

        print("ImagingPipeline initiated")

    
# load configuration for pipeline
    def load_config_file(self, config_path = None):
        try:
            pipeline_config = config_from_file(config_path)
            self.set_config(pipeline_config)
        except ValueError as e:
            print(f"Error loading config file: '{e}'.")
    
    def load_experimental_data(self):
        pipeline_general_config = self.get_config().get("pipe globals")
        imaging_directory = pipeline_general_config.get("directories").get("imaging root")
        experiment_directory = os.path.join(imaging_directory, pipeline_general_config.get("directories").get("experiment subdirectory"))
        print(f'Creating dictionaries from directory of imaging experiments: {experiment_directory}')
        # confirms the directory contains at least one folder re: the strain names passed to it
        strains_to_process = confirm_strains(experiment_directory, pipeline_general_config.get("strains"))
        all_experiments = process_strains(strains_to_process)

        self.add_experiments(all_experiments)

    def add_experiments(self, exper_dict:dict):
        current_experiments = self.get_experiments()
        current_experiments.update(exper_dict)
    
    def add_img_processor(self, img_proc_id, img_proc):
        current_img_processors = self.get_img_processors()
        current_img_processors.update({f'{img_proc_id}': img_proc})
        self.set_img_processors(current_img_processors)
    
    def retrieve_pipe_config(self, pipe_class:str):
        pipeline_config = self.get_config()
        pipe_node_list = pipeline_config.get("pipeline foreman").get(pipe_class)
        indiv_pipe_config = {}
        for pipe_node in pipe_node_list:
            indiv_pipe_config.update(pipeline_config.get(pipe_node))

        return indiv_pipe_config


    # SETTERS & GETTERS
          
    def set_device(self, device):
        self._current_device = device
    
    def get_device(self):
        return self._current_device

    def set_config(self, config):
        for key, value in config.items():
            self._cfg[key] = value
    
    def get_config(self):
        return self._cfg
    
    def get_img_processors(self):
        return self.img_processors

    def set_experiments(self, experiment_dict:dict):
        self._experiments = experiment_dict

    def get_experiments(self):
        return self._experiments
    
    def set_img_processors(self, img_processors):
        self._img_processors = img_processors
    
    def get_img_processors(self):
        return self._img_processors
