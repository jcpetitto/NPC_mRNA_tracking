# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:54:43 2025

@author: jctourtellotte
"""

# --- Outside Modules --- #
import argparse
import json

# --- Included Modules --- # 
# import pipeline configuration
import config
# configurations determined by detecting attributes 
#   of the system running the pipeline (ie. run torch on GPU, MPS, or CPU)

class YeastPipeline:
    
    def __init__(self):
# setting default options
# !!!: update when pipeline complete
        self._cfg = {
            "roisize": 16,
            "sigma": 0.92,
            "frames": [0,1000],
            "drift_bins": 4,
            "pixelsize": 128,
            "frametime": 0.02,
            "threshold_reg": 0.50,
            "cell_line_id": "823",
            "path": "",
            "resultsdir": "results/",
            "bright": "example_data/Calibration_Data/bright_images_green_channel_20ms_300EM.tiff",
            "dark": "example_data/Calibration_Data/dark_images_green_channel_20ms_300EM.tiff",
            "model_NE": "example_data/trained_networks/Modelweights_NE_segmentation.pt",
            "model_bg": "example_data/trained_networks/model_wieghts_background_psf.pth",
            "FoV_collection_path": "example_data/Example_raw_data/BMY823_7_25_23_aqsettings1_batchC/",
            "save_folder": "merged_results/",
            "trackdata": "tracks.data",
            "moviename": "test.mp4",
            "save_figures": False,
            "save_movies": False,
            "registration": {"frame_range": [0, 250]
                            },
            "calibration": {"frame_range": [0, 250],
                            "drift_bins": 4
                            },                    
            "ne_fit": {"use_merged_clusters": False,
                        "norm_line_length": 1,
                        "normal_sample_n": 100,
                        "frame_range": [0, 250]
            }
        }
        self.set_device(config.DEVICE)
 #       self.set_dfloat_torch(config.dfloat_torch)
 #       self.set_dfloat_np(config.dfloat_np)
        
        print("YeastPipeline initiated")

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
    
# load configuration for pipeline
    def load_config_file(self, config_path = None):
        if config_path is None:
    # Set default values
            print("No configuration file path provided. Default values have been loaded, which may not be appropriate for your data.")
        else:
            try:
                with open(config_path, 'r') as config_file:
                    config = json.load(config_file)
                    self.set_config(config)
            except FileNotFoundError:
                print(f"Error: JSON configuration file not found at '{config_path}'.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{config_path}'. Check the file format.")

 # Parse command-line argument(s); see: https://docs.python.org/3/library/argparse.html
    def parse_args():
        parser = argparse.ArgumentParser(description='Configure the yeast processing pipeline.')
        parser.add_argument('config_file_path', type=str, help='Path to the configuration file containing yeast data locaation, camera information, etc..')
        return parser.parse_args()