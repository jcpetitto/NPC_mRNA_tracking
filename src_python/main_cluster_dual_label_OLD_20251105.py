import sys
import pickle
import argparse
import os
import json
import traceback
import numpy as np
from pathlib import Path

# import pipeline pieces
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor
from tools.output_handling import save_output

if __name__ == "__main__":
    # --- JOB ARRAY SETUP ---
    # Get the job index from the command line
    # Default to 0 for local testing
    # Check for all required arguments
    if len(sys.argv) < 3: # script_name, job_index, config_path
        print("ERROR: Missing command-line arguments.")
        print("Usage: python main_cluster_dual_label.py <job_index> <config_path>")
        sys.exit(1)

    job_index = int(sys.argv[1]) - 1 
    config_path = sys.argv[2] # Use the argument from the command line

    # Now, initialize the pipeline with the dynamic path
    pipeline = ImagingPipeline(config_path)

    global_config = pipeline.retrieve_pipe_config("pipe globals")
    # making image processing specific configuration dictionary from configuration loaded into pipeline
    img_proc_cfg_dict = pipeline.retrieve_pipe_config("image processor")
    # Load all experiment data
    # TODO change so only the current experiment loads ...
    # ??? or maybe it could be set up so an overall pipeline manages all the img_proc instances ... maybe...?
    #           except they are all seperate jobs. So need to consider how to manage that as a flow
    all_experiments = pipeline.get_experiments()

    experiment_names = list(all_experiments.keys())

    # Check if the job_index is valid
    if job_index >= len(experiment_names):
        print(f"ERROR: Job index {job_index + 1} is out of bounds. Found only {len(experiment_names)} experiments.")
        sys.exit(1)

    # Select the specific experiment and its FoVs for this job
    experiment_name = experiment_names[job_index]
    fov_list = all_experiments[experiment_name]

    print(f"--- Processing Experiment (Job Index {job_index + 1}): {experiment_name} ---")

    # Define the base output directory

    output_base_dir = global_config['directories']['output root']
    os.makedirs(output_base_dir, exist_ok=True)

    # --- SINGLE EXPERIMENT PROCESSING ---
    # NOTE: The for loop has been removed. The script now processes only the one experiment selected above.
    
    if not fov_list:
        print(f" --> WARNING: No FoVs found for experiment {experiment_name}. Exiting.")
        sys.exit(0)

    try:
        img_proc = ImageProcessor(config_dict = img_proc_cfg_dict, FoV_dict = fov_list, device = pipeline.get_device())
        # add pointer to the pipeline for this ImageProcessor instance
        pipeline.add_img_processor(f'{experiment_name}', img_proc)

        # --- Initial NE Detection ---
        try:
            print("--- Running Initial NE Detection ---")
            img_proc.run_init_ne_detection()

            init_fit_subfolder = img_proc_cfg_dict['ne_fit']['initial']['output subdirectory']
            output_dir_init_fit = os.path.join(output_base_dir, init_fit_subfolder)
            os.makedirs(output_dir_init_fit, exist_ok=True)

            save_output(img_proc._get_ch1_cropped_imgs(), output_dir_init_fit, "ch1_crop", experiment_name)
            save_output(img_proc._get_ch1_init_bsplines(), output_dir_init_fit, "ch1_bsplines", experiment_name, is_pickle=True)

            save_output(img_proc._get_ch2_cropped_imgs(), output_dir_init_fit, "ch2_crop", experiment_name)
            save_output(img_proc._get_ch2_init_bsplines(), output_dir_init_fit, "ch2_bsplines", experiment_name, is_pickle=True)
            
        except Exception as e:
            print(f" --> ERROR: Failed during Initial NE Detection for {experiment_name}.")
            traceback.print_exc()

        # --- Image Registration ---
        try:
            print("\n--- Running Image Registration ---")
            reg_subfolder = img_proc_cfg_dict['registration']['output subdirectory']
            output_dir_reg = os.path.join(output_base_dir, reg_subfolder)
            os.makedirs(output_dir_reg, exist_ok=True)

            img_proc.register_images(reg_mode=1)
            save_output(img_proc.get_registration_results(reg_mode=1), output_dir_reg, "reg_results_mode1", experiment_name)

            img_proc.register_images(reg_mode=2)
            save_output(img_proc.get_registration_results(reg_mode=2), output_dir_reg, "reg_results_mode2", experiment_name)

        except Exception as e:
            print(f" --> ERROR: Failed during Image Registration for {experiment_name}.")
            traceback.print_exc()
        
        # --- Filtering based on registration values
        try:
            print("\n--- Running Registration Stability Analysis & Filtering ---")
            # This is the new function name
            img_proc.analyze_and_filter_by_registration() 
        except Exception as e:
            print(f" --> ERROR: Failed during Registration Analysis/Filtering for {experiment_name}.")
            traceback.print_exc()

        try: 
            print("\n--- Running Spline Refinement ---")
            img_proc.run_ne_refinement()
            ch1_ne_refined_bspline = img_proc._get_ch1_refined_bsplines()
            ch2_ne_refined_bspline = img_proc._get_ch2_refined_bsplines()

            refine_subfolder = img_proc_cfg_dict['ne_fit']['initial']['output subdirectory']
            output_dir_refine = os.path.join(output_base_dir, refine_subfolder)

            save_output(img_proc._get_ch1_refined_bsplines(), output_dir_refine, "refine_results_ch1", experiment_name)
            
            save_output(img_proc._get_ch2_refined_bsplines(), output_dir_refine, "refine_results_ch2", experiment_name)

        except Exception as e:
            print(f" --> ERROR: An unexpected error occurred while refining splines for {experiment_name}.")
            traceback.print_exc()
        try:
            print("\n--- Running Dual Label Difference Calculations ---")
            dual_dist_subfolder = img_proc_cfg_dict['dual_label']['output subdirectory']
            output_dir_dual_dist = os.path.join(output_base_dir, dual_dist_subfolder)
            os.makedirs(output_dir_dual_dist, exist_ok=True)

            img_proc.juxtapose_dual_labels()

            save_output(img_proc.get_dual_distances_by_FoV(), output_dir_dual_dist, "dual_dist_result", experiment_name)

        except Exception as e:
            print(f" --> ERROR: Failed during Dual Label Difference Calculations for {experiment_name}.")
            traceback.print_exc()
            
    except Exception as e:
        print(f" --> ERROR: An unexpected error occurred while processing {experiment_name}.")
        traceback.print_exc()
    pipeline.add_experiments(all_experiments)
    print(f"\n--- Job for experiment {experiment_name} finished. ---")