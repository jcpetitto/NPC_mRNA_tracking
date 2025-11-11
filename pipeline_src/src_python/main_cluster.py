import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path

# import pipeline pieces
from yeast_pipeline import YeastPipeline
from image_processor import ImageProcessor
from tools.json_handling import load_json_experiments_data, NumpyArrayEncoder

if __name__ == "__main__":
    # --- JOB ARRAY SETUP ---
    # Get the job index from the command line
    # Default to 0 for local testing
    if len(sys.argv) > 1:
        # LSF index starts at 1, Python lists start at 0. Subtract 1.
        job_index = int(sys.argv[1]) - 1 
    else:
        job_index = 0

    # Load all experiment data
    # TODO set this with a config
    all_experiments = load_json_experiments_data("/home/jocelyn.tourtellotte-umw/yeast_output/tracking_experiments/")
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
    # TODO set this with a config
    output_base_dir = "/home/jocelyn.tourtellotte-umw/yeast_output/tracking_experiments/"
    os.makedirs(output_base_dir, exist_ok=True)

    # --- SINGLE EXPERIMENT PROCESSING ---
    # NOTE: The for loop has been removed. The script now processes only the one experiment selected above.
    
    if not fov_list:
        print(f" --> WARNING: No FoVs found for experiment {experiment_name}. Exiting.")
        sys.exit(0)

    try:
        # Set up the pipeline for the specific experiment
        pipeline = YeastPipeline()
        # TODO make this an argument to submit 
        pipeline.load_config_file('/home/jocelyn.tourtellotte-umw/src_yeast_pipeline/config_options_cluster.json')
        
        # Set experiment-specific path
        # NOTE: Assumes all FoVs in an experiment share the same collection path
        pipeline._cfg["FoV_collection_path"] = fov_list[0]['FoV_collection_path']

        # Initialize the ImageProcessor with the experiment-specific config
        img_proc = ImageProcessor(config_dict=pipeline.get_config(), device=pipeline.get_device())

        # --- Initial NE Detection ---
        try:
            print("--- Running Initial NE Detection ---")
            init_fit_subfolder = "init_fit/"
            output_dir_init_fit = os.path.join(output_base_dir, init_fit_subfolder)
            os.makedirs(output_dir_init_fit, exist_ok=True)

            img_proc.run_init_ne_detection()
            initial_fit_result = img_proc._get_ne_init_fit()
            ne_crop_result = img_proc._get_ne_cropped_imgs()
            ne_bspline_results = img_proc._get_init_ne_bsplines()
                
            if initial_fit_result:
                initial_fit_output_path = os.path.join(output_dir_init_fit, f"initial_fit_result_{experiment_name}.json")
        
                print(f" --> Saving initial ne fit results to: {initial_fit_output_path}")
                with open(initial_fit_output_path, 'w') as f:
                    json.dump(initial_fit_result, f, indent=4, cls=NumpyArrayEncoder)

            if ne_crop_result:
                ne_crop_output_path = os.path.join(output_dir_init_fit, f"ne_crop_result_{experiment_name}.json")
                print(f" --> Saving ne crop results to: {ne_crop_output_path}")
                with open(ne_crop_output_path, 'w') as f:
                    json.dump(ne_crop_result, f, indent=4, cls=NumpyArrayEncoder)
            
            if ne_bspline_results:
                init_fit_bsplines_output_path = os.path.join(output_dir_init_fit, f"init_fit_bsplines_{experiment_name}.pkl")
                print(f" --> Saving refined fit results to: {init_fit_bsplines_output_path}")
                with open(init_fit_bsplines_output_path, 'wb') as f:
                    # Use pickle to dump the complex object to a file
                    pickle.dump(ne_bspline_results, f)
            
        except Exception as e:
            print(f" --> ERROR: Failed during Initial NE Detection for {experiment_name}.")
            print(f"     Error details: {e}")

        # --- Image Registration ---
        try:
            print("\n--- Running Image Registration ---")
            reg_subfolder = "registration/"
            output_dir_reg = os.path.join(output_base_dir, reg_subfolder)
            os.makedirs(output_dir_reg, exist_ok=True)

            img_proc.register_images()
            registration_result = img_proc._get_reg_diff()
            
            if registration_result:
                reg_output_path = os.path.join(output_dir_reg, f"reg_result_{experiment_name}.json")
                print(f" --> Saving registration results to: {reg_output_path}")
                with open(reg_output_path, 'w') as f:
                    json.dump(registration_result, f, indent=4, cls=NumpyArrayEncoder)

        except Exception as e:
            print(f" --> ERROR: Failed during Image Registration for {experiment_name}.")
            print(f"     Error details: {e}")
            
    except Exception as e:
        print(f" --> ERROR: An unexpected error occurred while processing {experiment_name}.")
        print(f"     Error details: {e}")

    print(f"\n--- Job for experiment {experiment_name} finished. ---")