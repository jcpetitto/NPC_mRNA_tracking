import os
import pickle
import json
import pickle
import numpy as np
from pathlib import Path

# import pipeline pieces
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor

from tools.output_handling import load_json_experiments_data, NumpyArrayEncoder

if __name__ == "__main__":

    # -- Overall Set-up -- #
    mRNA_tracking_data_list_dict = load_json_experiments_data("yeast_output/tracking_experiments/")

    # Define the base output directory
    output_base_dir = "yeast_output/tracking_experiments/"
    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # -- Per Experiment -- #
    # 2. Loop through each experiment
    for experiment_name, fov_list in mRNA_tracking_data_list_dict.items():
        print(f"\n--- Processing Experiment: {experiment_name} ---")
        
        if not fov_list:
            print(f" --> WARNING: No FoVs found for experiment {experiment_name}. Skipping.")
            continue

        try:
            # 3. Set up the pipeline for this specific experiment
            pipeline = ImagingPipeline()
            pipeline.load_config_file('config_options.json')

            # Set path for the current experiment from the first FoV's data
            # NOTE: Assumes all FoVs in an experiment share the same collection path
            new_path = fov_list[0]['FoV_collection_path']
            pipeline._cfg["FoV_collection_path"] = new_path

            # NOTE: This process (not yet complete) + (to be written) analysis will
            #            move into the ImagingPipeline Class with various pre-built configurations
            #             ex. FoV vs segment-based registration
            # Initialize the ImageProcessor with the experiment-specific config
            img_proc = ImageProcessor(config_dict=pipeline.get_config(), device=pipeline.get_device())
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
                print(f" --> ERROR: Failed during Initial NE Detection for {experiment_name}. Skipping to next step.")
                print(f"     Error details: {e}")
        
            try:
                print('--- Running Registration ---')
                reg_subfolder = "registration/"
                output_dir_reg = os.path.join(output_base_dir, reg_subfolder)
                os.makedirs(output_dir_reg, exist_ok=True)

                img_proc.register_images()
                registration_result = img_proc.get_registration_results()
                
                if registration_result:
                    reg_output_path = os.path.join(output_dir_reg, f"reg_result_{experiment_name}.json")
                    print(f" --> Saving registration results to: {reg_output_path}")
                    with open(reg_output_path, 'w') as f:
                        json.dump(registration_result, f, indent=4, cls=NumpyArrayEncoder)

            except Exception as e:
                print(f" --> ERROR: Failed during Registration for {experiment_name}.")
                print(f"     Error details: {e}")
            # try:
            #     print('--- Refining NE Splines ---')
            #     img_proc.run_ne_refinement()
            # except Exception as e:
            #     print(f" --> ERROR: Failed during Spline Refinement for {experiment_name}.")
            #     print(f"     Error details: {e}")

            # try: 
            #     print("\n--- Running Image Registration ---")
            #     img_proc.register_images()
            #     registration_result = img_proc.get_registration_results()
                
            #     if registration_result:
            #         reg_output_path = os.path.join(output_base_dir, f"reg_result_{experiment_name}.json")
            #         print(f" --> Saving registration results to: {reg_output_path}")
            #         with open(reg_output_path, 'w') as f:
            #             json.dump(registration_result, f, indent=4, cls=NumpyArrayEncoder)
            
            # except Exception as e:
            #     print(f" --> ERROR: Failed during Image Registration for {experiment_name}.")
            #     print(f"     Error details: {e}")
            
        except Exception as e:
            # Catch any other errors during processing
            print(f" --> ERROR: An unexpected error occurred while processing {experiment_name}. Skipping.")
            print(f"     Error details: {e}")

    print("\n--- All experiments processed. ---")