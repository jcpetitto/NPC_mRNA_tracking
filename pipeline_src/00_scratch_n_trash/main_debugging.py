# import external packages
import pickle
import argparse
import os

# import pipeline pieces
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="A script to process a file from a given location.") # create parser to accept the CL argument(s)
    # parser.add_argument("config_json", help="The full path to the json that contains the pipeline configuration.")
    # args = parser.parse_args()
    # json_path = args.config_json
    json_path = "/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/config_local_dual.json"
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Error: Unable to continue. The configuration file was not found at the path: {json_path}")

    pipeline = ImagingPipeline(json_path)

    # making image processing specific configuration dictionary from configuration loaded into pipeline
    img_proc_cfg_dict = pipeline.retrieve_pipe_config("image processor")
    all_experiments = pipeline.get_experiments()

    all_ne_crop_boxes = {}
    all_ne_bsplines = {}
    all_registration = {}
    all_refined_splines = {}
    all_distances = {}

    # -- Per Experiment -- #
    for experiment_name, fov_list in all_experiments.items():
        print(f"\n--- Processing Experiment: {experiment_name} ---")
        
        if not fov_list:
            print(f" --> WARNING: No FoVs found for experiment {experiment_name}. Skipping.")
            continue

        # NOTE: This process (not yet complete) + (to be written) analysis are/will be associated with the instance of the ImagingPipeline
        # NOTE: Want pre-built configurations re:
        # things like ex. FoV vs segment-based registration

        # Initialize the ImageProcessor with the experiment-specific config

        img_proc = ImageProcessor(config_dict = img_proc_cfg_dict, FoV_dict = fov_list, device = pipeline.get_device())

        # add pointer to the pipeline for this ImageProcessor instance
        pipeline.add_img_processor(f'{experiment_name}', img_proc)
        
        print("--- Running Initial NE Detection ---")
        img_proc.run_init_ne_detection()
        if img_proc._get_cfg()["ne_dual_label"]:
            ch1_ne_crop_result = img_proc._get_ch1_cropped_imgs()
            ch1_ne_bspline_results = img_proc._get_ch1_init_bsplines()

            ch2_ne_crop_result = img_proc._get_ch2_cropped_imgs()
            ch2_ne_bspline_results = img_proc._get_ch2_init_bsplines()

            all_ne_crop_boxes.update({f'{experiment_name}':
                                            {
                                                'ch1': ch1_ne_crop_result,
                                                'ch2': ch2_ne_crop_result
                                            }
                                    }) 
            all_ne_bsplines.update({f'{experiment_name}':
                                            {
                                                'ch1': ch1_ne_bspline_results,
                                                'ch2': ch2_ne_bspline_results
                                            }
                                    })  
        else:    
            ne_crop_result = img_proc._get_ch1_cropped_imgs()
            ne_bspline_results = img_proc._get_ch1_init_bsplines()

            all_ne_crop_boxes.update({f'{experiment_name}': ne_crop_result}) 
            all_ne_bsplines.update({f'{experiment_name}': ne_bspline_results}) 

        img_proc.register_images()
        registration_result = img_proc.get_registration_results()

        all_registration.update({f'{experiment_name}': registration_result})

        # !!! needs to include culling by registration results

        # Refinement function checks to see if the pipeline is configured for dual label to determine what channels it is being run on
        #   - channel 1 is assumed to be the nuclear pore label in a single label / mRNA tracking situation
        #   - dual label refines on both channels
        #       this requires the matches to be made as refinement is a computational waste to do for NE that aren't found in both channels
        #       matching is done within this step "run_ne_refinement" based on the overlap of crop boxes in both channels
        #       matching only occurs for dual labels
        img_proc.run_ne_refinement()

        # only for dual label experiments
        # !!! does need the results from registration

        if img_proc._get_cfg()["ne_dual_label"]:
            ch1_ne_refined_bspline = img_proc._get_ch1_refined_bsplines()
            ch2_ne_refined_bspline = img_proc._get_ch2_refined_bsplines()

            all_refined_splines.update({f'{experiment_name}':
                                            {
                                                'ch1': ch1_ne_refined_bspline,
                                                'ch2': ch2_ne_refined_bspline
                                            }
                                    })
        else:
            # mRNA tracking needs to refine only the NE label channel (ch1)
            ch1_ne_refined_bspline = img_proc._get_ch1_refined_bsplines()

            all_refined_splines.update({f'{experiment_name}':
                                            {
                                                'ch1': ch1_ne_refined_bspline,
                                            }
                                    })

        if img_proc._get_cfg()["ne_dual_label"]:
            img_proc.juxtapose_dual_labels()

            dual_distance_results = img_proc.get_dual_distances_by_FoV()
            all_distances.update({f'{experiment_name}': dual_distance_results})

    with open('output/dual_debug_img_proc.pkl', 'wb') as f:
        pickle.dump(img_proc, f)
        
    with open('output/dual_debug_all_exper.pkl', 'wb') as f:
        pickle.dump(all_experiments, f)
        
    with open('output/dual_debug_all_ne_bspl.pkl', 'wb') as f:
        pickle.dump(all_ne_bsplines, f)
        
    with open('output/dual_debug_all_ne_crop.pkl', 'wb') as f:
        pickle.dump(all_ne_crop_boxes, f)
        
    with open('output/dual_debug_all_reg.pkl', 'wb') as f:
        pickle.dump(all_registration, f)

    with open('output/dual_debug_all_refined.pkl', 'wb') as f:
        pickle.dump(all_refined_splines, f)

        
        # SELF: is differentiating the refinement needs to only include matched labels an improvement to the pipeline or was this a step included before


    pipeline.add_experiments(all_experiments)

    print("\n--- All experiments processed. ---")
