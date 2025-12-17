import sys
import logging
import argparse
from pathlib import Path
import traceback
import os

import pickle
import json

import numpy as np
import pandas as pd

# import pipeline pieces
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor
from tools.output_handling import save_output

# --- Helper Functions for Checkpointing ---

def load_checkpoint(path: Path):
    """Loads a pickled object from a checkpoint file."""
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"ERROR: Could not load checkpoint {path}. Error: {e}")
        return None

def save_checkpoint(obj: object, path: Path):
    """Saves an object to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# --- Pipe-specific functions --- #

def run_initial_ne(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """Runs the Initial NE Detection step and saves outputs."""
    logger.info("--- Running Initial NE Detection ---")
    img_proc.run_init_ne_detection()

    init_fit_subfolder = cfg_dict['ne_fit']['initial']['output subdirectory']
    output_dir = out_base / init_fit_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Your save_output calls
    save_output(img_proc._get_ch1_cropped_imgs(), output_dir, "ch1_crop", exp_name)
    save_output(img_proc._get_ch1_init_bsplines(), output_dir, "ch1_bsplines", exp_name, is_pickle=True)
    save_output(img_proc._get_ch2_cropped_imgs(), output_dir, "ch2_crop", exp_name)
    save_output(img_proc._get_ch2_init_bsplines(), output_dir, "ch2_bsplines", exp_name, is_pickle=True)

def run_registration(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """Runs the Image Registration step and saves outputs."""
    logger.info("--- Running Image Registration ---")
    reg_subfolder = cfg_dict['registration']['output subdirectory']
    output_dir = out_base / reg_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    img_proc.register_images(reg_mode=1)
    save_output(img_proc.get_registration_results(reg_mode=1), output_dir, "reg_results_mode1", exp_name)

    img_proc.register_images(reg_mode=2)
    save_output(img_proc.get_registration_results(reg_mode=2), output_dir, "reg_results_mode2", exp_name)

def run_filtering(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Runs the Registration Stability Analysis, saves reports, and
    filters the ImageProcessor's NE pair map.
    """
    logger.info("--- Running Registration Stability Analysis & Filtering ---")
    
    # 1. Run the analysis (this is the method from image_processor.py)
    # It returns the flat report and the global stats dict
    flat_report_data, global_stats = img_proc.analyze_registration_stability()

    # 2. Print the "Experiment as a whole" report to the log
    if global_stats:
        logger.info("--- EXPERIMENT-WIDE STABILITY REPORT ---")
        # .get() is safer, in case a level has no data
        print_global_stats_to_log("Global Slice-Level Stats", global_stats.get("slice_level", {}))
        print_global_stats_to_log("Global NE-Label-Level Stats", global_stats.get("ne_label_level", {}))
        print_global_stats_to_log("Global FoV-Level Stats", global_stats.get("fov_level", {}))
    
    # 3. Save the "flat form" reports (if data was generated)
    if flat_report_data:
        logger.info("Saving Registration Stability Reports...")
        # Get the output directory from the config
        reg_subfolder = cfg_dict['registration']['output subdirectory']
        output_dir_reg_report = out_base / reg_subfolder
        # The registration step should have already created this directory
        
        save_output(
            flat_report_data, 
            output_dir_reg_report, 
            "registration_stability_report", 
            exp_name
        )
        
        save_output(
            global_stats,
            output_dir_reg_report, 
            "registration_summary_stats", 
            exp_name
        )
    
    logger.info("--- Registration Analysis & Filtering Complete ---")

def run_refinement(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Tells the ImageProcessor to run the robust, restartable
    spline refinement step.
    
    This function is responsible for creating the paths, and the
    ImageProcessor handles the loop and partial file I/O.
    """
    logger.info("--- Preparing for Spline Refinement ---")
    
    # --- 1. Define paths ---
    refine_subfolder = cfg_dictcfg_dict['ne_fit']['refinement']['output subdirectory']
    output_dir = out_base / refine_subfolder
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Define paths for IN-PROGRESS checkpoints
    partial_ch1_path = output_dir / f"refine_results_ch1_{exp_name}_IN_PROGRESS.pkl"
    partial_ch2_path = output_dir / f"refine_results_ch2_{exp_name}_IN_PROGRESS.pkl"

    # --- 2. Call the encapsulated method ---
    # Pass the paths AND the I/O helper functions
    img_proc.run_ne_refinement(
        partial_ch1_path=partial_ch1_path,
        partial_ch2_path=partial_ch2_path,
        load_checkpoint_func=load_checkpoint,
        save_checkpoint_func=save_checkpoint
    )
    
    # --- 3. Save the FINAL, complete output files ---
    # The img_proc object is now populated with the final results.
    # We use save_output with is_pickle=True
    logger.info("Saving final (complete) refinement files...")
    save_output(img_proc._get_ch1_refined_bsplines(), output_dir, "refine_results_ch1", exp_name, is_pickle=True)
    save_output(img_proc._get_ch2_refined_bsplines(), output_dir, "refine_results_ch2", exp_name, is_pickle=True)

    logger.info("--- Spline Refinement Step Complete ---")

def run_bridging(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Runs the (optional) Bezier bridging step to connect refined spline segments
    into whole, closed membranes.
    """
    logger.info("--- Running Bezier Spline Bridging ---")
    
    # Get the output directory (we'll re-use the merged_splines folder)
    try:
        refine_subfolder = cfg_dictcfg_dict['ne_fit']['refinement']['output subdirectory']
    except KeyError:
        refine_subfolder = cfg_dict['ne_fit']['initial']['output subdirectory']
    output_dir = out_base / refine_subfolder
    
    # Run the new ImageProcessor method
    img_proc.run_bezier_bridging()

    # Save the bridged output
    logger.info("Saving bridged spline results...")
    # Always save Channel 1
    save_output(img_proc._get_ch1_bridged_splines(), output_dir, "bridged_splines_ch1", exp_name, is_pickle=True)

    if cfg_dict.get('ne_dual_label', False):
        save_output(img_proc._get_ch2_bridged_splines(), output_dir, "bridged_splines_ch2", exp_name, is_pickle=True)

def run_dual_label(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """Runs the Dual Label Difference Calculations step."""
    logger.info("--- Running Dual Label Difference Calculations ---")
    dual_dist_subfolder = cfg_dict['dual_label']['output subdirectory']
    output_dir = out_base / dual_dist_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        img_proc.juxtapose_dual_labels()
        logger.info("Juxtaposition complete!")  # ADD THIS
        
        distances = img_proc.get_dual_distances_by_FoV()
        logger.info(f"Retrieved distances: {len(distances) if distances else 0} FoVs")  # ADD THIS
        
        save_output(distances, output_dir, "dual_dist_result", exp_name)
        logger.info("Distance results saved!")  # ADD THIS
    except Exception as e:
        logger.error(f"ERROR in run_dual_label: {e}")
        logger.error(traceback.format_exc())
        raise

def run_mrna_tracking(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Runs the mRNA Particle Tracking step.
    
    *** PLACEHOLDER: You must implement: ***
    1.  `img_proc.run_mrna_tracking()` in ImageProcessor
    2.  `img_proc.get_mrna_tracking_results()` in ImageProcessor
    3.  `['mrna_tracking']` section in your config file
    """
    logger.info("--- Running mRNA Particle Tracking ---")
    
    # 1. Run the new tracking method
    # This method will need to use the refined splines
    # (e.g., img_proc._get_ch1_refined_bsplines()) as input.
    # img_proc.run_mrna_tracking() 
    logger.warning("run_mrna_tracking() is a placeholder. Please implement in ImageProcessor.")

    # 2. Get the new output subdirectory
    # try:
    #     mrna_subfolder = cfg_dict['mrna_tracking']['output subdirectory']
    #     output_dir = out_base / mrna_subfolder
    #     output_dir.mkdir(parents=True, exist_ok=True)
    # except KeyError:
    #     logger.error("Config file is missing ['mrna_tracking']['output subdirectory']")
    #     return

    # 3. Save the new results
    # save_output(img_proc.get_mrna_tracking_results(), output_dir, "mrna_track_results", exp_name)
    logger.warning("Skipping save for mRNA tracking (placeholder).")

def print_global_stats_to_log(title, stats_dict):
    """Prints a formatted summary of the global stats dictionaries."""
    print(f"\n--- {title} (N={stats_dict['n']}) ---")
    if stats_dict['n'] == 0:
        print("  No data found for this view.")
        return
        
    print(f"  Angle Delta (deg):   Mean = {stats_dict['angle_mean']:.4f},  StdDev = {stats_dict['angle_std']:.4f}")
    print(f"  Scale Delta (unit):  Mean = {stats_dict['scale_mean']:.4f},  StdDev = {stats_dict['scale_std']:.4f}")
    print(f"  Shift-Y Delta (px):  Mean = {stats_dict['rdif_y_mean']:.4f},  StdDev = {stats_dict['rdif_y_std']:.4f}")
    print(f"  Shift-X Delta (px):  Mean = {stats_dict['rdif_x_mean']:.4f},  StdDev = {stats_dict['rdif_x_std']:.4f}")
    print(f"  Combined Precision (sigma_reg) = {stats_dict['sigma_reg']:.4f}")

if __name__ == "__main__":
    # --- ARGPARSE SETUP --- #
    # handles CL input
    parser = argparse.ArgumentParser(description="Process a single experiment for the yeast pipeline.")
    parser.add_argument("job_index", type=int, help="Job array index (1-based).")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    parser.add_argument("--rerun", action="store_true", help="Force re-run all steps, ignoring checkpoints.")
    args = parser.parse_args()

    # --- SETUP LOGGING --- #
    # starts at 1 because jobs start at 1
    log_format = f'%(asctime)s - JOB {args.job_index} - [%(name)s] - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, 
                        format=log_format,
                        stream=sys.stdout)
    global logger
    logger = logging.getLogger(__name__) # Get logger for this main script
    logger.info(f"Script started with job_index={args.job_index}, config={args.config_path}, rerun={args.rerun}")

    # --- JOB ARRAY SETUP ---
    # Get the job index from the command line
    # Default to 0 for local testing
    # Check for all required arguments
    if len(sys.argv) < 3: # script_name, job_index, config_path
        print("ERROR: Missing command-line arguments.")
        print("Usage: python main_cluster_dual_label.py <job_index> <config_path>")
        sys.exit(1)
    try: 


        # Initialize the pipeline
        pipeline = ImagingPipeline(args.config_path)
        global_config = pipeline.retrieve_pipe_config("pipe globals")
        # making image processing specific configuration dictionary from configuration loaded into pipeline
        img_proc_cfg_dict = pipeline.retrieve_pipe_config("image processor")
        # Load all experiment data
        # TODO change so only the current experiment loads ...
        # ??? or maybe it could be set up so an overall pipeline manages all the img_proc instances ... maybe...?
        #           except they are all seperate jobs. So need to consider how to manage that as a flow
        all_experiments = pipeline.get_experiments()
        experiment_names = list(all_experiments.keys())

        # Use 0-based index for list access
        job_index_0based = args.job_index - 1
        # Check if the job_index is valid
        if job_index_0based >= len(experiment_names):
            logger.error(f"Job index {args.job_index} is out of bounds. Found {len(experiment_names)} experiments.")
            sys.exit(1)

        # Select the specific experiment and its FoVs for this job
        experiment_name = experiment_names[job_index_0based]
        fov_list = all_experiments[experiment_name]

        logger.info(f"--- Processing Experiment (Job Index {args.job_index}): {experiment_name} ---")

        output_base_dir = Path(global_config['directories']['output root']) # Define the base output directory

        # --- DEFINING CHECKPOINT PATHS --- #
        chkpt_dir = output_base_dir / "checkpoints" / experiment_name
        chkpt_init_ne = chkpt_dir / "state_after_init_ne.pkl"
        chkpt_reg = chkpt_dir / "state_after_registration.pkl"
        chkpt_filter = chkpt_dir / "state_after_filtering.pkl"
        chkpt_refine = chkpt_dir / "state_after_refinement.pkl"
        chkpt_bridge = chkpt_dir / "state_after_bridging.pkl"

        if not fov_list:
            logger.warning(f"No FoVs found for experiment {experiment_name}. Exiting.")
            sys.exit(0)
        
        # --- CHECKPOINT / RESTART LOGIC --- #

        img_proc = None # Initialize as None

        # If --rerun is used, clear all old checkpoints
        if args.rerun:
            logger.warning("RERUN flag detected. Clearing all old checkpoints.")
            for chkpt in [chkpt_init_ne, chkpt_reg, chkpt_filter, chkpt_refine, chkpt_bridge]:
                if chkpt.exists():
                    chkpt.unlink() # Delete the file
        
        # --- Load the latest state (if possible) --- #
        else:
            logger.info("Checking for existing checkpoints...")
            for chkpt in [chkpt_bridge, chkpt_refine, chkpt_filter, chkpt_reg, chkpt_init_ne]:
            # Try to load from each checkpoint, latest to earliest
                if img_proc is None:
                    img_proc = load_checkpoint(chkpt)
                    if img_proc:
                        logger.info(f"Loaded from checkpoint: {chkpt}")

        if img_proc is not None:
            # Diagnostic: Check what's actually in the loaded object
            ch1_refined = img_proc._get_ch1_refined_bsplines()
            ch2_refined = img_proc._get_ch2_refined_bsplines()
            ch1_bridged = img_proc._get_ch1_bridged_splines()
            ch2_bridged = img_proc._get_ch2_bridged_splines()
            
            logger.info(f"DIAGNOSTIC - Loaded object contents:")
            logger.info(f"  Ch1 refined: {len(ch1_refined) if ch1_refined else 0} FoVs")
            logger.info(f"  Ch2 refined: {len(ch2_refined) if ch2_refined else 0} FoVs")
            logger.info(f"  Ch1 bridged: {len(ch1_bridged) if ch1_bridged else 0} FoVs")
            logger.info(f"  Ch2 bridged: {len(ch2_bridged) if ch2_bridged else 0} FoVs")
        
            if ch1_refined:
                for fov_id, ne_labels in ch1_refined.items():
                    logger.info(f"    Ch1 FoV {fov_id}: {len(ne_labels) if ne_labels else 0} NE labels")
            if ch2_refined:
                for fov_id, ne_labels in ch2_refined.items():
                    logger.info(f"    Ch2 FoV {fov_id}: {len(ne_labels) if ne_labels else 0} NE labels")

        if img_proc is None:
            # --- Start from scratch (if state not restored) ---
            img_proc = ImageProcessor(config_dict=img_proc_cfg_dict, FoV_dict=fov_list, device=pipeline.get_device())
            pipeline.add_img_processor(f'{experiment_name}', img_proc) # Add to pipeline
        else:
            logger.info("Successfully loaded ImageProcessor state from checkpoint.")
            # We still need to add the loaded object to the pipeline
            pipeline.add_img_processor(f'{experiment_name}', img_proc)

        # --- Get the pipeline mode ---
        full_config = pipeline.get_config()
        # Read the mode from the config, default to "dual_label"
        pipeline_mode = full_config.get("pipeline foreman", {}).get("pipeline_mode", "dual_label")
        logger.info(f"PIPELINE MODE: {pipeline_mode}")

        # --- Run Pipeline --- #

        try:
            if chkpt_init_ne.exists() and not args.rerun:
                logger.info("Skipping Initial NE Detection (checkpoint exists).")
            else:
                run_initial_ne(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
                save_checkpoint(img_proc, chkpt_init_ne) # Save state
                logger.info(f"Saved checkpoint: {chkpt_init_ne}")

            if chkpt_reg.exists() and not args.rerun:
                logger.info("Skipping Image Registration (checkpoint exists).")
            else:
                run_registration(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
                save_checkpoint(img_proc, chkpt_reg) # Save state
                logger.info(f"Saved checkpoint: {chkpt_reg}")
            
            if chkpt_filter.exists() and not args.rerun:
                logger.info("Skipping Filtering (checkpoint exists).")
            else:
                # Call the updated function with all necessary arguments
                run_filtering(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
                save_checkpoint(img_proc, chkpt_filter) # Save state
                logger.info(f"Saved checkpoint: {chkpt_filter}")
            
            if chkpt_refine.exists() and not args.rerun:
                logger.info("Skipping Spline Refinement (checkpoint exists).")
            else:
                run_refinement(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
                save_checkpoint(img_proc, chkpt_refine) # Save state
                logger.info(f"Saved checkpoint: {chkpt_refine}")
            run_bridging_flag = img_proc_cfg_dict.get('ne_fit', {}).get('run_bezier_bridging', False)
            
            if img_proc_cfg_dict.get('ne_fit', {}).get('run_bezier_bridging', False):
                if not chkpt_bridge.exists() or args.rerun:
                    run_bridging(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
                    save_checkpoint(img_proc, chkpt_bridge)

            # --- CONDITIONAL FINAL STEP(S) --- #
            # pipeline_mode based on type of experiment:
            #   dual label versus mRNA tracking relative to an NPC label
            
            if pipeline_mode == "dual_label":
                logger.info("Running final step: Dual Label Calculations.")
                run_dual_label(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            
            elif pipeline_mode == "mrna_tracking":
                logger.info("Running final step: mRNA Particle Tracking.")
                run_mrna_tracking(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            
            else:
                logger.error(f"Unknown pipeline_mode: '{pipeline_mode}'. No final step executed.")

        except Exception as e:
            logger.error(f"A critical error occurred during pipeline execution for {experiment_name}.")
            logger.error(traceback.format_exc()) # Log the full traceback
            sys.exit(1) # Exit with error

        all_logs = []
        
        # Check if get_img_processors exists, otherwise get from the main 'pipeline' variable
        if hasattr(pipeline, 'get_img_processors'):
            processors_dict = pipeline.get_img_processors()
            if processors_dict: # Check if it's not None or empty
                # We just have one processor for this job, so get its logs
                if experiment_name in processors_dict:
                    proc = processors_dict[experiment_name]
                    all_logs.extend(proc.get_segment_logs())
        
        # --- SAVE LOGS --- #
        all_logs = []
        
        if all_logs:
            logger.info("--- Compiling segment refinement log... ---")
            df = pd.DataFrame(all_logs)
            
            # Reorder columns for clarity
            cols = ['fov_id', 'channel', 'ne_label', 'seg_label', 'profile_index', 'status', 'num_total_points']
            # Filter for only columns that exist
            cols_to_use = [col for col in cols if col in df.columns]
            df = df[cols_to_use]
            
            # Save to the main output directory
            log_save_path = output_base_dir / f"segment_refinement_log_{experiment_name}.csv"
            df.to_csv(log_save_path, index=False)
            logger.info(f"--- Segment log saved to {log_save_path} ---")
        else:
            logger.info("--- No segment logs were generated. ---")

        # ??? why did I do this? there was a reason... a future related reason, which is about all I recall
        pipeline.add_experiments(all_experiments)
        
        print(f"\n--- Job for experiment {experiment_name} finished. ---")

    except Exception as e:
        logger.error(f"An unexpected error occurred during *setup* for {experiment_name}.")
        logger.error(traceback.format_exc())
        sys.exit(1)

    