"""
Cluster Execution Script for Imaging Pipeline
=============================================

This script serves as the primary entry point for running the imaging pipeline in a distributed cluster environment. It parses command-line arguments to identify the specific job index (experiment) to process, loads the configuration, and orchestrates the sequential execution of pipeline steps.

It includes robust checkpointing mechanisms to save the state of the ``ImageProcessor`` after each major step (Responsivity, Detection, Registration, Filtering, Refinement, Bridging), allowing jobs to be resumed or restarted efficiently.

Usage
-----
.. code-block:: bash

    python main_cluster.py <job_index> <config_path> [--rerun]

Arguments
---------
job_index : int
    1-based index corresponding to the experiment to process from the configuration list.
config_path : str
    Path to the master JSON configuration file.
rerun : bool, optional
    If 'true', '1', or 'yes', forces a complete re-run by deleting existing checkpoints.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import traceback
import os

import pickle

# import pipeline pieces
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor
from tools.output_handling import save_output
from utils.report_visualizer import ReportVisualizer

# --- Helper Functions for Checkpointing ---

def load_checkpoint(path: Path):
    """
    Loads a pickled object from a checkpoint file.

    Parameters
    ----------
    path : pathlib.Path
        The file path to the pickle checkpoint.

    Returns
    -------
    object or None
        The loaded object if the file exists and is valid; otherwise None.
    """
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"ERROR: Could not load checkpoint {path}. Error: {e}")
        return None

def save_checkpoint(obj: object, path: Path):
    """
    Saves an object to a pickle file using the highest protocol.

    Parameters
    ----------
    obj : object
        The Python object to pickle (typically an ``ImageProcessor`` instance).
    path : pathlib.Path
        The destination file path. Parent directories are created if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# --- Pipeline Steps ---
# METHODS - considered attaching this to the pipeline rather than the processor
#but that does not account for different cameras being used for different
# experimentsltiple cameras 
def run_responsivity(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Executes the Detector Responsivity Calibration step.

    Calculates gain, offset, and read noise for the detector channels and saves the calibration statistics to disk.

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The 'image processor' sub-dictionary from the configuration.
    out_base : pathlib.Path
        The root output directory for the experiment.
    exp_name : str
        The unique name of the current experiment.
    """
    logger.info("--- Running Detector Responsivity Calibration ---")
    resp_subfolder = cfg_dict['responsivity']['output subdirectory']
    output_dir = out_base / resp_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_proc.determine_responsivity()
    
    # Save calibration stats
    cal_stats = img_proc._get_drd_stats()
    meanvar_data = img_proc._get_drd_meanvar()
    
    save_output(cal_stats, output_dir, "calibration_stats", exp_name)
    save_output(meanvar_data, output_dir, "meanvar_data", exp_name)
    
    logger.info("Responsivity > Channel 1:")
    logger.info(f"  Camera gain: {cal_stats['ch1']['gain']:.4f} e-/ADU")
    logger.info(f"  Dark offset: {cal_stats['ch1']['offset']:.2f} ADU")
    logger.info(f"  Read noise: {cal_stats['ch1']['dk_noise']:.4f} photons")
    logger.info("Responsivity > Channel 2:")
    logger.info(f"  Camera gain: {cal_stats['ch2']['gain']:.4f} e-/ADU")
    logger.info(f"  Dark offset: {cal_stats['ch2']['offset']:.2f} ADU")
    logger.info(f"  Read noise: {cal_stats['ch2']['dk_noise']:.4f} photons")

def run_initial_ne(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Executes the Initial Nuclear Envelope (NE) Detection step (via labeled Nups).

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The configuration dictionary.
    out_base : pathlib.Path
        The root output directory.
    exp_name : str
        The experiment name.
    """
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
    """
    Executes the Image Registration step.

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The configuration dictionary.
    out_base : pathlib.Path
        The root output directory.
    exp_name : str
        The experiment name.
    """
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
    Runs the Registration Stability Analysis, saves reports, and filters the ImageProcessor's NE pair map.

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The configuration dictionary.
    out_base : pathlib.Path
        The root output directory.
    exp_name : str
        The experiment name.
    """
    logger.info("--- Running Registration Stability Analysis & Filtering ---")
    reg_subfolder = cfg_dict['registration']['output subdirectory']
    output_dir = out_base / reg_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- FoV METHOD (Population) ---
    # "Did the whole FoV move more than the experiment average?"
    logger.info("Running FoV-Level (Population) Stability Analysis...")
    fov_report, fov_sigma = img_proc.analyze_fov_reg_stability(prune=False)
    
    save_output(fov_report, output_dir, "report_stability_FOV", exp_name)
    save_output({"population_sigma": fov_sigma}, output_dir, "stats_stability_FOV", exp_name)

    # --- INDIVIDUAL METHOD (Temporal / Local) ---
    # "Did this specific nucleus move more than its specific noise?"
    logger.info("Running Individual Nucleus (Temporal) Stability Analysis...")
    ne_report, active_pairs = img_proc.analyze_ne_reg_stability(prune=False)
    
    save_output(ne_report, output_dir, "report_stability_INDIVIDUAL", exp_name)
    save_output(active_pairs, output_dir, "final_paired_nuclei_map", exp_name, is_pickle=True)

    comp_report = img_proc.compare_stability_reports(fov_report, ne_report)

    save_output(comp_report, output_dir, "report_stability_COMPARISON", exp_name)

    exp_stats = comp_report['experiment']
    logger.info("--- Comparison Summary ---")
    logger.info(f"  Total Detected: {exp_stats['total_detected']}")
    logger.info(f"  Loss (FoV Method): {exp_stats['total_pruned_fov_method']}")
    logger.info(f"  Loss (NE Method):  {exp_stats['total_pruned_ne_method']}")
    logger.info(f"  Net Difference:    {exp_stats['net_loss_diff']} (Positive = FoV is harsher)")

    filter_mode = cfg_dict.get('registration', {}).get('stability_filter_mode', 'ne').lower()
    logger.info(f"Active Filter Mode: {filter_mode.upper()}")
    
    active_pairs = {}
    
    if filter_mode == 'fov':
        # "Nuke" Mode
        logger.info("  >> Executing FoV-Level Pruning...")
        _, active_pairs = img_proc.analyze_fov_reg_stability(prune=True)
        
    elif filter_mode == 'ne' or filter_mode == 'individual':
        # "Sniper" Mode
        logger.info("  >> Executing NE-Level Pruning...")
        _, active_pairs = img_proc.analyze_ne_reg_stability(prune=True)
        
    else:
        logger.warning(f"Unknown filter mode '{filter_mode}'. No pruning performed.")
        active_pairs = img_proc.get_ne_pairs_by_FoV()

    save_output(active_pairs, output_dir, "final_paired_nuclei_map", exp_name, is_pickle=True)
    logger.info("--- Registration Analysis & Filtering Complete ---")

def run_refinement(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Tells the ImageProcessor to run the robust, restartable spline refinement step.
    
    This function is responsible for creating the paths, and the
    ImageProcessor handles the loop and partial file I/O.

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The configuration dictionary.
    out_base : pathlib.Path
        The root output directory.
    exp_name : str
        The experiment name.
    """
    logger.info("--- Preparing for Spline Refinement ---")
    refine_subfolder = cfg_dict['ne_fit']['refinement']['output subdirectory']
    output_dir = out_base / refine_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_ch1_path = output_dir / f"refine_results_ch1_{exp_name}_IN_PROGRESS.pkl"
    partial_ch2_path = output_dir / f"refine_results_ch2_{exp_name}_IN_PROGRESS.pkl"
    img_proc.run_ne_refinement(partial_ch1_path=partial_ch1_path,  
                                partial_ch2_path=partial_ch2_path,
                                load_checkpoint_func=load_checkpoint, save_checkpoint_func=save_checkpoint)
    logger.info("Saving final (complete) refinement files...")
    save_output(img_proc._get_ch1_refined_bsplines(), output_dir, "refine_results_ch1", exp_name, is_pickle=True)
    save_output(img_proc._get_ch2_refined_bsplines(), output_dir, "refine_results_ch2", exp_name, is_pickle=True)

    logger.info("--- Spline Refinement Step Complete ---")

def run_bridging(img_proc: ImageProcessor, cfg_dict: dict, out_base: Path, exp_name: str):
    """
    Runs the (optional) Bezier bridging step to connect refined spline segments
    into whole, closed membranes.

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The configuration dictionary.
    out_base : pathlib.Path
        The root output directory.
    exp_name : str
        The experiment name.
    """
    logger.info("--- Running Bezier Spline Bridging ---")
    refine_subfolder = cfg_dict['ne_fit']['refinement']['output subdirectory']
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
    """Runs the Dual Label Difference Calculations step.

    Parameters
    ----------
    img_proc : ImageProcessor
        The active image processing object.
    cfg_dict : dict
        The configuration dictionary.
    out_base : pathlib.Path
        The root output directory.
    exp_name : str
        The experiment name.
    """
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
    print(f"--- {title} (N={stats_dict['n']}) ---")
    if stats_dict['n'] == 0:
        print("  No data found for this view.")
        return
    print(f"  Angle Delta (deg):   Mean = {stats_dict['angle_mean']:.4f},  StdDev = {stats_dict['angle_std']:.4f}")
    print(f"  Scale Delta (unit):  Mean = {stats_dict['scale_mean']:.4f},  StdDev = {stats_dict['scale_std']:.4f}")
    print(f"  Shift-Y Delta (px):  Mean = {stats_dict['rdif_y_mean']:.4f},  StdDev = {stats_dict['rdif_y_std']:.4f}")
    print(f"  Shift-X Delta (px):  Mean = {stats_dict['rdif_x_mean']:.4f},  StdDev = {stats_dict['rdif_x_std']:.4f}")
    print(f"  Combined Precision (sigma_reg) = {stats_dict['sigma_reg']:.4f}")

def run_visualization(cfg_dict: dict, out_base: Path, exp_name: str):
    logger.info("--- Running Report Visualization ---")
    
    # Define paths based on your config structure
    reg_subfolder = cfg_dict['registration']['output subdirectory']
    reg_dir = out_base / reg_subfolder
    
    # File definitions
    path_raw = reg_dir / f"reg_results_mode1_{exp_name}.json"
    path_ne = reg_dir / f"report_stability_NE_LABEL_{exp_name}.json"
    path_fov = reg_dir / f"report_stability_FOV_{exp_name}.json"
    
    # Initialize Visualizer
    viz = ReportVisualizer(cfg_dict, output_dir=reg_dir)
    
    # Generate Graphs
    try:
        viz.generate_dashboard(path_raw, path_ne, path_fov, exp_name=exp_name)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

# --- Main Function --- #

def run_pipeline(job_index: int, config_path: str, rerun: bool = False):
    log_format = f'%(asctime)s - JOB {job_index} - [%(name)s] - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
    global logger
    logger = logging.getLogger(__name__)
    logger.info(f"Script started with job_index={job_index}, config={config_path}, rerun={rerun}")

    pipeline = ImagingPipeline(config_path)
    global_config = pipeline.retrieve_pipe_config("pipe globals")
    img_proc_cfg_dict = pipeline.retrieve_pipe_config("image processor")
    all_experiments = pipeline.get_experiments()
    experiment_names = list(all_experiments.keys())
    job_index_0based = job_index - 1
    if job_index_0based >= len(experiment_names):
        logger.error(f"Job index {job_index} is out of bounds. Found {len(experiment_names)} experiments.")
        return
    experiment_name = experiment_names[job_index_0based]
    fov_list = all_experiments[experiment_name]
    output_base_dir = Path(global_config['directories']['output root'])
    chkpt_dir = output_base_dir / "checkpoints" / experiment_name
    chkpt_responsivity = chkpt_dir / "state_after_responsivity.pkl"
    chkpt_init_ne = chkpt_dir / "state_after_init_ne.pkl"
    chkpt_reg = chkpt_dir / "state_after_registration.pkl"
    chkpt_filter = chkpt_dir / "state_after_filtering.pkl"
    chkpt_refine = chkpt_dir / "state_after_refinement.pkl"
    chkpt_bridge = chkpt_dir / "state_after_bridging.pkl"

    # Add file logging
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    log_file = output_base_dir / f'{timestamp}_pipeline_run_{experiment_name}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    img_proc = None
    if rerun:
        logger.warning("RERUN flag detected. Clearing all old checkpoints.")
        for chkpt in [chkpt_init_ne, chkpt_reg, chkpt_filter, chkpt_refine, chkpt_bridge]:
            if chkpt.exists():
                chkpt.unlink()
    else:
        logger.info("Checking for existing checkpoints...")
        for chkpt in [chkpt_bridge, chkpt_refine, chkpt_filter, chkpt_reg, chkpt_init_ne]:
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
        img_proc = ImageProcessor(config_dict=img_proc_cfg_dict, FoV_dict=fov_list, device=pipeline.get_device())
        pipeline.add_img_processor(f'{experiment_name}', img_proc)
    else:
        pipeline.add_img_processor(f'{experiment_name}', img_proc)

    pipeline_mode = pipeline.get_config().get("pipeline foreman", {}).get("pipeline_mode", "dual_label")
    logger.info(f"PIPELINE MODE: {pipeline_mode}")

    try:
        if not chkpt_responsivity.exists() or rerun:
            run_responsivity(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            save_checkpoint(img_proc, chkpt_responsivity)
        if not chkpt_init_ne.exists() or rerun:
            run_initial_ne(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            save_checkpoint(img_proc, chkpt_init_ne)
        if not chkpt_reg.exists() or rerun:
            run_registration(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            save_checkpoint(img_proc, chkpt_reg)
        if not chkpt_filter.exists() or rerun:
            run_filtering(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            save_checkpoint(img_proc, chkpt_filter)
        if not chkpt_refine.exists() or rerun:
            run_refinement(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
            save_checkpoint(img_proc, chkpt_refine)
        if img_proc_cfg_dict.get('ne_fit', {}).get('run_bezier_bridging', False):
            if not chkpt_bridge.exists() or rerun:
                run_bridging(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
                save_checkpoint(img_proc, chkpt_bridge)
        if pipeline_mode == "dual_label":
            run_dual_label(img_proc, img_proc_cfg_dict, output_base_dir, experiment_name)
        elif pipeline_mode == "mrna_tracking":
            logger.warning("mRNA tracking placeholder not implemented.")
        else:
            logger.error(f"Unknown pipeline_mode: '{pipeline_mode}'.")

        run_visualization(img_proc_cfg_dict, output_base_dir, experiment_name)
    except Exception as e:
        logger.error(f"A critical error occurred during pipeline execution for {experiment_name}.")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Parse command-line arguments for cluster execution
    # Format: python main_cluster.py <job_index> <config_path> [--rerun]
    
    if len(sys.argv) < 3:
        print("ERROR: Insufficient arguments.")
        print("Usage: python main_cluster.py <job_index> <config_path> [<rerun_flag>]")
        print("  job_index: 1-based job array index")
        print("  config_path: Path to configuration JSON file")
        print("  rerun_flag: 'true' to force re-run (optional, default: false)")
        sys.exit(1)
    
    try:
        job_index = int(sys.argv[1])
        config_path = sys.argv[2]
        rerun = False
        
        # Check for rerun flag (flexible parsing)
        if len(sys.argv) > 3:
            rerun_arg = sys.argv[3].lower()
            rerun = rerun_arg in ['true', '1', 'yes', 'rerun']
        
        run_pipeline(job_index=job_index, config_path=config_path, rerun=rerun)
        
    except ValueError as e:
        print(f"ERROR: Invalid job_index. Must be an integer. Got: {sys.argv[1]}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error during argument parsing: {e}")
        traceback.print_exc()
        sys.exit(1)