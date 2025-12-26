"""
Created on Fri May  9 12:55:24 2025

@author: jctourtellotte
"""

# import external packages
import sys
import traceback
import os
from pathlib import Path
from datetime import datetime
import copy

import pickle
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

# import supporting pipeline classes (if any)
from utils.ne_improved_pairing import improved_ne_crop_and_pair
from tools.utility_functions import dict_update_or_replace, find_shared_keys, filter_channels_by_label_map
from tools.geom_tools import bspline_transformation, reconstruct_periodic_spline
from utils.spline_bridging import bridge_refined_splines
from utils.responsivity import detector_responsivity_determ
from utils.img_registration import image_registration, calculate_mode_reg_stats, calculate_drift_map, build_ne_stability_report, calculate_fov_population_stats, build_fov_stability_report, generate_stability_comparison_report
from utils.npc_detect_initial import detect_npc
from utils.ne_dual_labels import match_ne_labels_by_iou, calc_dual_distances
from utils.npc_spline_refinement import NESplineRefiner
from utils.distance_logging import DistanceCalculationLogger

logger = logging.getLogger(__name__)

# from utils.npc_spline_refinement import NESplineRefiner
# TODO update ALL times a path is used to load something to os.join or what have you
class ImageProcessor:

    # --- Primary Image Processing Functions --- #
    def __init__(self, config_dict, FoV_dict, device = torch.device('cpu'), threshold_regist = 0.5):
        self._set_cfg(config_dict)

        self._set_current_device(device)
        self._set_FoV_collection_dict(FoV_dict)

        self._drd_stats = None
        self._drd_meanvar = None

        # Bounds and initial guess for GLRT detector
        self._bounds_glrt = [
            [0, self._cfg['roisize'] - 1],  # x bounds
            [0, self._cfg['roisize']],      # y bounds
            [0, 1e9],                 # Photons bounds
            [0, 1e6]                  # Background bounds
        ]

        self._initial_guess = [
            self._cfg['roisize'] / 2,       # x initial guess
            self._cfg['roisize'] / 2,       # y initial guess
            0,                        # Photons initial guess
            60                        # Background initial guess
        ]

        self._reg_diff_mode1 = {}
        self._reg_diff_mode2 = {}
        self._drift_correction = {}

        self._ch1_cropped_imgs = {}
        self._ch1_init_bsplines = {}

        self._ch2_cropped_imgs = {}
        self._ch2_init_bsplines = {}

        self._ne_pairs_by_FoV = {}
        self._dual_distances_by_FoV = {}

        self._ch1_refined_bsplines = {}
        self._ch2_refined_bsplines = {}

        self._ch1_bridged_splines = {}
        self._ch2_bridged_splines = {}


        self._segment_logs = []
        
        print('Image Processor initiated')

    

    

    # --- Pipes --- #
    # ----- Responsivity ---- #
    # SELF - extend to N channels (not a priority for current use case)
    def determine_responsivity(self):
        # Detector Responsivity Determination
        #   uses bright (gain) and dark (offset) image pair
        #   return calibration statistics as well as mean/variance
        cfg = self._get_cfg()
        imaging_directory = cfg.get("directories").get("imaging root")
        responsivity_cfg = cfg['responsivity']
        responsivity_subdir = responsivity_cfg['input subdirectory']

        ch1_drd_stats, ch1_meanvar_plt_stats = \
                detector_responsivity_determ(
                    os.path.join(imaging_directory, responsivity_subdir, responsivity_cfg['ch1']['bright']),
                    os.path.join(imaging_directory, responsivity_subdir, responsivity_cfg['ch1']['dark'])
                    )
        
        ch2_drd_stats, ch2_meanvar_plt_stats = \
                detector_responsivity_determ(
                    os.path.join(imaging_directory, responsivity_subdir, responsivity_cfg['ch2']['bright']),
                    os.path.join(imaging_directory, responsivity_subdir, responsivity_cfg['ch2']['dark'])
                    )
        drd_stats = {   'ch1': ch1_drd_stats,
                        'ch2': ch2_drd_stats}
        meanvar_plt_stats = {   'ch1': ch1_meanvar_plt_stats,
                                'ch2': ch2_meanvar_plt_stats}
                
        self._set_drd_stats(drd_stats)
        self._set_drd_meanvar(meanvar_plt_stats)


    # ----- Initial NE Fitting ---- #
    # sets initial fit in motion and directs the results to the appropriate class variables
    def run_init_ne_detection(self, FoV_list=[], add_to_existing = True):
        # if FoV_list is non-empty (ie. contains 1+ FoV_id strings) then this runs for that subset of FoV_ids, otherwise, this step is run for all FoV loaded into existing FoV collection for this ImageProcessor instance
        print("--- Detecting Nuclear Envelope ----")
        if len(FoV_list) != 0:
            FoV_dict = self._get_FoV_collection_dict()
            FoV_for_npc_detection = \
                {FoV_id: FoV_dict[FoV_id] for FoV_id in FoV_list if FoV_id in FoV_dict}
        else:
            FoV_for_npc_detection = self._get_FoV_collection_dict()
        
        if not self._get_cfg()["ne_dual_label"]:
            ne_img_crop_results, ne_init_bspline_results = \
                self._run_ne_init_fit(FoV_for_npc_detection, img_track = 'fn_track_ch1')
            
            self._set_ch1_cropped_imgs( ne_img_crop_results )
            self._set_ch1_init_bsplines( ne_init_bspline_results )
            self._set_ch2_cropped_imgs( ne_img_crop_results )
            self._set_ch2_init_bsplines( ne_init_bspline_results )
        else:
            # Run detection for both channels (NOW returns labeled masks too!)
            ch1_img_crop_results, ch1_init_bspline_results, ch1_labeled_masks = \
                self._run_ne_init_fit(FoV_for_npc_detection, img_track = 'fn_track_ch1', id_suffix = '_ch1')
            
            self._set_ch1_cropped_imgs( ch1_img_crop_results )
            self._set_ch1_init_bsplines( ch1_init_bspline_results )

            ch2_img_crop_results, ch2_init_bspline_results, ch2_labeled_masks = \
                self._run_ne_init_fit(FoV_for_npc_detection, img_track = 'fn_track_ch2', id_suffix = '_ch2')
            
            self._set_ch2_cropped_imgs( ch2_img_crop_results )
            self._set_ch2_init_bsplines( ch2_init_bspline_results )

# TASK: tie these configs to the config json
        ch1_ch2_ne_label_pairs = {}
        for fov_id in ch1_labeled_masks.keys():
            pairs = improved_ne_crop_and_pair(
                ch1_labeled_img=ch1_labeled_masks[fov_id],
                ch2_labeled_img=ch2_labeled_masks[fov_id],
                ch1_existing_crops=ch1_img_crop_results[fov_id],
                ch2_existing_crops=ch2_img_crop_results[fov_id],
                min_iou_tight=0.7,
                max_centroid_distance_pixels=10
            )
            ch1_ch2_ne_label_pairs[fov_id] = pairs
        
        # DEBUG: Show initial pairing results
        print("\n" + "="*60)
        print("INITIAL NE PAIRING RESULTS")
        print("="*60)
        for fov_id_debug, pairs_debug in ch1_ch2_ne_label_pairs.items():
            print(f"FoV {fov_id_debug}: {len(pairs_debug)} pairs found")
            for ch1_label, ch2_label in pairs_debug.items():
                print(f"  Ch1 NE {ch1_label} ↔ Ch2 NE {ch2_label}")
        print("="*60 + "\n")

        self._set_ne_pairs_by_FoV(ch1_ch2_ne_label_pairs)
        
        # Initialize Ledger for DETECTED nuclei
        if not hasattr(self, '_ne_status_ledger'):
            self._ne_status_ledger = {}

        # Log Channel 1
        ch1_splines = self._get_ch1_init_bsplines()
        if ch1_splines:
            for fov_id, ne_dict in ch1_splines.items():
                for ne_label in ne_dict.keys():
                    self._log_status_change('ch1', fov_id, ne_label, 'DETECTED', 'initial_detection')

        # Log Channel 2 (if exists)
        ch2_splines = self._get_ch2_init_bsplines()
        if ch2_splines:
            for fov_id, ne_dict in ch2_splines.items():
                for ne_label in ne_dict.keys():
                    self._log_status_change('ch2', fov_id, ne_label, 'DETECTED', 'initial_detection')

    # orchestrates the steps for initial fitting for each entry in an FoV_dict
    def _run_ne_init_fit(self, FoV_dict, img_track = 'fn_track_ch1', id_suffix = ''):
        ne_trained_model = os.path.join(self._cfg['directories']['model root'], self._cfg['model_NE'])
        current_device = self._current_device
        
        ne_fit_cfg = self._cfg['ne_fit']
        ne_fit_initial = ne_fit_cfg['initial']
        frame_range = ne_fit_cfg['frame_range']
        bbox_dim = ne_fit_cfg['bbox_dim']

        plot_test_imgs = ne_fit_initial.get('plot_test_imgs', False)
        use_merged_clusters = ne_fit_initial.get('use_merged_clusters', True)
        masking_threshold = ne_fit_initial.get('masking_threshold', 0.5)
        max_merge_dist = ne_fit_initial.get('max_merge_dist', 10)
        bspline_smoothing = ne_fit_initial.get('bspline_smoothing', 1.6)
        init_sampling_density = ne_fit_initial.get('init_sampling_density', 10)
        
        ne_img_crop_dict = {} # dict to store cropped images coordinates based on initial fit
        ne_bspline_dict = {} # dict to store initial bspline objects
        labeled_masks_dict = {}

        # run initial detection for each FoV
        for key, entry in FoV_dict.items():
            FoV_id = key
            track_path = os.path.join(entry['FoV_collection_path'], entry['imgs'][img_track])
            if not os.path.exists(track_path):
                print(f" --> SKIPPING: No npc channel track found for {FoV_id}.") 
                continue
            else:
                ne_img_crop, ne_bsplines, labeled_mask = \
                    detect_npc(
                        img_track_path = track_path,
                        frame_range = frame_range,
                        NE_model = ne_trained_model,
                        device = current_device,
                        init_sampling_density = init_sampling_density,
                        bspline_smoothing = bspline_smoothing,
                        FoV_id = FoV_id,
                        masking_threshold = masking_threshold,
                        bbox_dim = bbox_dim,
                        use_merged_clusters = use_merged_clusters,
                        max_merge_dist = max_merge_dist,
                        plot_test_imgs = plot_test_imgs,
                        id_suffix = id_suffix
                        )
                #TODO: catch if none detected

                ne_img_crop_dict.update({f'{FoV_id}' : ne_img_crop})
                ne_bspline_dict.update({f'{FoV_id}' : ne_bsplines})
                labeled_masks_dict.update({f'{FoV_id}' : labeled_mask})

        return ne_img_crop_dict, ne_bspline_dict, labeled_masks_dict

    # ----- Refinement of NE Fit ---- #
    # handles dual label situation based on "ne_pairs_by_FoV" created during initial NE fitting.

    def run_ne_refinement(self, partial_ch1_path, partial_ch2_path, load_checkpoint_func, save_checkpoint_func):
        """
        Runs the Spline Refinement step with robust, per-FoV checkpointing.
        The looping and partial saving logic is encapsulated here.
        """
        print("--- Refining NE Splines (with restartable loop) ---")
        
        # --- Load partial results if they exist ---
        # We use the I/O functions passed from the main script
        print("Loading partial refinement checkpoints (if any)...")
        # --- Load partial results if they exist ---
        logger.info("Loading partial refinement checkpoints (if any)...")
        ch1_refined_splines = load_checkpoint_func(partial_ch1_path) or {}
        ch2_refined_splines = load_checkpoint_func(partial_ch2_path) or {}
        
        # ADD THIS DIAGNOSTIC:
        logger.info(f"Loaded {len(ch1_refined_splines)} Ch1 FoVs from checkpoint")
        logger.info(f"Loaded {len(ch2_refined_splines)} Ch2 FoVs from checkpoint")

        # --- Get the full list of FoVs to process ---
        fov_id_list = self.get_fovs_for_refinement()
        if not fov_id_list:
            print("Warning: No FoVs found to refine. Skipping refinement step.")
            return

        print(f"Found {len(fov_id_list)} FoVs to process for refinement.")
        if ch1_refined_splines:
            print(f"Loaded {len(ch1_refined_splines)} already-processed FoVs (Ch1).")

        # --- Main restartable loop ---
        for fov_id in fov_id_list:
            # Check if this FoV is already done
            if fov_id in ch1_refined_splines:
                print(f"Skipping FoV {fov_id} (already refined).")
                continue
            
            try:
                # This is the compute-heavy step for ONE FoV
                ch1_fov_result, ch2_fov_result = self.refine_splines_for_fov(fov_id)
                
                # Add results to the dictionaries
                if ch1_fov_result:
                    ch1_refined_splines[fov_id] = ch1_fov_result
                if ch2_fov_result:
                    ch2_refined_splines[fov_id] = ch2_fov_result
                
                # --- Save checkpoint AFTER this FoV is processed ---
                if ch1_fov_result:
                    save_checkpoint_func(ch1_refined_splines, partial_ch1_path)
                if ch2_fov_result:
                    save_checkpoint_func(ch2_refined_splines, partial_ch2_path)
                
                print(f"Processed and checkpointed refinement for FoV: {fov_id}")

            except Exception as e:
                print(f"CRITICAL: Refinement failed for FoV {fov_id}. Error: {e}")
                traceback.print_exc()
                # We continue to the next FoV, this one failed but others might work

        # --- Finalization after loop ---
        print("Refinement loop complete. Storing final data in object...")
        
        # Set the final, complete dictionaries into the img_proc object
        # --- Finalization after loop ---
        logger.info("Refinement loop complete. Storing final data in object...")

        logger.info(f"DIAGNOSTIC - About to store in object:")
        logger.info(f"  ch1_refined_splines has {len(ch1_refined_splines)} FoVs")
        logger.info(f"  ch2_refined_splines has {len(ch2_refined_splines)} FoVs")

        # Set the final, complete dictionaries into the img_proc object
        self._set_ch1_refined_bsplines(ch1_refined_splines)
        self._set_ch2_refined_bsplines(ch2_refined_splines)

        logger.info(f"DIAGNOSTIC - After storing:")
        logger.info(f"  self._ch1_refined_bsplines has {len(self._ch1_refined_bsplines)} FoVs")
        logger.info(f"  self._ch2_refined_bsplines has {len(self._ch2_refined_bsplines)} FoVs")
        
        # --- Clean up IN-PROGRESS files ---
        try:
            if partial_ch1_path.exists():
                partial_ch1_path.unlink()
            if partial_ch2_path.exists():
                partial_ch2_path.unlink()
            print("Cleaned up in-progress refinement files.")
        except OSError as e:
            print(f"Warning: Could not delete in-progress files. Error: {e}")

    def refine_splines_for_fov(self, fov_id):
        """
        Runs spline refinement for ONLY the specified fov_id.
        This is called by the new run_ne_refinement loop.
        
        Returns:
            (dict, dict): A tuple of (ch1_refined_splines, ch2_refined_splines)
                            for this FoV. ch2_splines will be None if not dual-label.
        """
        print(f"--- Refining splines for FoV: {fov_id} ---")
        ch1_refined_output = None
        ch2_refined_output = None

        # Check if this image processor is handling a dual label situation
        if self._get_cfg().get('ne_dual_label'):
            label_map = self.get_ne_pairs_by_FoV()
            if fov_id not in label_map:
                print(f"Warning: {fov_id} not in label_map. Skipping.")
                return None, None
                
            # Filter all data for just this FoV
            ch1_splines_fov = {fov_id: self._get_ch1_init_bsplines().get(fov_id)}
            ch2_splines_fov = {fov_id: self._get_ch2_init_bsplines().get(fov_id)}
            ch1_crops_fov = {fov_id: self._get_ch1_cropped_imgs().get(fov_id)}
            ch2_crops_fov = {fov_id: self._get_ch2_cropped_imgs().get(fov_id)}
            fov_label_map = {fov_id: label_map.get(fov_id)}

            # Filter channels by the pair map
            paired_splines = filter_channels_by_label_map(fov_label_map, ch1_splines_fov, ch2_splines_fov)
            paired_crops = filter_channels_by_label_map(fov_label_map, ch1_crops_fov, ch2_crops_fov)

            # --- Refine Channel 1 for this FoV ---
            if paired_splines['ch1']:
                ch1_refined_output = self._run_refinement_for_fov(
                    img_track_key = 'fn_track_ch1',
                    initial_bsplines_dict = paired_splines['ch1'],
                    crop_box_dict = paired_crops['ch1'],
                    channel = 'ch1',
                    fov_id = fov_id
                )

            # --- Refine Channel 2 for this FoV ---
            if paired_splines['ch2']:
                ch2_refined_output = self._run_refinement_for_fov(
                    img_track_key='fn_track_ch2',
                    initial_bsplines_dict = paired_splines['ch2'],
                    crop_box_dict = paired_crops['ch2'],
                    channel = 'ch2',
                    fov_id = fov_id
                )
        
        else:
            # --- Single Label ---
            ch1_splines_fov = {fov_id: self._get_ch1_init_bsplines().get(fov_id)}
            ch1_crops_fov = {fov_id: self._get_ch1_cropped_imgs().get(fov_id)}

            if ch1_splines_fov:
                ch1_refined_output = self._run_refinement_for_fov(
                    img_track_key='fn_track_ch1',
                    initial_bsplines_dict=ch1_splines_fov,
                    crop_box_dict=ch1_crops_fov,
                    channel='ch1',
                    fov_id = fov_id
                )
        
        # Return the results for just this FoV
        logger.info(f"DIAGNOSTIC - refine_splines_for_fov({fov_id}) returning:")
        logger.info(f"  ch1_refined_output type: {type(ch1_refined_output)}")
        logger.info(f"  ch1_refined_output value: {ch1_refined_output is not None}")
        if ch1_refined_output:
            logger.info(f"  ch1_refined_output keys: {list(ch1_refined_output.keys())}")
        logger.info(f"  ch2_refined_output type: {type(ch2_refined_output)}")
        logger.info(f"  ch2_refined_output value: {ch2_refined_output is not None}")
        if ch2_refined_output:
            logger.info(f"  ch2_refined_output keys: {list(ch2_refined_output.keys())}")
        
        return ch1_refined_output, ch2_refined_output


    def _run_refinement_for_fov(self, img_track_key, initial_bsplines_dict, crop_box_dict, channel, fov_id):
        """
        Internal function that runs NESplineRefiner for a single FoV.
        
        Returns:
            dict: The refined spline dictionary for the given fov_id,
                    or None if it fails.
        """
        if fov_id not in self._FoV_collection_dict: 
            logger.error(f"Error: {fov_id} not in FoV collection.")
            return None
        if fov_id not in initial_bsplines_dict:
            logger.error(f"Error: {fov_id} has no initial splines to refine.")
            return None

        # Get data for the *specific* FoV
        initial_splines = initial_bsplines_dict[fov_id]

        logger.info(f"DIAGNOSTIC - _run_refinement_for_fov starting for {fov_id}/{channel}")
        logger.info(f"  Initial splines has {len(initial_splines) if initial_splines else 0} NE labels")
        if initial_splines:
            for ne_label, segments in initial_splines.items():
                logger.info(f"    NE {ne_label}: {len(segments) if segments else 0} segments")


        fov_entry = self._FoV_collection_dict[fov_id]
        img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs'][img_track_key])
        
        cal_stats = self._get_drd_stats()
        if cal_stats is not None:
            camera_gain = cal_stats.get('gain', 1.0)
            print(f"Using calibrated camera gain: {camera_gain:.4f} e-/ADU")
        else:
            camera_gain = 1.0
            print("WARNING: Calibration not run. Using default gain=1.0 (Poisson filtering may be inaccurate)")


        refiner = NESplineRefiner(
            channel = channel,
            img_path=img_path,
            fov_id=fov_id,
            FoV_ne_crop_box_entry=crop_box_dict[fov_id],
            config_dict=self._cfg,
            device=self._current_device,
            camera_gain = camera_gain
        )
        
        # This is the compute-heavy step
        refined_splines_dict, segment_log, quality_metadata = refiner.refine_initial_bsplines(initial_splines)
        if quality_metadata:
            print(f"DEBUG: Got quality metadata with {len(quality_metadata)} NE labels")
        # Save refinement report for this FoV/channel
        try:
            img_proc_cfg = self._get_cfg()
            output_dir = Path(img_proc_cfg.get('ne_fit', {}).get('refined_fit', {}).get('output_subdirectory'))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Import pandas for report generation
            import pandas as pd
            df = pd.DataFrame(segment_log)
            
            # Create simple summary report
            report_path = output_dir / f"{timestamp}_{fov_id}_ch{channel}_refinement_report.txt"
            csv_path = output_dir / f"{timestamp}_{fov_id}_ch{channel}_profile_details.csv"
            
            with open(report_path, 'w') as f:
                f.write(f"FoV: {fov_id}, Channel: {channel}\n")
                f.write(f"Total profiles: {len(df)}\n")
                f.write(f"Successes: {(df['status'] == 'success').sum()}\n")
                f.write(f"\nFailure breakdown:\n")
                for status, count in df['status'].value_counts().items():
                    if status != 'success':
                        f.write(f"  {status}: {count}\n")
            
            df.to_csv(csv_path, index=False)
            print(f"Saved refinement report: {report_path.name}")
            
        except Exception as e:
            print(f"WARNING: Could not save refinement report: {e}")

        logger.info("DIAGNOSTIC - refiner.refine_initial_bsplines returned:")

        # Add the FoV and channel metadata to the log
        if segment_log:
            for log_entry in segment_log:
                log_entry['fov_id'] = fov_id
                log_entry['channel'] = channel
            self._segment_logs.extend(segment_log) # Add to our class list

            logger.info(f"DIAGNOSTIC - segment_log for {fov_id}/{channel}: {len(segment_log)} entries")
            # Count successes vs failures
            successes = sum(1 for entry in segment_log if entry.get('status') == 'success')
            failures = sum(1 for entry in segment_log if entry.get('status') == 'failed')
            logger.info(f"  Successes: {successes}, Failures: {failures}")
        else:
            logger.warning(f"DIAGNOSTIC - No segment_log returned for {fov_id}/{channel}")

        logger.info(f"DIAGNOSTIC - refiner.refine_initial_bsplines returned:")
        logger.info(f"  refined_splines_dict type: {type(refined_splines_dict)}")
        logger.info(f"  refined_splines_dict is None: {refined_splines_dict is None}")
        if refined_splines_dict:
            logger.info(f"  refined_splines_dict has {len(refined_splines_dict)} NE labels")
            for ne_label in refined_splines_dict:
                logger.info(f"    NE {ne_label}: present")
                return refined_splines_dict # NOTE added back from 11-13-2025
        logger.warning(f"DIAGNOSTIC - _run_refinement_for_fov returning None for {fov_id}")
        return None

    def run_bezier_bridging(self):
        """
        Takes refined spline segments and connects them
        with Bezier bridges to create whole, closed membranes
        using the logic in spline_bridging.py.
        """
        logger.info("--- Running Bezier Bridging ---")
        is_dual_label = self._get_cfg().get('ne_dual_label', False)
        ne_fit_config = self._get_cfg().get('ne_fit', {})

        # This contains the exact precision for every nucleus, used to trim the ends being bridged
        sigma_map = self._get_sigma_map()

        if not sigma_map:
            logger.warning("No precision map found! (Did you run Stability Analysis?) Using defaults.")

        # --- CONFIG: Trim Calculation Mode ---
        # Options: 'ne' (local precision), 'fov' (global average), 'compare' (log both)
        trim_mode = ne_fit_config.get('trim_calculation_mode', 'ne')
        logger.info(f"Precision Trimming Mode: {trim_mode.upper()}")

        # --- BUILD REG PREC MAP ---
        reg_mode1_data = self.get_registration_results(reg_mode=1)
        uncertainty_map = {}

        if reg_mode1_data:
            for fov_id, fov_reg_data in reg_mode1_data.items():
                uncertainty_map[fov_id] = {}
                
                # FoV-level Fallback
                if 'ch_reg_prec' in fov_reg_data:
                    uncertainty_map[fov_id]['sigma_reg'] = fov_reg_data['ch_reg_prec']
# !!!: make sure this prioritizes per-label AND global is a fallback AND there is a config flag to change this
                # Per-Label Stats (Prefered)
                # Iterate through NE label specific reg data. 
                # If the value is a dict and has 'slice_', it's an NE label with registration slices.
                for key, val in fov_reg_data.items():
                    if isinstance(val, dict) and 'shift_vector' in val:
                        # Found an NE label entry. 
                        # Now check for slices to calculate sigma.
                        slice_shifts = []
                        for subkey, subval in val.items():
                            if subkey.startswith('slice_') and 'shift_vector' in subval:
                                slice_shifts.append(subval['shift_vector'])
                        
                        if len(slice_shifts) > 1:
                            # Calculate standard deviation of the shift vectors
                            # slice_shifts is shape (N, 2) -> [[y1, x1], [y2, x2], ...]
                            shifts_arr = np.array(slice_shifts)
                            
                            # std dev along axis 0 (across slices)
                            std_y = np.std(shifts_arr[:, 0])
                            std_x = np.std(shifts_arr[:, 1])
                            
                            # Combine into radial precision magnitude
                            sigma_reg = np.sqrt(std_y**2 + std_x**2)
                            
                            uncertainty_map[fov_id][key] = {'sigma_reg': sigma_reg}
                            # logger.debug(f"  {fov_id}/{key}: Calculated sigma_reg = {sigma_reg:.4f} px from {len(slice_shifts)} slices")
                        else:
                            # Fallback if slices missing or insufficient
                            # Use the global FoV sigma we set earlier
                            uncertainty_map[fov_id][key] = {'sigma_reg': uncertainty_map[fov_id]['sigma_reg']}

        logger.info(f"Uncertainty map built for {len(uncertainty_map)} FoVs.")

        # Channel 1 (Always run)
        ch1_refined = self._get_ch1_refined_bsplines()
        if ch1_refined:
            logger.info("Bridging Channel 1...")
            ch1_bridged_data = bridge_refined_splines(ch1_refined, ne_fit_config)
            self._set_ch1_bridged_splines(ch1_bridged_data)
        else:
            logger.warning("No Ch1 refined splines found to bridge.")
            self._set_ch1_bridged_splines({})

        # Channel 2 (Conditional)
        if is_dual_label:
            ch2_refined = self._get_ch2_refined_bsplines()
            if ch2_refined:
                logger.info("Bridging Channel 2...")
                ch2_bridged_data = bridge_refined_splines(ch2_refined, ne_fit_config)
                self._set_ch2_bridged_splines(ch2_bridged_data)
            else:
                logger.warning("Dual label mode, but no Ch2 refined splines found to bridge.")
                self._set_ch2_bridged_splines({})
        else:
            # Not dual label, so just set to an empty dict
            self._set_ch2_bridged_splines({})

    # ---- Compare Dual Labeled NE Pipes --- #

    def juxtapose_dual_labels(self):
        """
        Calculates the 2D separation distances between paired Ch1/Ch2 nuclear 
        envelope splines in the image plane.

        Uses time-averaged (mean) images from live-cell imaging to improve SNR.
        Distances represent lateral offsets between fluorescent markers labeling
        Nups, measured in pixels.

        Note: The imaging dataset is 2D spatial + temporal (not z-stacks). 
        The z-dimension in raw data represents time, not depth.
        """
        # only import this if using the dual label option

        logger = logging.getLogger(__name__)
        logger.info("--- Running Dual Label Juxtaposition ---")

        cfg_dual = self._get_cfg().get('dual_label')
        output_base_dir = Path(self._get_cfg().get('directories', {}).get('output root', ''))
        output_dual_dir =  output_base_dir / Path(cfg_dual.get('output subdirectory', 'distances'))
        dist_logger = DistanceCalculationLogger(output_dual_dir)

        # --- Fallback Logic ---
        ch1_splines_to_use = self._get_ch1_bridged_splines()
        ch2_splines_to_use = self._get_ch2_bridged_splines()
        
        # Check if the bridged dictionaries are empty (i.e., step was skipped or failed)
        if not ch1_splines_to_use: 
            logger.info("...No bridged splines found. Falling back to refined segments.")
            ch1_splines_to_use = self._get_ch1_refined_bsplines()
            ch2_splines_to_use = self._get_ch2_refined_bsplines()
        else:
            logger.info("...Using bridged spline data.")
            # Note: We assume ch2 is also present if ch1 is,
            # because run_bezier_bridging handles the dual-label logic.

        if not ch1_splines_to_use or (self._get_cfg().get('ne_dual_label') and not ch2_splines_to_use):
            logger.error("Required splines (Ch1 or Ch2) not found. Cannot run juxtaposition.")
            return

        # Get config info
        cfg_dual = self._get_cfg().get('dual_label')
        if not cfg_dual:
            logger.error("`dual_label` configuration missing.")
            return
        
        ch1_crop_boxes = self._get_ch1_cropped_imgs()
        ch2_crop_boxes = self._get_ch2_cropped_imgs()

        # move NE splines from cropbox to FoV coordinate space
        # accounts for differences in cropbox location in FoV space
        ch1_splines_FoV_space = self._splines_to_FoV_space(ch1_splines_to_use, ch1_crop_boxes)
        ch2_splines_FoV_space = self._splines_to_FoV_space(ch2_splines_to_use, ch2_crop_boxes)

        # Get drift data if it exists
        drift_data = self._get_drift_data() # ??? Revisit
    
        # Get registration results for transformation
        reg_mode1_data = self.get_registration_results(reg_mode=1) # mode = 1 uses first set of BF images
        # TASK set up additional modes (i.e. 2)

        ch2_splines_final = {}

        # Registration
        # Apply per-NE-label registration transformation to Ch2 splines
        if reg_mode1_data:
            logger.info("Applying registration transformation to Ch2 splines...")
            ch2_splines_transformed = {}
            # Get NE pairing information
            ne_pairs_by_fov = self.get_ne_pairs_by_FoV()

            for fov_id in ch2_splines_FoV_space.keys():
                if fov_id not in reg_mode1_data:
                    logger.warning(f"No registration data for FoV {fov_id}, skipping transformation")
                    ch2_splines_transformed[fov_id] = ch2_splines_FoV_space[fov_id]
                    continue
                
                fov_reg_data = reg_mode1_data[fov_id]
                
                # Get NE pairs for this FoV
                if fov_id not in ne_pairs_by_fov:
                    logger.warning(f"No NE pairs for FoV {fov_id}")
                    ch2_splines_transformed[fov_id] = ch2_splines_FoV_space[fov_id]
                    continue
                
                ne_pairs = ne_pairs_by_fov[fov_id]  # {ch1_label: ch2_label}
                
                # Transform each Ch2 NE spline
                ch2_splines_transformed[fov_id] = {}
                for ch2_ne_label, ne_data in ch2_splines_FoV_space[fov_id].items():
                    
                    # Find the corresponding Ch1 label for this Ch2 label
                    ch1_ne_label = None
                    for ch1_label, ch2_label in ne_pairs.items():
                        if ch2_label == ch2_ne_label:
                            ch1_ne_label = ch1_label
                            break
                    
                    if ch1_ne_label is None:
                        logger.warning(f"  FoV {fov_id} Ch2 NE {ch2_ne_label}: No Ch1 pair found, skipping transformation")
                        ch2_splines_transformed[fov_id][ch2_ne_label] = ne_data
                        continue
                    
                    # Get per-NE-label registration for this Ch1 label
                    # Registration is keyed by Ch1 label since that's the reference
                    if ch1_ne_label not in fov_reg_data:
                        logger.warning(f"  FoV {fov_id} Ch1 NE {ch1_ne_label}: No registration data, using FoV-level fallback")
                        # Try FoV-level registration as fallback
                        if 'scale' in fov_reg_data and 'angle' in fov_reg_data and 'shift_vector' in fov_reg_data:
                            reg_params = {
                                'scale': fov_reg_data['scale'],
                                'angle': fov_reg_data['angle'],
                                'shift_vector': fov_reg_data['shift_vector']
                            }
                        else:
                            logger.warning("    No usable registration data, skipping")
                            ch2_splines_transformed[fov_id][ch2_ne_label] = ne_data
                            continue
                    else:
                        # Use per-NE-label registration
                        ne_reg_data = fov_reg_data[ch1_ne_label]
                        reg_params = {
                            'scale': ne_reg_data['scale'],
                            'angle': ne_reg_data['angle'],
                            'shift_vector': ne_reg_data['shift_vector']
                        }
                    
                    try:
                        if fov_id in ch2_crop_boxes and ch2_ne_label in ch2_crop_boxes[fov_id]:
                            crop_info = ch2_crop_boxes[fov_id][ch2_ne_label]
                            # The center of the 75x75 pixel box
                            center_y_px = crop_info['height'] / 2.0
                            center_x_px = crop_info['width'] / 2.0
                            
                            FoV_space_center_y_px = crop_info['final_top'] + center_y_px
                            FoV_space_center_x_px = crop_info['final_left'] + center_x_px
                            
                            rotation_center_FoV_px = [FoV_space_center_y_px, FoV_space_center_x_px]
                    except KeyError:
                        rotation_center_FoV_px = [0,0] # Fallback, though risky in FoV space
                    
                    # Get the full periodic spline
                    ch2_spline = ne_data['full_periodic_spline']
                    
                    # Apply transformation (align Ch2 to Ch1)
                    # Center should be the centroid of the Ch2 NE
                    ch2_spline_transformed = bspline_transformation(
                        ch2_spline, 
                        reg_params,
                        center_coords = rotation_center_FoV_px
                    )
                    
                    # Create transformed ne_data dict
                    ch2_splines_transformed[fov_id][ch2_ne_label] = {
                        'data_segments': ne_data['data_segments'],
                        'bridge_segments': ne_data['bridge_segments'],
                        'full_periodic_spline': ch2_spline_transformed,  # Use transformed spline
                        'u_ranges': ne_data['u_ranges']
                    }
                    
                    logger.info(f"  FoV {fov_id} Ch1 {ch1_ne_label}↔Ch2 {ch2_ne_label}: "
                                f"shift=[{reg_params['shift_vector'][0]:.2f}, {reg_params['shift_vector'][1]:.2f}] px, "
                                f"angle={reg_params['angle']:.3f}°, scale={reg_params['scale']:.4f}")
            
            # Use transformed splines for distance calculation
            ch2_splines_final = ch2_splines_transformed
            logger.info(f"Registration transformation complete")
        else:
            logger.warning("Skipping registration transformation (no registration data)")
            ch2_splines_final = ch2_splines_FoV_space

        # Call the calculation function from ne_dual_labels
        # ??? is passing the min_iou now redundant since the pairs themselves are being passed? I think so.
        distances_by_FoV = calc_dual_distances(
            FoV_dict = self._FoV_collection_dict,
            ch1_bsplines = ch1_splines_FoV_space,
            ch2_bsplines = ch2_splines_final,
            ne_pairs_map = self.get_ne_pairs_by_FoV(),
            ch1_key = cfg_dual.get('channel1'),
            ch2_key = cfg_dual.get('channel2'),
            min_iou = cfg_dual.get('min_iou', 0.9),
            N_dist_calc = cfg_dual.get('N_dist_calc', 200),
            drift_data = drift_data,
            distance_logger = dist_logger
        )
        
        self._set_dual_distances_by_FoV(distances_by_FoV)

        # Save distance reports after all FoVs processed
        for fov_id in distances_by_FoV.keys():
            dist_logger.save_reports(fov_id)

    def _splines_to_FoV_space(self, splines_local, crop_dict, channel_name = 'Ch2'):
        """
        (Global Coordinates):
            1. Loads splines in the local coordinate space (cropbox).
            2. Shifts splines to FoV coordinate space using crop box offsets (final_left, final_top). 
        """
        logger.info(f"Shifting {channel_name} splines from cropbox to FoV coordinate space...")
        splines_FoV_space = {}
        for fov_id, ne_dict in splines_local.items():
            splines_FoV_space[fov_id] = {}

            for label, ne_data in ne_dict.items():
                ne_data_FoV_space = copy.deepcopy(ne_data)

                try:
                    crop_info = crop_dict[fov_id][label]
                    offset_x = crop_info['final_left']
                    offset_y = crop_info['final_top']
                    
                    # Shift the BSpline coefficients (control points) by adding offset
                    ne_data_FoV_space['full_periodic_spline'].c[:, 0] += offset_x
                    ne_data_FoV_space['full_periodic_spline'].c[:, 1] += offset_y
                    
                    splines_FoV_space[fov_id][label] = ne_data_FoV_space
                except KeyError:
                            logger.warning(f"Missing crop info for FoV {fov_id} Label {label}. Skipping globalization.")
                            continue
        return splines_FoV_space

    # image registration
    # want to register the difference between a set of red frames and set of green frames
    # per ne label per FoV <- what we need for translating individual NE
    # AND/OR
    # either the whole FoV (for comparison)

    def register_images(self, reg_mode = 1, FoV_list=[], add_to_existing = True, detailed_output = False):
        # registers the images in a directory
        # apply image_registration to every member of the _FoV_collection_dict
        #   (default)
        # OR
        # FoV_list - use to register a subset of the existing FoV's in the
        #               collection dictionary
        # add_to_exisiting - can either add FoV registration to the existing
        #                       list (default) OR overwrite the list
        FoV_dict = self._get_FoV_collection_dict()

        # if else single line enlisted to handle dual label situation
        ref_cropped_img_dict = self._get_ch1_cropped_imgs()
        datum_cropped_img_dict = self._get_ch2_cropped_imgs() if self._get_cfg()["ne_dual_label"] else None

        ne_labels_by_FoV_dict = self.get_ne_pairs_by_FoV() if self._get_cfg()['ne_dual_label'] else None

        if len(FoV_list) != 0:
            FoV_to_register = {FoV_id: FoV_dict[FoV_id] for FoV_id in FoV_list if FoV_id in FoV_dict}

            FoV_ch1_crop_dim_dict = {FoV_id: ref_cropped_img_dict[FoV_id] for FoV_id in FoV_list if FoV_id in ref_cropped_img_dict}
            FoV_ch2_crop_dim_dict = {FoV_id: datum_cropped_img_dict[FoV_id] for FoV_id in FoV_list if FoV_id in datum_cropped_img_dict} if datum_cropped_img_dict is not None else None
        else:
            FoV_to_register = FoV_dict
            FoV_ch1_crop_dim_dict = ref_cropped_img_dict
            FoV_ch2_crop_dim_dict = datum_cropped_img_dict if datum_cropped_img_dict is not None else FoV_ch1_crop_dim_dict # if there isn't a dual label, the crop dimensiosn dictionary is set to be a pointer to the same dict as ch1

        registration_dict = {}

        matched_FoV_id = find_shared_keys(FoV_ch1_crop_dim_dict, FoV_ch2_crop_dim_dict)

        for FoV_id in matched_FoV_id:
        # for FoV_id, FoV_dict_entry in FoV_to_register.items():
            # Find the corresponding dictionary in FoV_crop_dimensions
            # If not dual label, use a pointer to avoid unncessary search
            FoV_ch1_crop_dim_entry = FoV_ch1_crop_dim_dict[FoV_id]
            FoV_ch2_crop_dim_entry = FoV_ch2_crop_dim_dict[FoV_id] if self._get_cfg()['ne_dual_label'] else FoV_ch1_crop_dim_entry

            ne_labels_by_FoV_entry = ne_labels_by_FoV_dict[FoV_id] if self._get_cfg()['ne_dual_label'] else None

            # returns dictionary entry, so append will add to the list
            #   (rather than using expand)
            current_FoV_reg, current_ne_labels_reg = \
                image_registration(
                    FoV_to_register[FoV_id],
                    FoV_ch1_crop_dim_entry,
                    FoV_ch2_crop_dim_entry,
                    self._cfg['registration']['frame_range'],
                    self._cfg['registration']['frames_per_average'],
                    self._cfg['registration']['padding'],
                    self._cfg['registration']['upsample_factor'],
                    self._cfg['registration']['upscale_factor'],
                    reg_mode,
                    ne_labels_by_FoV_entry,
                    detailed_output
                    )
            # registration_dict.update(
            #     {
            #         f'{FoV_id}':
            #         {
            #             'FoV_reg_data': current_FoV_reg,
            #             'ne_label_registration': current_ne_labels_reg
            #         }
            #     }
            #     )            

            registration_dict.update(
                {f'{FoV_id}': current_FoV_reg}
                )
            registration_dict[f'{FoV_id}'].update(current_ne_labels_reg)

            self._set_reg_diff(registration_dict, reg_mode, add_to_existing)

    # --- Helper Methods --- #

    # ---- For Registration ---- #

    def _calculate_stats_from_lists(self, delta_angles, delta_scales, rdif_vectors):
        """Helper function to calculate a standard set of stats from lists."""
        if not rdif_vectors:
            return {
                "n": 0, "sigma_reg": np.nan,
                "angle_mean": np.nan, "angle_std": np.nan,
                "scale_mean": np.nan, "scale_std": np.nan,
                "rdif_y_mean": np.nan, "rdif_y_std": np.nan,
                "rdif_x_mean": np.nan, "rdif_x_std": np.nan
            }
        
        d_angle_arr = np.array(delta_angles)
        d_scale_arr = np.array(delta_scales)
        rdif_arr = np.array(rdif_vectors)

        sigma_y = np.std(rdif_arr[:, 0])
        sigma_x = np.std(rdif_arr[:, 1])
        sigma_reg = np.sqrt(sigma_y**2 + sigma_x**2)
        
        return {
            "n": len(rdif_arr),
            "sigma_reg": sigma_reg,
            "angle_mean": np.mean(d_angle_arr), "angle_std": np.std(d_angle_arr),
            "scale_mean": np.mean(d_scale_arr), "scale_std": np.std(d_scale_arr),
            "rdif_y_mean": np.mean(rdif_arr[:, 0]), "rdif_y_std": sigma_y,
            "rdif_x_mean": np.mean(rdif_arr[:, 1]), "rdif_x_std": sigma_x
        }

    def _get_deltas(self, reg_before, reg_after):
        """Helper to get deltas from two registration dictionaries."""
        delta_angle = reg_after['angle'] - reg_before['angle']
        delta_scale = reg_after['scale'] - reg_before['scale']
        rdif_vec = np.array(reg_after['shift_vector']) - np.array(reg_before['shift_vector'])
        rdif_mag = np.linalg.norm(rdif_vec)
        return delta_angle, delta_scale, rdif_vec, rdif_mag

    def _get_effective_radius(self):
        """Retrieves the nucleus radius (half-width) for physics conversions."""
        try:
            bbox = self._get_cfg().get('ne_fit', {}).get('bbox_dim', {'width': 75})
            return bbox.get('width', 75) / 2.0
        except ValueError:
            return 37.5

    def analyze_ne_reg_stability(self, prune=False):
        """
        Orchestrates the Registration Stability analysis.

        Methodology:
        1. Loads Mode 1 (Pre) and Mode 2 (Post) registration data.
        2. Calculates Drift (Mode 2 - Mode 1) and Combined Precision (Sigma).
        3. Filters nuclei where Drift > 2 * Precision.
        
        Returns:
            report (dict): Full status report with metrics for every nucleus.
            pairs_map (dict): The active (pruned) pairs map used by the pipeline.
        """
        logger.info("--- Analyzing Registration Stability (Temporal / Individual Nuclear Envelopes) ---")

        # Gather data
        reg_m1 = self.get_registration_results(reg_mode=1)
        reg_m2 = self.get_registration_results(reg_mode=2)
        pairs_map = self.get_ne_pairs_by_FoV()

        if not reg_m1 or not reg_m2:
            logger.error("Missing Registration Data (Mode 1 or Mode 2). Cannot calculate drift.")
            # Return empty structure to prevent crashes downstream, but log error
            return {}, pairs_map

        # Configuration
        radius_px = self._get_effective_radius()
        logger.info(f"  Physics Radius: {radius_px} px")

        # Calculations
        stats_m1 = calculate_mode_reg_stats(reg_m1, radius_px)
        stats_m2 = calculate_mode_reg_stats(reg_m2, radius_px)
        drift_map = calculate_drift_map(reg_m1, reg_m2, radius_px)
        
        # Builds report
        report = build_ne_stability_report                                           (stats_m1, stats_m2, drift_map, pairs_map)
        
        # Cache precision for bridging
        sigma_map = {}
        for fov, nuclei in report.items():
            sigma_map[fov] = {}
            for label, data in nuclei.items():
                sigma_map[fov][label] = data['metrics']['sigma_combined']
        self._set_sigma_map(sigma_map)

        # SWITCH
        # Culling pairs based on (dynamic/data driven) registration threshold
        if prune:
            pruned_total, pairs_map = self._prune_ne_by_stability(report, pairs_map)
            logger.info(f"  NE Stability Check Complete. Pruned {pruned_total} unstable nuclei.")
            self.save_status_ledger()

        self.save_status_ledger()

        return report, pairs_map

    def     _prune_ne_by_stability(self, report, pairs_map):
        """
        Updates the Ledger and culls the provided pairs_map in place by individual ne_label
        Expects nested report: {fov_id: {ne_label: {...}}}
        """
        
        prune_count = 0
        
        for fov_id, label_dict in report.items():
            for ch1_label, entry in label_dict.items():
                
                status = entry['status']
                metrics = entry['metrics']
                ch2_label = entry['ch2_label']
                
                self._update_ne_state('ch1', fov_id, ch1_label, {
                    'registration': status,
                    'paired': 'yes' if entry['is_paired'] else 'no',
                    'refined_fit': 'pending' if entry['is_paired'] else 'not applicable'
                }, log_event=f"Reg Filter: {status}", metrics=metrics)
                
                if ch2_label:
                        self._update_ne_state('ch2', fov_id, ch2_label, {
                        'registration': status,
                        'paired': 'yes',
                        'refined_fit': 'pending'
                    }, log_event=f"Reg Filter: {status}", metrics=metrics)

                if status == 'failed':
                    prune_count += 1
                    if entry['is_paired'] and fov_id in pairs_map and ch1_label in pairs_map[fov_id]:
                        del pairs_map[fov_id][ch1_label]

        return prune_count, pairs_map

    def analyze_fov_reg_stability(self, prune = False):
        """
        Registration based stability, aggregated by FoV.
        
        Logic:
        1. Calculate Drift for every FoV (Mode 2 - Mode 1).
        2. Calculate Global Sigma (Std Dev of all FoV drifts).
        3. Filter: Reject FoV if Drift > 2 * Global Sigma.
        """
        logger.info("--- Analyzing FoV Stability (Population Method) ---")
        
        reg_m1 = self.get_registration_results(reg_mode=1)
        reg_m2 = self.get_registration_results(reg_mode=2)
        
        if not reg_m1 or not reg_m2:
            return {}, 0.0
            
        radius_px = self._get_effective_radius()
        
        # 1. Calculate Population Stats (from img_registration.py)    
        global_sigma, fov_drifts = calculate_fov_population_stats(reg_m1, reg_m2, radius_px)
        
        logger.info(f"  Population Sigma (across {len(fov_drifts)} FoVs): {global_sigma:.4f} eff. px")
        
        # 2. Build Report
        report = build_fov_stability_report(fov_drifts, global_sigma, reg_m1)

        # SWITCH: Prune Only if Active
        if prune:
            pairs_map = self.get_ne_pairs_by_FoV()
            pruned_total, pairs_map = self._prune_fov_by_stability(report, pairs_map)
            logger.info(f"  FoV Stability Check Complete. Pruned {pruned_total} entire FoVs.")
            self.save_status_ledger()

        return report, global_sigma
    
    def _prune_fov_by_stability(self, report, pairs_map):
        """
        [Renamed/New] Prunes ENTIRE FoVs from the pairs map.
        """
        pruned_fov_count = 0
        
        for fov_id, entry in report.items():
            status = entry['status']
            
            if status == 'failed':
                if fov_id in pairs_map:
                    # Log mass deletion
                    if hasattr(self, '_ne_status_ledger'):
                        for ch1_label in pairs_map[fov_id].keys():
                            self._update_ne_state('ch1', fov_id, ch1_label, 
                                {'registration': 'failed_fov', 'paired': 'no'}, 
                                log_event=f"Reg Filter (FoV): {status}")
                    
                    del pairs_map[fov_id]
                    pruned_fov_count += 1
                    
        return pruned_fov_count, pairs_map
    
    def compare_stability_reports(self, fov_report, ne_report):
        logger.info("--- Generating Method Comparison Report ---")
        comp_stats = generate_stability_comparison_report(fov_report, ne_report)

        logger.info(f"Comparison Complete. Net Loss Diff: {comp_stats['experiment']['net_loss_diff']} (FoV - NE)")

        return comp_stats

    # ---- For Refinement ---- #
    def get_fovs_for_refinement(self):
        """
        Helper function to get the list of FoVs that have valid 
        initial splines and are ready for refinement.
        """
        if self._get_cfg().get('ne_dual_label'):
            # For dual label, use the paired map as the source of truth
            return list(self.get_ne_pairs_by_FoV().keys())
        else:
            # For single label, just use the initial splines
            return list(self._get_ch1_init_bsplines().keys())
    
    # --- Status Tracking Helpers ---
    def save_status_ledger(self, filename="ne_status_ledger.json"):
        """
        Saves the current NE Status Ledger to the output root defined in config.
        """
        if not hasattr(self, '_ne_status_ledger') or not self._ne_status_ledger:
            return

        try:
            # Attempt to find the output root from the config
            # (Matches how you look up 'model root' in other functions)
            out_root = self._cfg.get('directories', {}).get('output root', '.')
            output_path = Path(out_root) / filename
            
            # Ensure the directory exists (just in case)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(self._ne_status_ledger, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to save status ledger: {e}")
            
    # --- NE --- #
    # ???: make log a class unto itself, then can have ne vs mRNA tracking options
    def _init_ledger_entry(self, channel, fov_id, ne_label):
        """Ensures a ledger entry exists with the State Vector structure."""
        if not hasattr(self, '_ne_status_ledger'): self._ne_status_ledger = {}
        if channel not in self._ne_status_ledger: self._ne_status_ledger[channel] = {}
        if fov_id not in self._ne_status_ledger[channel]: self._ne_status_ledger[channel][fov_id] = {}
        
        if ne_label not in self._ne_status_ledger[channel][fov_id]:
            self._ne_status_ledger[channel][fov_id][ne_label] = {
                'state': {
                    'registration': 'pending',
                    'paired': 'unknown',
                    'refined_fit': 'pending'
                },
                'history': []
            }

    def _update_ne_state(self, channel, fov_id, ne_label, state_updates: dict, log_event: str = None, metrics: dict = None):
        """
        Updates specific keys in the 'state' dictionary and optionally logs to history.
        Example: state_updates = {'registration': 'passed', 'paired': 'no'}
        """
        self._init_ledger_entry(channel, fov_id, ne_label)
        
        # 1. Update the State Vector
        entry = self._ne_status_ledger[channel][fov_id][ne_label]
        for k, v in state_updates.items():
            entry['state'][k] = v
            
        # 2. Log to History (Optional, keeps the timeline view)
        if log_event:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': log_event,
                'state_snapshot': state_updates
            }
            # Add metrics if provided (Crucial for debugging why something failed)
            if metrics:
                history_entry['metrics'] = metrics
            
            entry['history'].append(history_entry)

    def _log_status_change(self, channel, fov_id, ne_label, status, event):
        """
        Legacy wrapper to handle calls from run_init_ne_detection.
        Maps the 'DETECTED' status to a state update and logs the event.
        """
        # We record that detection occurred, which effectively initializes the entry via _update_ne_state
        self._update_ne_state(channel, fov_id, ne_label, {'detection': status}, log_event=event)

    # ---- For Dual Label ---- #
        # determines paired ne_labels based on crop_box overlap (crop_boxes determined in initial NE fitting)
    def pair_dual_labels(self, ch1_crop_dict, ch2_crop_dict, min_iou = 0.9):
        matched_FoV_id = find_shared_keys(ch1_crop_dict, ch2_crop_dict)
        
        ne_label_pairs_by_FoV = {}

        for FoV_id in matched_FoV_id:
            current_FoV_ne_label_pairs = match_ne_labels_by_iou(ch1_crop_dict[FoV_id], ch2_crop_dict[FoV_id], min_iou)
            ne_label_pairs_by_FoV.update({f'{FoV_id}': current_FoV_ne_label_pairs})
        
        return ne_label_pairs_by_FoV

    # --- Getters & Setters --- #
    # Error logging
    def get_segment_logs(self):
        return self._segment_logs

    # Configuration
    def _set_cfg(self, config):
        self._cfg = config
    
    def _get_cfg(self):
        return self._cfg

    # Torch Device
    def _set_current_device(self, device):
        self._current_device = device
    
    def _get_current_device(self):
        return self._current_device

    # FoV Dictionary
    def _set_FoV_collection_dict(self, FoV_dict):
        self._FoV_collection_dict = FoV_dict

    def _get_FoV_collection_dict(self):
        return self._FoV_collection_dict

    # Detector Responsivity Determination
    # TODO - impliment format control/check
    def _set_drd_stats(self, cal_stats):
        self._drd_stats = cal_stats

    def _get_drd_stats(self):
        return self._drd_stats
    
    def _set_drd_meanvar(self, meanvar):
        self._drd_meanvar = meanvar

    def _get_drd_meanvar(self):
        return self._drd_meanvar

    # Registration
    # !!! for NE level registration, requires initial NE fit to be run first
    def _set_reg_diff(self, new_reg_diff_mode1_dict, reg_mode = 1, add_to_existing = True):
        if reg_mode == 1:
            self._reg_diff_mode1 = dict_update_or_replace(self.get_registration_results(reg_mode),
                                                            new_reg_diff_mode1_dict,
                                                            add_to_existing)
        elif reg_mode == 2:
            self._reg_diff_mode2 = dict_update_or_replace(self.get_registration_results(reg_mode),
                                                            new_reg_diff_mode1_dict,
                                                            add_to_existing)
        else:
            print(f'{reg_mode} is not a valid option. The only valid options are 1 & 2.')

    def get_registration_results(self, reg_mode = 1):
        if reg_mode == 1:
            return self._reg_diff_mode1
        elif reg_mode == 2:
            return self._reg_diff_mode2
        else:
            print(f'{reg_mode} is not a valid option. The only valid options are 1 & 2.')

    def _set_sigma_map(self, sigma_map):
            self._sigma_map = sigma_map

    def _get_sigma_map(self):
        return getattr(self, '_sigma_map', {})

    # Drift Correction
    def _set_drift_correction(self, new_drift_dict, add_to_existing = True):
        self._drift_correction = dict_update_or_replace(self._get_drift_correction(),
                                                            new_drift_dict,
                                                            add_to_existing)

    def _get_drift_correction(self):
        return self._drift_correction


    def _get_drift_data(self):
        """Returns drift correction data, or None if not available."""
        return self._drift_correction if hasattr(self, '_drift_correction') and self._drift_correction else None

    # Initial NPC Fitting
    # Label 1

    def _set_ch1_cropped_imgs(self, new_ch1_cropped_imgs, add_to_existing = True):
        self._ch1_cropped_imgs = dict_update_or_replace(self._get_ch1_cropped_imgs(),
                                                            new_ch1_cropped_imgs,
                                                            add_to_existing)

    def _get_ch1_cropped_imgs(self):
        return self._ch1_cropped_imgs
    
    def _set_ch1_init_bsplines(self, new_ch1_init_bsplines, add_to_existing = True):
        self._ch1_init_bsplines = dict_update_or_replace(self._get_ch1_init_bsplines(), new_ch1_init_bsplines, add_to_existing)


    def _get_ch1_init_bsplines(self):
        return self._ch1_init_bsplines

    #    Label 2 (same as Label 1 if not an NE dual label experiment)
    def _set_ch2_cropped_imgs(self, new_ch2_cropped_imgs, add_to_existing = True):
        self._ch2_cropped_imgs = dict_update_or_replace(self._get_ch2_cropped_imgs(),
                                                            new_ch2_cropped_imgs,
                                                            add_to_existing)

    def _get_ch2_cropped_imgs(self):
        return self._ch2_cropped_imgs
    
    def _set_ch2_init_bsplines(self, new_ch2_init_bsplines, add_to_existing = True):
        self._ch2_init_bsplines = dict_update_or_replace(self._get_ch2_init_bsplines(), new_ch2_init_bsplines, add_to_existing)


    def _get_ch2_init_bsplines(self):
        return self._ch2_init_bsplines

    def _set_ne_pairs_by_FoV(self, new_ne_pair_dict, add_to_existing = True):
        self._ne_pairs_by_FoV = dict_update_or_replace(self.get_ne_pairs_by_FoV(),
                                                            new_ne_pair_dict,
                                                            add_to_existing)
    def get_ne_pairs_by_FoV(self):
        return self._ne_pairs_by_FoV

    # Refined NPC Fitting

    def _set_ch1_refined_bsplines(self, new_splines, add_to_existing=True):
        self._ch1_refined_bsplines = dict_update_or_replace(self._get_ch1_refined_bsplines(), new_splines, add_to_existing)

    def _get_ch1_refined_bsplines(self):
        return self._ch1_refined_bsplines

    def _set_ch2_refined_bsplines(self, new_splines, add_to_existing=True):
        self._ch2_refined_bsplines = dict_update_or_replace(self._get_ch2_refined_bsplines(), new_splines, add_to_existing)

    def _get_ch2_refined_bsplines(self):
        return self._ch2_refined_bsplines
    
    def _get_ch1_bridged_splines(self):
        return self._ch1_bridged_splines
    
    def _set_ch1_bridged_splines(self, bridged_splines):
        self._ch1_bridged_splines = bridged_splines
        
    def _get_ch2_bridged_splines(self):
        return self._ch2_bridged_splines
    
    def _set_ch2_bridged_splines(self, bridged_splines):
        self._ch2_bridged_splines = bridged_splines

    def _set_dual_distances_by_FoV(self, new_dual_distances_by_FoV, add_to_existing = True):
        self._dual_distances_by_FoV = dict_update_or_replace(self.get_dual_distances_by_FoV(),
                                                                new_dual_distances_by_FoV,
                                                                add_to_existing)
    def get_dual_distances_by_FoV(self):
        return self._dual_distances_by_FoV

    # representation given when an instance of the object is printed
    def __repr__(self):
        return f"---Image Processer Configuration---\nImageProcessor(configuration={self._cfg})"