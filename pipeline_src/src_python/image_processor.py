"""
Created on Fri May  9 12:55:24 2025

@author: jctourtellotte
"""

# import external packages
import sys
import traceback
import os

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
from utils.img_registration import image_registration, compute_drift
from utils.npc_detect_initial import detect_npc
from utils.ne_dual_labels import match_ne_labels_by_iou, calc_dual_distances
from utils.npc_spline_refinement import NESplineRefiner

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


    # ???: registration is going to happen AFTER the bounding boxes re: NE are determined
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
        
        self._set_ne_pairs_by_FoV(ch1_ch2_ne_label_pairs)

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
# !!!
# ??? WHAT IS THIS RETURNING 
# !!!
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
        refined_splines_dict, segment_log = refiner.refine_initial_bsplines(initial_splines)

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


    # def run_bezier_bridging(self):
    #     """
    #     Takes refined spline segments and connects them
    #     with Bezier bridges to create whole, closed membranes
    #     using the logic in spline_bridging.py.
        
    #     Conditionally processes Ch2 only if 'ne_dual_label' is True.
    #     """
    #     logger.info("--- Running Bezier Bridging ---")
    #     is_dual_label = self._get_cfg().get('ne_dual_label', False)
    #     ne_fit_config = self._get_cfg().get('ne_fit', {}) # Get the ne_fit config
        
    #     # --- Channel 1 (Always run) ---
    #     ch1_refined = self._get_ch1_refined_bsplines()
    #     if ch1_refined:
    #         logger.info("Bridging Channel 1...")
    #         ch1_bridged_data = bridge_refined_splines(ch1_refined, ne_fit_config)
    #         self._set_ch1_bridged_splines(ch1_bridged_data)
    #     else:
    #         logger.warning("No Ch1 refined splines found to bridge.")
    #         self._set_ch1_bridged_splines({})

    #     # --- Channel 2 (Conditional) ---
    #     if is_dual_label:
    #         ch2_refined = self._get_ch2_refined_bsplines()
    #         if ch2_refined:
    #             logger.info("Bridging Channel 2...")
    #             ch2_bridged_data = bridge_refined_splines(ch2_refined, ne_fit_config)
    #             self._set_ch2_bridged_splines(ch2_bridged_data)
    #         else:
    #             logger.warning("Dual label mode, but no Ch2 refined splines found to bridge.")
    #             self._set_ch2_bridged_splines({})
    #     else:
    #         # Not dual label, so just set to an empty dict
    #         self._set_ch2_bridged_splines({})


    # ---- Compare Dual Labeled NE Pipes --- #

    def juxtapose_dual_labels(self):
        """
        Calculates the 3D distances between the two channels' splines
        for each NE label.
        
        This method will use "bridged" splines if they exist,
        otherwise it will fall back to the refined segments.
        """
        logger.info("--- Running Dual Label Juxtaposition ---")
        
        # --- Smart Fallback Logic ---
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
            
        # Get drift data if it exists
        drift_data = self._get_drift_data()

        # Call the calculation function from ne_dual_labels
        # This function is now passed the "smart-selected" splines
        distances_by_FoV = calc_dual_distances(
            FoV_dict = self._FoV_collection_dict,
            ch1_bsplines = ch1_splines_to_use,
            ch2_bsplines = ch2_splines_to_use,
            ne_pairs_map = self.get_ne_pairs_by_FoV(),
            ch1_key = cfg_dual.get('channel1'),
            ch2_key = cfg_dual.get('channel2'),
            min_iou = cfg_dual.get('min_iou', 0.9),
            N_dist_calc = cfg_dual.get('N_dist_calc', 200),
            drift_data = drift_data
        )
        
        self._set_dual_distances_by_FoV(distances_by_FoV)

    # OLD JUX DUAL LABELS
        # def juxtapose_dual_labels(self):
        #     """
        #     Calculates the 3D distances between the two channels' splines
        #     for each NE label.
            
        #     This method will use "bridged" splines if they exist,
        #     otherwise it will fall back to the refined segments.
        #     """
        #     logger.info("--- Running Dual Label Juxtaposition ---")
        #     # Get all the necessary data dictionaries
        #     cfg_ne_fit = self._get_cfg()['ne_fit']
        #     bbox_dim_dict = cfg_ne_fit['bbox_dim']
        #     bbox_dim = np.array([bbox_dim_dict['height'], bbox_dim_dict['width']])
        #     ch1_refined_segments_dict = self._get_ch1_refined_bsplines()
        #     ch2_refined_segments_dict = self._get_ch2_refined_bsplines()
        #     FoV_registration = self.get_registration_results()
        #     ne_label_pairs_by_FoV = self.get_ne_pairs_by_FoV()
        #     N_dist_samples = self._get_cfg()["dual_label"]["N_dist_calc"]

        #     dual_distances_by_FoV = {}
            
        #     # Find FoVs that have data for both channels
        #     matched_FoV_ids = find_shared_keys(ch1_refined_segments_dict, ch2_refined_segments_dict)

        #     for FoV_id in matched_FoV_ids:
        #         dist_by_ne_pair = {}
                
        #         # Loop through each paired NE in the current FoV
        #         for ch1_ne_label, ch2_ne_label in ne_label_pairs_by_FoV[FoV_id].items():
                    
        #             # --- DUAL-PATH LOGIC BASED ON CONFIG ---
                    
        #             # PATH 1: Merge segments to get a full loop and SIGNED distance
        #             if cfg_ne_fit['use_merged_clusters']:
        #                 # Reconstruct a single, periodic spline from the refined segments
        #                 final_spline_A = reconstruct_periodic_spline(ch1_refined_segments_dict[FoV_id][ch1_ne_label])
        #                 final_spline_B = reconstruct_periodic_spline(ch2_refined_segments_dict[FoV_id][ch2_ne_label])

        #                 if final_spline_A is None or final_spline_B is None:
        #                     print(f"  --> Could not reconstruct a full loop for NE pair {ch1_ne_label}/{ch2_ne_label}. Skipping.")
        #                     continue
                        
        #                 is_periodic = True # The reconstructed spline is periodic
        #                 bspline_A = final_spline_A
        #                 bspline_B = final_spline_B

        #             # PATH 2: Use only the longest segment and get UNSIGNED distance
        #             else:
        #                 # The dictionary for each NE will contain only one segment ('segment_0')
        #                 try:
        #                     segment_A_data = ch1_refined_segments_dict[FoV_id][ch1_ne_label]['segment_0']
        #                     segment_B_data = ch2_refined_segments_dict[FoV_id][ch2_ne_label]['segment_0']
                            
        #                     # is_periodic = False # We are analyzing a single segment
        #                     # bspline_A = segment_A_data['bspline_object']
        #                     # bspline_B = segment_B_data['bspline_object']
        #                 except KeyError:
        #                     print(f"  --> Could not find 'segment_0' for NE pair {ch1_ne_label}/{ch2_ne_label}. Skipping.")
        #                     continue

        #             # --- COMMON CALCULATION LOGIC ---
        #             # Apply registration to the channel B spline
        #             current_reg_values = FoV_registration[FoV_id]['ne_label_registration'][ch1_ne_label]
        #             transformed_bspline_B = bspline_transformation(bspline_B, current_reg_values, center_coords=bbox_dim / 2)

        #             # Calculate the distance
        #             distances = calculate_signed_distances(
        #                 bspline_A, 
        #                 transformed_bspline_B, 
        #                 N_dist_samples, 
        #                 is_periodic=is_periodic
        #             )
                    
        #             dist_by_ne_pair[f'{ch1_ne_label}_vs_{ch2_ne_label}'] = distances
                
        #         if dist_by_ne_pair:
        #             dual_distances_by_FoV[FoV_id] = dist_by_ne_pair
                    
        #     self._set_dual_distances_by_FoV(dual_distances_by_FoV)
    
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

    def analyze_registration_stability(self):
        """
        Performs a full 3-level stability analysis (FoV, Label, Slice).
        
        1. Gathers stats for all three "views" to report on the "experiment as a whole".
        2. Calculates a global sigma_reg from all slice-level comparisons.
        3. Filters NE-Labels by comparing their *mean* r_dif against the global threshold.
        4. Returns a "flat form" list of dicts for reporting and the global stats.
        """
        print("--- Analyzing Registration Stability (Full Audit) ---")
        
        reg_mode1_data = self.get_registration_results(reg_mode=1)
        reg_mode2_data = self.get_registration_results(reg_mode=2)

        if not reg_mode1_data or not reg_mode2_data:
            print(" --> WARNING: Missing reg_mode=1 or reg_mode=2 data. Skipping.")
            return [], {}

        # --- Lists for "Experiment as a whole" stats ---
        all_fov_delta_angles, all_fov_delta_scales, all_fov_rdif_vectors = [], [], []
        all_label_delta_angles, all_label_delta_scales, all_label_rdif_vectors = [], [], []
        all_slice_delta_angles, all_slice_delta_scales, all_slice_rdif_vectors = [], [], []
        
        # --- List for "Flat Form" Report ---
        flat_report_data = []
        
        ne_pairs_map = self.get_ne_pairs_by_FoV()

        # --- Loop 1: Gather all data ---
        for fov_id, pairs in ne_pairs_map.items():
            if fov_id not in reg_mode1_data or fov_id not in reg_mode2_data:
                continue
            
            # --- 1. FoV-Level Data ---
            try:
                fov_d_angle, fov_d_scale, fov_rdif_vec, fov_rdif_mag = self._get_deltas(
                    reg_mode1_data[fov_id], reg_mode2_data[fov_id]
                )
                all_fov_delta_angles.append(fov_d_angle)
                all_fov_delta_scales.append(fov_d_scale)
                all_fov_rdif_vectors.append(fov_rdif_vec)
            except (KeyError, TypeError, AttributeError):
                fov_rdif_mag = np.nan # Mark as invalid

            # --- 2. NE-Label-Level and Slice-Level Data ---
            for ch1_label in list(pairs.keys()):
                try:
                    reg_before_label = reg_mode1_data[fov_id][ch1_label]
                    reg_after_label = reg_mode2_data[fov_id][ch1_label]

                    # --- 2a. NE-Label (Mean) Data ---
                    label_d_angle, label_d_scale, label_rdif_vec, label_rdif_mag = self._get_deltas(
                        reg_before_label, reg_after_label
                    )
                    all_label_delta_angles.append(label_d_angle)
                    all_label_delta_scales.append(label_d_scale)
                    all_label_rdif_vectors.append(label_rdif_vec)
                    
                    # --- 2b. Slice-Level Data ---
                    label_slice_deltas = [] # (angle, scale, rdif_vec)
                    slice_keys = {k for k in reg_before_label if k.startswith('slice_')} & \
                                    {k for k in reg_after_label if k.startswith('slice_')}

                    for slice_key in slice_keys:
                        s_d_angle, s_d_scale, s_rdif_vec, _ = self._get_deltas(
                            reg_before_label[slice_key], reg_after_label[slice_key]
                        )
                        label_slice_deltas.append((s_d_angle, s_d_scale, s_rdif_vec))
                        all_slice_delta_angles.append(s_d_angle)
                        all_slice_delta_scales.append(s_d_scale)
                        all_slice_rdif_vectors.append(s_rdif_vec)
                    
                    # --- 2c. Add to Flat Report ---
                    # Calculate stats for *this label's* slices
                    slice_stats = self._calculate_stats_from_lists(
                        [d[0] for d in label_slice_deltas],
                        [d[1] for d in label_slice_deltas],
                        [d[2] for d in label_slice_deltas]
                    )

                    flat_report_data.append({
                        "fov_id": fov_id,
                        "ne_label": ch1_label,
                        "mean_rdif_mag": label_rdif_mag,
                        "mean_delta_angle": label_d_angle,
                        "mean_delta_scale": label_d_scale,
                        "slice_n": slice_stats["n"],
                        "slice_sigma_reg": slice_stats["sigma_reg"],
                        "slice_rdif_y_std": slice_stats["rdif_y_std"],
                        "slice_rdif_x_std": slice_stats["rdif_x_std"],
                        "filtered_global": "pending" # Will be filled next
                    })
                        
                except (KeyError, TypeError, AttributeError):
                    pass # Skip this label if data is incomplete

        # --- Loop 2: Calculate Global Stats ("Experiment as a whole") ---
        global_stats = {
            "fov_level": self._calculate_stats_from_lists(
                all_fov_delta_angles, all_fov_delta_scales, all_fov_rdif_vectors
            ),
            "ne_label_level": self._calculate_stats_from_lists(
                all_label_delta_angles, all_label_delta_scales, all_label_rdif_vectors
            ),
            "slice_level": self._calculate_stats_from_lists(
                all_slice_delta_angles, all_slice_delta_scales, all_slice_rdif_vectors
            )
        }
        
        # --- Loop 3: Filter based on Global Slice Precision ---
        if global_stats["slice_level"]["n"] == 0:
            print("\n --> WARNING: No slice-level data found. Cannot filter.")
            return flat_report_data, global_stats

        print("\n--- Applying Filtering based on GLOBAL SLICE-LEVEL Precision ---")
        
        # This is the "ground truth" precision from all slices
        sigma_reg_global = global_stats["slice_level"]["sigma_reg"]
        threshold = 2 * sigma_reg_global

        print(f"  Global Precision (sigma_reg, from all slices) = {sigma_reg_global:.4f} pixels")
        print(f"  Filtering Threshold (2 * sigma_reg) = {threshold:.4f} pixels")
        print("  (Filtering NE labels based on their OVERALL MEAN r_dif vs. this threshold)")

        ne_pairs_map_to_filter = self.get_ne_pairs_by_FoV() 
        total_filtered = 0
        
        for entry in flat_report_data:
            mean_mag = entry["mean_rdif_mag"]
            if np.isnan(mean_mag):
                entry["filtered_global"] = "error"
                continue

            if mean_mag > threshold:
                entry["filtered_global"] = True
                total_filtered += 1
                fov_id, ch1_label = entry["fov_id"], entry["ne_label"]
                
                print(f"     ACTION: Filtering {fov_id} / {ch1_label} (||Mean_rdif|| = {mean_mag:.4f} > {threshold:.4f})")
                
                if fov_id in ne_pairs_map_to_filter and ch1_label in ne_pairs_map_to_filter[fov_id]:
                    del ne_pairs_map_to_filter[fov_id][ch1_label]
            else:
                entry["filtered_global"] = False
                    
        print(f" --> Filtering complete. Removed {total_filtered} unstable NE label pairs.")
        
        return flat_report_data, global_stats

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