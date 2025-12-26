# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 08:51:23 2025

@author: jctourtellotte
"""

# --- Outside Modules --- #
# import tifffile
# import math
import os
import signal
import torch
# import torch.nn as nn
# import torch.optim as optim
import traceback
from pathlib import Path
import theseus as th
from scipy.interpolate import make_splprep
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# --- Included Modules --- #
from tools.utility_functions import calc_tangent_endpts, extract_cropped_images, sample_bspline
from tools.geom_tools import get_u_range_from_bspline, bspline_from_tck

from utils.MLEstimation import model_error_func, construct_rich_gaus_initial_guess, construct_gaus_lin_initial_guess, MODEL_REGISTRY

def _timeout_handler(signum, frame):
    """Helper function to raise a TimeoutError."""
    raise TimeoutError("Theseus optimization call timed out.")

class NESplineRefiner:
    # Class - Why? bc image loading and mean frame config etc
    # def __init__(self, img_path, FoV_initial_fit_entry, FoV_initial_bspline_entry, FoV_ne_crop_box_entry, config_dict, device = torch.device('cpu')):
    def __init__(self, channel, img_path, fov_id, FoV_ne_crop_box_entry, config_dict, device = torch.device('cpu'), camera_gain = 1.0):
        print('initiating Spline Refinement')
        self._channel = channel
        self._FoV_id = fov_id
        self._current_device = device
        self._camera_gain = camera_gain

        ne_fit_cfg = config_dict.get('ne_fit', {})
        fit_refinement_cfg = ne_fit_cfg.get('refinement', {})

        self._cfg = {
            'directories': config_dict.get('directories', {}),
            'frame_range': ne_fit_cfg['frame_range'],
            'line_length': ne_fit_cfg['line_length'],
            'n_samples_along_normal': ne_fit_cfg['n_samples_along_normal'],
            'max_curvature_angle_deg': ne_fit_cfg['max_curvature_angle_deg'],
            'refinement': fit_refinement_cfg
        }
        # Refinement-specific parameters
        self._model_name = fit_refinement_cfg.get('default_model', "gaussian_linear")
        self._model_cfg = next((model for model in fit_refinement_cfg['model_list'] if model["name"] == self._model_name), None)
        if not self._model_cfg:
            print(f"No model found with name: {self._model_name}")
        self._lm_cfg = fit_refinement_cfg['lm_optimizer']

        # NPC fit paramaters
        self._ne_images = extract_cropped_images(img_path, ne_fit_cfg.get('frame_range', [0, 250]), FoV_ne_crop_box_entry, mean_img = True)

        half_length_line = self._get_line_length() / 2
# QUESTION: what is this for?
# did I figure this out and forget to answer myself?
        self._initial_bounds = [[-1e4, 1e4],
                                [-1e4, 1e4],
                                [-100, 100],
                                [-1e4, 1e4],
                                [-100, 100],
                                [-1e4, 1e4],
                                [half_length_line - 2, half_length_line + 2],
                                [half_length_line - 4, half_length_line + 4],
                                [-1e4, 1e4],
                                [-1e4, 1e4],
                                [-1000, 1000]]


    def refine_initial_bsplines(self, initial_bsplines, testing_mode = False):
        channel = self._get_channel()
        FoV_id = self._get_FoV_id()
        cropped_imgs = self._get_ne_imgs()
        prep_for_opt = self._get_prep_for_opt()
        
        final_refined_splines_dict = {}
        segment_log = [] # This is your new log

        print(f'Current FoV: {FoV_id} , {channel}')

        for ne_label, segments in initial_bsplines.items():
            print(f"Current NE mask label: {ne_label}")
            mean_ne_img = cropped_imgs.get(ne_label)
            if mean_ne_img is None:
                print(f"Warning: No cropped image found for NE label {ne_label}. Skipping.")
                continue
            
            refined_segments_for_this_ne = {}
            # ---- PER SEGEMENT ----
            for seg_label, bspline_obj in segments.items():
                print(f"  Processing segment: {seg_label}")
                
                # 1. Pull *all* profiles and the out-of-bounds mask
                try:
                    intensity_profiles_all, dist_along_norm_all, validity_mask_oom = self._pull_segment_profiles(
                        mean_ne_img, seg_label, bspline_obj
                    )
                except Exception as e:
                    print(f"    CRITICAL: Failed _pull_segment_profiles for {seg_label}. Error: {e}")
                    segment_log.append({
                        'ne_label': ne_label, 'seg_label': seg_label, 'profile_index': -1,
                        'status': 'fail:profile_extraction', 'num_total_points': -1
                    })
                    continue
                
                # 2. Get *all* original spline points
                u_values_full = get_u_range_from_bspline(bspline_obj, self._get_sampling_density())
                segment_points_xy_all = bspline_obj(u_values_full)
                segment_derivs_xy_all = bspline_obj(u_values_full, nu=1)

                total_points = len(validity_mask_oom)
                line_length = self._get_line_length()
                model_name = self._get_model_name()
                
                # Default mu is center of the line (no change)
                optimized_mu_full = np.full(total_points, line_length / 2.0)
                successful_fits = 0

                # 3. --- LOOP ONE-BY-ONE ---
                for i in range(total_points):
                    status = "unprocessed"

                    # Skip if out-of-bounds
                    if not validity_mask_oom[i]:
                        status = "skip:out_of_bounds"
                        segment_log.append({
                            'ne_label': ne_label, 'seg_label': seg_label, 'profile_index': i,
                            'status': status, 'num_total_points': total_points
                        })
                        continue
                    # pulling current intensity profile and distance arrays
                    single_intensity = intensity_profiles_all[i]
                    single_dist = dist_along_norm_all[i]

                    # Skip if degenerate (flat line)
                    if np.std(single_intensity) < 1e-3:
                        status = "skip:degenerate"
                        segment_log.append({
                            'ne_label': ne_label, 'seg_label': seg_label, 'profile_index': i,
                            'status': status, 'num_total_points': total_points
                        })
                        continue
                    # Skip if profile has insufficient contrast for edge detection
                    intensity_range = single_intensity.max() - single_intensity.min()
                    if intensity_range < 10:  # ADU - adjust threshold based on your data
                        status = "skip:low_contrast"
                        segment_log.append({
                            'ne_label': ne_label, 'seg_label': seg_label, 'profile_index': i,
                            'status': status, 'num_total_points': total_points
                        })
                        continue

                    if prep_for_opt:
                        # Prep *one* profile
                        noise = np.max(single_intensity) * self._get_noise_multiplier()
                        jitter = np.random.randn(*single_intensity.shape) * noise
                        single_intensity = single_intensity + jitter
                    
                    # TEMPORARY DIAGNOSTIC - Remove after debugging
                    if i % 100 == 0:  # Print every 100th profile to avoid spam
                        print(f"    Profile {i}: max={single_intensity.max():.1f}, "
                              f"min={single_intensity.min():.1f}, "
                              f"mean={single_intensity.mean():.1f}, "
                              f"std={single_intensity.std():.2f}")

                    # 4. --- OPTIMIZE ONE-BY-ONE ---
                    try:
                        # This function has its own try/except
                        mu_result, status, predicted_intensity = self._optimize_single_profile_adaptive(
                            single_intensity,
                            single_dist,
                            model_name
                        )
                        
                        # if status == "success":
                        #     # Poisson-based quality check
                        #     if not self._check_poisson_consistency(
                        #         predicted_intensity,    # Model prediction
                        #         single_intensity,       # Actual data
                        #         gain=self._get_camera_gain()
                        #     ):
                        #         status = "fail:poisson_outlier"
                        #         optimized_mu_full[i] = line_length / 2.0
                        #     else:
                        #         optimized_mu_full[i] = mu_result
                        #         successful_fits += 1
                        if status == "success":
                            # Likelihood ratio test for quality check
                            if not self._check_fit_quality_likelihood_ratio(
                                predicted_intensity,    # Model prediction
                                single_intensity,       # Actual data
                                gain=self._get_camera_gain(),
                                n_params=11,  # Richards-Gaussian has 11 parameters
                                threshold_delta_aic=10  # Standard threshold from Burnham & Anderson 2004
                            ):
                                status = "fail:likelihood_ratio"
                                optimized_mu_full[i] = line_length / 2.0
                            else:
                                optimized_mu_full[i] = mu_result
                                successful_fits += 1
                    except Exception as e:
                        # This catches failure in the *loop itself*
                        status = "fail:loop_critical"
                        error_msg = str(e)
                        print(f"    --> ERROR: Loop failed at profile {i}. Error: {error_msg[:50]}")
                    # Log the final status of this profile
                    log_entry = {
                        'ne_label': ne_label, 
                        'seg_label': seg_label, 
                        'profile_index': i,
                        'status': status, 
                        'num_total_points': total_points,
                        'intensity_max': float(single_intensity.max()),
                        'intensity_min': float(single_intensity.min()),
                        'intensity_mean': float(single_intensity.mean()),
                        'intensity_std': float(single_intensity.std()),
                        'intensity_range': float(single_intensity.max() - single_intensity.min())
                    }
                    segment_log.append(log_entry)
                    
                # 5. Check if we have enough *good* fits to make a spline
                if successful_fits < 4:
                    # print(f"    Warning: Not enough successful fits ({successful_fits}) for {seg_label}. Skipping segment.")
                    continue
                
                # 6. Apply curvature filter to prevent non-physical artifacts
                max_angle = self._get_cfg().get('max_curvature_angle_deg', 1.0)
                optimized_mu_full, n_reverted, curvature_mask = self.apply_curvature_filter(
                    optimized_mu_full, 
                    segment_points_xy_all, 
                    segment_derivs_xy_all,
                    line_length, 
                    max_angle_change_deg=max_angle
                )

                # Update segment_log with curvature info
                for i, is_valid in enumerate(curvature_mask):
                    if i < len(segment_log):
                        segment_log[i]['passed_curvature'] = bool(is_valid)
                        segment_log[i]['was_reverted'] = not is_valid

                # 7. Re-assemble and fit the final spline 
                try:
                    refined_points = adjust_spline_points(
                        optimized_mu_full,        # Full (N_total,) mu array
                        line_length,
                        segment_points_xy_all,    # Full (2, N_total) points
                        segment_derivs_xy_all     # Full (2, N_total) derivatives
                    )
                    
                    crop_margin = 10  # pixels
                    x_min, x_max = -crop_margin, 75 + crop_margin
                    y_min, y_max = -crop_margin, 75 + crop_margin
                    
                    n_clipped_x = np.sum((refined_points[0, :] < x_min) | (refined_points[0, :] > x_max))
                    n_clipped_y = np.sum((refined_points[1, :] < y_min) | (refined_points[1, :] > y_max))
                    
                    if n_clipped_x > 0 or n_clipped_y > 0:
                        print(f"    CLIPPING: {seg_label} had {n_clipped_x} X and {n_clipped_y} Y points out of bounds")
                    
                    refined_points[0, :] = np.clip(refined_points[0, :], x_min, x_max)
                    refined_points[1, :] = np.clip(refined_points[1, :], y_min, y_max)

                    bspline_refined, _ = make_splprep([refined_points[0, :], refined_points[1, :]], s=1.0, k=3)
                    
                    # STEP 3: Validate control points are within acceptable bounds
                    bspline_obj = bspline_from_tck(bspline_refined)
                    
                    # Check control points with slightly tighter bounds
                    ctrl_x_min, ctrl_x_max = bspline_obj.c[:, 0].min(), bspline_obj.c[:, 0].max()
                    ctrl_y_min, ctrl_y_max = bspline_obj.c[:, 1].min(), bspline_obj.c[:, 1].max()
                    
                    # Allow slightly larger margin for control points (smoothing can extrapolate a bit)
                    ctrl_margin = 20  # pixels
                    ctrl_bounds_ok = (
                        ctrl_x_min >= -ctrl_margin and ctrl_x_max <= 75 + ctrl_margin and
                        ctrl_y_min >= -ctrl_margin and ctrl_y_max <= 75 + ctrl_margin
                    )
                    
                    if not ctrl_bounds_ok:
                        print(f"    WARNING: {seg_label} control points escaped bounds:")
                        print(f"      X: [{ctrl_x_min:.1f}, {ctrl_x_max:.1f}] (should be ~[0, 75])")
                        print(f"      Y: [{ctrl_y_min:.1f}, {ctrl_y_max:.1f}] (should be ~[0, 75])")
                        print("      Retrying with tighter smoothing...")
                        
                        # STEP 4: Retry with progressively tighter smoothing
                        smoothing_attempts = [0.5, 0.1, 0.01]
                        for attempt_smoothing in smoothing_attempts:
                            bspline_refined, _ = make_splprep(
                                [refined_points[0, :], refined_points[1, :]], 
                                s=attempt_smoothing, 
                                k=3
                            )
                            
                            bspline_obj = bspline_from_tck(bspline_refined)
                            ctrl_x_min, ctrl_x_max = bspline_obj.c[:, 0].min(), bspline_obj.c[:, 0].max()
                            ctrl_y_min, ctrl_y_max = bspline_obj.c[:, 1].min(), bspline_obj.c[:, 1].max()
                            
                            ctrl_bounds_ok = (
                                ctrl_x_min >= -ctrl_margin and ctrl_x_max <= 75 + ctrl_margin and
                                ctrl_y_min >= -ctrl_margin and ctrl_y_max <= 75 + ctrl_margin
                            )
                            
                            if ctrl_bounds_ok:
                                print(f"      SUCCESS with smoothing={attempt_smoothing}")
                                print(f"        X: [{ctrl_x_min:.1f}, {ctrl_x_max:.1f}]")
                                print(f"        Y: [{ctrl_y_min:.1f}, {ctrl_y_max:.1f}]")
                                break
                        
                        if not ctrl_bounds_ok:
                            print("      FAILED: Even with s=0.01, control points still out of bounds")
                            print(f"      Skipping {seg_label} (refinement produced invalid geometry)")
                            continue  # Don't add this segment to results
                    
                    # STEP 5: Final validation - check for extreme control points
                    max_allowed_extent = 150  # Absolute maximum (2× crop box size)
                    if (abs(ctrl_x_min) > max_allowed_extent or abs(ctrl_x_max) > max_allowed_extent or
                        abs(ctrl_y_min) > max_allowed_extent or abs(ctrl_y_max) > max_allowed_extent):
                        print(f"    REJECTING: {seg_label} has extreme control points (>150 pixels)")
                        print(f"      X: [{ctrl_x_min:.1f}, {ctrl_x_max:.1f}]")
                        print(f"      Y: [{ctrl_y_min:.1f}, {ctrl_y_max:.1f}]")
                        continue
                    
                    # Success! Add to results
                    refined_segments_for_this_ne[seg_label] = bspline_refined
                
                except Exception as e:
                    print(f"    --> ERROR: Spline fitting failed for {seg_label} after refinement.")
                    print(f"        Error: {e}")
                    traceback.print_exc()

            if refined_segments_for_this_ne:
                final_refined_splines_dict[ne_label] = refined_segments_for_this_ne

        # Before return
        summary = self.summarize_segment_log(segment_log)
        print("\n=== REFINEMENT SUMMARY ===")
        print(f"Total profiles attempted: {summary['total_profiles']}")
        print(f"Success rate: {summary['by_status'].get('success', 0) / summary['total_profiles']:.1%}")

        if summary['failure_breakdown']:
            print("\nFailure breakdown:")
            for fail_type, count in summary['failure_breakdown'].items():
                print(f"  {fail_type}: {count}")

        # Save detailed refinement report
        try:
            print(f"DEBUG: Starting save_refinement_report for {self._FoV_id} ch{self._channel}")
            
            output_base_dir = Path(self._get_cfg().get('directories', {}).get('output root', ''))
            output_dir = output_base_dir / Path(self._get_cfg().get('refinement',{}).get('output subdirectory', 'refined_fit'))
            print(f"DEBUG: Output dir will be: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True) 

            report_stats = self.save_refinement_report(
                segment_log, 
                output_dir, 
                channel=self._channel,
                fov_id=self._FoV_id
            )

            print(f"DEBUG: Saved report with {report_stats['n_success']}/{report_stats['total_profiles']} successes")

        except Exception as e:
            print(f"ERROR: Could not save refinement report: {e}")
            traceback.print_exc()

        # === ADD QUALITY METADATA TO SPLINES ===
        quality_metadata = {}
        for ne_label in final_refined_splines_dict.keys():
            quality_metadata[ne_label] = {}
            
            for seg_label in final_refined_splines_dict[ne_label].keys():
                # Extract stats for this specific segment from segment_log
                seg_log = [entry for entry in segment_log 
                        if entry.get('ne_label') == ne_label and entry.get('seg_label') == seg_label]
                
                if seg_log:
                    n_total = len(seg_log)
                    n_success = sum(1 for e in seg_log if e['status'] == 'success')
                    n_likelihood_fail = sum(1 for e in seg_log if e['status'] == 'fail:likelihood_ratio')
                    n_optimization_fail = sum(1 for e in seg_log if e['status'] == 'fail:all_step_sizes')
                    n_curvature_fail = sum(1 for e in seg_log if e.get('was_reverted', False))
                    
                    quality_metadata[ne_label][seg_label] = {
                        'n_total': n_total,
                        'n_success': n_success,
                        'n_curvature_fail': n_curvature_fail,
                        'n_likelihood_fail': n_likelihood_fail,
                        'n_optimization_fail': n_optimization_fail
                    }

        return final_refined_splines_dict, segment_log, quality_metadata
    
    def _optimize_single_profile_adaptive(self, single_intensity_profile, single_dist_profile, model_name):
        """
        Attempts optimization with adaptive step size strategy.
        
        Reference:
            Marquardt (1963) - Adaptive damping improves convergence
            in ill-conditioned problems by adjusting between gradient
            descent (large λ) and Gauss-Newton (small λ).
        """
        device = self._get_current_device()
        line_length = self._get_line_length()
        default_mu = line_length / 2.0
        
        # Try multiple step sizes: coarse to fine
        step_sizes = [1.0, 0.5, 0.1, 0.01]
        
        best_result = None
        best_residual = np.inf
        
        for step_size in step_sizes:
            try:
                result_mu, status, predicted = self._try_optimization_with_params(
                    single_intensity_profile, 
                    single_dist_profile, 
                    model_name,
                    step_size=step_size,
                    timeout_sec=5  # Increased from 2 to account for Richards-Gaussian complexity
                )
                
                if status == "success" and predicted is not None:
                    residual = self._calculate_fit_residual(
                        predicted, 
                        single_intensity_profile
                    )
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_result = (result_mu, status, predicted)
                        
                        # If residual is good enough, accept and stop trying
                        mean_intensity = np.mean(single_intensity_profile)
                        if residual < 0.1 * mean_intensity:
                            break
                            
            except Exception as e:
                continue
        
        if best_result is not None:
            return best_result
        else:
            return default_mu, "fail:all_step_sizes", None


    def _try_optimization_with_params(self, single_intensity_profile, single_dist_profile, model_name, step_size=0.1, timeout_sec=5):
        """
        Single optimization attempt with specified parameters.
        
        This is essentially your current _optimize_single_profile but with
        configurable step_size and timeout.
        """
        device = self._get_current_device()
        line_length = self._get_line_length()
        default_mu = line_length / 2.0

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)

        try:
            intensity_batch = single_intensity_profile[np.newaxis, :]
            dist_batch = single_dist_profile[np.newaxis, :]
            
            theta_variable, dist_variable, intensity_variable, cost_weight = \
                self._setup_theseus_model(intensity_batch, dist_batch, model_name)
            
            cost_function = self._setup_th_ad_cost_fn(
                theta_variable, 
                intensity_variable, 
                dist_variable, 
                cost_weight
            )
            
            objective = th.Objective()
            objective.add(cost_function)
            
            # Use the passed step_size instead of config default
            cfg = self._get_lm_cfg()
            optimizer = th.LevenbergMarquardt(
                objective,
                max_iterations=cfg['iterations'],
                step_size=step_size,  # ← Use passed parameter
                step_radius=cfg['step_radius']
            )
            
            theseus_layer = th.TheseusLayer(optimizer)
            theseus_layer.to(device)
            initial_guess_tensor = theta_variable.tensor.to(device)
            inputs = {"theta": initial_guess_tensor}
            
            final_state, info = theseus_layer.forward(inputs)
            signal.alarm(0)
            
            final_params = final_state["theta"].detach()
            
            if torch.isnan(final_params).any():
                return default_mu, "fail:nan", None

            # Generate model prediction
            final_theta_var = th.Vector(tensor=final_params, name="theta_final")
            model_function = MODEL_REGISTRY[model_name]
            
            with torch.no_grad():
                predicted_intensity = model_function(dist_variable, final_theta_var)
                predicted_intensity = predicted_intensity.cpu().numpy().squeeze()

            final_params_np = final_params.cpu().numpy()
            mu_raw = final_params_np[0, 7]
            mu_clamped = np.clip(mu_raw, 0, self._get_line_length())

            if abs(mu_raw - mu_clamped) > 0.1:
                return default_mu, "fail:bounds", None

            return mu_clamped, "success", predicted_intensity

        except TimeoutError:
            signal.alarm(0)
            return default_mu, "fail:timeout", None
        except Exception as e:
            signal.alarm(0)
            error_type = type(e).__name__
            return default_mu, f"fail:{error_type}", None


    def _calculate_fit_residual(self, predicted, observed):
        """
        Calculate a scalar measure of fit quality.
        
        Uses normalized root mean square error (NRMSE) so that
        residuals are comparable across profiles with different
        intensity ranges.
        
        Args:
            predicted: Model prediction array
            observed: Measured data array
            
        Returns:
            float: NRMSE value (lower is better)
        """
        residuals = predicted - observed
        
        # Root mean square error
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Normalize by data range for comparability
        data_range = np.ptp(observed)  # peak-to-peak (max - min)
        
        if data_range < 1e-6:
            # Avoid division by zero for flat profiles
            return np.inf
        
        nrmse = rmse / data_range
        
        return nrmse

    def _optimize_single_profile(self, single_intensity_profile, single_dist_profile, model_name):
        device = self._get_current_device()
        line_length = self._get_line_length()
        default_mu = line_length / 2.0

        # --- TIMEOUT TRAP ---
        # Set the signal handler for the alarm
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)  # NOTE Set a 5-second timeout

        try:
            # --- Create a "batch of one" ---
            # Theseus expects a batch dimension
            intensity_batch = single_intensity_profile[np.newaxis, :]
            dist_batch = single_dist_profile[np.newaxis, :]
            
            # --- Run the full Theseus setup for this *one profile* ---
            theta_variable, dist_variable, intensity_variable, cost_weight = \
                self._setup_theseus_model(intensity_batch, dist_batch, model_name)
            
            cost_function = self._setup_th_ad_cost_fn(
                theta_variable, 
                intensity_variable, 
                dist_variable, 
                cost_weight
            )
            
            objective = th.Objective()
            objective.add(cost_function)
            optimizer = self._setup_th_lm_optimizer(objective)
            
            theseus_layer = th.TheseusLayer(optimizer)
            theseus_layer.to(device)
            initial_guess_tensor = theta_variable.tensor.to(device)
            inputs = {"theta": initial_guess_tensor}
            
            # --- Run optimization ---
            final_state, info = theseus_layer.forward(inputs)
            signal.alarm(0)  # Disable the alarm on success
            
            final_params = final_state["theta"].detach().cpu().numpy()
            
            if np.isnan(final_params).any():
                print(f"      → NaN failure! Intensity: max={single_intensity_profile.max():.1f}, "
                        f"min={single_intensity_profile.min():.1f}, range={single_intensity_profile.max()-single_intensity_profile.min():.1f}")
                return default_mu, "fail:nan", None
            
            # --- START: Model Intensity Prediction --- #
            # theseus var from final params to pass to model
            final_theta_var = th.Vector(
                tensor=torch.tensor(final_params, dtype=torch.float32, device=device),
                name="theta_final"
            )
            model_function = MODEL_REGISTRY[model_name]

            # Evaluate the model with final parameters
            # This returns the predicted intensities at each distance point
            with torch.no_grad():
                predicted_intensity = model_function(dist_variable, final_theta_var)
                predicted_intensity = predicted_intensity.cpu().numpy().squeeze()
            # --- END: Model Intensity Prediction --- #

            # Clamp mu to valid range [0, line_length]
            mu_raw = final_params[0, 7]
            mu_clamped = np.clip(mu_raw, 0, self._get_line_length())

            if abs(mu_raw - mu_clamped) > 0.1:
                # Mu was out of bounds, treat as failed optimization
                return default_mu, "fail:bounds", None

            return mu_clamped, "success", predicted_intensity

        except TimeoutError as e:
            # This is the trap catching the timeout
            print("    --> TIMEOUT: Theseus took too long. Reverting.")
            return default_mu, "fail:timeout", None

        except Exception as e:
            signal.alarm(0) # Disable alarm on other errors too
            error_type = type(e).__name__
            print(f"    --> Warning: Theseus failed with {error_type}: {str(e)[:100]}")  # First 100 chars
            return default_mu, f"fail:{error_type}", None

    def _pull_segment_profiles(self, mean_ne_img, seg_label, bspline_obj):
        print(f"Extracting all profiles for {seg_label}...")
        line_length = self._get_line_length()
        # Sample points and derivatives ONLY from the current segment
        u_values_full = get_u_range_from_bspline(bspline_obj, self._get_sampling_density())
        segment_points_xy = bspline_obj(u_values_full)
        segment_derivs_xy = bspline_obj(u_values_full, nu=1)
        num_points_in_segment = segment_points_xy.shape[1]
        num_samples_on_normal = self._get_samples_on_normal()

        _, normal_endpoints = calc_tangent_endpts(
            segment_points_xy, segment_derivs_xy, line_length, True
        )

        # Extract intensity profiles points on this segment
        intensity_profiles, dist_along_norm, validity_mask = \
        extract_profile_along_norm(
                                    mean_ne_img, 
                                    # ssegment_points_xy, 
                                    normal_endpoints, 
                                    num_points_in_segment, 
                                    num_samples_on_normal)

        return intensity_profiles, dist_along_norm, validity_mask

    def _prep_seg_profiles_for_optim(self, intensity_profiles, noise_multiplier = 0.005):
        print("prepping profile for optimization")
        noise_level = np.max(intensity_profiles) * noise_multiplier
        jitter = np.random.randn(*intensity_profiles.shape) * noise_level
        intensity_profiles = intensity_profiles + jitter
        return intensity_profiles


    def _setup_theseus_model(self, intensity_array, dist_array, model_name):
        # establish model and setup initial guess accordingly
        model_name = self._get_model_name()
        model_cfg = self._get_model_cfg()
        line_length = self._get_line_length()
        device = self._get_current_device()

        intensity_tensor = torch.tensor(intensity_array, dtype=torch.float32, device=device)
        dist_tensor = torch.tensor(dist_array, dtype=torch.float32, device=device)

        if model_name == "richards_gaussian":
            initial_guess_tensor = construct_rich_gaus_initial_guess(
                dist_along_norm_tensor=dist_tensor,
                intensity_init_tensor=intensity_tensor,
                line_length=line_length,
                num_parameters=model_cfg['parameters'],
                intensity_params=model_cfg['initial_guess'],
                device=device
            )
        elif model_name == "gaussian_linear":
            initial_guess_tensor = construct_gaus_lin_initial_guess(
                dist_tensor, intensity_tensor, device
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # --- Theseus Variables ---        
        theta_var = th.Vector(tensor=initial_guess_tensor, name="theta")
        dist_var = th.Variable(tensor=dist_tensor, name="distances")
        intensity_var = th.Variable(tensor=intensity_tensor, name="intensities")

        # ** Define the Cost Function and Weight **
        poisson_w_diag = 1.0 / torch.sqrt(intensity_tensor + 1e-6)
        cost_wt = th.DiagonalCostWeight(poisson_w_diag)

        return theta_var, dist_var, intensity_var, cost_wt
    
    def _setup_th_ad_cost_fn(self, theta_var, intensity_var, dist_var, cost_wt):
        model_name = self._get_model_name()
        # Model Function
        model_function = MODEL_REGISTRY[model_name]
        # --- Use a lambda to pass the chosen model_function to the error function ---
        cost_fn = th.AutoDiffCostFunction(
                optim_vars=[theta_var],
                err_fn=lambda optim_vars, aux_vars: model_error_func(
                    optim_vars, aux_vars, model_function
                ),
                dim=intensity_var.tensor.shape[1],  # = S
                aux_vars=[dist_var, intensity_var],
                cost_weight=cost_wt,
                name=f"{model_name}_single_fit"
            )
        return cost_fn

    def _setup_th_lm_optimizer(self, objective_to_optimize):
        cfg = self._get_lm_cfg()
        
        optimizer = th.LevenbergMarquardt(
            objective_to_optimize,  # Use the objective that was passed in
            max_iterations=cfg['iterations'],
            step_size=cfg['step_size'],
            step_radius=cfg['step_radius']
        )
        return optimizer


    def _refine_single_model(self, intensity_profiles, dist_along_norm, model_function, model_config):
        device = self._current_device

        theta_variable, dist_variable, intensity_variable, cost_weight = \
            self._setup_theseus_model(intensity_profiles, dist_along_norm, model_config['name'])
        
        cost_function = self._setup_th_ad_cost_fn(theta_variable, intensity_variable, dist_variable, model_function, cost_weight, model_config['name'])

        objective = th.Objective()
        objective.add(cost_function)

        optimizer = self._setup_th_lm_optimizer(objective)
        
        # Create the Theseus Layer, which makes the solver a callable module
        theseus_layer = th.TheseusLayer(optimizer)
        
        # Move the layer and data to the correct device
        theseus_layer.to(device)
        # QUESTION do I need these?
        # dist_tensor = dist_variable.tensor.to(device)
        # intensity_tensor = intensity_variable.tensor.to(device)
        initial_guess_tensor = theta_variable.tensor.to(device)

        # Provide the initial values for the variables to be optimized
        inputs = {"theta": initial_guess_tensor}
        
        # ================== NEW FORCED DEBUGGING BLOCK ==================
        print("\n\n*** DEBUG: REACHED THE LINE RIGHT BEFORE THE CRASH. ***")
        
        # Save the inputs to a file for inspection
        debug_data = {
            "initial_guess": theta_variable.tensor.cpu(),
            "intensities": intensity_variable.tensor.cpu(),
            "distances": dist_variable.tensor.cpu()
        }
        torch.save(debug_data, "debug_theseus_inputs.pt")
        print("*** DEBUG: Successfully saved 'debug_theseus_inputs.pt'. Exiting now. ***\n\n")
        
        import sys
        sys.exit() # Force the script to stop here
        # ==============================================================

        final_state, info = theseus_layer.forward(inputs)
        
        # --- Calculate Goodness-of-Fit ---
        final_params = final_state["theta"].detach()
        log_likelihood = info.best_err.item() # Note: This is related to log-likelihood
        n_params = model_config['parameters']
        n_datapoints = intensity_profiles.shape[1]
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_datapoints) - 2 * log_likelihood
        print(f'AIC:{aic}\nBIC: {bic}')
        return {
            "model_name": model_config['name'],
            "final_params": final_params.cpu().numpy(),
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic
        }


    # Define a handler that raises an exception
    def _timeout_handler(signum, frame):
        raise TimeoutError("Theseus optimization call timed out.")

    def _check_poisson_consistency(self, predicted, observed, gain=1.0, sigma_threshold=5.0, max_outlier_fraction=0.05):
        """
        Check if residuals are consistent with Poisson noise.
        
        For Poisson-distributed photon counts with expected value μ,
        Var(counts) = μ. 
        Conversion to ADU by gain g:
            For a detector with gain g (ADU/photon or e-/ADU):
                Var(ADU) = Mean(ADU) / g
                \(\sigma \)_ADU = sqrt(Mean_ADU / g)
            
        Args:
            predicted: Model prediction (ADU)
            observed: Measured data (ADU)  
            gain: Camera gain (e-/ADU), from calibration
            sigma_threshold: Rejection threshold in units of \(\sigma \)
            
        Returns:
            bool: True if residuals are consistent with Poisson noise
        
        Note:
            The theoretical outlier rate for k*σ threshold is erfc(k/√2):
            - 3σ: 0.27%
            - 4σ: 0.0063%  
            - 5σ: 5.7e-5%
            
            max_outlier_fraction should be set considering both:
            1. Theoretical rate (very small for high σ)
            2. Expected model imperfection (dominant for real data)
            
            A typical choice is 5% regardless of sigma_threshold, acknowledging
            that model limitations dominate over statistical outliers.

        Reference:
            Thompson, Larson & Webb. Biophys J. 2002;82(5):2775-2783
            "Precise nanometer localization analysis for individual 
            fluorescent probes"
        """

        # Expected standard deviation from Poisson statistics
        #   (uses predicted values as expected)
        expected_std = np.sqrt(np.maximum(predicted, 1.0) / gain)
        
        # Actual residuals
        residuals = np.abs(observed - predicted)
        
        # Fraction of points exceeding threshold
        outlier_fraction = np.mean(residuals > sigma_threshold * expected_std)

        is_consistent = outlier_fraction < max_outlier_fraction
    
        if not is_consistent:
            print(f"      Poisson check failed: {outlier_fraction:.2%} outliers "
                f"(threshold: {max_outlier_fraction:.2%} at {sigma_threshold}σ)")
        
        return is_consistent
    
    def _check_fit_quality_likelihood_ratio(self, predicted, observed, gain=1.0, n_params=11, threshold_delta_aic=10):
        """
        Likelihood ratio test for fit quality assessment.
        
        Compares the fitted model against a null model (constant background).
        Uses Akaike Information Criterion (AIC) to penalize model complexity.
        
        This approach is more appropriate for complex multi-parameter models
        than fixed outlier thresholds, and is consistent with GLRT-based
        particle detection used elsewhere in the pipeline.
        
        References:
            1. Ober, R.J., Ram, S. & Ward, E.S. (2004). "Localization accuracy in 
            single-molecule microscopy." Biophysical Journal, 86(2), 1185-1200.
            - Establishes likelihood ratio framework for localization quality
            
            2. Smith, C.S., Joseph, N., Rieger, B. & Lidke, K.A. (2010). "Fast, 
            single-molecule localization that achieves theoretically minimum 
            uncertainty." Nature Methods, 7(5), 373-375.
            - Uses log-likelihood for quality metrics in single-molecule localization
            
            3. Burnham, K.P. & Anderson, D.R. (2004). "Multimodel inference: 
            Understanding AIC and BIC in model selection." Sociological Methods 
            & Research, 33(2), 261-304.
            - Guidelines for interpreting Δ AIC values (Δ AIC > 10 = strong evidence)
        
        Args:
            predicted: Model prediction in ADU (fitted Richards-Gaussian model)
            observed: Measured data in ADU
            gain: Camera gain in e-/ADU from responsivity calibration
            n_params: Number of model parameters (11 for Richards-Gaussian)
            threshold_delta_aic: Minimum AIC improvement required (default: 10)
            
        Returns:
            bool: True if fitted model is significantly better than null model
            
        Statistical Basis:
            For Poisson-distributed photon counts, the log-likelihood is:
                log L = Σ [n_i × log(λ_i) - λ_i - log(n_i!)]
            
            where n_i = observed counts, λ_i = expected counts (in photons)
            
            AIC = 2k - 2×log(L), where k = number of parameters
            
            Lower AIC indicates better model. Δ AIC = AIC_null - AIC_fitted
            Δ AIC > 10 indicates strong evidence for the fitted model.
        """
        
        # Convert ADU to photon counts using gain
        observed_photons = observed * gain
        predicted_photons = predicted * gain
        
        # Ensure positive values (Poisson requires λ > 0)
        predicted_photons = np.maximum(predicted_photons, 1e-3)
        observed_photons = np.maximum(observed_photons, 0)
        
        # Log-likelihood for fitted model (H₁: Richards-Gaussian)
        # Using Poisson distribution: log P(n|λ) = n×log(λ) - λ - log(n!)
        # We drop the log(n!) term as it's constant for both models
        log_L_fitted = np.sum(
            observed_photons * np.log(predicted_photons) - predicted_photons
        )
        
        # Null model (H₀): constant background = mean intensity
        null_model = np.full_like(observed_photons, np.mean(observed_photons))
        null_model = np.maximum(null_model, 1e-3)
        
        log_L_null = np.sum(
            observed_photons * np.log(null_model) - null_model
        )
        
        # Calculate AICs (Akaike Information Criterion)
        # AIC = 2k - 2×log(L), where lower AIC = better model
        aic_fitted = 2 * n_params - 2 * log_L_fitted
        aic_null = 2 * 1 - 2 * log_L_null  # Null model has 1 parameter (mean)
        
        # Δ AIC = AIC_null - AIC_fitted
        # Positive Δ AIC means fitted model is better
        delta_aic = aic_null - aic_fitted
        
        # Burnham & Anderson (2004) guidelines:
        # Δ AIC < 2: Weak evidence
        # Δ AIC 4-7: Moderate evidence  
        # Δ AIC > 10: Strong evidence for better model
        
        is_good_fit = delta_aic > threshold_delta_aic
        
        if not is_good_fit:
            print(f"      Likelihood test failed: ΔAIC={delta_aic:.1f} (need >{threshold_delta_aic})")
        
        return is_good_fit

    def identify_nonphysical_regions(self, spline_points, spline_derivs, max_angle_change_deg=1.0):
        """
        Identify regions of spline with non-physical curvature.
        
        The nuclear envelope has bending modulus κ ≈ 10-20 k_BT, giving persistence 
        length L_p ≈ 50-100 nm. At sampling distance ~20 nm, expected angular 
        deviation is ~0.3°. We use 1° threshold (3× safety margin).
        
        References:
            1. Zimmerberg, J. & Kozlov, M.M. (2006). "How proteins produce cellular 
            membrane curvature." Nature Reviews Molecular Cell Biology, 7(1), 9-19.
            - Box 1: Membrane bending modulus κ ≈ 10-20 k_BT
            
            2. Helfrich, W. (1973). "Elastic properties of lipid bilayers: theory 
            and possible experiments." Zeitschrift für Naturforschung C, 
            28(11-12), 693-703.
            - Bending energy theory: E = (κ/2) ∫(c₁ + c₂)² dA
        
        Args:
            spline_points: (2, N) array of [y, x] coordinates
            spline_derivs: (2, N) array of tangent vectors
            max_angle_change_deg: Maximum allowed angle change between consecutive points
            
        Returns:
            valid_mask: (N,) boolean array, True for physically reasonable points
            angle_changes_deg: (N-1,) array of angle changes in degrees
        """
    
        # Calculate tangent angles
        angles = np.arctan2(spline_derivs[1, :], spline_derivs[0, :])
        
        # Angle differences (normalized to [-π, π])
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        angle_changes_deg = np.abs(np.rad2deg(angle_diffs))
        
        # Flag transitions that exceed threshold
        bad_transitions = angle_changes_deg > max_angle_change_deg
        
        # Mark points on BOTH sides of bad transition as invalid
        valid_mask = np.ones(len(angles), dtype=bool)
        bad_indices = np.where(bad_transitions)[0]
        
        for idx in bad_indices:
            valid_mask[idx] = False      # Point before transition
            valid_mask[idx + 1] = False  # Point after transition
        
        return valid_mask, angle_changes_deg


    def apply_curvature_filter(self, optimized_mu_full, segment_points_xy, 
                            segment_derivs_xy, line_length, max_angle_change_deg=1.0):
        """
        Apply curvature-based filtering to refined positions.
        
        Reverts points with non-physical curvature to their original positions
        (no refinement applied). This prevents optimization artifacts while 
        preserving good fits.
        
        Args:
            optimized_mu_full: (N,) array of refined mu positions
            segment_points_xy: (2, N) array of original spline points
            segment_derivs_xy: (2, N) array of original tangent vectors
            line_length: Length of normal line for refinement
            max_angle_change_deg: Curvature threshold
            
        Returns:
            filtered_mu: (N,) array with bad points reverted to default
            n_reverted: Number of points that failed curvature check
        """
        
        # Compute refined spline points
        refined_points = adjust_spline_points(
            optimized_mu_full, 
            line_length, 
            segment_points_xy, 
            segment_derivs_xy
        )
        
        # Compute tangents via finite differences
        refined_derivs = np.zeros_like(refined_points)
        refined_derivs[:, 1:-1] = (refined_points[:, 2:] - refined_points[:, :-2]) / 2.0
        refined_derivs[:, 0] = refined_points[:, 1] - refined_points[:, 0]
        refined_derivs[:, -1] = refined_points[:, -1] - refined_points[:, -2]
        
        # Check curvature
        valid_mask, angle_changes = self.identify_nonphysical_regions(
            refined_points, 
            refined_derivs, 
            max_angle_change_deg
        )
        
        # Revert invalid points to default (no refinement)
        default_mu = line_length / 2.0
        filtered_mu = optimized_mu_full.copy()
        filtered_mu[~valid_mask] = default_mu
        
        n_reverted = np.sum(~valid_mask)
        
        if n_reverted > 0:
            max_angle = np.max(angle_changes[~valid_mask[:-1]])
            print(f"    Curvature filter: {n_reverted} points reverted (max angle: {max_angle:.2f}°)")
        
        return filtered_mu, n_reverted, valid_mask

    def save_refinement_report(self, segment_log, output_dir, channel, fov_id):
        """
        Save comprehensive refinement statistics for analysis and comparison.
        
        Outputs:
            1. Text summary report
            2. CSV with per-profile details for plotting
            3. JSON with aggregate statistics
        """
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert segment_log to DataFrame
        df = pd.DataFrame(segment_log)
        
        # === 1. AGGREGATE STATISTICS ===
        summary_stats = {
            'timestamp': timestamp,
            'fov_id': fov_id,
            'channel': channel,
            'total_profiles': len(df),
            # 'n_success': (df['status'] == 'success').sum(),
            'n_success': int((df['status'] == 'success').sum()),
            'success_rate': (df['status'] == 'success').sum() / len(df) if len(df) > 0 else 0,
            'camera_gain': self._get_camera_gain(),
            'model_name': self._get_model_name(),
            'n_parameters': 11 if self._get_model_name() == 'richards_gaussian' else 5
        }
        
        # Failure breakdown
        # failure_counts = df[df['status'] != 'success']['status'].value_counts().to_dict()
        # summary_stats['failure_breakdown'] = failure_counts
        failure_counts = {k: int(v) for k, v in df[df['status'] != 'success']['status'].value_counts().to_dict().items()}

        # Intensity statistics by status
        status_groups = df.groupby('status')
        for status, group in status_groups:
            if 'intensity_mean' in group.columns:
                summary_stats[f'{status}_intensity_mean'] = group['intensity_mean'].mean()
                summary_stats[f'{status}_intensity_std'] = group['intensity_std'].mean()
                summary_stats[f'{status}_intensity_range'] = group['intensity_range'].mean()
        
        # === 2. SAVE TEXT REPORT ===
        report_path = output_dir / f"{timestamp}_{fov_id}_ch{channel}_refinement_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"SPLINE REFINEMENT REPORT - {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"FoV ID: {fov_id}\n")
            f.write(f"Channel: {channel}\n")
            f.write(f"Camera Gain: {summary_stats['camera_gain']:.4f} e-/ADU\n")
            f.write(f"Model: {summary_stats['model_name']} ({summary_stats['n_parameters']} parameters)\n")
            f.write(f"\n")
            
            f.write(f"OVERALL PERFORMANCE\n")
            f.write(f"-" * 40 + "\n")
            f.write(f"Total profiles attempted: {summary_stats['total_profiles']}\n")
            f.write(f"Successful fits: {summary_stats['n_success']}\n")
            f.write(f"Success rate: {summary_stats['success_rate']:.1%}\n")
            f.write(f"\n")
            
            f.write(f"FAILURE BREAKDOWN\n")
            f.write(f"-" * 40 + "\n")
            for fail_type, count in failure_counts.items():
                pct = count / summary_stats['total_profiles'] * 100
                f.write(f"  {fail_type}: {count} ({pct:.1f}%)\n")
            f.write(f"\n")
            
            f.write(f"INTENSITY STATISTICS BY STATUS\n")
            f.write(f"-" * 40 + "\n")
            for status in df['status'].unique():
                if f'{status}_intensity_mean' in summary_stats:
                    f.write(f"\n{status}:\n")
                    f.write(f"  Mean intensity: {summary_stats[f'{status}_intensity_mean']:.1f}\n")
                    f.write(f"  Std dev: {summary_stats[f'{status}_intensity_std']:.1f}\n")
                    f.write(f"  Range: {summary_stats[f'{status}_intensity_range']:.1f}\n")
        
        # === 3. SAVE CSV (for plotting) ===
        csv_path = output_dir / f"{timestamp}_{fov_id}_ch{channel}_profile_details.csv"
        df.to_csv(csv_path, index=False)
        
        # === 4. SAVE JSON (for programmatic access) ===
        json_path = output_dir / f"{timestamp}_{fov_id}_ch{channel}_summary_stats.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n=== Refinement report saved ===")
        print(f"  Text: {report_path.name}")
        print(f"  CSV: {csv_path.name}")
        print(f"  JSON: {json_path.name}")
        
        return summary_stats

    def summarize_segment_log(self, segment_log):
        """
        Generate a summary report of refinement results.
        
        Args:
            segment_log: List of dicts with status for each profile
            
        Returns:
            summary: Dict with counts and statistics
        """
        import pandas as pd
        
        df = pd.DataFrame(segment_log)
        
        summary = {
            'total_profiles': len(df),
            'by_status': df['status'].value_counts().to_dict(),
            'by_ne_label': {},
            'by_segment': {}
        }
        
        # Per NE label stats
        for ne_label in df['ne_label'].unique():
            ne_df = df[df['ne_label'] == ne_label]
            summary['by_ne_label'][ne_label] = {
                'total': len(ne_df),
                'success': (ne_df['status'] == 'success').sum(),
                'success_rate': (ne_df['status'] == 'success').sum() / len(ne_df) if len(ne_df) > 0 else 0
            }
        
        # Failure breakdown
        fail_types = df[df['status'].str.startswith('fail')]['status'].value_counts()
        summary['failure_breakdown'] = fail_types.to_dict() if not fail_types.empty else {}
        
        return summary

    # --- Getters and Setters --- #

    # config contains (global) directories and the necessary remindment and shared ne_fit key-value pairs
    def _get_cfg(self):
        return self._cfg

    def _get_model_name(self):
        return self._model_name
    
    def _get_model_cfg(self):
        return self._model_cfg

    def _get_current_device(self):
        return self._current_device
    
    def _get_camera_gain(self):
        return self._camera_gain

    def _get_FoV_id(self):
        return self._FoV_id

    def _get_ne_imgs(self):
        return self._ne_images
    
    def _get_channel(self):
        return self._channel
    
    def _get_sampling_density(self):
        return self._cfg.get('refinement', {}).get('final_sampling_density', 16)

    def _get_line_length(self):
        return self._cfg.get('line_length', 12)

    def _get_samples_on_normal(self):
        return self._cfg.get('n_samples_along_normal', 100)

    def _get_prep_for_opt(self):
        return self._cfg.get('refinement', {}).get('prep_for_opt', True)

    def _get_noise_multiplier(self):
        return self._cfg.get('refinement', {}).get('noise_multiplier', 0.005)
    
    def _get_lm_cfg(self):
        return self._lm_cfg

def adjust_spline_points(optimized_mu, line_length, points, derivatives):
    position_change = optimized_mu - (line_length / 2)

    # Calculate the unit normal vectors (this can be a helper function)
    norm_magnitudes = np.linalg.norm(derivatives, axis=0)
    unit_normals_yx = np.vstack((-derivatives[1, :], derivatives[0, :])) / (norm_magnitudes + 1e-9)

    # Update the original sample points by moving them along their normal vectors
    refined_points = points + unit_normals_yx * position_change

    return refined_points


def extract_profile_along_norm(img_mean, normal_endpts, normal_lines_n, n_samples_along_normal = 10):
    img_shape = img_mean.shape
    zi_array = np.zeros((normal_lines_n, n_samples_along_normal))
    dist_array = np.zeros((normal_lines_n, n_samples_along_normal))
    validity_mask = np.ones(normal_lines_n, dtype=bool)

    for i in range(normal_lines_n):
        y0, x0 = normal_endpts[0][0][i], normal_endpts[0][1][i]
        y1, x1 = normal_endpts[1][0][i], normal_endpts[1][1][i]
        
        # Check if the line is out of bounds
        if not (0 <= np.round(x0) < img_shape[1] - 2 and 0 <= np.round(y0) < img_shape[0] - 2 and
                0 <= np.round(x1) < img_shape[1] - 2 and 0 <= np.round(y1) < img_shape[0] - 2):
            validity_mask[i] = False # Mark this profile as bad
            continue
        
        y, x = np.linspace(y0, y1, n_samples_along_normal), \
            np.linspace(x0, x1, n_samples_along_normal)

        # --- LINEAR INTERPOLATION ---
        # safety net against crapping out with extreme changes in intensity
        coords = np.vstack((y, x))
        zi = scipy.ndimage.map_coordinates(img_mean, coords, order=1, mode='nearest')

        # zi = img_mean[np.round(y).astype(int), np.round(x).astype(int)]
        zi_array[i,:] = zi

    # equal space accounts for line slope by weighting as an emergent property
    #   of more points associated with pixels the normal line traverses a longer
    #   distance through

        dist_alongline = np.cumsum(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
        dist_alongline = np.append(0, dist_alongline)
        dist_array[i,:] = dist_alongline

    return zi_array, dist_array, validity_mask


