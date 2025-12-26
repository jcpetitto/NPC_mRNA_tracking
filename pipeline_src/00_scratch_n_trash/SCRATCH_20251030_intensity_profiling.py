import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile
import torch
import theseus as th
from typing import Dict, Any
from datetime import datetime

from scipy.optimize import minimize_scalar
from scipy.interpolate import splprep
import traceback

from utils.MLEstimation import (
    model_error_func, 
    construct_rich_gaus_initial_guess, 
    construct_gaus_lin_initial_guess, 
    MODEL_REGISTRY
)

from utils.npc_spline_refinement import NESplineRefiner, extract_profile_along_norm, adjust_spline_points
from tools.utility_functions import extract_cropped_images, filter_data_by_indices, find_segments, calc_tangent_endpts
from tools.geom_tools import get_u_range_from_bspline, bspline_from_tck, points_by_arc_len_parm_robust, calc_bspline_curvature, build_curve_bridge

DESIRED_SAMPLING_INTERVAL_NM = 0.5
NM_PER_PIXEL = 128.0

# loading data created and saved during a run of main_debugging.py specifically for testing dual label functionality
with open('output/dual_debug_img_proc.pkl', 'rb') as f:
    img_proc = pickle.load(f)

with open('output/dual_debug_all_exper.pkl', 'rb') as f:
    all_experiments = pickle.load(f)

with open('output/dual_debug_all_ne_bspl.pkl', 'rb') as f:
    all_ne_bsplines = pickle.load(f)

with open('output/dual_debug_all_ne_crop.pkl', 'rb') as f:
    all_ne_crop_boxes = pickle.load(f)

with open('output/dual_debug_all_reg.pkl', 'rb') as f:
    all_registration = pickle.load(f)

# --- Set-up for testing: Data Loading and what not --- #

frame_range = img_proc._get_cfg()['ne_fit']['frame_range']
FoV_dict = all_experiments['BMY9999_99_99_9999']
# FoV 0083
FoV_id = '0083'
img_path_ch1 = os.path.join(FoV_dict[FoV_id]['FoV_collection_path'], FoV_dict[FoV_id]['imgs']['fn_track_ch1'])
img_path_ch2 = os.path.join(FoV_dict[FoV_id]['FoV_collection_path'], FoV_dict[FoV_id]['imgs']['fn_track_ch2'])

raw_img_stack_ch1 = tifffile.imread(img_path_ch1)[frame_range[0]:frame_range[1], :, :]
img_ch1 = np.mean(raw_img_stack_ch1, axis=0, dtype='uint16')
global_min_ch1 = np.min(img_ch1)
global_max_ch1 = np.max(img_ch1)
fig, ax = plt.subplots()
ax.imshow(img_ch1, cmap='inferno', origin='upper', vmin=global_min_ch1, vmax=global_max_ch1)
fig.show()
plt.savefig("/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/test_intensity_profile/0083_ch1_mean_img.png")
plt.close(fig)

raw_img_stack_ch2 = tifffile.imread(img_path_ch2)[frame_range[0]:frame_range[1], :, :]
img_ch2 = np.mean(raw_img_stack_ch2, axis=0, dtype='uint16')
global_min_ch2 = np.min(img_ch2)
global_max_ch2 = np.max(img_ch2)
fig, ax = plt.subplots()
ax.imshow(img_ch2, cmap='inferno', origin='upper', vmin=global_min_ch2, vmax=global_max_ch2)
fig.show()
plt.savefig("/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/test_intensity_profile/0083_ch2_mean_img.png")
plt.close(fig)


FoV_ch1_crop_dict = all_ne_crop_boxes['BMY9999_99_99_9999']['ch1'][FoV_id]
FoV_ch2_crop_dict = all_ne_crop_boxes['BMY9999_99_99_9999']['ch2'][FoV_id]
spline_refiner_ch1 = NESplineRefiner('ch1', img_path_ch1, FoV_id, FoV_ch1_crop_dict, img_proc._get_cfg(), img_proc._get_current_device())
spline_refiner_ch2 = NESplineRefiner('ch2', img_path_ch2, FoV_id, FoV_ch2_crop_dict, img_proc._get_cfg(), img_proc._get_current_device())
init_bsplines_ch1 = all_ne_bsplines['BMY9999_99_99_9999']['ch1'][FoV_id]
init_bsplines_ch2 = all_ne_bsplines['BMY9999_99_99_9999']['ch2'][FoV_id]

cropped_imgs_ch1 = extract_cropped_images(full_img_path = img_path_ch1, frame_range = [0, 250], crop_boxes = FoV_ch1_crop_dict)
cropped_imgs_ch2 = extract_cropped_images(full_img_path = img_path_ch2, frame_range = [0, 250], crop_boxes = FoV_ch2_crop_dict)


# --- Function for Refinement --- #


def test_refinement_with_filtering(
    spline_refiner,
    evaluated_points_xy,     # Shape (2, N) [x, y]
    evaluated_derivatives_xy, # Shape (2, N) [dx, dy]
    intensity_profiles,      # Shape (N, S) numpy array
    dist_along_norm,       # Shape (N, S) numpy array
    model_name,
    is_periodic
):
    """
    Standalone test function that replicates _refine_ne_bsplines
    but includes the pre-filtering "wonky point" check.
    """
    
    # 0. Get common objects from the refiner instance
    device = spline_refiner._current_device
    model_function = MODEL_REGISTRY[model_name]
    cfg_fit = spline_refiner._cfg_fit

    # 1. Create Full Tensors
    intensity_tensor = torch.tensor(intensity_profiles, 
                                    dtype=torch.float32, device=device)

    # 2. NEW: Create the Filter Mask
    # Check 1: Must be finite (no NaNs or Infs)
    valid_mask = torch.isfinite(intensity_tensor).all(dim=1)
    
    # Check 2: Must not be a flat line (std dev > tiny number)
    std_devs = torch.std(intensity_tensor, dim=1)
    # valid_mask = valid_mask & (std_devs > 1e-3)
    valid_mask = valid_mask & (std_devs > 0.5)

    num_total = intensity_tensor.shape[0]
    num_valid = valid_mask.sum().item()
    
    if num_valid == 0:
        raise ValueError("No valid profiles found after filtering. Skipping segment.")
    
    print(f"    Filtering profiles: {num_valid} / {num_total} are valid.")
    
    # 3. Create "Good" Tensors (as *numpy arrays* for the setup function)
    valid_mask_cpu = valid_mask.cpu() # For numpy indexing
    good_intensity_numpy = intensity_profiles[valid_mask_cpu]
    good_dist_numpy = dist_along_norm[valid_mask_cpu]

    # --- 4. YOUR JITTER FIX (MOVED HERE) ---
    # Add jitter *only* to the "good" profiles
    noise_level = np.max(good_intensity_numpy) * 0.005 
    jitter = np.random.randn(*good_intensity_numpy.shape) * noise_level
    good_intensity_numpy = good_intensity_numpy + jitter
    # --- END JITTER FIX ---

    # 5. Run Optimization ONLY on "Good, Jittered" Data
    # We call the refiner's setup methods directly
    theta_variable, dist_variable, intensity_variable, cost_weight = \
        spline_refiner._setup_theseus_model(
            good_intensity_numpy,
            good_dist_numpy,
            model_name
        )
    
    cost_function = spline_refiner._setup_th_ad_cost_fn(
        theta_variable, intensity_variable, dist_variable, 
        model_function, cost_weight, (f'{model_name}_fit')
    )
    
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = spline_refiner._setup_th_lm_optimizer(objective)
    theseus_layer = th.TheseusLayer(optimizer).to(device)

    inputs = {"theta": theta_variable.tensor.to(device)}
    
    # --- SAVE THE DEBUG FILE *BEFORE* THE CRASH ---
    print("    Saving inputs to 'debug_theseus_inputs.pt'...")
    try:
        debug_data = {
            "initial_guess": theta_variable.tensor.cpu(),
            "intensities": intensity_variable.tensor.cpu(), # The "good" intensities
            "distances": dist_variable.tensor.cpu()
        }
        torch.save(debug_data, "debug_theseus_inputs.pt")
        print("    ... Debug file saved.")
    except Exception as e:
        print(f"    ... FAILED to save debug file: {e}")
    # --- END DEBUG BLOCK ---
    
    final_state, info = theseus_layer.forward(inputs)
    final_params_good = final_state["theta"].detach() # Shape [num_valid, 11]

    # 5. NEW: Map Results Back to Full Spline
    mu_good = final_params_good[:, 7] # Index 7 is 'mu'
    line_length = cfg_fit['line_length']
    
    # Create a full-sized array for *all* N original points
    # Initialize with the "no change" offset (center of the line)
    mu_all = torch.full(
        (num_total,), 
        fill_value=(line_length / 2.0), 
        device=device, 
        dtype=torch.float32
    )
    
    # Place the "good" results into the full array
    mu_all[valid_mask] = mu_good
    
    # 6. Adjust Points Using the Full 'mu_all' Array
    # We use the *original* full point/derivative sets
    # adjust_spline_points expects (x, y) points and (dx, dy) derivatives
    refined_points_xy = adjust_spline_points(
        mu_all.cpu().numpy(),
        line_length, 
        evaluated_points_xy,       # Pass (x,y)
        evaluated_derivatives_xy   # Pass (dx,dy)
    )
    
    # splprep expects [x_coords, y_coords]
    # refined_points_xy is already (2, N) with [x, y]
    tck_refined, _ = splprep(
        [refined_points_xy[0, :], refined_points_xy[1, :]], 
        s=1.0, 
        k=3, 
        per=is_periodic  # Correct keyword
    )
    
    return tck_refined


# --- Helper Functions --- #

def plot_bspline_segments(
    image: np.ndarray, 
    segment_dict: Dict[str, Dict[str, Any]],
    title: str = "Initial B-Spline Segments",
    save_path: str = None
):
    """
    Plots a dictionary of B-spline segments on a background image.

    Args:
        image: The 2D numpy array to use as the background.
        segment_dict: The dictionary of segment data, e.g.,
                      {'segment_0': {'bspline_object': <obj>}, ...}
        title: Optional title for the plot.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Plot the background image
    ax.imshow(image, cmap='inferno', origin='upper', alpha=0.7)

    # 2. Get a color map to make each segment distinct
    num_segments = len(segment_dict)
    colors = plt.cm.get_cmap('jet', num_segments)

    # 3. Iterate and plot each segment
    for i, (segment_name, segment_data) in enumerate(segment_dict.items()):
        
        bspline_obj = segment_data.get('bspline_object')
        if bspline_obj is None:
            print(f"Skipping {segment_name}: no 'bspline_object' found.")
            continue

        # Sample the spline to get plotting points
        u_range = np.linspace(0, 1, 100)
        points_xy = bspline_obj(u_range) # Shape (N, 2) [x, y]
        
        x_coords = points_xy[:, 0]
        y_coords = points_xy[:, 1]
        
        # Plot the segment
        ax.plot(
            x_coords, y_coords, 
            color=colors(i), 
            linewidth=2.5, 
            label=segment_name
        )
            
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(fontsize='small')
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax


def plot_refinement_comparison(
    background_image,
    original_points_xy, # Expects (N, 2) [x, y] from spline
    refined_points_xy, # Expects (N, 2) [x, y] from spline
    signal_mask,
    title="Spline Refinement Comparison",
    save_path=None
):
    """
    Plots original spline (x,y) points, color-coded by signal,
    and the final refined (x,y) spline points.
    """
    
    # Convert (x, y) spline coords to (y, x) image coords for plotting
    original_points_yx = np.fliplr(original_points_xy)
    refined_points_yx = np.fliplr(refined_points_xy)
    
    # Separate original points based on signal
    signal_points_yx = original_points_yx[signal_mask]
    no_signal_points_yx = original_points_yx[~signal_mask]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background_image, cmap='inferno', origin='upper', alpha=0.7)

    # 1. Plot Original Points
    ax.scatter(
        no_signal_points_yx[:, 1], no_signal_points_yx[:, 0], 
        c='gray', s=10, alpha=0.5, label='Original (No Signal)'
    )
    ax.scatter(
        signal_points_yx[:, 1], signal_points_yx[:, 0], 
        c='cyan', s=15, label='Original (With Signal)'
    )

    # 2. Plot New Spline Points
    ax.plot(
        refined_points_yx[:, 1], refined_points_yx[:, 0], 
        'r-', linewidth=2, label='Refined Spline'
    )
    ax.scatter(
        refined_points_yx[:, 1], refined_points_yx[:, 0], 
        c='red', s=5, zorder=10
    )

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def extract_profiles_for_segment(mean_ne_img, segment_points_xy, 
                                 segment_derivs_xy, cfg_fit):
    """
    Helper to extract profiles for a single segment's points.
    """
    line_length = cfg_fit['line_length']
    n_samples_along_normal = cfg_fit['n_samples_along_normal']
    
    # Get number of points *from the segment itself*
    num_points_in_segment = segment_points_xy.shape[0]
            
    # calc_tangent_endpts expects (2, N)
    _, normal_endpoints = calc_tangent_endpts(
        segment_points_xy.T, segment_derivs_xy.T, line_length, True
    )
    # SELF somewhere (in the function called here?) is also an interpolation, which is noteworthy             
    # extract_profile_along_norm expects (2, N)
    intensity_profiles, dist_along_norm, error_flag = extract_profile_along_norm(
        mean_ne_img, 
        segment_points_xy.T, 
        normal_endpoints, 
        num_points_in_segment, # Use the correct number of points
        n_samples_along_normal
    )
    # # SELF note idea of jitter re: stairs
    # if not error_flag:
    #     # Add a tiny amount of noise (jitter) to break flat plateaus
    #     # This prevents the zero-gradient Cholesky error
    #     # noise_level = 1e-3  # Very small noise (too small)
    #     # noise_level = 1e-1  # 100x more noise (still small: 0.1 intensity)
    #     noise_level = np.max(intensity_profiles) * 0.005
    #     jitter = np.random.randn(*intensity_profiles.shape) * noise_level
    #     intensity_profiles = intensity_profiles + jitter
    
    return intensity_profiles, dist_along_norm, error_flag

def build_curvature_constrained_bridge(
    p_end_xy, p_start_xy, 
    tangent_end_xy, tangent_start_xy, 
    target_mean_curvature,
    tangent_scale_bounds=(0.05, 1.5)
):
    """
    Optimizes the tangent_scale of a Bezier bridge to match
    a target mean curvature. Assumes all inputs are (x, y) coords.
    """
    
    def objective_func(tangent_scale):
        # build_curve_bridge (from geom_tools) expects (y, x) coords,
        # so we must flip our (x, y) inputs.
        bridge_points_yx = build_curve_bridge(
            np.flip(p_end_xy), np.flip(p_start_xy), 
            np.flip(tangent_end_xy), np.flip(tangent_start_xy), 
            tangent_scale=tangent_scale
        )
        
        if bridge_points_yx.size < 4: return 1e6
        
        bridge_points_xy = np.fliplr(bridge_points_yx)
        
        try:
            tck_bridge, _ = splprep(
                [bridge_points_xy[:, 0], bridge_points_xy[:, 1]], 
                s=0, k=3, per=False
            )
            spline_bridge = bspline_from_tck(tck_bridge, is_periodic=False)
            u_sample = np.linspace(0, 1, 20)
            bridge_curvatures = calc_bspline_curvature(spline_bridge, u_sample)
            mean_curvature = np.nanmean(bridge_curvatures[bridge_curvatures > 0])
            error = (mean_curvature - target_mean_curvature)**2
            return error if np.isfinite(error) else 1e6
        except Exception:
            return 1e6

    opt_result = minimize_scalar(
        objective_func, bounds=tangent_scale_bounds, method='bounded'
    )
    optimal_scale = opt_result.x if opt_result.success else 0.3 # Fallback
    
    return build_curve_bridge(
        np.flip(p_end_xy), np.flip(p_start_xy), 
        np.flip(tangent_end_xy), np.flip(tangent_start_xy), 
        tangent_scale=optimal_scale
    )


def reconstruct_periodic_spline(segments_dict):
    """
    Takes a dictionary of refined, non-periodic spline segments and stitches them
    together to form a single, periodic B-spline using constrained bridges.
    """
    if not segments_dict: return None
    if len(segments_dict) < 2:
        print("    Only one segment, returning as-is (non-periodic).")
        return list(segments_dict.values())[0]['bspline_object']

    sorted_segments = sorted(segments_dict.values(), key=lambda s: s['original_u_range'][0])
    
    segment_only_contour_xy = np.concatenate(
        [points_by_arc_len_parm_robust(s['bspline_object'], n_points=100) 
        for s in sorted_segments], 
        axis=0
    )
    
    tck_segments, _ = splprep(
        [segment_only_contour_xy[:, 0], segment_only_contour_xy[:, 1]], 
        s=1.0, k=3, per=True
    )
    spline_of_segments = bspline_from_tck(tck_segments, is_periodic=True)

    u_full = np.linspace(0, 1, 1000)
    target_curvatures = calc_bspline_curvature(spline_of_segments, u_full)
    target_mean_curv = np.nanmean(target_curvatures[target_curvatures > 0])
    print(f"    Target mean curvature for bridges: {target_mean_curv:.4f}")

    all_points_with_bridges_xy = []
    for i in range(len(sorted_segments)):
        current_segment = sorted_segments[i]
        next_segment = sorted_segments[(i + 1) % len(sorted_segments)] # Wrap around

        current_points_xy = points_by_arc_len_parm_robust(
            current_segment['bspline_object'], n_points=100
        )
        all_points_with_bridges_xy.append(current_points_xy)
        
        p_end_xy = current_points_xy[-1]
        p_start_xy = points_by_arc_len_parm_robust(
            next_segment['bspline_object'], n_points=2
        )[0]
        
        tangent_end_xy = current_segment['bspline_object'].derivative(1)(1.0)
        tangent_start_xy = next_segment['bspline_object'].derivative(1)(0.0)

        bridge_points_yx = build_curvature_constrained_bridge(
            p_end_xy, p_start_xy, 
            tangent_end_xy, tangent_start_xy,
            target_mean_curvature=target_mean_curv 
        )
        
        if bridge_points_yx.size > 0:
            bridge_points_xy = np.fliplr(bridge_points_yx)
            all_points_with_bridges_xy.append(bridge_points_xy[1:-1])

    full_contour_xy = np.concatenate(all_points_with_bridges_xy, axis=0)
    tck_final, _ = splprep(
        [full_contour_xy[:, 0], full_contour_xy[:, 1]], 
        s=1.0, k=3, per=True
    )
    final_periodic_spline = bspline_from_tck(tck_final, is_periodic=True)
    
    return final_periodic_spline

# -----------------------------------------------------------------
# --- 2. MAIN SCRIPT (This is the new workflow) ---
# -----------------------------------------------------------------

# --- A: VISUALIZE THE INPUT (Verify the 'surgery' worked) ---
print("--- Plotting NEW (un-bridged) Initial Segment Results ---")
for ne_label in init_bsplines_ch1:
    
    # This is our NEW data structure from the pickle file
    unbridged_segments_dict = init_bsplines_ch1[ne_label]
    mean_ne_img = cropped_imgs_ch1.get(ne_label)
    
    if mean_ne_img is None:
        print(f"Skipping plot for {ne_label}, no image found.")
        continue
    
    # Check if the data is in the NEW format (a dict of dicts)
    if not (isinstance(unbridged_segments_dict, dict) and 
            'bspline_object' not in unbridged_segments_dict and
            all(isinstance(v, dict) and 'bspline_object' in v for v in unbridged_segments_dict.values())):
        print(f"Skipping plot for {ne_label}, data is not a segment dictionary (likely old format).")
        continue

    print(f"Plotting {len(unbridged_segments_dict)} initial segments for NE {ne_label}...")

    # Uses your plot_bspline_segments function (already in your file)
    plot_bspline_segments(
        mean_ne_img,
        unbridged_segments_dict,
        title=f"Initial Un-Bridged Segments (NE {ne_label})",
        save_path=f"./output/{FoV_id}_ch1_{ne_label}_INITIAL_SEGMENTS.png"
    )

print("--- Initial segment plotting complete. Proceeding to refinement... ---")


# --- B: PHASE 1 (Refine the Segments) ---
print("\n--- PHASE 1: Refining Individual Segments ---")
# This dictionary will hold the results of Phase 1
all_refined_segments = {}

for ne_label_to_test in init_bsplines_ch1:
    
    unbridged_segments_dict = init_bsplines_ch1[ne_label_to_test]
    mean_ne_img = cropped_imgs_ch1.get(ne_label_to_test)
    cfg_fit = spline_refiner_ch1._cfg_fit

    # Skip if no image or not the new data structure
    if mean_ne_img is None: continue
    if not (isinstance(unbridged_segments_dict, dict) and 
            'bspline_object' not in unbridged_segments_dict and
            all(isinstance(v, dict) and 'bspline_object' in v for v in unbridged_segments_dict.values())):
        continue
    
    print(f"Refining {len(unbridged_segments_dict)} un-bridged segments for NE {ne_label_to_test}...")
    
    # This will hold the refined segments *for this NE*
    refined_segments_for_ne = {}

    # 2. Manually run the refinement loop for EACH segment
    for i, (segment_key, bspline_data) in enumerate(unbridged_segments_dict.items()):
        
        print(f"  -> Refining {segment_key}...")
        bspline_obj = bspline_data['bspline_object']
        
        u_values_full = get_u_range_from_bspline(bspline_obj, spline_refiner_ch1._get_sampling_density())
        num_points_full = len(u_values_full)

        if num_points_full < 4:
            print(f"  -> Skipping {segment_key}, not enough points (n={num_points_full}).")
            continue

        try:
            # 1. Sample, 2. Extract Profiles
            segment_points_xy = bspline_obj(u_values_full)
            segment_derivs_xy = bspline_obj.derivative(1)(u_values_full)
            
            all_intensity_profiles, all_dist_along_norm, error_flag = \
                extract_profiles_for_segment(
                    mean_ne_img, segment_points_xy, segment_derivs_xy, cfg_fit
                )
            if error_flag:
                raise ValueError("Error extracting intensity profiles.")

            # 3. Create a "signal mask" that is just ALL TRUE (for plotting)
            signal_mask = np.ones(num_points_full, dtype=bool)

            # 4. RUN THE REFINEMENT FUNCTION
            refined_tck = test_refinement_with_filtering(
                spline_refiner_ch1,
                segment_points_xy.T,       # (2, N) [x, y]
                segment_derivs_xy.T,       # (2, N) [dx, dy]
                all_intensity_profiles, 
                all_dist_along_norm, 
                cfg_fit['default_model'], 
                is_periodic=False 
            )
            if np.isnan(refined_tck[0]).any() or np.isnan(refined_tck[1]).any():
                raise ValueError("Optimizer returned NaNs, resulting in invalid spline TCK.")

            # 5. Get points from the new refined TCK
            refined_bspline_obj = bspline_from_tck(refined_tck, is_periodic=False)
            refined_points_xy = refined_bspline_obj(np.linspace(0, 1, num_points_full))

            # 6. PLOT THE COMPARISON
            plot_save_path = f"./output/{FoV_id}_ch1_{ne_label_to_test}_{segment_key}_refinement.png"
            plot_refinement_comparison(
                background_image=mean_ne_img,
                original_points_xy=segment_points_xy,
                refined_points_xy=refined_points_xy,
                signal_mask=signal_mask,
                title=f"Refinement: {ne_label_to_test} {segment_key} (n={num_points_full})",
                save_path=plot_save_path
            )
            print(f"    ... Plot saved to {plot_save_path}")

            # 7. Store the result, ready for Phase 2
            refined_segments_for_ne[segment_key] = {
                'bspline_object': refined_bspline_obj,
                'is_periodic': False,
                'original_u_range': (i, i+1) # Use index for sorting
            }
            
        except Exception as e:
            print(f"  --> Warning: Refining of {segment_key} failed. {e}.")
            traceback.print_exc()
            continue
            
    all_refined_segments[ne_label_to_test] = refined_segments_for_ne

print(f"\nPhase 1 complete. Refined {len(all_refined_segments)} NEs.")


# --- C: PHASE 2 (Stitch the Refined Segments) ---
print("\n--- PHASE 2: Stitching Refined Segments ---")
all_final_splines = {}

for ne_label, refined_segments_dict in all_refined_segments.items():
    
    if not refined_segments_dict:
        print(f"Skipping stitching for {ne_label}, no refined segments found.")
        continue
    
    print(f"Stitching {len(refined_segments_dict)} refined segments for NE {ne_label}...")
    mean_ne_img = cropped_imgs_ch1[ne_label]
    
    # 1. Run the new reconstruction
    final_spline = reconstruct_periodic_spline(refined_segments_dict)
    all_final_splines[ne_label] = final_spline
    
    # 2. Plot the 'Ta-Da!'
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(mean_ne_img, cmap='inferno', origin='upper', alpha=0.7)

    # Plot Original (the one from detect_initial)
    # We plot the *pieces* of the original for a fair comparison
    orig_segments_dict = init_bsplines_ch1[ne_label]
    for i, (seg_name, seg_data) in enumerate(orig_segments_dict.items()):
        u_orig = np.linspace(0, 1, 100)
        orig_points_xy = seg_data['bspline_object'](u_orig)
        ax.plot(orig_points_xy[:, 0], orig_points_xy[:, 1], 'c--', 
                label='Original Segments' if i == 0 else None)
    
    # Plot Final Reconstructed
    u_final = np.linspace(0, 1, 1000)
    final_points_xy = final_spline(u_final)
    ax.plot(final_points_xy[:, 0], final_points_xy[:, 1], 'r-', 
            linewidth=2, label='Final Reconstructed Spline')

    ax.set_title(f"Final Reconstruction (NE {ne_label})")
    ax.set_aspect('equal')
    ax.legend()
    
    final_plot_path = f"./output/{FoV_id}_ch1_{ne_label}_FINAL_RECONSTRUCTION.png"
    plt.savefig(final_plot_path)
    print(f"    ... Final reconstruction plot saved to {final_plot_path}")
    plt.close()

print(f"\n--- Workflow Complete. Final splines generated for {len(all_final_splines)} NEs. ---")

# print("\n--- Inspecting Debug File ---")
# try:
#     # This will load the .pt file from the *last* crash
#     debug_data = torch.load("debug_theseus_inputs.pt")
    
#     print(f"Debug file keys: {debug_data.keys()}")
    
#     intensities = debug_data["intensities"].cpu().numpy()
#     distances = debug_data["distances"].cpu().numpy()
    
#     num_profiles_in_batch = intensities.shape[0]
#     print(f"Data shape (profiles, samples): {intensities.shape}")
    
#     # --- New Plotting Logic ---
#     # We'll plot a 3x3 grid of random samples from this batch
    
#     num_to_plot = min(num_profiles_in_batch, 9) # Plot up to 9
#     plot_indices = np.random.choice(
#         num_profiles_in_batch, num_to_plot, replace=False
#     )
    
#     fig, axes = plt.subplots(
#         3, 3, figsize=(15, 12), constrained_layout=True
#     )
#     fig.suptitle(
#         f"Random Sample of {num_to_plot} 'Bad' Profiles from Batch", 
#         fontsize=16
#     )
    
#     # Flatten axes array for easy iteration
#     for i, ax in enumerate(axes.flat):
#         if i < num_to_plot:
#             idx = plot_indices[i]
#             profile_to_plot = intensities[idx, :]
#             dist_to_plot = distances[idx, :]
            
#             ax.plot(dist_to_plot, profile_to_plot, 'b-')
#             ax.set_title(f"Profile Index: {idx}")
#             ax.grid(True, linestyle='--', alpha=0.6)
#         else:
#             ax.axis('off') # Hide unused subplots
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_path = f"./output/debug_profile_GRID_plot_{timestamp}.png"        
#     plt.savefig(save_path)
#     plt.close()
#     print(f"... Saved grid of debug profiles to {save_path}")
#     # --- End New Plotting Logic ---

# except Exception as e:
#     print(f"Could not load or plot debug data. (This is OK if it's the first run)")
#     print(f"Error: {e}")
# --- REPLACEMENT FOR "Inspecting Debug File" (line 678) ---

print("\n--- Inspecting Last Failed Batch (debug_theseus_inputs.pt) ---")

def plot_good_vs_bad(good_data, bad_data, timestamp):
    """Plots a grid of good vs. bad profiles."""
    
    num_good = len(good_data)
    num_bad = len(bad_data)
    
    if num_good == 0 and num_bad == 0:
        print("    ... No good or bad profiles found to plot.")
        return

    fig, axes = plt.subplots(2, max(num_good, num_bad), 
                             figsize=(max(num_good, num_bad) * 5, 10), 
                             constrained_layout=True)
    fig.suptitle(f"Good vs. Bad Profiles (Batch @ {timestamp})", fontsize=16)

    # Plot Good Profiles
    for i in range(max(num_good, num_bad)):
        ax = axes[0, i] if max(num_good, num_bad) > 1 else axes[0]
        if i < num_good:
            idx, dist, prof = good_data[i]
            ax.plot(dist, prof, 'g-') # Green for good
            ax.set_title(f"GOOD: Profile Index {idx}")
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.axis('off')

    # Plot Bad Profiles
    for i in range(max(num_good, num_bad)):
        ax = axes[1, i] if max(num_good, num_bad) > 1 else axes[1]
        if i < num_bad:
            idx, dist, prof = bad_data[i]
            ax.plot(dist, prof, 'r-') # Red for bad
            ax.set_title(f"BAD (CRASHED): Profile Index {idx}")
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.axis('off')

    save_path = f"./output/debug_GOOD_vs_BAD_plot_{timestamp}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"... Saved Good vs. Bad plot to {save_path}")

try:
    debug_data = torch.load("debug_theseus_inputs.pt")
    
    intensities = debug_data["intensities"].cpu().numpy()
    distances = debug_data["distances"].cpu().numpy()
    
    num_profiles_in_batch = intensities.shape[0]
    print(f"Data shape (profiles, samples): {intensities.shape}")

    # --- Find Good vs. Bad ---
    good_profiles_data = []
    bad_profiles_data = []
    max_to_find = 5 # Let's find 5 of each

    spline_refiner_config = img_proc._get_cfg()['ne_fit']
    model_config = next(
        filter(lambda d: d.get('name') == "richards_gaussian", 
               spline_refiner_config['model_list'])
    )
    device = img_proc._get_current_device()

    print("Sorting profiles (this may take a moment)...")
    for i in range(num_profiles_in_batch):
        if len(good_profiles_data) >= max_to_find and len(bad_profiles_data) >= max_to_find:
            break # We have enough of both

        single_intensity = intensities[i]
        single_distance = distances[i]
        
        try:
            # We run the single-profile tester (defined on line 388)
            test_single_profile_optimization(
                single_intensity,
                single_distance,
                model_config,
                spline_refiner_config,
                device
            )
            # If it doesn't crash, it's "good"
            if len(good_profiles_data) < max_to_find:
                good_profiles_data.append((i, single_distance, single_intensity))
                
        except Exception as e:
            # If it *does* crash, it's "bad"
            if len(bad_profiles_data) < max_to_find:
                bad_profiles_data.append((i, single_distance, single_intensity))

    print(f"... Found {len(good_profiles_data)} good profiles and {len(bad_profiles_data)} bad profiles.")
    
    # --- Plot the comparison ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_good_vs_bad(good_profiles_data, bad_profiles_data, timestamp)

except Exception as e:
    print(f"Could not load or plot debug data. (This is OK if it's the first run)")
    print(f"Error: {e}")
