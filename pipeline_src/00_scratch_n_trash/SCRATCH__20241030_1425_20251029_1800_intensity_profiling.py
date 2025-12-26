# working on refinement after running through registration s.t. variables are available in the session
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import theseus as th

from scipy import stats
from scipy.interpolate import make_splprep
import traceback

from utils.MLEstimation import (
    model_error_func, 
    construct_rich_gaus_initial_guess, 
    construct_gaus_lin_initial_guess, 
    MODEL_REGISTRY
)

from utils.npc_spline_refinement import NESplineRefiner, extract_profile_along_norm, adjust_spline_points
from tools.utility_functions import extract_cropped_images, filter_data_by_indices, find_segments, calc_tangent_endpts
from tools.geom_tools import get_u_range_from_bspline, bspline_from_tck

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

#### --- Refining Spline Fit --- ####
# associate sections of spline with signal / no signal
# TODO sample signal sections based on arc length & signal density constant (256 slices per pixel width)
# get profiles for signal sections
    # intensity_data_only = {
    #     key: val['intensity_profile'] 
    #     for key, val in intensity_profiles.items() 
    #     if 'intensity_profile' in val
    # }
    # filtered_intensity_profiles = filter_data_by_indices(intensity_data_only, norm_intensity_mask)
    # filtered_distance_profiles = filter_data_by_indices(norm_distances, norm_intensity_mask)
    # filtered_spline_points = filter_ne_dict_by_signal()

def test_single_profile_optimization(
    single_intensity_profile: np.ndarray,
    single_dist_profile: np.ndarray,
    model_config: dict,
    spline_refiner_config: dict,
    device: torch.device
) -> tuple:
    """
    Runs a Theseus optimization on a single intensity profile.
    
    Args:
        single_intensity_profile: (S,) numpy array of intensity values.
        single_dist_profile: (S,) numpy array of distance values.
        model_config: The config entry for the model (e.g., from 'model_list').
        spline_refiner_config: The 'ne_fit' config dictionary.
        device: The torch device to run on.
        
    Returns:
        A tuple of (final_state, info, initial_guess_tensor, dist_var, theta_var)
    """
    
    print(f"--- Testing single profile optimization for model: {model_config['name']} ---")
    
    # --- 1. Prep Tensors ---
    # The model functions expect a batch dimension.
    # We create a "batch" of 1 by adding a .unsqueeze(0).
    intensity_tensor = torch.tensor(
        single_intensity_profile, dtype=torch.float32, device=device
    ).unsqueeze(0) # Shape [1, S]
    dist_tensor = torch.tensor(
        single_dist_profile, dtype=torch.float32, device=device
    ).unsqueeze(0) # Shape [1, S]
    
    # --- 2. Get Initial Guess ---
    line_length = spline_refiner_config['line_length']
    model_name = model_config['name']

    if model_name == "richards_gaussian":
        initial_guess_tensor = construct_rich_gaus_initial_guess(
            dist_along_norm_tensor=dist_tensor,
            intensity_init_tensor=intensity_tensor,
            line_length=line_length,
            num_parameters=model_config['parameters'],
            intensity_params=model_config['initial_guess'],
            device=device
        )
    elif model_name == "gaussian_linear":
        initial_guess_tensor = construct_gaus_lin_initial_guess(
            dist_tensor, intensity_tensor, device
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Initial Guess: {initial_guess_tensor.cpu().numpy().squeeze()}")

    # --- 3. Setup Theseus Objects (from _setup_theseus_model) ---
    theta_var = th.Vector(tensor=initial_guess_tensor, name="theta")
    dist_var = th.Variable(tensor=dist_tensor, name="distances")
    intensity_var = th.Variable(tensor=intensity_tensor, name="intensities")
    
    # Cost Weight
    poisson_w_diag = 1.0 / torch.sqrt(intensity_tensor + 1e-6)
    cost_w = th.DiagonalCostWeight(poisson_w_diag)
    
    # Model Function
    model_function = MODEL_REGISTRY[model_name]

    # --- 4. Cost Function (from _setup_th_ad_cost_fn) ---
    cost_fn = th.AutoDiffCostFunction(
        optim_vars=[theta_var],
        err_fn=lambda optim_vars, aux_vars: model_error_func(
            optim_vars, aux_vars, model_function
        ),
        dim=intensity_var.tensor.shape[1],  # = S
        aux_vars=[dist_var, intensity_var],
        cost_weight=cost_w,
        name=f"{model_name}_single_fit"
    )

    # --- 5. Objective & Optimizer (from _setup_th_lm_optimizer) ---
    objective = th.Objective()
    objective.add(cost_fn)
    
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=spline_refiner_config['lm_optimizer']['iterations'],
        step_size=spline_refiner_config['lm_optimizer']['step_size'],
        step_radius=spline_refiner_config['lm_optimizer']['step_radius']
    )

    # --- 6. Create Layer and Run ---
    theseus_layer = th.TheseusLayer(optimizer).to(device)
    inputs = {"theta": initial_guess_tensor}
    
    print("Running optimizer...")
    final_state, info = theseus_layer.forward(inputs)
    
    print("Optimization complete.")
    print(f"Final Params: {final_state['theta'].detach().cpu().numpy().squeeze()}")
    
    return final_state, info, initial_guess_tensor, dist_var, theta_var, model_function


## TESTING SINGLE PROFILE OPTIMIZATION

    # ne_label_to_test = '12'
    # profile_index = 0
    # model_name_to_test = 'richards_gaussian'

    # try:
    #     spline_refiner_config = img_proc._get_cfg()['ne_fit']
    #     model_config = next(
    #         filter(lambda d: d.get('name') == model_name_to_test, 
    #                 spline_refiner_config['model_list'])
    #     )
    #     device = img_proc._get_current_device()

    #     single_intensity = intensity_profiles[ne_label_to_test]['intensity_profile'][profile_index]
    #     single_distance = norm_distances[ne_label_to_test][profile_index]

    #     # 3. Get your configs (you already have access to img_proc)
    #     spline_refiner_config = img_proc._get_cfg()['ne_fit']
    #     model_config = next(
    #         filter(lambda d: d.get('name') == model_name_to_test, 
    #                 spline_refiner_config['model_list'])
    #     )
    #     device = img_proc._get_current_device()

    #     # 4. Run the test
    #     (final_state, 
    #         info, 
    #         initial_guess_tensor, 
    #         dist_var, 
    #         theta_var, 
    #         model_function) = test_single_profile_optimization(
    #         single_intensity,
    #         single_distance,
    #         model_config,
    #         spline_refiner_config,
    #         device
    #     )
    #     # 5. PLOT THE RESULT
    #     final_params_tensor = final_state['theta'].detach()
        
    #     # Get model predictions
    #     with torch.no_grad():
    #         # Pass Theseus variables to the model function
    #         initial_pred = model_function(dist_var, theta_var)
    #         final_pred = model_function(
    #             dist_var, th.Vector(tensor=final_params_tensor, name="theta_final")
    #         )
                
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(single_distance, single_intensity, 'ko', label='Raw Data', markersize=4)
    #     plt.plot(
    #         single_distance, 
    #         initial_pred.cpu().numpy().squeeze(), 
    #         'b--', 
    #         label='Initial Guess'
    #     )
    #     plt.plot(
    #         single_distance, 
    #         final_pred.cpu().numpy().squeeze(), 
    #         'r-', 
    #         linewidth=2, 
    #         label='Final Fit'
    #     )
    #     plt.legend()
    #     plt.title(f"Single Profile Fit (Label {ne_label_to_test}, Index {profile_index})")
    #     plt.xlabel("Distance along norm (pixels)")
    #     plt.ylabel("Intensity")
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.show()

    # except KeyError:
    #     print(f"Could not find data for NE Label: {ne_label_to_test}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # # --- comparing optimized based on profile bspline to old bspline ---

    # bspline_data = init_bsplines_ch1[ne_label_to_test]
    # mean_ne_img = cropped_imgs_ch1[ne_label_to_test]

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

# def test_refinement_with_filtering(
    #     spline_refiner,
    #     evaluated_points_xy,     # Shape (2, N) [x, y]
    #     evaluated_derivatives_xy, # Shape (2, N) [dx, dy]
    #     intensity_profiles,      # Shape (N, S) numpy array
    #     dist_along_norm,       # Shape (N, S) numpy array
    #     model_name
    # ):
    #     """
    #     Standalone test function that replicates _refine_ne_bsplines
    #     but includes the pre-filtering "wonky point" check.
    #     """
        
    #     # 0. Get common objects from the refiner instance
    #     device = spline_refiner._current_device
    #     model_function = MODEL_REGISTRY[model_name]
    #     cfg_fit = spline_refiner._cfg_fit

    #     # 1. Create Full Tensors
    #     intensity_tensor = torch.tensor(intensity_profiles, 
    #                                     dtype=torch.float32, device=device)
    #     dist_tensor = torch.tensor(dist_along_norm, 
    #                                dtype=torch.float32, device=device)
        
    #     # 2. NEW: Create the Filter Mask
    #     # Check 1: Must be finite (no NaNs or Infs)
    #     valid_mask = torch.isfinite(intensity_tensor).all(dim=1)
        
    #     # Check 2: Must not be a flat line (std dev > tiny number)
    #     std_devs = torch.std(intensity_tensor, dim=1)
    #     valid_mask = valid_mask & (std_devs > 1e-3) 

    #     num_total = intensity_tensor.shape[0]
    #     num_valid = valid_mask.sum().item()
        
    #     if num_valid == 0:
    #         raise ValueError("No valid profiles found after filtering. Skipping segment.")
        
    #     print(f"    Filtering profiles: {num_valid} / {num_total} are valid.")
        
    #     # 3. Create "Good" Tensors (as *numpy arrays* for the setup function)
    #     valid_mask_cpu = valid_mask.cpu() # For numpy indexing
    #     good_intensity_numpy = intensity_profiles[valid_mask_cpu]
    #     good_dist_numpy = dist_along_norm[valid_mask_cpu]

    #     # 4. Run Optimization ONLY on "Good" Data
    #     # We call the refiner's setup methods directly
    #     theta_variable, dist_variable, intensity_variable, cost_weight = \
    #         spline_refiner._setup_theseus_model(
    #             good_intensity_numpy,
    #             good_dist_numpy,
    #             model_name
    #         )
        
    #     cost_function = spline_refiner._setup_th_ad_cost_fn(
    #         theta_variable, intensity_variable, dist_variable, 
    #         model_function, cost_weight, (f'{model_name}_fit')
    #     )
        
    #     objective = th.Objective()
    #     objective.add(cost_function)
    #     optimizer = spline_refiner._setup_th_lm_optimizer(objective)
    #     theseus_layer = th.TheseusLayer(optimizer).to(device)

    #     inputs = {"theta": theta_variable.tensor.to(device)}
        
    #     final_state, info = theseus_layer.forward(inputs)
    #     final_params_good = final_state["theta"].detach() # Shape [num_valid, 11]

    #     # 5. NEW: Map Results Back to Full Spline
    #     mu_good = final_params_good[:, 7] # Index 7 is 'mu'
    #     line_length = cfg_fit['line_length']
        
    #     # Create a full-sized array for *all* N original points
    #     # Initialize with the "no change" offset (center of the line)
    #     mu_all = torch.full(
    #         (num_total,), 
    #         fill_value=(line_length / 2.0), 
    #         device=device, 
    #         dtype=torch.float32
    #     )
        
    #     # Place the "good" results into the full array
    #     mu_all[valid_mask] = mu_good
        
    #     # 6. Adjust Points Using the Full 'mu_all' Array
    #     # We use the *original* full point/derivative sets
    #     # adjust_spline_points expects (x, y) points and (dx, dy) derivatives
    #     refined_points_xy = adjust_spline_points(
    #         mu_all.cpu().numpy(),
    #         line_length, 
    #         evaluated_points_xy,       # Pass (x,y)
    #         evaluated_derivatives_xy   # Pass (dx,dy)
    #     )
        
    #     # splprep expects [x_coords, y_coords]
    #     # refined_points_xy is already (2, N) with [x, y]
    #     tck_refined, _ = make_splprep(
    #         [refined_points_xy[0, :], refined_points_xy[1, :]], 
    #         s=1.0, 
    #         k=3
    #     )
        
    #     return tck_refined


# -----------------------------------------------------------------
## TESTING SINGLE SEGMENT (NEW SEGMENT-AWARE VERSION)
# -----------------------------------------------------------------

ne_label_to_test = '12'
segment_key_to_test = 'segment_0'  # <-- Specify which segment to test
# profile_index = 100               # <-- Pick a profile *within* that segment
model_name_to_test = 'richards_gaussian'

print(f"--- Setting up test for {ne_label_to_test}, {segment_key_to_test} ---")

try:
    # 1. Get configs and device
    spline_refiner_config = img_proc._get_cfg()['ne_fit']
    model_config = next(
        filter(lambda d: d.get('name') == model_name_to_test, 
                spline_refiner_config['model_list'])
    )
    device = img_proc._get_current_device()

    # --- Get individual segment data
    bspline_obj = init_bsplines_ch1[ne_label_to_test][segment_key_to_test]
    mean_ne_img = cropped_imgs_ch1[ne_label_to_test]
    cfg_fit = img_proc._get_cfg()['ne_fit']

    # --- Get all profiles for this segment
    print(f"Extracting all profiles for {segment_key_to_test}...")
    u_values_full = get_u_range_from_bspline(bspline_obj, spline_refiner_ch1._get_sampling_density())
    segment_points_xy = bspline_obj(u_values_full)
    segment_derivs_xy = bspline_obj.derivative(1)(u_values_full)

    line_length = cfg_fit['line_length']
    n_samples_along_normal = cfg_fit['n_samples_along_normal']
    num_points_in_segment = segment_points_xy.shape[1]
            
    _, normal_endpoints = calc_tangent_endpts(
        segment_points_xy, segment_derivs_xy, line_length, True
    )
            
    intensity_profiles, dist_along_norm, error_flag = extract_profile_along_norm(
        mean_ne_img, 
        segment_points_xy, 
        normal_endpoints, 
        num_points_in_segment, 
        n_samples_along_normal
    )
    
    if error_flag:
        raise ValueError("Error extracting profiles for the test segment.")

    # --- YOUR JITTER FIX (to match the main loop) ---
    noise_level = np.max(intensity_profiles) * 0.005 
    jitter = np.random.randn(*intensity_profiles.shape) * noise_level
    intensity_profiles = intensity_profiles + jitter
    print("...Profiles extracted and jittered.")
    
    # 4. NOW we can select the single profile
    for profile_index in range(len(intensity_profiles)):
        # Get the profile data *using* the index
        single_intensity = intensity_profiles[profile_index]
        single_distance = dist_along_norm[profile_index]
    
    # 5. Run the test (this part is unchanged)
        (   final_state, 
            info, 
            initial_guess_tensor, 
            dist_var, 
            theta_var, 
                model_function) = test_single_profile_optimization(
                                    single_intensity,
                                    single_distance,
                                    model_config,
                                    spline_refiner_config,
                                    device
                                )
    
    # 6. PLOT THE RESULT (this part is unchanged)
    final_params_tensor = final_state['theta'].detach()
    
    with torch.no_grad():
        initial_pred = model_function(dist_var, theta_var)
        final_pred = model_function(
            dist_var, th.Vector(tensor=final_params_tensor, name="theta_final")
        )
            
    plt.figure(figsize=(10, 6))
    plt.plot(single_distance, single_intensity, 'ko', label='Raw Data', markersize=4)
    plt.plot(
        single_distance, 
        initial_pred.cpu().numpy().squeeze(), 
        'b--', 
        label='Initial Guess'
    )
    plt.plot(
        single_distance, 
        final_pred.cpu().numpy().squeeze(), 
        'r-', 
        linewidth=2, 
        label='Final Fit'
    )
    plt.legend()
    plt.title(f"Single Profile Fit (Label {ne_label_to_test}, {segment_key_to_test}, Index {profile_index})")
    plt.xlabel("Distance along norm (pixels)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

except KeyError:
    print(f"Could not find data for NE Label: {ne_label_to_test} or Segment: {segment_key_to_test}")
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()

# -----------------------------------------------------------------
# --- END OF TEST BLOCK ---
# -----------------------------------------------------------------

failed_profile_indices = []
successful_fits = {} # Use a dict to store params by index

print("\n--- Starting loop to fit all single profiles (this may take a while)... ---")

for profile_index in range(len(intensity_profiles)):
    try:
        # Get the profile data *using* the index
        single_intensity = intensity_profiles[profile_index]
        single_distance = dist_along_norm[profile_index]
    
        # 5. Run the test (this is the part that might crash)
        (final_state, 
         info, 
         initial_guess_tensor, 
         dist_var, 
         theta_var, 
         model_function) = test_single_profile_optimization(
                            single_intensity,
                            single_distance,
                            model_config,
                            spline_refiner_config,
                            device
                        )
        
        # If it succeeds, store the results
        successful_fits[profile_index] = final_state['theta'].detach().cpu().numpy().squeeze()
        
    except Exception as e:
        # If it crashes, log the index and continue
        print(f"--> WARNING: Fit failed for profile_index {profile_index}.")
        failed_profile_indices.append(profile_index)
        continue # Explicitly move to the next iteration

# --- Add this *after* your loop ---
print("\n" + "="*30)
print("--- Single Profile Fit Summary ---")
print(f"Successfully fit {len(successful_fits)} profiles.")
print(f"Failed to fit {len(failed_profile_indices)} profiles.")
if failed_profile_indices:
    print(f"First 10 failed indices: {failed_profile_indices[:10]}")
print("="*30 + "\n")

failed_profiles = intensity_profiles[failed_profile_indices,:]

with open(f'./output/{FoV_id}_ch1_{ne_label_to_test}_{segment_key_to_test}_dual_debug_failed_profiles.pkl', 'wb') as f:
            pickle.dump(failed_profiles, f)

with open(f'./output/{FoV_id}_ch1_{ne_label_to_test}_{segment_key_to_test}_dual_debug_succesful_profiles.pkl', 'wb') as f:
            pickle.dump(successful_fits, f)
            

# failed profiles
bad_profiles = intensity_profiles[failed_profile_indices]
bad_labels = torch.zeros(len(bad_profiles), dtype=torch.float32)

# successful profiles
good_indices = list(successful_fits.keys())
good_profiles = intensity_profiles[good_indices]

# balance the dataset re: sampling
num_to_match = len(bad_profiles)
indices_to_sample = np.random.choice(len(good_profiles), num_to_match, replace=False)

good_profiles_balanced = good_profiles[indices_to_sample]
good_labels_balanced = torch.ones(num_to_match, dtype=torch.float32)

# combine data and create loader
X = torch.tensor(np.vstack((good_profiles_balanced, bad_profiles)), dtype=torch.float32)
y = torch.cat((good_labels_balanced, bad_labels))

# Shuffle the combined dataset
shuffle_indices = torch.randperm(len(X))
X = X[shuffle_indices]
y = y[shuffle_indices]

# Create a TensorDataset and DataLoader
# batch size of 32 is a starting point
batch_size = 32
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#nn.Module with one linear layer and a sigmoid. in_features must match the number of samples in your profile (e.g., 100)
class LogisticClassifier(nn.Module):
    def __init__(self, num_features):
        super(LogisticClassifier, self).__init__()
        # One simple linear layer: 100 inputs -> 1 output
        self.linear = nn.Linear(in_features=num_features, out_features=1)
    
    def forward(self, x):
        # Pass the output through a sigmoid to get a 0-1 probability
        return torch.sigmoid(self.linear(x))

# Get the number of features (e.g., 100)
num_features = X.shape[1]
model = LogisticClassifier(num_features)

# --- Training Setup ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

print("\n--- Training Classifier ---")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for profiles, labels in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get model predictions
        outputs = model(profiles)
        
        # Calculate the loss
        loss = criterion(outputs, labels.unsqueeze(1)) # Labels need to be [batch_size, 1]
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("--- Training Complete ---")

# --- 1. Extract the trained weights ---
# The weights are at model.linear.weight
# .data gets the tensor, .squeeze() removes extra dims
weights = model.linear.weight.data.squeeze().cpu().numpy()

# Get a representative distance vector (e.g., from the first failed profile)
distances = dist_along_norm[failed_profile_indices[0]]

# --- 2. Plot the weights ---
plt.figure(figsize=(10, 6))
plt.plot(distances, weights, 'bo-', label='Learned Weight')
plt.axhline(0, color='gray', linestyle='--')
plt.title("What the Model Learned: Feature Importance")
plt.xlabel("Distance along norm (pixels)")
plt.ylabel("Weight (Importance)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 3. How to Read This Plot ---
print("Interpreting the plot:")
print("  - POSITIVE weights: Intensity at this distance predicts a GOOD fit.")
print("  - NEGATIVE weights: Intensity at this distance predicts a BAD fit (a crash).")
print("  - Weights near ZERO: Intensity at this distance is irrelevant.")

# --- FEATURE REDUCTION MODEL --- #
# --- 2. Get the Raw Profile Data ---
bad_profiles = intensity_profiles[failed_profile_indices]
bad_labels = torch.zeros(len(bad_profiles), dtype=torch.float32)

good_indices = list(successful_fits.keys())
good_profiles = intensity_profiles[good_indices]

# --- 3. BALANCE THE DATASET ---
num_to_match = len(bad_profiles)
indices_to_sample = np.random.choice(len(good_profiles), num_to_match, replace=False)

good_profiles_balanced = good_profiles[indices_to_sample]
good_labels_balanced = torch.ones(num_to_match, dtype=torch.float32)

# --- 4. *** NEW: FEATURE ENGINEERING (6 Features) *** ---
print("Engineering features...")
all_profiles_balanced = np.vstack((good_profiles_balanced, bad_profiles))
# We need the distance array for the centroid calculation
distances_for_centroid = dist_along_norm[0] # All distance arrays are the same

features = []
for profile in all_profiles_balanced:
    # 1. Amplitude
    amplitude = np.max(profile) - np.min(profile)
    
    # 2. Peak Position (Argmax)
    peak_position = np.argmax(profile)
    
    # 3. Kurtosis (Flatness)
    kurtosis = stats.kurtosis(profile)
    
    # 4. Skewness (Lopsidedness)
    skewness = stats.skew(profile)
    
    # 5. Standard Deviation (Your suggestion)
    std_dev = np.std(profile)
    
    # 6. Centroid (Your "combo" suggestion)
    # Ensure no division by zero if profile is all zero
    sum_intensity = np.sum(profile)
    if sum_intensity == 0:
        centroid = np.mean(distances_for_centroid) # fallback to center
    else:
        centroid = np.sum(profile * distances_for_centroid) / sum_intensity
    
    features.append([
        amplitude, peak_position, kurtosis, skewness, std_dev, centroid
    ])

feature_names = [
    "Amplitude", "Peak Position (argmax)", "Kurtosis (Flatness)", 
    "Skewness", "Std Deviation", "Centroid (Weighted Mean)"
]
# --- END OF NEW BLOCK ---

# --- 5. Combine and Create DataLoader ---
X = torch.tensor(features, dtype=torch.float32)
y = torch.cat((good_labels_balanced, bad_labels))

# Shuffle
shuffle_indices = torch.randperm(len(X))
X = X[shuffle_indices]
y = y[shuffle_indices]

batch_size = 32
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_features = X.shape[1]
model = LogisticClassifier(num_features)



# --- Training Setup ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

print("\n--- Training Classifier ---")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for profiles, labels in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get model predictions
        outputs = model(profiles)
        
        # Calculate the loss
        loss = criterion(outputs, labels.unsqueeze(1)) # Labels need to be [batch_size, 1]
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("--- Training Complete ---")


# --- 1. Extract the trained weights ---
weights = model.linear.weight.data.squeeze().cpu().numpy()
bias = model.linear.bias.data.cpu().numpy()[0]

print("\n--- Model Interpretation ---")
print(f"Bias (Base likelihood): {bias:.4f}")
for name, weight in zip(feature_names, weights):
    print(f"  - Weight for {name}: {weight:.4f}")

# --- 2. How to Read This ---
print("\nInterpreting the weights:")
print("  - POSITIVE weight: A high value for this feature (e.g., high Amplitude) predicts a GOOD fit.")
print("  - NEGATIVE weight: A high value for this feature (e.g., high Skewness) predicts a BAD fit (a crash).")

# --- 3. Plot the weights ---
plt.figure(figsize=(10, 6))
plt.bar(feature_names, weights, color='blue', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')
plt.title("What the Model Learned: Feature Importance")
plt.ylabel("Weight (Importance)")
plt.xticks(rotation=15)
plt.show()