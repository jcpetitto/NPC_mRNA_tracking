# working on refinement after running through registration s.t. variables are available in the session
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import tifffile
import torch
import theseus as th

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

# # 0190_ch1_05_segment_0_intensity_profile
    # test_ch1_bspline = all_ne_bsplines['BMY1408_12_14_2023']['ch1']['0190']['05']['bspline_object']
    # test_ch2_bspline = all_ne_bsplines['BMY1408_12_14_2023']['ch2']['0190']['03']['bspline_object']

    # u_range = np.linspace(0,1,1000)

    # test_ch1_bspline_d1 = test_ch1_bspline.derivative(nu=1)
    # test_ch2_bspline_d1 = test_ch2_bspline.derivative(nu=1)

    # test_ch1_bspline_d2 = test_ch1_bspline.derivative(nu=2)
    # test_ch2_bspline_d2 = test_ch2_bspline.derivative(nu=2)

    # eval_test_ch1_bspline = test_ch1_bspline(u_range)
    # eval_test_ch1_bspline_d1 = test_ch1_bspline_d1(u_range)
    # eval_test_ch1_bspline_d2 = test_ch1_bspline_d2(u_range)

    # y = eval_test_ch1_bspline[:,1]
    # x = eval_test_ch1_bspline[:,0]

    # dy = eval_test_ch1_bspline_d1[:,1]
    # dx = eval_test_ch1_bspline_d1[:,0]

    # ddy = eval_test_ch1_bspline_d2[:,1]
    # ddx = eval_test_ch1_bspline_d2[:,0]

    # curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
    # curvature_std = np.std(curvature)
    # curvature_mean = np.mean(curvature)

def calc_bspline_curvature(bspline_obj, u_param_to_test):
    eval_bspline_obj_d1 = bspline_obj(u_param_to_test, nu=1)
    eval_bspline_obj_d2 = bspline_obj(u_param_to_test, nu=2)

    # assign to variables for readability
    dy = eval_bspline_obj_d1[:,1]
    dx = eval_bspline_obj_d1[:,0]

    ddy = eval_bspline_obj_d2[:,1]
    ddx = eval_bspline_obj_d2[:,0]

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    return curvature

# FUNCTION TESTING
    # curvature_ch1 = calc_bspline_curvature(test_ch1_bspline, u_range)
    # curvature_ch2 = calc_bspline_curvature(test_ch2_bspline, u_range)

    # curvature_ch1.std()
    # curvature_ch1.mean()

    # curvature_ch2.std()
    # curvature_ch2.mean()

    # plt.hist(curvature_ch1)
    # plt.hist(curvature_ch2)
    # plt.close()

    # curvature_0080_ch2_11 = calc_bspline_curvature(bspline_0080_ch2_11, u_range)
    # print(curvature_0080_ch2_11.std())
    # print(curvature_0080_ch2_11.mean())
    # plt.hist(curvature_0080_ch2_11)
    # plt.close()

def plot_curvature_dist(curvature_points):
    mean_value = np.mean(curvature_points)
    std_value = np.std(curvature_points)

    fig, ax = plt.subplots()
    ax.hist(curvature_points, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Curvature')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    annotation_text = f'$\mu$ = {mean_value:.2f}\n$\sigma$ = {std_value:.2f}'

    ax.annotate(annotation_text, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=12, color='blue',
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="k", lw=1, alpha=0.8))

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show(fig)

# plot_curvature_dist(curvature)

def plot_curvature_on_curve(y_coords, x_coords, curvature_values, cmap='viridis', title='Curve with Points Colored by Curvature'):

    if not (len(y_coords) == len(x_coords) == len(curvature_values)):
        raise ValueError("Input arrays y_coords, x_coords, and curvature_values must have the same length.")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot where 'c' is the color argument, mapped by 'cmap'
    scatter = ax.scatter(y_coords, x_coords, c=curvature_values, cmap=cmap, s=50, zorder=2)

    # Create a colorbar to serve as a legend for the curvature values
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Curvature', rotation=270, labelpad=20, fontsize=12)

    ax.set_title(title, fontsize=16)

    ax.invert_yaxis()

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6, zorder=1)

    plt.show()

# # testing with variables calculated at the start of this script
# plot_curvature_on_curve(y,x, curvature)

# set-up for tesitng on FoV_id 0191 because it sems to have a variaty of viable NE labels in both channels
# SELF for R50 note nature of data structure
# TODO ... could be better re: channels in the initial experiment data
    # FoV_0191_dict = all_experiments['BMY1408_12_14_2023']['0191']
    # img_path_ch1 = os.path.join(FoV_0191_dict['FoV_collection_path'], FoV_0191_dict['imgs']['fn_track_ch1'])
    # img_path_ch2 = os.path.join(FoV_0191_dict['FoV_collection_path'], FoV_0191_dict['imgs']['fn_track_ch2'])

    # FoV_0191_ch1_crop_dict = all_ne_crop_boxes['BMY1408_12_14_2023']['ch1']['0191']
    # FoV_0191_ch2_crop_dict = all_ne_crop_boxes['BMY1408_12_14_2023']['ch2']['0191']

    # spline_refiner_ch1 = NESplineRefiner('ch1', img_path_ch1, '0191', FoV_0191_ch1_crop_dict,img_proc._get_cfg(), img_proc._get_current_device())

    # spline_refiner_ch2 = NESplineRefiner('ch2', img_path_ch1, '0191', FoV_0191_ch2_crop_dict,img_proc._get_cfg(), img_proc._get_current_device())

def extract_intensity_profiles(spline_refiner, init_bsplines):
    cropped_imgs = spline_refiner._get_ne_imgs()
    sampling_density = spline_refiner._get_sampling_density()
    
    intensity_results = {}
    dist_norm_results = {}
    for ne_label, bspline_data in init_bsplines.items():
        print(f"Current NE mask label: {ne_label}")
        # determine sampling parameter range based on the length of the bspline arc
        u_range = get_u_range_from_bspline(bspline_data['bspline_object'], sampling_density)
        # Get the corresponding cropped image using the ne_label_to_test as the key
        mean_ne_img = cropped_imgs.get(ne_label)
        if mean_ne_img is None:
            print(f"Warning: No cropped image found for NE label {ne_label}. Skipping.")
            continue
        i = 0

# !!! "normal_lines_n" sets the number of normal lines sampled, not a u related parameter... ...
        segment_points = bspline_data['bspline_object'](u_range)
        segment_derivs = bspline_data['bspline_object'](u_range, nu = 1)
        intensity_profiles, dist_along_norm = spline_refiner._sample_norm_intensity(ne_label, i, mean_ne_img, segment_points, segment_derivs)
        intensity_results.update(
            {
                f'{ne_label}': {
                                'u_range': u_range,
                                'intensity_profile': intensity_profiles
                                }
            })
        dist_norm_results.update({f'{ne_label}': dist_along_norm})
        # print(np.max(intensity_profiles))
        # print(2*np.std(intensity_profiles))
    return intensity_results, dist_norm_results


# made test dataset BMY9999 with FoV from different yeast strain dual label datasets that have a variety of nucleus situations
u_range = np.linspace(0, 1, 1000)

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


intensity_profiles, norm_distances = extract_intensity_profiles(spline_refiner_ch1, init_bsplines_ch1)

# Returns
#       'u_range'   - all parameters 
#       '##'        -   key: NE label
#                       values: 'u_params': calculated based on the initial npc fit bspline object
#                       values: 'is_signal': indices where there is signal

def establish_signal_regions(intensity_profiles, u_range, n_subset = 100):
    norm_intensity_thresh_masks = {'u_range': u_range}

    for key, value in intensity_profiles.items():
        profile_u_params = value['u_range']
        profile_1d_sum = np.sum(value['intensity_profile'], axis=1)
        try:
            threshold = threshold_otsu(profile_1d_sum)
        except ValueError as e:
            # This can happen if all values are identical (e.g., all zero)
            print(f"Otsu's method failed: {e}. Using mean as a fallback.")
            threshold = np.mean(profile_1d_sum)

        is_signal_mask = profile_1d_sum > threshold

        signal_indices = np.where(is_signal_mask)[0]
        signal_u_params = profile_u_params[signal_indices]
        norm_intensity_thresh_masks.update({f'{key}': signal_indices})

        # Plotting results
            # n_points = len(u_range)
            # spline_values = init_bsplines_ch1[key]['bspline_object'](u_range)
            # y_spline = spline_values[:,1]
            # x_spline = spline_values[:,0]
            # y_signal = y_spline[is_signal_mask]
            # x_signal = x_spline[is_signal_mask]

            # y_no_signal = y_spline[~is_signal_mask]
            # x_no_signal = x_spline[~is_signal_mask]
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            # ax1.plot(profile_1d_sum, label='1D Summed Profile')
            # ax1.axhline(y=threshold, color='r', linestyle='--', label="Otsu's Threshold")
            # ax1.set_title('1D Profile with Signal Threshold')
            # ax1.set_xlabel('Normal Line Index (0-999)')
            # ax1.set_ylabel('Summed Intensity')
            # ax1.legend()

            # ax2.imshow(cropped_imgs_ch1[key], cmap='inferno', origin='upper', vmin=global_min_ch1, vmax=global_max_ch1)
            # ax2.scatter(y_no_signal, x_no_signal, c='red', label='No Signal (Background)', s=10, alpha = 0.5)
            # ax2.scatter(y_signal, x_signal, c='cyan', label='Signal (Otsu)', s=10, alpha = 0.5)
            # ax2.set_title(f'Spline Signal Classification for {key}')
            # ax2.set_xlabel('X Coordinate')
            # ax2.set_ylabel('Y Coordinate')
            # ax2.set_aspect('equal', 'box')
            # ax2.legend()
            # fig.show()
            # plt.savefig(f"/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/test_intensity_profile/{filename}")
            # plt.close(fig)

    return norm_intensity_thresh_masks

norm_intensity_mask = establish_signal_regions(intensity_profiles, u_range)

#### --- Refining Spline Fit --- ####
# associate sections of spline with signal / no signal
# TODO sample signal sections based on arc length & signal density constant (256 slices per pixel width)
# get profiles for signal sections
intensity_data_only = {
    key: val['intensity_profile'] 
    for key, val in intensity_profiles.items() 
    if 'intensity_profile' in val
}
filtered_intensity_profiles = filter_data_by_indices(intensity_data_only, norm_intensity_mask)
filtered_distance_profiles = filter_data_by_indices(norm_distances, norm_intensity_mask)
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


ne_label_to_test = '12'
profile_index = 0
model_name_to_test = 'richards_gaussian'

try:
    spline_refiner_config = img_proc._get_cfg()['ne_fit']
    model_config = next(
        filter(lambda d: d.get('name') == model_name_to_test, 
                spline_refiner_config['model_list'])
    )
    device = img_proc._get_current_device()

    single_intensity = intensity_profiles[ne_label_to_test]['intensity_profile'][profile_index]
    single_distance = norm_distances[ne_label_to_test][profile_index]

    # 3. Get your configs (you already have access to img_proc)
    spline_refiner_config = img_proc._get_cfg()['ne_fit']
    model_config = next(
        filter(lambda d: d.get('name') == model_name_to_test, 
                spline_refiner_config['model_list'])
    )
    device = img_proc._get_current_device()

    # 4. Run the test
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
    # 5. PLOT THE RESULT
    final_params_tensor = final_state['theta'].detach()
    
    # Get model predictions
    with torch.no_grad():
        # Pass Theseus variables to the model function
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
    plt.title(f"Single Profile Fit (Label {ne_label_to_test}, Index {profile_index})")
    plt.xlabel("Distance along norm (pixels)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

except KeyError:
    print(f"Could not find data for NE Label: {ne_label_to_test}")
except Exception as e:
    print(f"An error occurred: {e}")

# --- comparing optimized based on profile bspline to old bspline ---

bspline_data = init_bsplines_ch1[ne_label_to_test]
mean_ne_img = cropped_imgs_ch1[ne_label_to_test]

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

def test_refinement_with_filtering(
    spline_refiner,
    evaluated_points_xy,     # Shape (2, N) [x, y]
    evaluated_derivatives_xy, # Shape (2, N) [dx, dy]
    intensity_profiles,      # Shape (N, S) numpy array
    dist_along_norm,       # Shape (N, S) numpy array
    model_name
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
    dist_tensor = torch.tensor(dist_along_norm, 
                               dtype=torch.float32, device=device)
    
    # 2. NEW: Create the Filter Mask
    # Check 1: Must be finite (no NaNs or Infs)
    valid_mask = torch.isfinite(intensity_tensor).all(dim=1)
    
    # Check 2: Must not be a flat line (std dev > tiny number)
    std_devs = torch.std(intensity_tensor, dim=1)
    valid_mask = valid_mask & (std_devs > 1e-3) 

    num_total = intensity_tensor.shape[0]
    num_valid = valid_mask.sum().item()
    
    if num_valid == 0:
        raise ValueError("No valid profiles found after filtering. Skipping segment.")
    
    print(f"    Filtering profiles: {num_valid} / {num_total} are valid.")
    
    # 3. Create "Good" Tensors (as *numpy arrays* for the setup function)
    valid_mask_cpu = valid_mask.cpu() # For numpy indexing
    good_intensity_numpy = intensity_profiles[valid_mask_cpu]
    good_dist_numpy = dist_along_norm[valid_mask_cpu]

    # 4. Run Optimization ONLY on "Good" Data
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
    tck_refined, _ = make_splprep(
        [refined_points_xy[0, :], refined_points_xy[1, :]], 
        s=1.0, 
        k=3
    )
    
    return tck_refined

### END FUNCTION
# 2. Get the full, density-sampled u_range and signal mask
if ne_label_to_test not in intensity_profiles:
    print(f"Error: NE Label {ne_label_to_test} not found in intensity_profiles. Exiting.")
else:
    u_values_full = intensity_profiles[ne_label_to_test]['u_range']
    
    profile_1d_sum = np.sum(intensity_profiles[ne_label_to_test]['intensity_profile'], axis=1)
    try:
        threshold = threshold_otsu(profile_1d_sum)
    except ValueError:
        threshold = np.mean(profile_1d_sum)
    is_signal_mask = profile_1d_sum > threshold
    
    # Use find_segments on the boolean mask
    segment_index_ranges = find_segments(is_signal_mask, True)
    
    print(f"Found {len(segment_index_ranges)} signal segments for NE {ne_label_to_test}.")

    refined_segments_for_ne = {}

    # 3. Manually run the refinement loop
    for i, (start_idx, end_idx) in enumerate(segment_index_ranges):
        segment_label = f"segment_{i}"
        
        u_segment = u_values_full[start_idx : end_idx + 1]
        num_points_in_segment = len(u_segment)
        u_range_tuple = (u_values_full[start_idx], u_values_full[end_idx])

        if num_points_in_segment < 4:
            print(f"  -> Skipping {segment_label}, not enough points (n={num_points_in_segment}).")
            continue
            
        print(f"  -> Refining {segment_label} using {num_points_in_segment} points...")

        try:
            # 1. Sample the *original* spline
            segment_points_xy = bspline_data['bspline_object'](u_segment) # (N, 2) [x, y]
            segment_derivs_xy = bspline_data['bspline_object'].derivative(1)(u_segment) # (N, 2) [dx, dy]

            # 2. Extract intensity profiles (using the "hacky but correct" logic)
            cfg_fit = spline_refiner_ch1._cfg_fit
            line_length = cfg_fit['line_length']
            n_samples_along_normal = cfg_fit['n_samples_along_normal']
            current_normal_lines_n = num_points_in_segment 
            
            _, normal_endpoints = calc_tangent_endpts(
                segment_points_xy.T, segment_derivs_xy.T, line_length, True
            )
            
            intensity_profiles, dist_along_norm, error_flag = extract_profile_along_norm(
                mean_ne_img, 
                segment_points_xy.T, 
                normal_endpoints, 
                current_normal_lines_n,
                n_samples_along_normal
            )
            if error_flag:
                raise ValueError("Error extracting intensity profiles.")
            
            # 3. Create signal mask (for plotting)
            signal_mask = np.ones(num_points_in_segment, dtype=bool)

            # 4. *** RUN THE NEW TEST FUNCTION ***
            refined_segment_tck = test_refinement_with_filtering(
                spline_refiner_ch1,      # Pass the refiner instance
                segment_points_xy.T,       # (2, N) [x, y]
                segment_derivs_xy.T,       # (2, N) [dx, dy]
                intensity_profiles, 
                dist_along_norm, 
                cfg_fit['default_model'], 
                is_periodic=False 
            )
            # *** END OF CHANGE ***
            
            # 5. Get points from the new refined TCK
            refined_bspline_obj = bspline_from_tck(refined_segment_tck, is_periodic=False)
            refined_points_xy = refined_bspline_obj(np.linspace(0, 1, num_points_in_segment))

            # 6. PLOT THE COMPARISON
            plot_save_path = f"./output/{FoV_id}_ch1_{ne_label_to_test}_{segment_label}_refinement.png"
            plot_refinement_comparison(
                background_image=mean_ne_img,
                original_points_xy=segment_points_xy,
                refined_points_xy=refined_points_xy,
                signal_mask=signal_mask,
                title=f"Refinement: {ne_label_to_test} {segment_label} (n={num_points_in_segment})",
                save_path=plot_save_path
            )
            print(f"    ... Plot saved to {plot_save_path}")

            # 7. Store the result
            refined_segments_for_ne[segment_label] = {
                'bspline_object': refined_bspline_obj,
                'is_periodic': False,
                'original_u_range': u_range_tuple
            }
            
        except Exception as e:
            print(f"  --> Warning: Refining of segment {i} failed. {e}.")
            traceback.print_exc()
            continue

print(f"\nPhase 1 complete. Refined {len(refined_segments_for_ne)} segments.")


print("--- Loading debug data to find profile 91 ---")
try:
    # This file was created by your *failed* run.
    debug_data = torch.load("debug_theseus_inputs.pt") 
    intensities = debug_data["intensities"].cpu().numpy()
    distances = debug_data["distances"].cpu().numpy()

    bad_profile_intensity = intensities[91, :]
    bad_profile_distance = distances[91, :]

    plt.figure()
    plt.plot(bad_profile_distance, bad_profile_intensity, 'ro-')
    plt.title("The Culprit: Profile 91")
    plt.xlabel("Distance")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Could not load or plot debug data: {e}")
