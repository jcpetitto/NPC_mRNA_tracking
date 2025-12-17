import pickle
import os
import traceback
import tifffile
import torch

import numpy as np
import matplotlib.pyplot as plt
import theseus as th

from scipy.interpolate import splprep
from skimage.filters import threshold_otsu

from utils.MLEstimation import (
    model_error_func, 
    construct_rich_gaus_initial_guess, 
    construct_gaus_lin_initial_guess, 
    MODEL_REGISTRY
)

from utils.npc_spline_refinement import (
    NESplineRefiner,
    extract_profile_along_norm,
    adjust_spline_points
)
from tools.utility_functions import (
    extract_cropped_images,
    filter_data_by_indices,
    find_segments,
    calc_tangent_endpts
)
from tools.geom_tools import (
    get_u_range_from_bspline,
    bspline_from_tck
)

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

def profile_optim_setup(single_intensity_profile: np.ndarray, single_dist_profile: np.ndarray, model_config: dict, spline_refiner_config: dict, device: torch.device):
    # Model functions expect a batch dimension -> create a "batch" of 1 via .unsqueeze(0)
    intensity_tensor = torch.tensor(
        single_intensity_profile, dtype=torch.float32, device=device
    ).unsqueeze(0) # Shape [1, S]
    dist_tensor = torch.tensor(
        single_dist_profile, dtype=torch.float32, device=device
    ).unsqueeze(0) # Shape [1, S]
# --- Initial Guess ---
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

def test_single_profile_optimization(single_intensity_profile: np.ndarray, single_dist_profile: np.ndarray, model_config: dict, spline_refiner_config: dict, device: torch.device) -> tuple:
    # Model functions expect a batch dimension -> create a "batch" of 1 via .unsqueeze(0)
    intensity_tensor = torch.tensor(
        single_intensity_profile, dtype=torch.float32, device=device
    ).unsqueeze(0) # Shape [1, S]
    dist_tensor = torch.tensor(
        single_dist_profile, dtype=torch.float32, device=device
    ).unsqueeze(0) # Shape [1, S]
    
    # --- Initial Guess ---
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

    # --- Setup Theseus Objects ---
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
