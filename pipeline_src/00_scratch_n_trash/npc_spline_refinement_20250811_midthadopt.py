# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 08:51:23 2025

@author: jctourtellotte
"""

# --- Outside Modules --- #
import tifffile
import math
import torch
import torch.nn as nn
import torch.optim as optim
import theseus as th
from scipy.interpolate import make_splprep
import numpy as np

# --- Included Modules --- #
from tools.utility_functions import calc_tangent_endpts, extract_cropped_images, sample_bspline

from utils.MLEstimation import NPC_FitPredictor, LM_MLE_forSpline, ModelFitCost

#from utils.ne_fit_utils import npcfit_class, [[\]]
# ???: check functions in utils.ne_fit_utils module


class NESplineRefiner:
    # Class - Why? bc image loading and mean frame config etc
    # def __init__(self, img_path, FoV_initial_fit_entry, FoV_initial_bspline_entry, FoV_ne_crop_box_entry, config_dict, device = torch.device('cpu')):
    def __init__(self, img_path, FoV_ne_crop_box_entry, config_dict, device = torch.device('cpu')):
        print('initiating Spline Refinement')
        self._cfg_fit = config_dict['ne_fit']
        self._FoV_id = FoV_ne_crop_box_entry['FoV_id']
        # self._initial_fit = FoV_initial_fit_entry['initial_fit']     
        # self._initial_bsplines = FoV_initial_bspline_entry['bsplines']
        # self._registration = reg_values
        self._current_device = device

        self._ne_images = extract_cropped_images(img_path, self._cfg_fit['frame_range'], FoV_ne_crop_box_entry['cropped_img'])
        
# QUESTION: what is this for?
        half_length_line = self._cfg_fit['line_length'] / 2
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

# --- Getters and Setters --- #
    def _get_FoV_id(self):
        return self._FoV_id

    def _get_ne_imgs(self):
        return self._ne_images

# --------------------------- #

    def refine_from_initial_fit(self, initial_fit_data):
        # TODO: img_path_rnp - to cover "dual image" case was included in original version; make new function for this; the functionality needed in is in the original code for figure 1
        cropped_imgs = self._get_ne_imgs()

        ne_refine_fit_list = []

        # for each ne in initial fit dictionary
        for ne_init_fit in initial_fit_data:
            # refine the fit for the ne & return fit
            print(f"Current NE mask label: {ne_init_fit['ne_label']}")
            try:
                refined_fit = \
                    refine_single_ne_fit(img_mean = [entry for entry in cropped_imgs if entry['ne_label'] ==  ne_init_fit['ne_label']],
                                         init_spl_pts = ne_init_fit['init_spline_points'],
                                         init_spl_deriv = ne_init_fit['init_spline_deriv'],
                                         config_fit = self._cfg_fit,
                                         init_bounds = self._initial_bounds,
                                         device = self._current_device)
                    
                single_refined_fit = {"ne_label": ne_init_fit['ne_label'],
                                      "parameters": refined_fit['parameters'],
                                      "param_history": refined_fit['param_history']}
                print(f"Returned: single refined fit for mask label: {single_refined_fit['ne_label']}")
                ne_refine_fit_list.append(single_refined_fit)
            except ValueError as e:
                print(f"Warning: Refining of indiivdual NE spline fit failed. \n{e}. \nSkipping.")
                continue

        return ne_refine_fit_list

    def refine_initial_bsplines(self, initial_bsplines):
        # TODO: img_path_rnp - to cover "dual image" case was included in original version; make new function for this; the functionality needed in is in the original code for figure 1
        cropped_imgs = self._get_ne_imgs()

        ne_refine_bspline_list = []

        # for each ne in initial fit dictionary
        for ne_init_bspline in initial_bsplines['bsplines']:
            # refine the fit for the ne & return fit
            ne_label = ne_init_bspline['ne_label']
            print(f"Current NE mask label: {ne_label}")

            mean_ne_img = mean_ne_img = next((entry['cropped_img'] for entry in cropped_imgs if entry['ne_label'] == ne_label), None)
            if mean_ne_img is None:
                raise ValueError(f"No cropped image found for NE label {ne_label}")

            try:
                # define evenly spaced points to evaluate
                evaluated_points, evaluated_derivatives = sample_bspline(ne_init_bspline['bspline_object'], self._cfg_fit['normal_lines_n'])

                intensity_profiles, dist_along_norm = self._sample_norm_intensity(mean_ne_img, evaluated_points, evaluated_derivatives)
                
                refined_bspline = self.refine_ne_bsplines(evaluated_points, evaluated_derivatives, intensity_profiles, dist_along_norm)
                single_refined_spline = {'ne_label': mean_ne_img, 'reflined_spline': refined_bspline}

                print(f"Returned: single refined fit for mask label: {ne_label}")
                ne_refine_bspline_list.append(single_refined_spline)
            except ValueError as e:
                print(f"Warning: Refining of indiivdual NE spline fit failed. \n{e}. \nSkipping.")
                continue

        return ne_refine_bspline_list

    def _sample_norm_intensity(self, mean_ne_img, evaluated_points, evaluated_derivatives):
        line_length = self._cfg_fit['line_length']
        n_samples_along_normal = self._cfg_fit['n_samples_along_normal']
        normal_lines_n = self._cfg_fit['normal_lines_n']

        tangent_endpoints, normal_endpoints = \
        calc_tangent_endpts(evaluated_points, evaluated_derivatives, line_length, True)

        # calculate intensity profile
        intensity_profiles, dist_along_norm, error_flag = \
            extract_profile_along_norm(mean_ne_img, evaluated_points, normal_endpoints, normal_lines_n, n_samples_along_normal)
        
        if error_flag:
            raise ValueError("Error extracting intensity profiles from the image.")
            return
        
        return intensity_profiles, dist_along_norm

    def refine_ne_bsplines(self, evaluated_points, evaluated_derivatives, intensity_profiles, dist_along_norm):
        line_length = self._cfg_fit['line_length']
        intensity_params = self._cfg_fit['initial_guess']
        # learning_rate = self._cfg_fit['learning_rate']
        # iterations = self._cfg_fit['iterations']
        # convergence_tolerance = self._cfg_fit['convergence_tolerance']
        # convergence_patience = self._cfg_fit['convergence_patience']
        device = self._current_device

        # define evenly spaced points to evaluate
        # u_values_to_sample = np.linspace(0, 1, normal_lines_n, endpoint=False)
        # evaluated_points = init_bspline(u_values_to_sample)
        # evaluated_derivatives = init_bspline.derivative(1)(u_values_to_sample)

        # tangent_endpoints, normal_endpoints = \
        # calc_tangent_endpts(evaluated_points, evaluated_derivatives, line_length, True)

        #     # calculate intensity profile
        # intensity_profiles, dist_along_norm, error_flag = \
        #     extract_profile_along_norm(mean_ne_img, evaluated_points, normal_endpoints, normal_lines_n, n_samples_along_normal)
        # throw error if  error_flag > 0 and return None; return values if < 0
        # function catching error should continue to the next iteration if an error without adding a reflined spline
        # QUESTION should it do anything else to keep track of this?
        # if error_flag:
        #     raise ValueError("Error extracting intensity profiles from the image.")
        #     return

        intensity_tensor = torch.tensor(intensity_profiles, dtype=torch.float32, device=device)
        dist_tensor = torch.tensor(dist_along_norm, dtype=torch.float32, device=device)

        initial_guess_tensor = construct_initial_guess(
            dist_along_norm_tensor = dist_tensor,
            intensity_init_tensor = intensity_tensor,
            line_length = line_length,
            num_parameters = len(self._initial_bounds),
            intensity_params = intensity_params,
            device = device
            )
        # # Wrap the parameters in nn.Parameter to make them learnable
        # theta = nn.Parameter(initial_guess_tensor)

        # npc_prediction_model = NPC_FitPredictor(dist_tensor)

        # # QUESTION parameter tuning
        # optimizer = optim.Adam([theta], lr=learning_rate) # Use Adam
        # loss_fn = nn.PoissonNLLLoss(log_input=False) # Use Poisson NLL for MLE

        # last_loss = float('inf')
        # patience_counter = 0 # used to check for stablization of loss over several iterations

        # for i in range(iterations):
        #     optimizer.zero_grad() # Reset gradients

        #     # Forward pass: get model prediction
        #     predicted_intensity, _ = npc_prediction_model(theta)
        #     # TODO look into how the methods compare re: replacing need for Jacobian
        #     loss = loss_fn(predicted_intensity.squeeze(-1), intensity_tensor)

        #     # Backward pass: compute gradients automatically
        #     loss.backward()

        #     # --- ADD THIS DEBUGGING BLOCK ---
        #     # Check for NaN in gradients before the optimizer step
        #     if torch.isnan(theta.grad).any():
        #         print(f"NaN detected in GRADIENTS on iteration {i}!")
        #         breakpoint() # The debugger will stop here
        #     # --------------------------------

        #     optimizer.step() # Update parameters
            
        #     # Check for NaN in parameters after the optimizer step
        #     if torch.isnan(theta).any():
        #         print(f"NaN detected in PARAMETERS on iteration {i}!")
        #         breakpoint() # Or here
        #     # --------------------------------
            
        #     # --- Check Convergence ---
        #     if abs(last_loss - loss.item()) < convergence_tolerance:
        #         patience_counter += 1
        #     else:
        #         patience_counter = 0
            
        #     if patience_counter >= convergence_patience:
        #         print(f"Converged after {i+1} iterations.")
        #         break
                
        #     last_loss = loss.item()

        # # --- Creating the Refined Spline ---
        # final_params = theta.detach().cpu().numpy()
        
        theta_variable = th.Vector(tensor=initial_guess_tensor, name="theta")

        # --- 2. Setup and Run the Theseus Optimizer ---
        # ** Define the Cost Function and Weight **
        poisson_weight_diag = 1.0 / torch.sqrt(intensity_tensor + 1e-6)
        cost_weight = th.DiagonalCostWeight(poisson_weight_diag)
        cost_function = ModelFitCost(dist_tensor, intensity_tensor, cost_weight, theta_variable)
        
        # ** Define the Optimization Problem **
        objective = th.Objective()
        objective.add(cost_function)

        # Define the optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=self._cfg_fit['iterations'],
            step_size=0.5 # Damping parameter
        )
        
        # Create the Theseus Layer, which makes the solver a callable module
        theseus_layer = th.TheseusLayer(optimizer)
        
        # Move the layer and data to the correct device
        theseus_layer.to(device)
        dist_tensor = dist_tensor.to(device)
        intensity_tensor = intensity_tensor.to(device)
        initial_guess_tensor = initial_guess_tensor.to(device)

        # Provide the initial values for the variables to be optimized
        inputs = {"theta": initial_guess_tensor}

        # ** Run the optimization **
        final_state, info = theseus_layer.forward(inputs)

        # --- 3. Post-Processing (largely the same) ---
        final_params = final_state["theta"].detach().cpu().numpy()

        if np.isnan(final_params).any():
            raise ValueError("Theseus optimization produced NaN values.")

        # Determine refinded points (will use to create refined bspline)
        # 'mu' @ index 7 is the offset relative to the center of the line
        refined_points = adjust_spline_points(final_params[:, 7], line_length, evaluated_points, evaluated_derivatives)
        tck_refined, _ = make_splprep([refined_points[1, :], refined_points[0, :]], s=1.0, k=3) # assumes periodic

        return tck_refined

def adjust_spline_points(optimized_mu, line_length, points, derivatives):
    position_change = optimized_mu - (line_length / 2)

    # Calculate the unit normal vectors (this can be a helper function)
    norm_magnitudes = np.linalg.norm(derivatives, axis=0)
    unit_normals = np.vstack((-derivatives[1, :], derivatives[0, :])) / (norm_magnitudes + 1e-9)

    # Update the original sample points by moving them along their normal vectors
    refined_points = points + unit_normals * position_change

    return refined_points


# normal_lines_n - number of sample points at which to find normal line
# !!!: can't exceed # of sampled splines - should this throw an error?
# !!!: OR have loading the config check for potential errors such as this!
# normal_samples_n - number of samples to take along the normal line;
# !!!:  100 was used; 10 does not provide enough information for the nuumber of parameters (11)
def refine_single_ne_fit(img_mean, init_spl_pts, init_spl_deriv, config_fit, init_bounds, device, correct_offset = False):
    normal_lines_n = config_fit['normal_lines_n']
    normal_samples_n = config_fit['n_samples_along_normal']
    line_length = config_fit['line_length']
    intensity_parameters = config_fit['initial_guess']
    little_lambda = config_fit['little_lambda']
    iterations = config_fit['iterations']
    correct_offset = config_fit['offset_correction']
    
    num_init_points = init_spl_pts.shape[1]
    # set endpoint = False to exclude index of num_init_points, as its inclusion
    #   will be beyond the last index (num_init_points - 1)
    normal_sample_indices = np.linspace(0, num_init_points, normal_lines_n, endpoint = False, dtype = "uint16")
    sample_points = init_spl_pts[:, normal_sample_indices]
    sample_deriv = init_spl_deriv[:, normal_sample_indices]

    tangent_endpoints, normal_endpoints = \
        calc_tangent_endpts(sample_points, sample_deriv, line_length, True)

    # calculate intensity profile
    intensity_profile, dist_along_norm, error_flag = \
        extract_profile_along_norm(img_mean, sample_points, normal_endpoints, normal_lines_n, normal_samples_n)
    # set to check for error_flag
    # throw error if  > 0 and return None
    # return values if < 0
    # function catching error should continue to the next iteration if an error without adding a reflined spline
    # ???: should it do anything else to keep track of this?
    if error_flag > 0:
        raise ValueError("Reason: more than one error raised prior to applying LM MLE.")
        return

    # converting to tensors
    init_spl_pts_tensor = torch.tensor(init_spl_pts, dtype=torch.float32, device=device)
    intensity_tensor = torch.tensor(intensity_profile, dtype=torch.float32, device=device)
    dist_on_norm_tensor = torch.tensor(dist_along_norm, dtype=torch.float32, device=device)
    param_bounds_tensor = torch.tensor(init_bounds.copy(), dtype=torch.float32, device=device)

    # construct initial guess
    initial_guess_tensor = \
        construct_initial_guess(dist_along_norm_tensor = dist_on_norm_tensor,
                                intensity_init_tensor = intensity_tensor,
                                line_length = line_length,
                                num_parameters = len(init_bounds),
                                intensity_params = intensity_parameters,
                                device = device)

    # !!!: "where the magic happens" - PvV
    npc_prediction_model = NPC_FitPredictor(dist_on_norm_tensor)

    # bounds_ext = init_bounds.copy()
    # directly convert a copy to a tensor instead (above, see param_bounds_tensor)

    mle_forspline_model = LM_MLE_forSpline(npc_prediction_model, device)

    params_tensor, _, history_tensor = \
        mle_forspline_model.forward(initial_guess_tensor,
                                    intensity_tensor.unsqueeze(-1),
                                    param_bounds_tensor,
                                    iterations,
                                    little_lambda,
                                    device,
                                    correct_offset)
        
    # # -- Post Processing of LM MLE fitting -- #
    # # Find index of parameter fitting (ie. where it converges)
    # convergence_iter_tensor = find_fit_convergence(history_tensor,
    #                                                iterations,
    #                                                device)
    # # Filter out unacceptable fits based on parameter bounds and iteration number
    # try:
    #     accepted_fits_tensor = bound_and_iter_lim_filter(params_tensor,
    #                                                      convergence_iter_tensor,
    #                                                      param_bounds_tensor,
    #                                                      normal_lines_n,
    #                                                      iterations,
    #                                                      device
    #                                                      )
    # except ValueError as e:
    #     raise ValueError({e})
    # # Apply the filter to all relevant tensors
    # filtered_params_tensor = params_tensor[accepted_fits_tensor, :]
    
    # filtered_init_spl_pts_tensor = init_spl_pts_tensor[accepted_fits_tensor]
    
    # filtered_dist_on_norm_tensor = dist_on_norm_tensor[accepted_fits_tensor]
    
    # filtered_intensity_tensor = intensity_tensor[accepted_fits_tensor]
    
    
    
    #######
    
    
    ne_single_label_refined_fit = {"parameters": params_tensor,
                                   "param_history": history_tensor}
    
    return ne_single_label_refined_fit

def find_fit_convergence(history_tensor, iterations, device):
    # Find iteration count for each fit (parameter convergence)
    #   finding the iteration(s) when the individual parameter stops 
    #   changing (ie. equals zero) rather than when they all stop changing
    history_tensor_permuted = history_tensor.permute(1, 0, 2)

    zero_mask = (history_tensor_permuted[:, :, 0] == 0)
    # Find first True in each row (batch)
    first_zero_idx = zero_mask.float().argmax(dim=1)
    # Check if there actually was a zero (vs argmax returning 0 for all False)
    has_zero = zero_mask.any(dim=1)
    # Set iterations_vector
    fit_convergence_tensor = torch.where(has_zero,
                                    first_zero_idx.float(),
                                    torch.tensor(iterations,
                                                 dtype=torch.float32,
                                                 device=device)
                                    )
    return fit_convergence_tensor


def find_per_param_convergence(history_tensor, iterations, device, threshold=1e-6):

    # Calculate parameter changes between iterations
    param_diffs = torch.diff(history_tensor, dim=0)  # [iteration-1, batch, parameter]
    param_diffs_permuted = param_diffs.permute(1, 0, 2)  # [batch, iteration-1, parameter]

    # Initialize convergence matrix
    convergence_matrix = torch.full((param_diffs_permuted.size(0), param_diffs_permuted.size(2)),iterations, dtype=torch.float32, device=device)
    # Check each parameter separately

    for param_idx in range(param_diffs_permuted.size(2)):

        param_changes = param_diffs_permuted[:, :, param_idx].abs()  # [batch, iteration-1]

        converged_mask = (param_changes < threshold)

        

        # Find first iteration where parameter stops changing

        first_converged_idx = converged_mask.float().argmax(dim=1)

        has_converged = converged_mask.any(dim=1)

        

        convergence_matrix[:, param_idx] = torch.where(has_converged,

                                                       first_converged_idx.float() + 1,

                                                       torch.tensor(iterations, dtype=torch.float32, device=device))

    

    return convergence_matrix

def bound_and_iter_lim_filter(params_tensor, convergence_iter_tensor, param_bounds_tensor, normal_lines_n, iterations, device):
    # 1. Filter based on parameter bounds
    # Check if each parameter is within 95% of bounds
    bounds_lower = param_bounds_tensor[:, 0] * 0.95
    bounds_upper = param_bounds_tensor[:, 1] * 0.95
    
    params_outof_bds_tensor = torch.zeros(normal_lines_n, dtype=torch.bool, device=device)
    
    for param_idx in range(params_tensor.shape[1]):
        outside = torch.logical_or(
            params_tensor[:, param_idx] < bounds_lower[param_idx],
            params_tensor[:, param_idx] > bounds_upper[param_idx]
            )
        params_outof_bds_tensor = torch.logical_or(params_outof_bds_tensor, outside)

    # 2. Combined filtering
    # Mark fits that hit iteration limit or bounds
    bad_fits_tensor = torch.logical_or(params_outof_bds_tensor,
                                       convergence_iter_tensor == iterations
                                       )
    
    filtered_fit_mask_tensor = ~bad_fits_tensor
    
    n_good_fits = filtered_fit_mask_tensor.sum().item()
    
    print(f"Filtering NPC on bounds: {params_outof_bds_tensor.sum().item()}/{normal_lines_n}")
    print(f"Filtering NPC on iterations: {(convergence_iter_tensor == iterations).sum().item()}/{normal_lines_n}")
    print(f"Total good fits: {n_good_fits}/{normal_lines_n}")
    
    # !!!: throw and catch this error
    if n_good_fits == 0:
        raise ValueError(f"Reason: No acceptable fits based on bound and iteration conditions")
    
    
    return filtered_fit_mask_tensor


def extract_profile_along_norm(img_mean, sample_points, normal_endpts, normal_lines_n, n_samples_along_normal = 10):
    img_shape = img_mean.shape
    zi_array = np.zeros((normal_lines_n, n_samples_along_normal))
    dist_array = np.zeros((normal_lines_n, n_samples_along_normal))
    error_flag = False

    for i in range(normal_lines_n):
        y0, x0 = normal_endpts[0][0][i], normal_endpts[0][1][i]
        y1, x1 = normal_endpts[1][0][i], normal_endpts[1][1][i]

        y, x = np.linspace(y0, y1, n_samples_along_normal), \
            np.linspace(x0, x1, n_samples_along_normal)

        # Check if the line is out of bounds
        if not (0 <= np.round(x0) < img_shape[1] - 2 and 0 <= np.round(y0) < img_shape[0] - 2 and
                0 <= np.round(x1) < img_shape[1] - 2 and 0 <= np.round(y1) < img_shape[0] - 2):
            error_flag = True
            return None, None, error_flag

        zi = img_mean[np.round(y).astype(int), np.round(x).astype(int)]
        zi_array[i,:] = zi

    ## Alternative if are not looking for equal spacing re: line**
    # canvas = np.zeros(img_mean.shape, dtype=float)
    # for i in range(len(start_y)):
    #     line_rows, line_cols = skimage_line(start_y[i], start_x[i], end_y[i], end_x[i])
    #     canvas[line_rows, line_cols] = i + 1

    # equal space accounts for line slope by weighting as an emergent property
    #   of more points associated with pixels the normal line traverses a longer
    #   distance through

        dist_alongline = np.cumsum(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
        dist_alongline = np.append(0, dist_alongline)
        dist_array[i,:] = dist_alongline

    return zi_array, dist_array, error_flag

def construct_initial_guess(dist_along_norm_tensor, intensity_init_tensor, line_length, num_parameters, intensity_params, device):

    def line(dist_along_line, intensity_values):
        a = (intensity_values[:, -1] - intensity_values[:, 0]) / (dist_along_line[:, -1] - dist_along_line[:, 0])
        b = intensity_values[:, 0] - a * dist_along_line[:, 0]
        return a[..., None] * dist_along_line + b[..., None]

    # area between the measured intensity curve and a straight line drawn between the end points
    area = torch.sum(intensity_init_tensor - \
                     line(dist_along_norm_tensor,
                          intensity_init_tensor), dim=-1) * \
        (dist_along_norm_tensor[:,1] - dist_along_norm_tensor[:,0])

    # highest of curve (intensity) at the center of the curve
    mid_idx =  dist_along_norm_tensor.shape[1] // 2
    amp = intensity_init_tensor[:, mid_idx] - line(dist_along_norm_tensor,
                                                   intensity_init_tensor)[:, mid_idx]

    # amp = intensity_init_tensor[:, int(dist_along_norm_tensor.shape[1] / 2)] - \
    #     line(dist_along_norm_tensor[:, int(dist_along_norm_tensor.shape[1] / 2)],
    #          intensity_init_tensor,
    #          dist_along_norm_tensor)[:, 0]
    epsilon = 1e-9

    init_guess = torch.zeros((dist_along_norm_tensor.shape[0], num_parameters)).to(device)

    init_guess[:, 0] = intensity_init_tensor[:, 0] #A
    init_guess[:, 1] = intensity_init_tensor[:, -1] #K
    init_guess[:, 2] = 0.4 #B
    init_guess[:, 3] = 1 #C
    init_guess[:, 4] = 1 #nu
    init_guess[:, 5] = 0.5 #Q
    init_guess[:, 6] = line_length
    init_guess[:, 7] = line_length
    # ???: What is the difference? Why is the difference?
    # if makefig:
    #     init_guess[:, 8] = area / amp * 0.28 # sigma
    # else:
    #     init_guess[:, 8] = area / amp * 0.35 # sigma
    init_guess[:, 8] = area / (amp + epsilon)  * 0.35 # sigma
    init_guess[:, 9] = amp # amp
    init_guess[:, 10] = 0 # offset

    return init_guess