# -*- coding: utf-8 -*-

# In the Yeast_processor class...

def _prepare_refinement_inputs(self, dual_strain, makefig):
    """
    Prepares the necessary inputs for the spline refinement process based on mode flags.
    Handles loading of images and selection of initial spline data.
    """
    if dual_strain:
        # Logic for handling dual strain data
        points = self.pointsnpc
        deriv = self.derivnpc
        groups = [[0, np.size(p[0], axis=1)] for p in points]
        npc_image_stack = tifffile.imread(self.path + self.fn_track_ch1)
    else:
        # Standard logic
        points = self.initial_spline_points
        deriv = self.initial_spline_derivative
        groups = self.initial_spline_groups
        npc_image_stack = tifffile.imread(self.path + self.fn_track_ch2)
    
    # Special handling for 'makefig' mode which may use a subset of data
    if makefig:
        points = points[3:5]
        deriv = deriv[3:5]
        groups = groups[3:5]
        
    # Process image stack and calculate mean
    npc_image_stack = npc_image_stack[self.frames_npcfit[0]:self.frames_npcfit[1], :, :]
    self.imgshape = np.shape(npc_image_stack) # Set imgshape attribute
    npc_mean = np.mean(npc_image_stack, axis=0)
    
    return npc_mean, points, deriv, groups

def _process_and_plot_results(self, points_allcells, npc_mean, registration=True, save_fig=True):
    """
    Processes raw spline results: applies registration transform and plots the final output.
    """
    fig, ax = plt.subplots()
    ax.imshow(npc_mean, cmap='gray')
    ax.axis('off')

    transformed_points = []
    for cell_idx, cell_groups in enumerate(points_allcells):
        transformed_cell_groups = []
        for group_idx, points_refined in enumerate(cell_groups):
            
            # Apply registration transform if enabled
            if registration:
                points_refined_morphed = self.transform_coordinates(
                    points_refined.T,
                    self.registration['scale'],
                    self.registration['angle'],
                    self.registration['tvec'][::-1],
                    [npc_mean.shape[1] / 2, npc_mean.shape[0] / 2]
                ).T
                transformed_cell_groups.append(points_refined_morphed)
                xi, yi = points_refined_morphed[0, :], points_refined_morphed[1, :]
            else:
                transformed_cell_groups.append(points_refined)
                xi, yi = points_refined[0, :], points_refined[1, :]
            
            ax.plot(xi, yi, linewidth=0.5, color='green')
        
        transformed_points.append(transformed_cell_groups)
            
    if save_fig:
        fig.savefig(self.resultprefix + 'refined_spline.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return transformed_points

def _calculate_and_adjust_bounding_boxes(self):
    """Calculates and adjusts bounding boxes around the final refined NPC splines."""
    bounding_boxes = []
    for cell_groups in self.pointsnpc:
        min_r, max_r, min_c, max_c = 1e6, 0, 1e6, 0
        for points_refined in cell_groups:
            min_c = min(np.min(points_refined[0, :]), min_c)
            max_c = max(np.max(points_refined[0, :]), max_c)
            min_r = min(np.min(points_refined[1, :]), min_r)
            max_r = max(np.max(points_refined[1, :]), max_r)
        bounding_boxes.append([min_r, max_r, min_c, max_c])
    
    self.bbox_NE = np.array(bounding_boxes)
    
    # Adjust bounding boxes (expansion and rounding)
    track_rnp_full = tifffile.imread(self.path + self.fn_track_ch1)
    if track_rnp_full.ndim == 2:
        track_rnp_full = track_rnp_full[None, ...]
    
    img_h, img_w = track_rnp_full.shape[1], track_rnp_full.shape[2]
    
    for i in range(len(self.bbox_NE)):
        bbox = self.bbox_NE[i].copy()
        expansion = self.roisize / 2 + 16 # Magic number 16 could be a parameter
        
        bbox[0] -= expansion  # min_r
        bbox[2] -= expansion  # min_c
        bbox[1] += expansion  # max_r
        bbox[3] += expansion  # max_c

        # Clip to image bounds and round to nearest even number
        bbox[0] = int(np.ceil(max(bbox[0], 0) / 2) * 2)
        bbox[2] = int(np.ceil(max(bbox[2], 0) / 2) * 2)
        bbox[1] = int(np.ceil(min(bbox[1], img_h) / 2) * 2)
        bbox[3] = int(np.ceil(min(bbox[3], img_w) / 2) * 2)
        self.bbox_NE[i] = bbox

def _run_precision_estimation(self, deriv, length_line, bounds, number_points, sampling, groups, points, movie, sampling_normal, registration, smoothness, Lambda, number_mean):
    """Runs the fitting process on odd/even frames to estimate precision."""
    npc_stack = tifffile.imread(self.path + self.fn_track_ch2)
    
    # Select even and odd frames
    npc_mean1 = np.mean(npc_stack[::2, :, :][:number_mean, :, :], axis=0)
    npc_mean2 = np.mean(npc_stack[1::2, :, :][:number_mean, :, :], axis=0)
    
    return self.fit_per_meanforprecision(
        npc_mean1, deriv, length_line, bounds, number_points, sampling, 
        groups, points, movie, sampling_normal, registration, 0, 
        smoothness, Lambda, npc_mean2=npc_mean2
    )

# Helper function to extract intensity profiles normal to a spline
def _extract_normal_profiles(npc_mean, points, normals, selection, length_line, sampling_normal, img_shape):
    """
    Extracts 1D intensity profiles from an image along normal vectors at selected spline points.
    
    Returns:
        - zi_array (np.array): Array of intensity profiles.
        - dist_array (np.array): Array of distance axes for each profile.
        - error_flag (bool): True if any profile went out of image bounds.
    """
    num_profiles = len(selection)
    zi_array = np.zeros((num_profiles, sampling_normal))
    dist_array = np.zeros((num_profiles, sampling_normal))
    error_flag = False

    for k, select_idx in enumerate(selection):
        point = points[:, select_idx]
        normal_slope = normals[:, select_idx] / np.linalg.norm(normals[:, select_idx])

        start = point - length_line * normal_slope
        end = point + length_line * normal_slope
        
        x0, y0 = start[0], start[1]
        x1, y1 = end[0], end[1]

        # Check if the line is out of bounds
        if not (0 <= np.round(x0) < img_shape[1] - 2 and 0 <= np.round(y0) < img_shape[0] - 2 and
                0 <= np.round(x1) < img_shape[1] - 2 and 0 <= np.round(y1) < img_shape[0] - 2):
            error_flag = True
            return None, None, error_flag

        x, y = np.linspace(x0, x1, sampling_normal), np.linspace(y0, y1, sampling_normal)
        zi = npc_mean[np.round(y).astype(int), np.round(x).astype(int)]
        
        dist_alongline = np.cumsum(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
        dist_alongline = np.append(0, dist_alongline)
        
        zi_array[k, :] = zi
        dist_array[k, :] = dist_alongline
        
    return zi_array, dist_array, error_flag

# Helper to construct the initial guess for the MLE fit
def _construct_initial_guess(dist_array, zi_array, bounds, length_line, dev, makefig=False):
    """Constructs initial parameter guesses for the intensity profile model."""
    smp = torch.from_numpy(zi_array).to(dev)
    x_line = torch.from_numpy(dist_array).to(dev)

    def line(x, z, r): # Helper for background line
        a = (z[:, -1] - z[:, 0]) / (r[:, -1] - r[:, 0])
        b = z[:, 0]
        return a[..., None] * x + b[..., None]

    area = torch.sum(smp - line(x_line, smp, x_line), dim=-1) * (x_line[:, 1] - x_line[:, 0])
    amp = smp[:, int(x_line.shape[1] / 2)] - line(x_line[:, int(x_line.shape[1] / 2)], smp, x_line)[:, 0]

    initial_guess = torch.zeros((x_line.shape[0], len(bounds))).to(dev)
    initial_guess[:, 0] = smp[:, 0]  # A
    initial_guess[:, 1] = smp[:, -1] # K
    initial_guess[:, 2] = 0.4      # B
    initial_guess[:, 3] = 1.0      # C
    initial_guess[:, 4] = 1.0      # nu
    initial_guess[:, 5] = 0.5      # Q
    initial_guess[:, 6] = length_line # M
    initial_guess[:, 7] = length_line # mu
    sigma_factor = 0.28 if makefig else 0.35
    initial_guess[:, 8] = area / amp * sigma_factor # sigma
    initial_guess[:, 9] = amp      # amp
    initial_guess[:, 10] = 0.0     # offset
    return initial_guess

# Helper for the MLE fitting step
def _fit_profiles_mle(dist_array, zi_array, initial_guess, bounds, iterations, Lambda, dev):
    """Performs the MLE fit for the intensity profiles."""
    smp_tensor = torch.from_numpy(zi_array).to(dev)
    dist_tensor = torch.from_numpy(dist_array).to(dev)
    param_range = torch.Tensor(bounds).to(dev)
    
    # !!!: "where the magic happens" 
    # This part initializes a class containing the model function and then
    # calls a custom Levenberg-Marquardt MLE fitter (LM_MLE_forspline_new)
    # to find the best-fit parameters for the 'intensity_func_sig' model.
    model = npcfit_class(dist_tensor) 
    mle = LM_MLE_forspline_new(model)
    
    params, _, traces = mle.forward(initial_guess.type(torch.cuda.FloatTensor),
                                     smp_tensor[..., None].type(torch.cuda.FloatTensor),
                                     param_range, iterations, Lambda)
    
    torch.cuda.empty_cache()
    return params.cpu().detach().numpy(), traces.cpu().detach().numpy()

# Helper to filter the fit results based on various criteria
def _filter_fits(params, traces, selection, bounds, iterations, zi_array, dist_array):
    """
    Applies multiple filters to the results of the MLE fit.
    
    Returns:
        - A boolean mask of the same size as the original selection.
    """
    # 1. Filter based on hitting parameter bounds or max iterations
    iterations_vector = np.full(params.shape[0], iterations)
    for loc in range(params.shape[0]):
        try:
            # Find first occurrence of 0 in traces, which indicates stop of fitting
            iterations_vector[loc] = np.where(traces[:, loc, 0] == 0)[0][0]
        except IndexError:
            pass # No zero found, it ran for all iterations
    
    hit_max_iterations = (iterations_vector == iterations)
    
    hit_bounds = np.zeros(params.shape[0], dtype=bool)
    for col_idx in range(params.shape[1]):
        # Check if parameters are too close to the bounds
        on_border = np.logical_or(params[:, col_idx] < bounds[col_idx][0] * 0.95,
                                  params[:, col_idx] > bounds[col_idx][1] * 0.95)
        hit_bounds = np.logical_or(hit_bounds, on_border)
        
    initial_filter = np.logical_not(np.logical_or(hit_bounds, hit_max_iterations))
    
    # Apply initial filter
    params_f1 = params[initial_filter]
    selection_f1 = selection[initial_filter]
    zi_array_f1 = zi_array[initial_filter]
    dist_array_f1 = dist_array[initial_filter]

    if len(selection_f1) == 0:
        return np.zeros_like(selection, dtype=bool) # Return all False if nothing passed

    # 2. Filter based on residual error
    def intensity_func_sig_np(t, A, K, B, C, nu, Q, M, mu, sigma, amplitude,offset):
        # A numpy version of the intensity function for error calculation
        exponent_richards = -B * (t - M)
        denominator_richards = C + Q * np.exp(exponent_richards)
        power_richards = 1 / nu
        richards_curve = A + (K - A) / denominator_richards ** power_richards
        exponent_gaussian = -(t - mu) ** 2 / (2 * sigma ** 2)
        gaussian_curve = amplitude * np.exp(exponent_gaussian)
        return richards_curve + gaussian_curve + offset

    error_filter_mask = np.ones(len(selection_f1), dtype=bool)
    for qq in range(len(selection_f1)):
        fitted_curve = intensity_func_sig_np(dist_array_f1[qq], *params_f1[qq])
        error_non_squared = (zi_array_f1[qq] - fitted_curve) / fitted_curve
        if np.max(np.abs(error_non_squared)) > 0.50:
            error_filter_mask[qq] = False
            
    # Create the final filter mask relative to the original `selection` array
    final_filter = np.zeros_like(selection, dtype=bool)
    original_indices_f1 = np.where(initial_filter)[0]
    final_passing_indices = original_indices_f1[error_filter_mask]
    final_filter[final_passing_indices] = True
    
    return final_filter

# Reusable utility to find the longest consecutive sequence of points
def _find_longest_sequence(selection, max_gap):
    """Finds the longest sequence of points with gaps smaller than max_gap."""
    if len(selection) == 0:
        return np.array([], dtype=int)

    longest_sequence = []
    current_sequence = [selection[0]]

    for i in range(1, len(selection)):
        if selection[i] - current_sequence[-1] < max_gap:
            current_sequence.append(selection[i])
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence.copy()
            current_sequence = [selection[i]]
    
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence
        
    return np.array(longest_sequence)

# Helper to refit a spline from refined points and filter it by its geometry
def _refit_and_filter_spline(points_refined, is_periodic, smoothness, sampling, max_signs, max_angle_diff):
    """
    Fits a new spline to refined points and filters it based on geometric properties.
    
    Returns:
        - A dictionary with new spline data, or None if filtering fails.
    """
    if len(points_refined) < 4:
        return None

    tck, u = scipy.interpolate.splprep([points_refined[:, 0], points_refined[:, 1]],
                                       per=is_periodic, k=3, s=smoothness, quiet=3)
    
    num_samples = max(len(points_refined), sampling)
    xi, yi = scipy.interpolate.splev(np.linspace(0, 1, num_samples), tck)
    dxi, dyi = scipy.interpolate.splev(np.linspace(0, 1, num_samples), tck, der=1)
    ddxi, ddyi = scipy.interpolate.splev(np.linspace(0, 1, num_samples), tck, der=2)
    
    # Cut ends for open splines
    if not is_periodic:
        cut = min(100, len(xi) // 4) # Safer cut
        xi, yi = xi[cut:-cut], yi[cut:-cut]
        dxi, dyi = dxi[cut:-cut], dyi[cut:-cut]
        ddxi, ddyi = ddxi[cut:-cut], ddyi[cut:-cut]
        if len(xi) < 4: return None

    # Filter based on change in angle (curvature proxy)
    angles = np.arctan2(dyi, dxi)
    angle_diffs = np.abs(np.rad2deg((np.diff(angles) + np.pi) % (2 * np.pi) - np.pi))
    if np.amax(angle_diffs) >= max_angle_diff:
        return None

    # Filter based on sign changes of curvature (inflection points)
    normal_slope = np.array([dyi, -dxi]) # Simplified normal
    dotproduct = (normal_slope[:, 1:] * normal_slope[:, :-1]).sum(0)
    signchange = ((np.roll(np.sign(dotproduct), 1) - np.sign(dotproduct)) != 0).astype(int)
    if sum(signchange) >= max_signs:
        return None
        
    return {
        'points': np.array([xi, yi]),
        'deriv': np.array([dxi, dyi]),
        'closed': 1 if is_periodic else 0,
        'tck_amplitude': tck_amplitude, # This would require fitting amplitude/sigma splines here too
        'tck_sigma': tck_sigma
    }