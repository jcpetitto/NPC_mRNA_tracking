"""
Created on Sat May 17 18:49:35 2025

@author: jctourtellotte
"""

# outside packages
import math
import os
import tifffile
import skimage
import numpy as np
import imreg_dft as ird
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import zoom

from tools.utility_functions import crop_image


# Function operates on a given FoV, using path info passed as its path_dict
#    and corresponding crop_dim_dict
def image_registration(path_dict, ch1_crop_dim_dict, ch2_crop_dim_dict, frame_range, frames_per_average, padding, upsample_factor, upscale_factor, reg_mode = 1, ne_label_pair_dict = None, detailed_output = False):
    reg_all_ne_label = {}
    # need to register each sub-image (defined by crop dimensions) in a field of view
    # registration mode sets whether using the pre or post fluorescent microscopy bright field images
    try:
        channel1_path = os.path.join(path_dict['FoV_collection_path'], path_dict['imgs'][f'fn_ch1_reg{reg_mode}'])
        channel2_path = os.path.join(path_dict['FoV_collection_path'], path_dict['imgs'][f'fn_ch2_reg{reg_mode}'])
    except ValueError as e:
        print(f"Warning: files do not exist for the given registration mode - {reg_mode} - and/or it is an invalid number (options are 1 or 2) {e}. Skipping.")
        return

    channel1_bf = tifffile.imread(channel1_path)
    channel2_bf = tifffile.imread(channel2_path)
    
    ch1_bf_mean = np.mean(channel1_bf, axis = 0)
    ch2_bf_mean = np.mean(channel2_bf, axis = 0)

    FoV_reg_data = compute_registration(ch1_bf_mean, ch2_bf_mean, padding, upsample_factor, upscale_factor)

    # ne_label_list is all of the labels for ch1 if a paired dictionary
    #   is NOT include (ie. is the default value None) as this indicates
    #   the ch1/ch2 registration crops are NOT different (as they were not
    #   created based on different NPC associated fluorophores)
    ne_label_list = ch1_crop_dim_dict.keys() if ne_label_pair_dict is None else ne_label_pair_dict.keys()

    # for ne_label, crop_bounds_ch1 in ch1_crop_dim_dict.items():
    for ne_label in ne_label_list:
        # Access the nested dictionary for the original position
        crop_bounds_ch1 = ch1_crop_dim_dict[ne_label]
        crop_bounds_ch2 = ch1_crop_dim_dict[ne_label] if ne_label_pair_dict is None else ch2_crop_dim_dict[ne_label_pair_dict[ne_label]]

        ch1_bf_crop = crop_image(channel1_bf, crop_bounds_ch1, pad_to_size=(75, 75))
        ch2_bf_crop = crop_image(channel2_bf, crop_bounds_ch2, pad_to_size=(75, 75))
        
        ch1_bf_crop_mean = np.mean(ch1_bf_crop, axis = 0)
        ch2_bf_crop_mean = np.mean(ch2_bf_crop, axis = 0)

        # dictionary for registration of the mean image re: current ne label

        current_reg_calc = compute_registration(ch1_bf_crop_mean, ch2_bf_crop_mean, padding, upsample_factor, upscale_factor)

        if current_reg_calc is None:
            print(f"Warning: Initial registration failed for NE label {ne_label}, likely due to mismatched crop shapes. Skipping this label.")
            continue  # Skips to the next ne_label
        

        # create list to store registration data based on the average of subsets of size 'frames_per_average'
        
        slice_n = 1
        for i in range(frame_range[0], frame_range[1], frames_per_average):
            slice_start = i
            slice_end = i+frames_per_average
            ch1_temp_mean = np.mean(ch1_bf_crop[slice_start:slice_end], axis = 0)
            ch2_temp_mean = np.mean(ch2_bf_crop[slice_start:slice_end], axis = 0)

            temp_ne_reg_data = compute_registration(ch1_temp_mean, ch2_temp_mean, padding, upsample_factor, upscale_factor)

            current_reg_calc.update({f'slice_{slice_n}': temp_ne_reg_data})
            slice_n += 1

        reg_all_ne_label.update({f'{ne_label}': current_reg_calc})

    return FoV_reg_data, reg_all_ne_label

def compute_registration(channel1_img, channel2_img, padding = 5, upsample_factor = 1000, upscale_factor = 1):
    # expects images of dimensions (rows, columns)
    # channel 1 is used as the reference channel

    # !!!: Add: Raise an error
    if channel1_img.shape != channel2_img.shape:
        print("Error: Image stacks must have the same shape.")
        return None
    ch1_padded = np.pad(array = channel1_img, pad_width = padding, mode = 'constant', constant_values = 0)
    ch2_padded = np.pad(array = channel2_img, pad_width = padding, mode = 'constant', constant_values = 0)    
    # 1. apply difference of Gaussians filter with standard paramaters
    #       to increase contrast
    ch1_filtered = skimage.filters.difference_of_gaussians(ch1_padded, 0.5, 1.5)
    ch2_filtered = skimage.filters.difference_of_gaussians(ch2_padded, 0.5, 1.5)

    # upscaling for bilinear interpolation; determining rotation is improved
    #   by upscaling before applying ird._get_ang_scale
    ch1_upscaled = zoom(ch1_filtered, upscale_factor, order = 1)  # Bilinear interpolation
    ch2_upscaled = zoom(ch2_filtered, upscale_factor, order = 1)
        
    scale, angle = ird.imreg._get_ang_scale([ch1_upscaled, ch2_upscaled], None)
        
    # rotate and scale the image based on the results of ird.imreg._get_ang_scale
    ch2_scaled_rotated = ird.transform_img(ch2_filtered, angle = angle, scale = scale)
    
    # Calculate the translation required to align channel 2 to channel 1
    #   This particular translation function works better after
    #   angle and scale factors are accounted for, as was done above
    #   using ird._get_ang_scale & ird.transform_img
    
    # translation = ird.imreg.translation(ch1_filtered, ch2_scaled_rotated)['tvec']
    # skimage version of phase_cross_correlation is likely the better choice here based on available parameters re: control, including built in upsampling, and efficiency
    translation, _, _ = skimage.registration.phase_cross_correlation(
        ch1_filtered,
        ch2_scaled_rotated,
        upsample_factor = upsample_factor
        )
        
    # apply translation vector to the rotated and scaled image
    # ch2_sca_rot_trans = ird.transform_img(ch2_scaled_rotated, tvec = translation)
    results = {
        "scale": scale,
        "angle": math.radians(angle),
        "shift_vector": translation
    }
    
    return results

def calculate_rss(vec_y, vec_x, angle_rad, scale_val, radius_px):
    """
    Calculates the root sum of squares in pixels
    
    Args:
        angle_rad: Angle in RADIANS.
    """
    var_trans = vec_y**2 + vec_x**2 # in pixels
    var_angle = (angle_rad * radius_px)**2 # convert to pixels
    var_scale = (scale_val * radius_px)**2 # convert to pixels

    total_variance = var_trans + var_angle + var_scale

    return np.sqrt(total_variance)

def get_slice_sigma(ne_data, radius_px):
    """
    Calculates the internal precision (sigma) of a single nucleus from its slices.
    """
    slice_metrics = []
    
    # Extract slice data
    for k, v in ne_data.items():
        if k.startswith('slice_') and 'shift_vector' in v:
            slice_metrics.append([
                v['shift_vector'][0], 
                v['shift_vector'][1],
                v.get('angle', 0.0), 
                v.get('scale', 1.0)
            ])
            
    if len(slice_metrics) < 2:
        return 0.5 # Default fallback

    arr = np.array(slice_metrics)
    
    # Calculate Component Sigmas
    std_y = np.std(arr[:, 0])
    std_x = np.std(arr[:, 1])
    std_angle = np.std(arr[:, 2])
    std_scale = np.std(arr[:, 3])
    
    # RSS of the Sigmas
    return calculate_rss(std_y, std_x, std_angle, std_scale, radius_px)

def calculate_mode_reg_stats(reg_data, radius_px):
    """
    Calculates individual Precision (Sigma) and Signal (Offset) for a single mode.
    Returns:
        FoV_sigma_dict: Mean precision across all nuclei in a FoV
        {fov_id: {ne_label: {'sigma': float, 'signal': float}}}
    """
    nucleus_stats = {}

    for fov_id, fov_data in reg_data.items():
        if fov_id not in nucleus_stats:
            nucleus_stats[fov_id] = {}

        # --- PASS ONE: Systemic Offset (FoV Mean Shift) ---
        ne_keys = [k for k in fov_data.keys() 
                    if isinstance(fov_data[k], dict) and 'shift_vector' in fov_data[k]]

        for label in ne_keys:
            data = fov_data[label]

            # --- Absolute Offset (Signal) ---
            # "How far are the cameras apart?"
            raw_scale = data.get('scale', 1.0)
            scale_delta = raw_scale - 1.0
            signal = calculate_rss(
                            vec_y=data['shift_vector'][0], 
                            vec_x=data['shift_vector'][1],
                            angle_rad=data.get('angle', 0.0), 
                            scale_val=scale_delta,
                            radius_px=radius_px
                        )

            # --- Precision (Slice Noise) ---
            # collect metrics: [y, x, angle, scale]
            slice_metrics = []
            for k, v in data.items():
                if k.startswith('slice_') and 'shift_vector' in v:
                    slice_metrics.append([
                        v['shift_vector'][0], v['shift_vector'][1],
                        v.get('angle', 0.0), v.get('scale', 1.0)
                    ])
            
            sigma = 0.5 # create default/fallback
            if len(slice_metrics) > 1:
                arr = np.array(slice_metrics)
                # Std Devs
                std_y, std_x = np.std(arr[:, 0]), np.std(arr[:, 1])
                std_angle, std_scale = np.std(arr[:, 2]), np.std(arr[:, 3])
                
                sigma = calculate_rss(std_y, std_x, std_angle, std_scale, radius_px)
                
            nucleus_stats[fov_id][label] = {'sigma': sigma, 'signal': signal}

    return nucleus_stats

def calculate_drift_map(reg_m1, reg_m2, radius_px):
    """
    Calculates the Drift Vector (Mode 2 - Mode 1) for every nucleus.
    Returns: {fov_id: {ne_label: drift_rss}}
    """
    drift_map = {}
    
    for fov_id, m1_data in reg_m1.items():
        if fov_id not in drift_map: drift_map[fov_id] = {}
        
        m2_data = reg_m2.get(fov_id, {})
        
        ne_keys = [k for k in m1_data.keys() 
                    if isinstance(m1_data[k], dict) and 'shift_vector' in m1_data[k]]

        for label in ne_keys:
            if label in m2_data:
                d1 = m1_data[label]
                d2 = m2_data[label]
                
                # Delta Calculation (Mode 2 - Mode 1)
                dy = d2['shift_vector'][0] - d1['shift_vector'][0]
                dx = d2['shift_vector'][1] - d1['shift_vector'][1]
                da = d2.get('angle', 0.0) - d1.get('angle', 0.0)
                ds = d2.get('scale', 1.0) - d1.get('scale', 1.0)
                
                # RSS of the Difference
                rss = calculate_rss(dy, dx, da, ds, radius_px)

                drift_map[fov_id][label] = {
                    'rss': rss,
                    'dy': dy,
                    'dx': dx,
                    'da': da,
                    'ds': ds
                }
            else:
                drift_map[fov_id][label] = None # Flag as missing
                
    return drift_map

def build_ne_stability_report(stats_m1, stats_m2, drift_map, pairs_map):
    """
    Compiles the report and determines Pass/Fail status.
    Logic:
    1. Precision = sqrt(sigma_m1^2 + sigma_m2^2)
    2. Threshold = 2.0 * Precision
    3. Filter: Fail if Drift > Threshold
    """
    report = {}
    
    for fov_id, label_dict in stats_m1.items():
        if fov_id not in report:
                report[fov_id] = {}

        for label, m1_metrics in label_dict.items():
            sigma_m1 = m1_metrics['sigma']
            
            if fov_id in stats_m2 and label in stats_m2[fov_id]:
                sigma_m2 = stats_m2[fov_id][label]['sigma']
                sigma_combined = np.sqrt(sigma_m1**2 + sigma_m2**2)
            else:
                # If M2 is missing, we estimate (though it will fail drift check anyway)
                sigma_combined = sigma_m1 * 1.414 
                
            threshold = 2.0 * sigma_combined

            drift_data = drift_map.get(fov_id, {}).get(label, None)

            # Set status
            status = "passed"
            reason = None
            drift_rss = -1.0
            drift_dx = 0.0
            drift_dy = 0.0
            
            if drift_data is None:
                status = "failed"
                reason = "missing_in_mode2"
                drift = -1.0
            else:
                drift_rss = drift_data['rss']
                drift_dx = drift_data['dx']
                drift_dy = drift_data['dy']
            
                if drift_rss > threshold:
                    status = "failed"
                    reason = f"Unstable_Drift ({drift_rss:.3f} > {threshold:.3f})"
            
            # Check channel pairing
            is_paired = False
            ch2_label = None
            if fov_id in pairs_map and label in pairs_map[fov_id]:
                is_paired = True
                ch2_label = pairs_map[fov_id][label]
            
            # Build report entry
            metrics_package = {
                'drift': drift_rss,
                'drift_dx': drift_dx,
                'drift_dy': drift_dy,
                'sigma_m1': sigma_m1,
                'sigma_combined': sigma_combined,
                'threshold': threshold
            }

            report[fov_id][label] = {
                'ch2_label': ch2_label,
                'is_paired': is_paired,
                'status': status,
                'fail_reason': reason,
                'metrics': metrics_package
            }

    return report

def calculate_fov_population_stats(reg_m1, reg_m2, radius_px):
    """
    Calculates the GLOBAL sigma based on the population of FoV drifts.
    
    Returns:
        global_sigma (float): The std dev of drifts across all FoVs.
        fov_drifts (dict): {fov_id: drift_magnitude}
    """
    fov_stats = {}
    drift_vectors_mag = []
    
    for fov_id, m1_data in reg_m1.items():
        if fov_id not in reg_m2:
            continue
        
        m2_data = reg_m2[fov_id]
        
        # --- Get FoV-level Vectors
        if 'shift_vector' not in m1_data or 'shift_vector' not in m2_data:
            continue
            
        v1 = np.array(m1_data['shift_vector'])
        v2 = np.array(m2_data['shift_vector'])
        
        # --- Calculate Drift Vector (Unified)
        
        dy = v2[0] - v1[0]
        dx = v2[1] - v1[1]
        da = m2_data.get('angle', 0) - m1_data.get('angle', 0)
        ds = m2_data.get('scale', 1) - m1_data.get('scale', 1)
        
        # Magnitude
        drift_mag = calculate_rss(dy, dx, da, ds, radius_px)
        
        # Store components for plotting (Gaussian distributions)
        fov_stats[fov_id] = {
            'rss': drift_mag,
            'dy': dy,
            'dx': dx,
            'da': da,
            'ds': ds
        }

        drift_vectors_mag.append(drift_mag)

    # --- Calculate Population Sigma
    # sigma_reg - std of the drifts
    if drift_vectors_mag:
        # Root Mean Square of the drifts serves as the spread metric
        # ??? OR simple std dev.
        global_sigma = np.sqrt(np.mean(np.array(drift_vectors_mag)**2))
    else:
        global_sigma = 0.5
        
    return global_sigma, fov_stats

def build_fov_stability_report(fov_drifts, global_sigma, reg_m1 = None):
    """
    Builds a report filtering entire FoVs.
    If reg_m1 is provided, each ne_label inherets status of the parent FoV (for direct comparison with per-ne method).
    """
    report = {}
    threshold = 2.0 * global_sigma
    
    for fov_id, data in fov_drifts.items():
        status = "passed"
        reason = None

        drift_mag = data['rss']

        if drift_mag > threshold:
            status = "failed"
            reason = f"FoV_Drift ({drift_mag:.3f} > {threshold:.3f})"
            
        report[fov_id] = {
            'drift': drift_mag,
            'drift_dx': data['dx'],
            'drift_dy': data['dy'],
            'threshold': threshold,
            'status': status,
            'reason': reason
        }

        # Expansion to Nuclei (for Comparison/Pruning consistency)
        if reg_m1 and fov_id in reg_m1:
            ne_keys = [k for k in reg_m1[fov_id].keys() 
                        if isinstance(reg_m1[fov_id][k], dict) and 'shift_vector' in reg_m1[fov_id][k]]
            
            report[fov_id]['nuclei_expansion'] = {}
            for label in ne_keys:
                report[fov_id]['nuclei_expansion'][label] = {
                    'status': status, # Inherits FoV status
                    'reason': reason,
                    'drift': drift_mag,
                    'drift_dx': data['dx'],
                    'drift_dy': data['dy']
                }

    return report

# --- COMPARISON REPORT ---
def generate_stability_comparison_report(fov_report, ne_report):
    """
    Generates the 'Head-to-Head' comparison of the two stability methods.
    Returns a dictionary with per-FoV and experiment-wide stats.
    """        
    stats = {
        'per_fov': {},
        'experiment': {
            'total_detected': 0,
            'total_pruned_fov_method': 0,
            'total_pruned_ne_method': 0,
            'net_loss_diff': 0, # Positive = FoV is harsher
            'method_agreement': 0 # Nuclei where both said Pass or both said Fail
        }
    }
    
    # Get set of all FoVs
    all_fovs = set(fov_report.keys()) | set(ne_report.keys())
    
    for fov in all_fovs:
        # 1. Get Data for this FoV
        fov_entry_fov_method = fov_report.get(fov, {})
        fov_nuclei_fov_method = fov_entry_fov_method.get('nuclei_expansion', {})
        
        fov_nuclei_ne_method = ne_report.get(fov, {})
        
        # 2. Count Detected (Union of labels)
        labels = set(fov_nuclei_fov_method.keys()) | set(fov_nuclei_ne_method.keys())
        count_detected = len(labels)
        
        # 3. Count Pruned
        # FoV Method Pruning (Did the whole FoV fail?)
        count_pruned_fov = 0
        if fov_entry_fov_method.get('status') == 'failed':
            count_pruned_fov = count_detected # All detected in this FoV are dead
        
        # NE Method Pruning (Count individual fails)
        count_pruned_ne = sum(1 for v in fov_nuclei_ne_method.values() if v.get('status') == 'failed')
        
        stats['per_fov'][fov] = {
            'detected': count_detected,
            'pruned_fov_method': count_pruned_fov,
            'pruned_ne_method': count_pruned_ne,
            'diff': count_pruned_fov - count_pruned_ne
        }
        
        stats['experiment']['total_detected'] += count_detected
        stats['experiment']['total_pruned_fov_method'] += count_pruned_fov
        stats['experiment']['total_pruned_ne_method'] += count_pruned_ne
        
    # Final Stats
    loss_fov = stats['experiment']['total_pruned_fov_method']
    loss_ne = stats['experiment']['total_pruned_ne_method']
    stats['experiment']['net_loss_diff'] = loss_fov - loss_ne
    
    return stats

# !!! LOOK AT ALL THESE FUNCTIONS....

def visualize_channel_alignment(ch1_stack, ch2_stack, shift_vector, n_frames, title_prefix=""):
    """
    Generates plots to visualize the channel alignment every N frames.

    Args:
        ch1_stack: The image stack for the reference channel.
        ch2_stack: The image stack for the channel to be aligned.
        shift_vector: The array of calculated (y, x) shifts.
        n_frames: The interval at which to generate a plot (e.g., plot every 50th frame).
        title_prefix: A string to prepend to the plot titles.
    """
    num_total_frames = ch1_stack.shape[0]
    
    for i in range(0, num_total_frames, n_frames):
        ch1_frame = ch1_stack[i]
        ch2_frame = ch2_stack[i]
        
        # Get the transform for this specific frame
        tvec = shift_vector[i]
        
        # Apply the inverse translation to align channel 2 to channel 1
        ch2_aligned = ird.transform_img(ch2_frame, tvec=-tvec)
        
        # --- Create the plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot Channel 1
        axes[0].imshow(ch1_frame, cmap='gray')
        axes[0].set_title(f'Channel 1 (Frame {i})')
        axes[0].axis('off')
        
        # Plot Channel 2 (Original)
        axes[1].imshow(ch2_frame, cmap='gray')
        axes[1].set_title(f'Channel 2 (Frame {i}, Original)')
        axes[1].axis('off')
        
        # Plot Overlay of Aligned Images
        # Normalize for visualization
        ch1_norm = (ch1_frame - ch1_frame.min()) / (ch1_frame.max() - ch1_frame.min())
        ch2_aligned_norm = (ch2_aligned - ch2_aligned.min()) / (ch2_aligned.max() - ch2_aligned.min())
        
        # Create an RGB image for the overlay
        # Channel 1 in green, Aligned Channel 2 in magenta
        overlay = np.stack([ch2_aligned_norm, ch1_norm, ch2_aligned_norm], axis=-1)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Aligned Overlay (Frame {i})')
        axes[2].axis('off')
        
        fig.suptitle(f"{title_prefix} - Frame {i} Alignment", fontsize=16)
        plt.tight_layout()
        plt.show()

def compute_drift(path_dict, drift_bins, estimate_precision = False):
    full_img = tifffile.imread(path_dict['FoV_collection_path'] + path_dict['imgs']['fn_track_ch2'])

    size = np.shape(full_img)[0]
    bins = np.linspace(0, size, drift_bins + 1)

    ref_img = np.mean(full_img[int(bins[0]):int(bins[1])], axis = 0)
    translation_array = np.zeros((drift_bins, 2))
    translation_array1 = np.zeros((drift_bins, 2))
    translation_array2 = np.zeros((drift_bins, 2))

    for i in range(drift_bins):
        temp_img = np.mean(full_img[int(bins[i]):int(bins[i + 1])], axis = 0)
        translation, _, _ = skimage.registration.phase_cross_correlation(ref_img,
                                                                         temp_img,
                                                                         upsample_factor = 1000)

        translation_array[i, :] = translation + translation_array[i - 1, :]
        ref_img = temp_img.copy()

    bincenters = (bins[0:-1]+ bins[1::])/2
    smooth_x = scipy.interpolate.CubicSpline(bincenters,
                                             translation_array[:,1],
                                             extrapolate = True,
                                             bc_type = 'natural')
    smooth_y = scipy.interpolate.CubicSpline(bincenters,
                                             translation_array[:,0],
                                             extrapolate = True,
                                             bc_type = 'natural')
    frames = np.arange(0, size)
    drift = {"x": smooth_x(frames),
             "y": smooth_y(frames)
            }

    if estimate_precision:
        translation_array1, translation_array2 = precision_estimation(full_img,
                                                                      drift_bins,
                                                                      translation_array1,
                                                                      translation_array2)
        # ???: Trying to determine why these were calculated and returned in the original (YeastProcessor) version

    return drift, translation_array

def precision_estimation(f_img, d_bins, t_array_1, t_array_2):
    set_1 = f_img[::2,:,:] # even
    set_2 = f_img[1::2,:,:] # odd

    size_prec = np.shape(set_2)[0]
    bins_prec = np.linspace(0, size_prec, d_bins + 1)

    ref_img_1 = np.mean(set_1[int(bins_prec[0]):int(bins_prec[1])], axis = 0)
    ref_img_2 = np.mean(set_2[int(bins_prec[0]):int(bins_prec[1])], axis = 0)

    for i in range(d_bins):
        temp_img_1  = np.mean(set_1[int(bins_prec[i]):int(bins_prec[i + 1])], axis=0)
        temp_img_2 = np.mean(set_2[int(bins_prec[i]):int(bins_prec[i + 1])], axis=0)

        translation1, _, _ = skimage.registration.phase_cross_correlation(ref_img_1,
                                                                            temp_img_1,
                                                                            upsample_factor = 1000)
        translation2, _, _ = skimage.registration.phase_cross_correlation(ref_img_2,
                                                                            temp_img_2,
                                                                            upsample_factor = 1000)
        t_array_1[i, :] = translation1 + t_array_1[i - 1, :]
        t_array_2[i, :] = translation2 + t_array_2[i - 1, :]

        ref_img_1 = temp_img_1.copy()
        ref_img_2 = temp_img_2.copy()

    return t_array_1, t_array_2


## PLOTTING FUNCTIONS


# # NOTE JT (NICE) TO DO
# #       - make the output options into a seperate file
# #       - return plots rather than save to file (maybe... probably...)
# def registration_figs(output_path, mean_red, mean_green):
#     alpha = 1.0  # Increase alpha to make the image more opaque

#     # Plot and save the first image
#     fig, ax = plt.subplots()

#     # Display the 'mean_green' image with 'cmap_green' colormap
#     ax.imshow(mean_green, cmap='gray')
#     ax.axis('off')
#     plt.savefig(output_path+ 'green_original.svg',
#                 format='svg', )

#     plt.close()
#     # Display the 'mean_red_t' image with 'cmap_red' colormap and 0.5 alpha
#     # Plot and save the first image
#     fig, ax = plt.subplots()

#     # Display the 'mean_green' image with 'cmap_green' colormap
#     ax.axis('off')
#     ax.imshow(mean_red, cmap='gray')
#     plt.savefig(output_path+ 'red_original.svg',
#                 format='svg', )

#     plt.close()

#     # Plot and save the second image
#     fig, ax = plt.subplots()
#     alpha = 0.5  # Adjust this value as needed

#     # Create a new image by blending the two images with the same transparency
#     overlay_image = alpha * mean_green + (1 - alpha) * mean_red

#     # Plot the overlay image
#     ax.imshow(overlay_image, cmap='gray', alpha=1)

#     # Remove x and y axes
#     ax.axis('off')

#     plt.savefig(output_path+ 'registration_original.svg',
#                 format='svg',)
#     plt.close()
#     plt.close('all')
#     # Plot and save the second image
#     fig, ax = plt.subplots()
#     alpha = 0.5  # Adjust this value as needed

#     # Create a new image by blending the two images with the same transparency
#     overlay_image = alpha * mean_green + (1 - alpha) * mean_red_t

#     # Plot the overlay image
#     ax.imshow(overlay_image, cmap='gray', alpha=1)

#     # Remove x and y axes
#     ax.axis('off')

#     plt.savefig(output_path+ 'registration_finished.svg',
#                 format='svg',)

#     plt.close('all')

# def plot_drift_correction(t_array, drift, pixel_size, output_path):
#     plt.figure(figsize = [2, 2])
#     plt.scatter(bincenters, t_array[:,1] * pixel_size, marker = 'x')
#     plt.scatter(bincenters, t_array[:, 0] * pixel_size, marker = 'x')
#     plt.plot(frames, drift['x'] * pixel_size, label = r'$D_x$')
#     plt.plot(frames, drift['y'] * pixel_size, label = r'$D_y$')
#     plt.legend()
#     plt.ylabel('Drift $D$ [nm]')
#     plt.xlabel('Frame')
#     plt.savefig(output_path + 'drift.svg',
#                 format = 'svg')
#     plt.close('all')
