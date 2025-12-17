"""
Created on Sat May 17 18:49:35 2025

@author: jctourtellotte
"""

# outside packages
import json
import pandas as pd
import os
import tifffile
import skimage
import numpy as np
import imreg_dft as ird
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import zoom

from tools.utility_functions import parse_filename_split, crop_image


# Function operates on a given FoV, using path info passed as its path_dict
#    and corresponding crop_dim_dict
def image_registration(path_dict, ch1_crop_dim_dict, ch2_crop_dim_dict, frame_range, frames_per_average, padding, upsample_factor, upscale_factor, reg_mode = 1, ne_label_pair_dict = None, detailed_output = False):
    reg_all_ne_label = {}
    # # 2. Check registration difference
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

    # FoV_reg_stats = compute_reg_stats(FoV_reg_data['scale'], FoV_reg_data['angle'], FoV_reg_data['shift_vector'])
    # FoV_registration.update({'reg_stats':FoV_reg_stats})

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

        # ch1_bf_crop = channel1_bf[
        #     frame_range[0]:frame_range[1],
        #     crop_bounds_ch1['final_top']:crop_bounds_ch1['final_bottom'],
        #     crop_bounds_ch1['final_left']:crop_bounds_ch1['final_right']
        #     ]
        # ch2_bf_crop = channel2_bf[
        #     frame_range[0]:frame_range[1],
        #     crop_bounds_ch2['final_top']:crop_bounds_ch2['final_bottom'],
        #     crop_bounds_ch2['final_left']:crop_bounds_ch2['final_right']
        #     ]

        ch1_bf_crop = crop_image(channel1_bf, crop_bounds_ch1,'crop')
        ch2_bf_crop = crop_image(channel2_bf, crop_bounds_ch2,'crop')
        
        ch1_bf_crop_mean = np.mean(ch1_bf_crop, axis = 0)
        ch2_bf_crop_mean = np.mean(ch2_bf_crop, axis = 0)

        # dictionary for registration of the mean image re: current ne label
        current_reg_calc = compute_registration(ch1_bf_crop_mean, ch2_bf_crop_mean, padding, upsample_factor, upscale_factor)

        # current_ne_reg_stats = compute_reg_stats( current_ne_reg_data['scale'], current_ne_reg_data['angle'], current_ne_reg_data['shift_vector'])
        # ne_crop_registration.update({'reg_stats': current_ne_reg_stats})

        # create list to store registration data based on the average of subsets of size 'frames_per_average'
        current_reg_calc.update({'ne_subset_reg': {}})
        slice_n = 1
        for i in range(frame_range[0], frame_range[1], frames_per_average):
            slice_start = i
            slice_end = i+frames_per_average
            ch1_temp_mean = np.mean(ch1_bf_crop[slice_start:slice_end], axis = 0)
            ch2_temp_mean = np.mean(ch2_bf_crop[slice_start:slice_end], axis = 0)

            temp_ne_reg_data = compute_registration(ch1_temp_mean, ch2_temp_mean, padding, upsample_factor, upscale_factor)
            # temp_subset_avg = {'start': slice_start, 'stop': slice_end}
            # temp_subset_avg.update({'subset_reg_data': temp_ne_reg_data})

            # temp_ne_reg_stats = compute_reg_stats( temp_ne_reg_data['scale'], temp_ne_reg_data['angle'], temp_ne_reg_data['shift_vector'])
            # temp_subset_avg.update({'reg_stats': temp_ne_reg_stats})
            current_reg_calc['ne_subset_reg'].update({f'slice_{slice_n:02}': temp_ne_reg_data})
            slice_n += 1

        reg_all_ne_label.update({f'{ne_label}': current_reg_calc})

    return FoV_reg_data, reg_all_ne_label

def compute_registration(channel1_img, channel2_img, padding = 5, upsample_factor = 1000, upscale_factor = 1):
    # expects images of dimensions (rows, columns)
    # channel 1 is used as the reference channel

    # !!!: Add: Raise an error
    if channel1_img.shape != channel2_img.shape:
        print(f"Error: Image stacks must have the same shape.")
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
        "angle": angle,
        "shift_vector": translation
    }
    
    return results
    
def compute_reg_stats(scale_array, angle_array, y_shift_array, x_shift_array):
    # Calculate the mean and standard deviation of scale and angle
    mean_scale = np.mean(scale_array)
    std_dev_scale = np.std(scale_array)
    mean_angle = np.mean(angle_array)
    std_dev_angle = np.std(angle_array)
    
    # Calculate the mean shift between channels across all frames
    mean_shift_y = np.mean(y_shift_array)
    mean_shift_x = np.mean(x_shift_array)
   
    # Calculate the standard deviation of the shifts
    std_dev_y = np.std(y_shifts)
    std_dev_x = np.std(x_shifts)

    # Calculate the precision of the channel registration
    # This represents the radial consistency of the alignment over time.
    ch_reg_prec = np.sqrt(std_dev_y**2 + std_dev_x**2)

    summary_stats = {
        "mean_scale": mean_scale,
        "std_dev_scale": std_dev_scale,
        "mean_angle": mean_angle,
        "std_dev_angle": std_dev_angle,
        "mean_shift_y": mean_shift_y,
        "std_dev_y": std_dev_y,
        "mean_shift_x": mean_shift_x,
        "std_dev_x": std_dev_x,
        "ch_reg_prec": ch_reg_prec
    }

    return summary_stats   

def check_reg_prec(reg_dict_list):
    FoV_id_list = np.array([FoV['FoV_id'] for FoV in reg_dict_list], dtype='str')
    rdiffmag_scalers = np.array([FoV['rdiff_magnitude'] for FoV in reg_dict_list], dtype='float32')
    difference_vectors = np.array([FoV['diff_vec'] for FoV in reg_dict_list], dtype='float32')
    
    # take std on 0 axis; index wise between pairs rather than within pairs
    std_diff_vector = np.std(np.array(difference_vectors), axis = 0)
    
    reg_prec = np.sqrt(np.sum(np.square(std_diff_vector)))
    print(f"{2*reg_prec} is 2sigma_reg.")
    is_outlier = rdiffmag_scalers > 2 * reg_prec
    FoV_to_retain = FoV_id_list[~is_outlier]
    
    return std_diff_vector, reg_prec, FoV_to_retain, is_outlier    

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

    if(estimate_precision == True):
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


## Registration JSON Handling
#      designed for importing registration json and 
#       better integration with R (via pandas)
# For testing:
# '/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_imaging_pipeline/yeast_output/tracking_experiments/registration/reg_result_BMY823_BMY823_7_25_23_aqsettings1_batchC.json'

# returns 3 dataframes: df_fov, df_ne_label, df_ne_subset
def FoV_reg_json_to_df(json_path):
    # Load the JSON data from the file
    json_path="/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/yeast_output/dual_label_experiments/registration/reg_results_mode1_BMY1408.json"
    with open(json_path, 'r') as f:
        reg_data = json.load(f)
    
    if reg_data:
        try:
            exper_dict = parse_filename_split(json_path, 'reg_result_')
            df_fov, df_ne_label, df_ne_subset = FoV_reg_dict_to_df(reg_data, exper_dict)
            return df_fov, df_ne_label, df_ne_subset
        except ValueError as e:
            print(f"Warning: unable to convert loaded json to FoV registration dataframes. {e}. Skipping.")
            return
    else:
        raise Exception("Unable to load JSON located at {json_path}")
        return

# !!! Rewrite so works with dictionary of dictionaries, was originally written for a list of dictionaries
# def FoV_reg_dict_to_df(FoV_reg_dict, experiment_dict = {'strain': '', 'date': '', 'aqsettings': '', 'batch': ''}):
#     # Create empty lists to hold the data for each DataFrame
#     fov_rows = []
#     ne_label_rows = []
#     ne_subset_rows = []

#     # --- Loop through the entire data structure once ---
#     for key, fov_data in FoV_reg_dict.items():
#         fov_id = key

#         # 1. Extract data for the FoV DataFrame
#         fov_reg = fov_data['FoV_reg_data']
#         fov_rows.append({
#             'FoV_id': fov_id,
#             'scale': fov_reg['scale'],
#             'angle': fov_reg['angle'],
#             'y_shift': fov_reg['shift_vector'][0],
#             'x_shift': fov_reg['shift_vector'][1],
#             'strain': experiment_dict['strain'],
#             'date': experiment_dict['date'],
#             'aqsettings': experiment_dict['aqsettings'],
#             'batch': experiment_dict['batch']
#         })

#         # 2. Loop through the 'ne_label_registration' list
#         for key, ne_reg in fov_data['ne_label_registration'].items():
#             ne_label = key

#             # Add a row for the ne_label DataFrame
#             ne_label_rows.append({
#                 'FoV_id': fov_id,
#                 'ne_label': ne_label,
#                 'scale': ne_reg['scale'],
#                 'angle': ne_reg['angle'],
#                 'y_shift': ne_reg['shift_vector'][0],
#                 'x_shift': ne_reg['shift_vector'][1],
#                 'strain': experiment_dict['strain'],
#                 'date': experiment_dict['date'],
#                 'aqsettings': experiment_dict['aqsettings'],
#                 'batch': experiment_dict['batch']
#             })

#             # 3. Loop through the 'ne_subset_reg' list
#             # Use enumerate to get the subset number (starting from 1)
#             for i, subset_reg in enumerate(ne_reg.get('ne_subset_reg', [])):
#                 ne_subset_rows.append({
#                     'FoV_id': fov_id,
#                     'ne_label': ne_label,
#                     'ne_subset_number': i + 1,
#                     'scale': subset_reg['scale'],
#                     'angle': subset_reg['angle'],
#                     'y_shift': subset_reg['shift_vector'][0],
#                     'x_shift': subset_reg['shift_vector'][1],
#                     'strain': experiment_dict['strain'],
#                     'date': experiment_dict['date'],
#                     'aqsettings': experiment_dict['aqsettings'],
#                     'batch': experiment_dict['batch']
#                 })
#         # --- Create the pandas DataFrames from the lists of dictionaries ---
#         df_fov = pd.DataFrame(fov_rows)
#         df_ne_label = pd.DataFrame(ne_label_rows)
#         df_ne_subset = pd.DataFrame(ne_subset_rows)

#     return df_fov, df_ne_label, df_ne_subset


## PLOTTING FUNCTIONS


# # JT (NICE) TO DO
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
