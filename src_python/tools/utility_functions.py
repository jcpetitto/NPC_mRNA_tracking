
import os
import re
import torch
import tifffile
import math
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union, List

# --- DEVICE HANDLING --- #

# empty cache for the given device
def clear_device_cache(device):
    # can pass device as a string OR the object itself
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()  # Empty CUDA cache
    elif device.type == 'mps':
        torch.mps.empty_cache() # Empty MPS cache
    # No specific empty cache function for CPU (generally handled by system garbage collection)
    # Other device types can be addressed here if needed.

###                   ###
# --- FILE HANDLING --- #
###                   ###

# # !!!: Move to config after testing
# counter_dict = {'total_cells': 0,
#                 'total_detections': 0,
#                 'detections_per_cell': []}


def apply_to_dir_of_FoVs(directory_path: str, FoV_fn: callable, FoV_fn_params:dict = None, FoV_prefix: str = "FoV_"):
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    FoV_return_dict = {}
    # Iterate through items in the top-level directory
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        try:
            # Only process sub-directories
            if os.path.isdir(item_path) and item_name.startswith(FoV_prefix):
                print(f"Applying FoV_fn to sub-folder: '{item_name}'")
                full_FoV_fn_params = {'FoV_id': extract_digits(item_name)} \
                    if FoV_fn_params == None else FoV_fn_params # default to pass a dictionary containing the FoV_id, assuming it is the last set of digits in the sub-folder name
            
                FoV_return_dict.update(FoV_fn(full_FoV_fn_params))
        except Exception as e:
            print(f'--- apply_to_dir_of_FoVs : An error occurred: {e} ---')
    try:
        if not len(FoV_return_dict) == 0:
            return FoV_return_dict
    except Exception as e_big: 
        print(f'--- Error returning data, no data to return: {e_big}')


###                         ###
# --- DICTIONARY HANDLING --- #
###                         ###

def dict_update_or_replace(original_dict, new_items, add_to_existing = True):
    if (add_to_existing & len(original_dict) != 0):
        original_dict.update(new_items) # add entries
    else:
        original_dict = new_items # overwrite entries
    return original_dict    


def find_segments(y, trigger_val, stopind_inclusive = True):
    """
    # finds contiguous segments (islands) of True values in y.
    # returns a list of [start_index, end_index] pairs for these segments
    """
    # append "False" to either end of the array of segments to establish a "transition"
    # so that any segment at the start or end of the array is appropriately accounted for
    # re: inclusion or exclusion of the endpoint as part of the left- or right-most island
    y_ext = np.r_[False, y == trigger_val, False]

    # Compares all elements of the array to the one before it
    # store indices of changes (False to True, True to False), which represent
    # the start and stop indices of each segment island
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # idx[:-1:2] every island start position (ie. odd indices)
    # idx[1::2] every island stop position (ie. even indices)
    # idx[1::2] - int(stopind_inclusive) boolean determines if stop index the change or not
    #               ie. [False, True, True, False] ... inclusion would have i = 4 for the stop
    return list(zip(idx[:-1:2], idx[1::2] - int(stopind_inclusive)))

# returns first match, best for unique entries

def pull_entry_by_value(value_to_find, dict_to_search, key_to_search = 'FoV_id'):
    return  next((entry for entry in dict_to_search \
                  if entry.get(key_to_search) == value_to_find), None)

def filter_data_by_indices(data_dict, indices_dict):
    """
    Filters a dictionary of arrays based on a dictionary of signal indices.
    
    Args:
        data_dict: A dict like {'07': np.array_of_data, ...}
        indices_dict: A dict like {'07': np.array_of_indices, ...}
    """
    filtered_profiles = {}
    
    for ne_label, data_array in data_dict.items():
        if ne_label in indices_dict:
            
            valid_indices = indices_dict[ne_label]
            
            # Check if data_array is actually an array
            if not hasattr(data_array, '__getitem__'):
                print(f"Cannot filter {ne_label}, data is not indexable.")
                continue

            # This is the core logic: filter the array with the indices
            filtered_profiles[ne_label] = data_array[valid_indices]
            
        else:
            print(f"Warning: No signal indices found for NE label {ne_label}. Skipping.")
            
    return filtered_profiles

def find_shared_keys(dictA, dictB, message = 'Unmatched keys: '):
    # finding the symmetric difference (^)
    unmatched_keys = set(dictA.keys()) ^ set(dictB.keys())

    if unmatched_keys:
        print(f"{message}{unmatched_keys}")
        matched_keys = set(dictA.keys()) & set(dictB.keys())
    else:
        matched_keys = dictA.keys()
    
    return matched_keys



###                   ###
# --- TEXT HANDLING --- #
###                   ###

def extract_digits(text_str):
    digits_match = re.search(r'\d+', text_str) 
    return digits_match.group() if digits_match else None


def unpack_nested_lists(nested_list):
    nested_length = len(nested_list)
    new_list = []
    for i in range(nested_length):
        member = nested_list[i]
        while (len(member) == 1):
            member = member[0]
            print('in loop')
        new_list.append(member)
    return(new_list)

# TODO: set-up number formatting re: leading zeros with config and/or by looking at the # of FoV
    # str(number).zfill(total_width)

def filter_channels_by_label_map(label_map, ch1_data, ch2_data):
    """
    Filters data from two simple channel dictionaries based on a label map.

    This version assumes the input dictionaries do not have a top-level
    'experiment_id' and start directly with the 'parent' keys. The output
    is a single dictionary containing the filtered data for both channels,
    which can be easily accessed via the 'ch1' and 'ch2' keys.

    Args:
        label_map (dict): A dictionary where keys are 'parent' identifiers and
                          values are dictionaries mapping a 'ch1' label to a
                          corresponding 'ch2' label.
                          Example: {'0190': {'05': '03'}}
                          
        ch1_data (dict): A dictionary for channel 1 with the structure
                         {parent_key: {label: data}}.

        ch2_data (dict): A dictionary for channel 2 with the same structure
                         as ch1_data.

    Returns:
        dict: A new dictionary with a unified structure {'ch1': {...}, 'ch2': {...}},
              containing only the data for the labels specified in `label_map`.
    """
    # Initialize the output dictionary with keys for each channel.
    filtered_output = {
        'ch1': {},
        'ch2': {}
    }

    # Iterate over each 'parent' key and its label pairs in the map.
    for parent_key, label_pairs in label_map.items():
        # Check if this parent key exists in both channel data dictionaries.
        if parent_key in ch1_data and parent_key in ch2_data:
            
            # Now, iterate over the specific ch1 and ch2 label pairs to find.
            for ch1_label, ch2_label in label_pairs.items():
                
                # Check for and process the channel 1 label.
                # Safely access the dictionary for the parent key's labels.
                ch1_parent_data = ch1_data.get(parent_key, {})
                if ch1_label in ch1_parent_data:
                    # Initialize the parent key in the output if it's not there yet.
                    if parent_key not in filtered_output['ch1']:
                        filtered_output['ch1'][parent_key] = {}
                    
                    # Copy the data.
                    filtered_output['ch1'][parent_key][ch1_label] = ch1_parent_data[ch1_label]

                # Check for and process the channel 2 label.
                # Safely access the dictionary for the parent key's labels.
                ch2_parent_data = ch2_data.get(parent_key, {})
                if ch2_label in ch2_parent_data:
                    # Initialize the parent key in the output if it's not there yet.
                    if parent_key not in filtered_output['ch2']:
                        filtered_output['ch2'][parent_key] = {}

                    # Copy the data.
                    filtered_output['ch2'][parent_key][ch2_label] = ch2_parent_data[ch2_label]

    return filtered_output

###                 ###
# --- DO THE MATH --- #
###                 ###


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def intensity_func_sig(t, A, K, B, C, nu, Q, M, mu, sigma, amplitude, offset):

    exponent_richards = -B * (t - M)
    denominator_richards = C + Q * np.exp(exponent_richards)
    power_richards = 1 / nu
    richards_curve = A + (K - A) / denominator_richards ** power_richards

    exponent_gaussian = -(t - mu) ** 2 / (2 * sigma ** 2)
    gaussian_curve = amplitude * np.exp(exponent_gaussian)

    combined_curve = richards_curve + gaussian_curve+ offset

    return combined_curve

# OLD VERSION
    # def calc_tangent_endpts(points, derivs, line_length, include_normal=True):

    #     h_line_length = line_length/2
    #     # add a very small number to avoid dividing by zero when normalizing
    #     tangent_magnitudes = np.sqrt(derivs[0]**2 + derivs[1]**2) + 1e-8
        
    #     tangent_unit_vectors = derivs / tangent_magnitudes
    #     tangent_endpoints_start = points + h_line_length * tangent_unit_vectors
    #     tangent_endpoints_end = points - h_line_length * tangent_unit_vectors
    #     tangent_ends = np.array([tangent_endpoints_start, tangent_endpoints_end])
        
    #     # transform (rotate 90 degrees) by broadcasting normalized tangent unit vectors 
    #     normal_unit_vectors = np.array([[-1], [1]]) * tangent_unit_vectors
    #     normal_endpoints_start = points + h_line_length * normal_unit_vectors
    #     normal_endpoints_end = points - h_line_length * normal_unit_vectors
    #     normal_ends = np.array([normal_endpoints_start, normal_endpoints_end])
        
    #     return tangent_ends, normal_ends


def calc_tangent_endpts(points, derivs, line_length, include_normal=True):
    """
    Calculates the endpoints of lines tangent and normal to a curve.
    Assumes points and derivs are shape (2, N), where N is the number of points.
    """
    h_line_length = line_length / 2
    
    # Calculate magnitudes for each vector (column-wise)
    tangent_magnitudes = np.sqrt(derivs[0]**2 + derivs[1]**2) + 1e-9
    
    # Normalize to get unit vectors
    tangent_unit_vectors = derivs / tangent_magnitudes

    # Calculate tangent endpoints
    tangent_endpoints_start = points + h_line_length * tangent_unit_vectors
    tangent_endpoints_end = points - h_line_length * tangent_unit_vectors
    tangent_ends = np.array([tangent_endpoints_start, tangent_endpoints_end])
    
    normal_ends = None
    if include_normal:
        # --- THIS IS THE FIX ---
        # Correctly rotate the tangent vectors [dx, dy] to get the normal vectors [-dy, dx]
        normal_unit_vectors = np.vstack((-tangent_unit_vectors[1], tangent_unit_vectors[0]))
        # ---------------------

        normal_endpoints_start = points + h_line_length * normal_unit_vectors
        normal_endpoints_end = points - h_line_length * normal_unit_vectors
        normal_ends = np.array([normal_endpoints_start, normal_endpoints_end])

    return tangent_ends, normal_ends

def sample_bspline(bspline_to_sample, n_samples):
    u_values_to_sample = np.linspace(0, 1, n_samples, endpoint=False)
    evaluated_points = bspline_to_sample(u_values_to_sample)
    evaluated_derivatives = bspline_to_sample.derivative(1)(u_values_to_sample)

    return evaluated_points, evaluated_derivatives





###                    ###
# --- IMAGE HANDLING --- #
###                    ###


def get_centered_crop_indices(original_size: int, target_size: int) -> tuple[int, int]:
    """
    Calculates the start and end indices for a centered crop.

    Args:
        original_size: The total size of the dimension (e.g., 16).
        target_size: The desired size of the cropped dimension (e.g., 10).

    Returns:
        A tuple containing the start_index and end_index for slicing.
        Returns (-1, -1) if target_size is larger than original_size.

    Raises:
        ValueError: If target_size is less than or equal to 0.
    """
    if target_size <= 0:
        raise ValueError("Target size must be positive.")
    if target_size > original_size:
        print(f"Warning: Target size ({target_size}) is larger than original size ({original_size}). Cannot crop.")
        return -1, -1 # Indicate invalid crop

    padding_total = original_size - target_size
    padding_start = padding_total // 2  # Integer division handles odd padding

    start_index = padding_start
    end_index = start_index + target_size

    return start_index, end_index
def coord_of_crop_box(binary_img: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculates coordinates for a centered crop box based on True values in a binary image.

    Args:
        binary_img: The 2D boolean numpy array.
        width: The desired width of the crop box.
        height: The desired height of the crop box.

    Returns:
        A tuple (top, bottom, left, right) of integer coordinates,
        or None if no True values are found in binary_img.
    """
    rows, cols = np.where(binary_img)

    if rows.size == 0: # More idiomatic check for empty array
        return None

    # 1. Calculate the center of the bounding box of True values
    y_center = (rows.min() + rows.max()) / 2
    x_center = (cols.min() + cols.max()) / 2

    # 2. Calculate ideal box coordinates (can be float)
    half_height = height / 2
    half_width = width / 2
    top_f = y_center - half_height
    bottom_f = y_center + half_height
    left_f = x_center - half_width
    right_f = x_center + half_width

    # 3. Round UP to integers to ensure the box contains the center region
    #    Use int() for explicit type conversion after ceiling.
    #    Clip coordinates to image bounds using np.clip for conciseness.
    img_height, img_width = binary_img.shape
    top = int(np.clip(math.ceil(top_f), 0, img_height))
    bottom = int(np.clip(math.ceil(bottom_f), 0, img_height))
    left = int(np.clip(math.ceil(left_f), 0, img_width))
    right = int(np.clip(math.ceil(right_f), 0, img_width))

    # Ensure bottom >= top and right >= left (handles cases near edges)
    bottom = max(top, bottom)
    right = max(left, right)

    return top, bottom, left, right



#-----------------------------------------------------------------------------

def extract_cropped_images(
    full_img_path: str,
    frame_range: List[int],
    crop_boxes: Dict[str, Dict], # Expecting dict like {'label': {'final_top':.., 'final_bottom':.., ...}}
    mean_img: bool = True,
    pad_to_size: Optional[Union[str, Tuple[int, int]]] = None
) -> Dict[str, np.ndarray]:
    """
    Reads an image stack, optionally averages frames, and extracts multiple cropped regions.

    Args:
        full_img_path: Path to the TIF image stack.
        frame_range: List/tuple [start_frame, end_frame(exclusive)].
        crop_boxes: Dictionary where keys are labels and values are dicts
                    containing crop coordinates ('final_top', 'final_bottom',
                    'final_left', 'final_right', and optionally 'width', 'height').
        mean_img: If True, average the frames before cropping. If False, crop the stack.
        pad_to_size: If None, no padding. If 'crop', pad to original box dimensions
                     ('width', 'height' must be in crop_boxes entry). If tuple (h, w),
                     pad to specified dimensions.

    Returns:
        A dictionary where keys are labels and values are the cropped (and padded) images.
    """
    raw_img_stack = tifffile.imread(full_img_path)[frame_range[0]:frame_range[1]]

    img_to_crop = np.mean(raw_img_stack, axis=0, dtype=raw_img_stack.dtype) if mean_img else raw_img_stack

    # Use dictionary comprehension for cleaner creation
    cropped_images = {
        key: crop_image(img_to_crop, entry, pad_to_size)
        for key, entry in crop_boxes.items()
    }

    return cropped_images

#-----------------------------------------------------------------------------

def crop_image(
    img: np.ndarray,
    crop_box: Dict,
    pad_to_size: Optional[Union[str, Tuple[int, int]]] = None
) -> np.ndarray:
    """
    Crops a 2D image or 3D stack and optionally pads it to a target size.

    Args:
        img: The image (2D: HxW) or stack (3D: FxHxW) to crop.
        crop_box: Dictionary with 'final_top', 'final_bottom', 'final_left', 'final_right'.
                  Must include 'width' and 'height' if pad_to_size='crop'.
        pad_to_size: Controls padding (see extract_cropped_images).

    Returns:
        The cropped and potentially padded image/stack.
    """
    top = crop_box['final_top']
    bottom = crop_box['final_bottom']
    left = crop_box['final_left']
    right = crop_box['final_right']

    # Crop using slicing based on dimensions
    if img.ndim == 2: # HxW
        cropped_img = img[top:bottom, left:right]
        h_idx, w_idx = 0, 1
    elif img.ndim == 3: # FxHxW
        cropped_img = img[:, top:bottom, left:right]
        h_idx, w_idx = 1, 2
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    # Determine target height and width for padding
    if pad_to_size == 'crop':
        if 'height' not in crop_box or 'width' not in crop_box:
            raise ValueError("crop_box must contain 'height' and 'width' when pad_to_size='crop'")
        target_height = crop_box['height']
        target_width = crop_box['width']
    elif isinstance(pad_to_size, tuple) and len(pad_to_size) == 2:
        target_height, target_width = pad_to_size
    else: # No padding needed or invalid pad_to_size
        return cropped_img

    # Calculate padding amounts (ensure non-negative)
    current_height = cropped_img.shape[h_idx]
    current_width = cropped_img.shape[w_idx]

    pad_h_total = max(0, target_height - current_height)
    pad_w_total = max(0, target_width - current_width)

    pad_h_top = pad_h_total // 2
    pad_h_bottom = pad_h_total - pad_h_top
    pad_w_left = pad_w_total // 2
    pad_w_right = pad_w_total - pad_w_left

    # Construct padding tuple for np.pad
    if img.ndim == 2:
        pad_width = ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
    else: # ndim == 3
        pad_width = ((0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))

    # Apply padding if necessary
    if pad_h_total > 0 or pad_w_total > 0:
        padded_img = np.pad(cropped_img, pad_width, mode='constant', constant_values=0)
        return padded_img
    else:
        # If target size is smaller or equal, return the (potentially smaller) crop
        return cropped_img # Or raise an error/warning if target size is smaller?

###                        ###
# --- SEARCH ALGORITHMS  --- #
###                        ###

def nearest_neighbor(coordinates, distances):
    num_coordinates = len(coordinates)
    visited = [False] * num_coordinates
    order = [0]  # Starting from the first coordinate
    visited[0] = True  # Mark the starting point as visited

    for _ in range(num_coordinates - 1):
        current_point = order[-1]
        min_distance = float('inf')
        next_point = None

        for i in range(num_coordinates):
            if not visited[i] and distances[current_point, i] < min_distance:
                min_distance = distances[current_point, i]
                next_point = i

        visited[next_point] = True
        order.append(next_point)

    return order


def two_opt(initial_path, distances):

    best_path = initial_path[:]  # Make a copy to work with
    num_points = len(best_path)
    improvement_found = True

    # The main loop continues as long as we are finding improvements
    while improvement_found:
        improvement_found = False

        for i in range(1, num_points - 1):
            for j in range(i + 1, num_points):

                # The current edges are (i-1, i) and (j, j+1)
                # We are considering swapping them for (i-1, j) and (i, j+1)
                # We use (j + 1) % num_points to handle the edge from the last point back to the first

                current_node_i = best_path[i]
                prev_node_i = best_path[i-1]
                current_node_j = best_path[j]
                next_node_j = best_path[(j + 1) % num_points]


                # Calculate and compare the change in distance

                current_distance = distances[prev_node_i, current_node_i] + distances[current_node_j, next_node_j]
                new_distance = distances[prev_node_i, current_node_j] + distances[current_node_i, next_node_j]


                if new_distance < current_distance:

                    # Perform the 2-Opt swap (reverse the segment)
                    # The slice from i to j (inclusive) is reversed.
                    best_path[i : j + 1] = best_path[i : j + 1][::-1]
                    improvement_found = True

                    break
            if improvement_found:

                break

    return best_path

###              ###
# --- PLOTTING --- #
###              ###

def plot_image_heatmap(an_image, color_palette ="hot", title_str = "", legend_str = "", plot_box = None, save_path = None):
    image_size = an_image.shape

    fig, ax = plt.subplots()
    ax.imshow(an_image, cmap=color_palette, interpolation='nearest')

    # legend / colorbar
    im = ax.imshow(an_image, cmap=color_palette, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(legend_str)

    ax.set_title(title_str)

    if(plot_box is None):
        ax.set_xlim(0, image_size[1])
        ax.set_ylim(image_size[0], 0)
    else:
        ax.set_ylim([plot_box[2],plot_box[0]])
        ax.set_xlim([plot_box[1],plot_box[3]])
    
    if save_path != None:
        fig.savefig(save_path)
        
    return fig, ax

def plot_points_on_image(an_image, x_coord, y_coord, plot_box = None, save_path = None):
    image_size = an_image.shape
    fig, ax = plt.subplots()
    
    ax.imshow(an_image, cmap='gist_yarg', origin='upper')
    ax.scatter(x_coord, y_coord, color='yellowgreen', marker='o')
    
    if(plot_box == None):
        ax.set_xlim(0, image_size[1])
        ax.set_ylim(image_size[0], 0)
    else:
        ax.set_ylim([plot_box[2],plot_box[0]])
        ax.set_xlim([plot_box[1],plot_box[3]])
    
    if save_path != None:
        fig.savefig(save_path)
        
    return fig, ax
        

def plot_for_norm_qc(ne_image, curve_data, tangents, normals, plot_box = None):
    # Can reshape for convenience: (num_points, start/end, y/x)
    tangents = tangents.transpose(2,0,1)
    normals = normals.transpose(2,0,1)
    
    image_size = ne_image.shape
    fig, ax = plt.subplots()
    
    ax.imshow(ne_image, cmap='gist_yarg', origin='upper')
    
    ax.scatter(curve_data[0], curve_data[1], marker='o', edgecolors='yellowgreen', alpha = 0.5, linewidth=1.5, s=30)
    # Plot the tangent and normal lines
    for i in range(curve_data.shape[1]):
        # Tangent line for point i
        tangent_i = tangents[i]
        # tangent_i is [[y1, x1], [y2, x2]], so we plot x's vs y's
        ax.plot(tangent_i[:, 1], tangent_i[:, 0], 'c-', linewidth=1.5, label='Tangent' if i==0 else "")
        
        # Normal line for point i
        normal_i = normals[i]
        ax.plot(normal_i[:, 1], normal_i[:, 0], 'm-', linewidth=1.5, label='Normal' if i==0 else "")
    if(plot_box == None):
        ax.set_xlim([0, image_size[1]])
        ax.set_ylim([image_size[0], 0])
    else:
        ax.set_ylim([plot_box[2],plot_box[0]])
        ax.set_xlim([plot_box[1],plot_box[3]])
    
    return fig, ax
    
def plot_boxes_on_image(an_image, color_palette, box_edge_color, crop_box_list, plot_box = None, save_path = None):
    image_size = an_image.shape
    fig, ax = plt.subplots()
    
    ax.imshow(an_image, cmap=color_palette, origin='upper')
    
    for crop_box in crop_box_list:
        box_top, box_bottom, box_left, box_right = crop_box['final_top'], crop_box['final_bottom'], crop_box['final_left'], crop_box['final_right']
        box_height = box_bottom - box_top
        box_width = box_right - box_left
        current_box = patches.Rectangle(xy = (box_left, box_top), width = box_width, height = box_height, lw = 2, alpha = 0.5, edgecolor = box_edge_color, facecolor = 'none')

        ax.add_patch(current_box)
    
    if(plot_box == None):
        ax.set_xlim(0, image_size[1])
        ax.set_ylim(image_size[0], 0)
    else:
        ax.set_ylim([plot_box[2],plot_box[0]])
        ax.set_xlim([plot_box[1],plot_box[3]])
    
    if save_path != None:
        fig.savefig(save_path)
        
    return fig, ax


###               ###
# --- IMPORTING --- #
###               ###

def parse_filename_split(file_path, file_prefix:str):
    # Remove the .json extension and the optional 'reg_result_' prefix
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    if base_name.startswith(file_prefix):
        base_name = base_name.removeprefix(file_prefix)

    # Split the remaining string into parts
    parts = base_name.split('_')
    
    # Check if the list of parts has the expected length
    if len(parts) < 7:
        return None # Or raise an error

    # Extract the information based on their position
    strain = parts[1]
    date = f"{parts[2]}_{parts[3]}_{parts[4]}"
    aqsettings = parts[5].replace('aqsettings', '')
    batch = parts[6].replace('batch', '')

    return {
        'strain': strain,
        'date': date,
        'aqsettings': aqsettings,
        'batch': batch
    }

def config_from_file(config_path = None, node_list = None):
    if config_path is None:
        raise ValueError("No configuration file path provided.")
    else:
        try:
            with open(config_path, 'r') as config_file:
                main_config = json.load(config_file)

                if node_list is not None:
                    custom_config = {}
                    for node_name in node_list:
                        custom_config.update(main_config.get(node_name))
                    return custom_config
                else:
                    return main_config
        except FileNotFoundError:
            raise ValueError(f"Error: JSON configuration file not found at '{config_path}'.")
        except json.JSONDecodeError:
            raise ValueError(f"Error: Could not decode JSON from '{config_path}'. Check the file format.")

###               ###
# --- EXPORTING --- #
###               ###

# !!!: make so handles ndarrays
def dict_to_json(dict_to_save, file_name="test_file.json"):
    with open(file_name, 'w') as json_file:
        json.dump(dict_to_save, json_file, indent=4)




def flatten_data_to_dataframe(data: dict, float_type = 'float32') -> pd.DataFrame:
    flat_data = []

    # Iterate through the first level (e.g., '0191')
    for l1_key, l1_value in data.items():
        l1_data = {
            'FoV_id': l1_key,
            'FoV_scale': l1_value.get('scale'),
            'FoV_angle': l1_value.get('angle'),
            'FoV_shift_x': l1_value.get('shift_vector', [None, None])[0],
            'FoV_shift_y': l1_value.get('shift_vector', [None, None])[1]
        }
        
        # Iterate through the second level (e.g., '04', '05')
        for l2_key, l2_value in l1_value.items():
            if not isinstance(l2_value, dict):
                continue
            
            l2_data = {
                'ne_label': l2_key,
                'ne_label_scale': l2_value.get('scale'),
                'ne_label_angle': l2_value.get('angle'),
                'ne_label_shift_x': l2_value.get('shift_vector', [None, None])[0],
                'ne_label_shift_y': l2_value.get('shift_vector', [None, None])[1]
            }
            
            # Iterate through the third level (e.g., 'slice_01')
            for l3_key, l3_value in l2_value.items():
                if not isinstance(l3_value, dict):
                    continue
                
                l3_data = {
                    'slice_id': l3_key,
                    'slice_scale': l3_value.get('scale'),
                    'slice_angle': l3_value.get('angle'),
                    'slice_shift_x': l3_value.get('shift_vector', [None, None])[0],
                    'slice_shift_y': l3_value.get('shift_vector', [None, None])[1]
                }
                
                # Combine data from all levels into a single flat record
                record = {**l1_data, **l2_data, **l3_data}
                flat_data.append(record)
    pandas_df = pd.DataFrame(flat_data)
    float64_cols = list(pandas_df.select_dtypes(include=['float64','float32','float16']))
    for col in float64_cols:
        pandas_df[col] = pandas_df[col].astype(float_type)

    return pandas_df
