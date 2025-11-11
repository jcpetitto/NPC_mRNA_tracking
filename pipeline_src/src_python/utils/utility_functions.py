
import os
import re
import torch
import tqdm
import tifffile
import math
import json

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist


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
    
    FoV_return_list = []
    # Iterate through items in the top-level directory
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        try:
            # Only process sub-directories
            if os.path.isdir(item_path) and item_name.startswith(FoV_prefix):
                print(f"Applying FoV_fn to sub-folder: '{item_name}'")
                full_FoV_fn_params = {'FoV_id': extract_digits(item_name)} \
                    if FoV_fn_params == None else FoV_fn_params # default to pass a dictionary containing the FoV_id, assuming it is the last set of digits in the sub-folder name
            
                FoV_return_list.append(FoV_fn(full_FoV_fn_params))
        except Exception as e:
            print(f'--- apply_to_dir_of_FoVs : An error occurred: {e} ---')
    try:
        if not len(FoV_return_list) == 0:
            return FoV_return_list
    except Exception as e_big: 
        print(f'--- Error returning data, no data to return: {e_big}')


def check_for_img_files(path_dict, dircheck = False, dir_name = 'resultsdir'):
    
    err = False
    
    for key in path_dict['imgs']:
        print(f'Checking for {key}')
        if not os.path.exists(path_dict['path'] + path_dict['imgs'][key]):
            err = True
            
    if dircheck==True:
        if len(os.listdir(path_dict[dir_name])) != 0:
            err = True

    return err

def make_path_dict_entry(id_digits,
                         id_length:int,
                         add_leading_zeros = False,
                         id_key_str = 'FoV_id',
                         input_dir_path = {'FoV_collection_path': 'FoV_directory/'},
                         output_subdir = {'resultsdir': 'results/'},
                         input_subdir_suffix = 'cell ',
                         files_key = 'imgs',
                         files_pieces = [{'fn_reg_npc1':  {'prefix': 'BF1red',
                                                           'suffix': '.tiff'},
                                          'fn_reg_rnp1':  {'prefix': 'BF1green',
                                                           'suffix':'.tiff'},
                                          'fn_reg_npc2':  {'prefix': 'BF2red',
                                                           'suffix':'.tiff'},
                                          'fn_reg_rnp2':  {'prefix': 'BF2green',
                                                           'suffix':'.tiff'},
                                          'fn_track_rnp': {'prefix': 'RNAgreen',
                                                           'suffix':'.tiff'},
                                          'fn_track_npc': {'prefix': 'NEred',
                                                           'suffix':'.tiff'}}]):
    
    id_string = str(id_digits).zfill(id_length) if add_leading_zeros else str(id_digits)
    constructed_dict = {files_key:
                            { # key: f"{input_subdir_suffix.strip()}{id_string}/{value['prefix']}{id_string}{value['suffix']}" \
                             key: f"{input_subdir_suffix.strip()}{id_string}/{value['prefix']}{id_string}{value['suffix']}" \
                             for key, value in files_pieces[0].items()
                             }
                        }
    path_dict_entry = {id_key_str: id_string}
    path_dict_entry.update(input_dir_path, output_subdir, constructed_dict)

    return path_dict_entry

###                   ###
# --- LIST HANDLING --- #
###                   ###

def list_update_or_replace(original_list, new_items, add_to_existing = True):
    if (add_to_existing & len(original_list) != 0):
        original_list.extend(new_items) # add entries
    else:
        original_list = new_items # overwrite entries
    return original_list

def select_from_existing(existing_dict_list, ids_to_select=[], id_str='FoV_id'):
    # registers the images in a directory
    # apply image_registration to every member of the _FoV_collection_dict (default) OR
    # FoV_to_select - use to register a subset of the existing FoV's in the collection dictionary

    if not len(ids_to_select) == 0:
            ids_to_register = [item_id for item_id in existing_dict_list \
                         if item_id[id_str] in ids_to_select]
            if(len(ids_to_register) == 0):
                print(f'Warning: no entries added')
                print(f'All requested items lacked prerequisite data.')
            else:
                id_missing =  set(ids_to_select) - {id_to_reg[id_str] for id_to_reg in ids_to_register}
                if(len(id_missing) != 0):
                    print(f'Warning: not all items as they lacked prerequisite data.')
                    print(f'Items without prerequisite data: {id_missing}')
    else:
        ids_to_register = existing_dict_list
    
    return ids_to_register

###                         ###
# --- DICTIONARY HANDLING --- #
###                         ###

# returns first match, best for unique entries

def pull_entry_by_value(value_to_find, dict_to_search, key_to_search = 'FoV_id'):
    return  next((entry for entry in dict_to_search \
                  if entry.get(key_to_search) == value_to_find), None)

# def pull_sub_entry(key_value_pairs_dict):
#     for key, value in key_value_pairs_dict.items():
    

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


def calc_tangent_endpts(points, derivs, line_length, include_normal=True):
  
    h_line_length = line_length/2
    # add a very small number to avoid dividing by zero when normalizing
    tangent_magnitudes = np.sqrt(derivs[0]**2 + derivs[1]**2) + 1e-8
    
    tangent_unit_vectors = derivs / tangent_magnitudes
    tangent_endpoints_start = points + h_line_length * tangent_unit_vectors
    tangent_endpoints_end = points - h_line_length * tangent_unit_vectors
    tangent_ends = np.array([tangent_endpoints_start, tangent_endpoints_end])
    
    # transform (rotate 90 degrees) by broadcasting normalized tangent unit vectors 
    normal_unit_vectors = np.array([[-1], [1]]) * tangent_unit_vectors
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

def coord_of_crop_box(binary_img, width, height):
    # Find the indices of all True values
    rows, cols = np.where(binary_img)

    # return None if no True values found
    if not rows.any():
        return None

    # 1. Find the center of the bounding box
    y_center = (rows.min() + rows.max()) / 2
    x_center = (cols.min() + cols.max()) / 2

    # 2. Calculate box coordinates from the center
    # Note: These can be floats initially, they are converted in the next step
    new_top = y_center - height / 2
    new_bottom = y_center + height / 2
    new_left = x_center - width / 2
    new_right = x_center + width / 2

    # 3. Clip coordinates to ensure they are within image bounds
    img_height, img_width = binary_img.shape
    final_top = max(0, round(new_top))
    final_bottom = min(img_height, round(new_bottom))
    final_left = max(0, round(new_left))
    final_right = min(img_width, round(new_right))
    
    return final_top, final_bottom, final_left, final_right
    
def extract_cropped_images(full_img_path, frame_range, crop_boxes, id_str = 'ne_label', mean_img = True):
    raw_img_stack = tifffile.imread(full_img_path)[frame_range[0]:frame_range[1], :, :]
    if mean_img:
        img_to_crop = np.mean(raw_img_stack, axis=0, dtype='uint16')
    else:
        img_to_crop = raw_img_stack

    cropped_images = []

    for entry in crop_boxes:
        final_top = entry['orig_pos']['final_top']
        final_bottom = entry['orig_pos']['final_bottom']
        final_left = entry['orig_pos']['final_left']
        final_right = entry['orig_pos']['final_right']
        ne_cropped_img = \
            {f'{id_str}': entry[f'{id_str}'],
            'cropped_img': img_to_crop[final_top:final_bottom,final_left:final_right]}
        cropped_images.append(ne_cropped_img)

    return cropped_images

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
    
    ax.scatter(curve_data[0], curve_data[1], marker='o', edgecolors='yellowgreen', linewidth=1.5, s=30)
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
    
        
###               ###
# --- EXPORTING --- #
###               ###
 
# !!!: make so handles ndarrays
def dict_to_json(dict_to_save, file_name="test_file.json"):
     with open(file_name, 'w') as json_file:
         json.dump(dict_to_save, json_file, indent=4)