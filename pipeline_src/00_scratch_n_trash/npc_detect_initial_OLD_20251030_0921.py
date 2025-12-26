"""
@author: Jocelyn Petitto
"""

# --- Outside Modules --- #
import tifffile
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import skimage
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max

import scipy
from scipy.interpolate import splprep
from scipy.spatial.distance import cdist

# --- Included Modules --- # 
from utils.Neural_networks import Segment_NE
from tools.utility_functions import sigmoid, plot_points_on_image, coord_of_crop_box, find_segments
from tools.geom_tools import bspline_from_tck, find_closest_u_on_spline, u_values_to_ranges, build_curve_bridge, angular_sort_alg, get_spline_arc_length, get_u_range_from_bspline


# Initial detection

def detect_npc(img_track_path,
               frame_range,
               NE_model,
               device,
               img_width = 256, # ???: best way/place to set this? default this?
               init_sampling_density = 50,
               bspline_smoothing = 1.6,
               qc_min_labeled = 75,
               masking_threshold = 0.5,
               bbox_dim = {"width": 75, "height": 75},
               use_skel = True,
               use_merged_clusters = True,
               max_merge_dist = 10, 
               plot_test_imgs = False,
               FoV_id = 'NNN',
               id_suffix = ''): # id suffix for use in multiple NE label situations
    
    npc_track_img = tifffile.imread(img_track_path)[frame_range[0]:frame_range[1]]
    npc_img_mean = np.mean(npc_track_img, axis = 0)
    npc_img_relative_mean = npc_img_mean/np.max(npc_img_mean)
    
    pad_size = int((img_width - np.shape(npc_img_mean)[0])/2) # limits image size

   # define model architecture
    architecture = "FPN"
    encoder = "resnet34"

    
    # create (via masking_threshold) and label (integer values) initial NPC masks
    # 1st. apply model, 2. uniquely label unconnected areas
    # return both the binary response (logits) as well as the uniquely labeled mask(s)
    npc_labeled_img, _ = apply_segment_ne_model(NE_model,
                                                     npc_img_relative_mean,
                                                     pad_size,
                                                     masking_threshold,
                                                     architecture,
                                                     encoder,
                                                     device)

    if(plot_test_imgs):
        fig, axes = plt.subplots()
        axes.imshow(npc_labeled_img, cmap='hot', interpolation='nearest', origin='upper')
        axes.set_title(f"All NE labels {FoV_id}{id_suffix}")
        fig.savefig(f'output/{FoV_id}{id_suffix}_all_ne_labels.png')
    

    # get all unique label values and the count per label
    ne_mask_label_set = np.unique(npc_labeled_img, return_counts = True)

    # checks that the label isn't the background (0) and if the label count 
    #   meets or exceeds a threshold (qc_min_labeled, default 75)
    #   function for use with "apply_along_axis"
    def check_mask_fn(label_with_count):
        if label_with_count[0] != 0 and label_with_count[1] >= qc_min_labeled:
            return True
        else:
            return False
    
    # applies above function, selects only the labels that meet critera and
    #   saves them to a list
    usable_mask_labels = \
        ne_mask_label_set[0][np.apply_along_axis(check_mask_fn, 0, ne_mask_label_set)].tolist()

    npc_labeled_img[~np.isin(npc_labeled_img, list(usable_mask_labels))] = 0

    # test by visualizing labeled image
    if(plot_test_imgs):
        fig, ax = plt.subplots()
        ax.imshow(npc_labeled_img, cmap='hot', interpolation='nearest', origin='upper')
        ax.set_title(f'All usable NE labels {FoV_id}{id_suffix}')
        fig.savefig(f'output/{FoV_id}{id_suffix}_all_useable_ne_labels.png')
    
    indiv_ne_cropped_imgs = {}
    indiv_ne_bsplines = {}

    # For each set of labeled points (masked potential NE) that met criteria
    for ne_label in usable_mask_labels:
        current_img = npc_labeled_img == ne_label
        ne_mask_label_str = str(ne_label).zfill(2)
        # simultaneously select only the specified label AND change to boolean
        #   for improved performance re: binary dilation
        current_img = skimage.morphology.binary_dilation(current_img,
                                                   np.array([[1, 1, 1],
                                                             [1, 1, 1],
                                                             [1, 1, 1]])
                                                   )
        
        final_top, final_bottom, final_left, final_right = \
            coord_of_crop_box(current_img, bbox_dim["width"], bbox_dim["height"])
        current_img_crop = current_img[final_top:final_bottom, final_left:final_right]
        npc_img_rel_mean_crop = npc_img_relative_mean[final_top:final_bottom, final_left:final_right]

        # saving crop parameters for the image associated with the ne_label        
        indiv_ne_cropped_imgs.update({f'{ne_mask_label_str}':
                                                {
                                                    'height': bbox_dim["height"],
                                                    'width': bbox_dim["width"],
                                                    'final_top': final_top,
                                                    'final_left': final_left,
                                                    'final_bottom': final_bottom,
                                                    'final_right': final_right}
                                                })
        current_img_intensity_mask = np.where(current_img_crop, npc_img_rel_mean_crop, False)
        # test by visualizing
        if(plot_test_imgs):
            fig, ax = plt.subplots()
            ax.imshow(current_img_intensity_mask, cmap='inferno', origin='upper')
            ax.set_title(f"{FoV_id}{id_suffix} Mask with Binary Dilation (cropped)\nLabel: {ne_mask_label_str}")


            fig.savefig(f'output/{FoV_id}{id_suffix}_{ne_mask_label_str}_rel_mean_mask_combo.png')
            
            plt.close(fig)
        try:
            current_ne_bspline, is_periodic = \
                single_ne_init_fit(
                    current_img_crop,
                    current_img_intensity_mask,
                    use_skel,
                    init_sampling_density,
                    bspline_smoothing,
                    plot_test_imgs = plot_test_imgs,
                    FoV_id = FoV_id,
                    ne_mask_label_str = ne_mask_label_str,
                    use_merged_clusters = use_merged_clusters,
                    id_suffix = id_suffix
                    )
            # exception raised by/within single_ne_init_fit
        except ValueError:
            # skip to the next iteration if there are fewer than 5 zero values 
            #   within the region
            continue

        indiv_ne_bsplines.update({f'{ne_mask_label_str}': {
            'bspline_object': current_ne_bspline,
            'is_periodic': is_periodic
                }
            })

#    return npc_init_fit_results, indiv_ne_cropped_imgs, indiv_ne_bsplines
    return indiv_ne_cropped_imgs, indiv_ne_bsplines    

### --- Support Functions: NPC Detection and Fitting --- ###

def apply_segment_ne_model(trained_NE_model, npc_img_relative_mean, padsize, threshold, architecture = "FPN", encoder = "resnet34", device = torch.device('cpu'), plot_test_imgs = False):

    model = Segment_NE(architecture, encoder, in_channels = 3, out_classes = 1)
    
    # setting to make predictions while maintaining model integrity
    with torch.no_grad():   
        # Load (pre-existing) trained model
        state_dict = torch.load(trained_NE_model, map_location = device)
        model.load_state_dict(state_dict)
        (model.eval()).to(device)
        
        # convert to tensor with dimensions (1, 1, Height, Width)
        npc_img_mean_ = torch.tensor(npc_img_relative_mean, dtype=torch.float32).to(device)[None, None, ...] 
        # pad last 2 dimensions, height and width
        npc_img_mean_ = F.pad(npc_img_mean_, (padsize, padsize, padsize, padsize), mode='reflect').type(torch.float32)
        # ???: note on potential non-deterministic behavior re: running on CUDA: 
        #       https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        
        # inference step
        logits = model(npc_img_mean_) 
        logits = sigmoid(logits.detach().cpu().numpy()[:,0,:,:])
        
        # setting each value in the logit tensor to 0 or 1 based on 
        #   proximity (below or above) a given threshold
        logits[logits < threshold] = 0
        logits[logits >= threshold] = 1

        if padsize !=0:
            logits = logits[0, padsize:-padsize, padsize:-padsize]
        else:
            logits = logits[0, :, :]

        # clear object borders by assigning the background value
        #   for mask to narrow down area to search for peaks
        cleared = clear_border(logits)
        
    # Assigns unique labels to connected groups of integers in an array
    labeled_img = label(cleared)
    
    return labeled_img, logits

# !!! max merge in config!!!

# TODO - pair down debug mode within this function
# TODO - break function into smaller functions
# ??? - re: distance for filling gaps when defining segments versus merging segments
def single_ne_init_fit(current_ne_label_img: np.ndarray, current_ne_intensity_mask: np.ndarray, use_skel: bool = True, init_sampling_density: int = 256, bspline_smoothing = 1.6, max_merge_dist: int = 10, min_peaks_for_hull: int = 4, min_vertices_for_spline: int = 4, use_merged_clusters: bool = True, plot_test_imgs = False, FoV_id = 'NNN', ne_mask_label_str = 'NN', id_suffix = ''):
    
    # --- Plotting re: Debugging ---
    # Helper for plotting points
    def _debug_plot(image, points, title, filename, points2 = None):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='inferno', origin='upper')
        if points is not None and points.shape[0] > 0:
            # Assumes points are (N, 2) with (y, x) columns
            ax.scatter(points[:, 1], points[:, 0], c='blue', s=15, zorder=2, alpha = 0.50)
        if points2 is not None and points.shape[0] > 0:
            ax.scatter(points2[:, 1], points2[:, 0], c='green', s=15, zorder=2, alpha = 0.50)
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        print(f"output/{filename}")
        plt.savefig(f"output/{filename}")
        plt.close(fig)

    # Helper for plotting paths
    def _debug_plot_path(image, points, title, filename):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='inferno', origin='upper')
        if points is not None and points.shape[0] > 1:
            # Assumes points are (N, 2) with (y, x) columns
            ax.plot(points[:, 1], points[:, 0], 'r-', zorder=2, linewidth=1) # Draw lines
            ax.scatter(points[:, 1], points[:, 0], c='cyan', s=15, zorder=3) # Draw points
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        plt.savefig(f"output/{filename}")
        plt.close(fig)

    # prefix for debug file names, etc.
    prefix = f"{FoV_id}{id_suffix}_{ne_mask_label_str}"
    
    # 1. Find key points (peaks)
    if use_skel:
        skeleton_points = skeletonize(current_ne_label_img, method='lee')
        skeleton_points = np.column_stack(np.where(skeleton_points))
        intensity_peaks = peak_local_max(current_ne_intensity_mask, min_distance=1, exclude_border=False, p_norm=2)
        points_for_spline = skeleton_points
    else:
        intensity_peaks = peak_local_max(current_ne_intensity_mask, min_distance=1, exclude_border=False, p_norm=2)
        points_for_spline = intensity_peaks
    
    # print(f"{prefix}_01_skel_peaks.png")
    # _debug_plot(current_ne_intensity_mask, skeleton_points, "Step 1: Skeletonize (compared to max intensity peaks)", f"{prefix}_01_skel_peaks.png", intensity_peaks)

    if (len(skeleton_points) < min_peaks_for_hull):
        raise ValueError("Not enough peaks.")
    try:
        xi, yi, dxi, dyi, good_values = initial_spline_fitting(current_ne_label_img, points_for_spline, init_sampling_density, bspline_smoothing, min_vertices_for_spline)
    
    except ValueError as e:
        raise e
        
    initial_spline_coords = np.vstack((xi, yi))
    # initial_spline_deriv = np.vstack((dxi, dyi))
    
    # 3. Find contiguous segments that are actually on the mask
    segment_indices = find_segments(good_values, True)

    if not segment_indices:
        raise ValueError("No valid segments found.")

    segment_coords = [np.vstack((yi[start:end+1], xi[start:end+1])) for start, end in segment_indices]
    # segment_deriv = [initial_spline_deriv[:, start:end+1] for start, end in segment_indices]
    # Format: [start_y, start_x, end_y, end_x]
    segment_endpoints = np.array([[coords[0, 0], coords[1, 0], coords[0, -1], coords[1, -1]] for coords in segment_coords])

    # sampling_n = len(xi)
    # final_spline_points = np.zeros((2, sampling_n))
    # final_spline_derivs = np.zeros((2, sampling_n))
    # final_u = np.linspace(0, 1, sampling_n)

    # Create a mapping from an endpoint tuple to its original full coordinate data
    endpoint_map = {tuple(end_pt): seg_coords \
                    for end_pt, seg_coords in zip(segment_endpoints, segment_coords)}
    # deriv_endpoint_map = {tuple(end_pt): seg_deriv \
    #                         for end_pt, seg_deriv in zip(segment_endpoints, segment_deriv)}

    # MERGE (completed curve) path logic
    if use_merged_clusters:
        # segment_endpoints = np.array([[c[0, 0], c[1, 0], c[0, -1], c[1, -1]] for c in segment_coords])
        # endpoint_map = {tuple(ep): sc for ep, sc in zip(segment_endpoints, segment_coords)}
        deriv_endpoint_map = {tuple(ep): np.vstack((dyi[start:end+1], dxi[start:end+1])) for (start, end), ep in zip(segment_indices, segment_endpoints)}
        
        segment_clusters = find_segment_clusters(segment_endpoints, max_merge_dist)
        if not segment_clusters:
            raise ValueError("Could not form clusters.")
        
        largest_cluster = max(segment_clusters, key=len)
        points_in_cluster = [endpoint_map[tuple(ep)] for ep in largest_cluster]
        derivs_in_cluster = [deriv_endpoint_map[tuple(ep)] for ep in largest_cluster]
        
        merged_points = np.hstack(points_in_cluster).T
        merged_derivs = np.hstack(derivs_in_cluster).T

        # _debug_plot(current_ne_label_img, merged_points, "Step 4: Merged Cluster Points", f"{prefix}_04_merged_points.png")

        unique_points, unique_indices = np.unique(merged_points, axis=0, return_index=True)
        unique_derivs = merged_derivs[unique_indices]

        # _debug_plot(current_ne_label_img, unique_points, "Step 5: Unique Points", f"{prefix}_05_unique_points.png")

        if unique_points.shape[0] <= min_vertices_for_spline:
            raise ValueError("Not enough unique points.")

        sort_indices = angular_sort_alg(unique_points)
        ordered_path_points = unique_points[sort_indices]
        ordered_path_derivs = unique_derivs[sort_indices]

        # _debug_plot_path(current_ne_label_img, ordered_path_points, "Step 6: Angularly Sorted Path", f"{prefix}_06_angular_sort.png")

        # 7. Bridge the Final Gap
        bezier_bridge = build_curve_bridge( ordered_path_points[-1],
                                            ordered_path_points[0],
                                            ordered_path_derivs[-1],
                                            ordered_path_derivs[0]
                                            )

        # !!! HOW to impact the bridge's curvature? want to be able to force similar curvature as the existing curve
        
        # Combine the main path and the bridge; bezier_bridge[1:-1] to exclude its start and end points, which are already in ordered_path_points
        final_points_to_fit = np.vstack([ordered_path_points, bezier_bridge[1:-1]]) if bezier_bridge.size > 0 else ordered_path_points

        _debug_plot_path(current_ne_label_img, final_points_to_fit, "Step 7: Bridged Path", f"{prefix}_07_bridged_path.png")

        # 8. Fit Final Spline - includes bezier bridge points
        tck_final, _ = splprep([final_points_to_fit[:, 0],
                                final_points_to_fit[:, 1]],
                                s=bspline_smoothing, k=3, per=True) # splprep needs [x, y]
        final_bspline = bspline_from_tck(tck_final, is_periodic=True)
        is_periodic = True
# !!! NOTE: bspline is NOT refined at this point and therefore is built on the initial sampling density and, by virtue of the bridge being added AFTER the initial sampling, and arc length note computed for this graph, it is not equidistance betwen points
        final_spline_eval_points = final_bspline(np.linspace(0, 1, 200)) # Shape (N, 2) with (x, y)

        _debug_plot_path(current_ne_label_img, final_spline_eval_points, "Step 8: Final Fitted Spline", f"{prefix}_08_final_spline.png")
        
        # final_u = np.linspace(0, 1, init_sampling_density)
        # evaluated_points = final_bspline(final_u)
        # evaluated_derivatives = final_bspline.derivative(1)(final_u)
        # final_spline_points = np.vstack((evaluated_points[:,1], evaluated_points[:,0]))
        # final_spline_derivs = np.vstack((evaluated_derivatives[:,1], evaluated_derivatives[:,0]))

    else: # Non-merged path
        segment_lengths = [end - start for start, end in segment_indices]
        if not segment_lengths:
            raise ValueError("No segments found.")
        longest_segment_indices = segment_indices[np.argmax(segment_lengths)]
        start_idx, end_idx = longest_segment_indices
        points_in_segment = initial_spline_coords[:, start_idx : end_idx + 1]
        tck_final, _ = splprep([points_in_segment[1, :], points_in_segment[0, :]], s=0, k=3, per=False)
        final_bspline = bspline_from_tck(tck_final, is_periodic=False)
        is_periodic = False
        
        # ???: is outputting these final points even necessary?
        # final_bspline_length = get_spline_arc_length(final_bspline)
        # final_u_N = final_bspline_length * init_sampling_density
        # final_u = np.linspace(0, 1, final_u_N)
        # evaluated_points = final_bspline(final_u)
        # evaluated_derivatives = final_bspline(final_u, nu = 1)
        # final_spline_points = np.vstack((evaluated_points[:,1], evaluated_points[:,0]))
        # final_spline_derivs = np.vstack((evaluated_derivatives[:,1], evaluated_derivatives[:,0]))
        # _debug_plot_path(current_ne_label_img, np.fliplr(evaluated_points), "Final Spline (Non-Merged)", f"{prefix}_final_spline.png")

    # return final_spline_points, final_spline_derivs, final_bspline, is_periodic
    return final_bspline, is_periodic

def initial_spline_fitting(single_ne_img: np.ndarray, initial_peaks: np.ndarray, init_sampling_density: int, bspline_smoothing, min_vertices: int):
    
    # ne_img_dim: tuple (height, width) AKA (rows, columns)
    ne_img_dim = (single_ne_img.shape)

    hull = scipy.spatial.ConvexHull(initial_peaks)

    if len(hull.vertices) < min_vertices:
        raise ValueError(f"Fewer than {min_vertices} convex hull vertices found.")
    
    # make_: periodic spline (think: closed shapes)
    #       - smoothing factor (s) of 0 forces interpolation through points 
    #       - expects [x, y]
    # ??? Why are points lost somewhere re: that pesky broken-jaw of a PacMan

    tck, _ = splprep(
            [initial_peaks[hull.vertices, 1], initial_peaks[hull.vertices, 0]],
            k = 3,
            s = bspline_smoothing,
            per = True
        )
    init_bspline = bspline_from_tck(tck, is_periodic = True)

    # TODO evaluate the length of the arc made to determine the number of values using a pre-defined parameter density

    u_range = get_u_range_from_bspline(init_bspline, init_sampling_density)

    # Evaluate the spline object directly
    evaluated_points = init_bspline(u_range)
    evaluated_derivatives = init_bspline.derivative(1)(u_range)

    # Unpack the results
    xi, yi = evaluated_points[:,0], evaluated_points[:,1]
    dxi, dyi = evaluated_derivatives[:,0], evaluated_derivatives[:,1]
    
    
    # use image dimensions for clipping
    #   values below 0 -> 0
    #   values above the max for either dimension -> dim's max value
    # good_values: values > 0 = True
    rounded_coords = np.round(np.array([yi, xi])).astype(np.int16)
    
    # reminder: y : rows, x : columns
    # This is how rounded_coords is formatted, making index 0 y and index 1 x for indexing re: assigning good_values (below)

    # if spline evaluates to points beyond the img limits,
    #   np.clip forces them back into range by assigning such points
    #   the value at the boundary
    good_values = single_ne_img[np.clip(rounded_coords[0, :], 0, ne_img_dim[0] - 1),
                                np.clip(rounded_coords[1, :], 0, ne_img_dim[1] - 1)] > 0
    

    return xi, yi, dxi, dyi, good_values
    


### --- Helper functions --- ###



###                      ###
# -- SEGMENT CLUSTERING -- #
###                      ###

def find_adjacent_segments(segment_endpoints, dist_threshold):
    # finds neighoring segments as defined by a threshold (inclusive)
    #       re: maxmimum distance (in pixels) between the endpoints of segements
   
    num_segments = len(segment_endpoints)
    
    adj_list = {i: [] for i in range(num_segments)}
    
    start_points = segment_endpoints[:, 0:2]
    end_points = segment_endpoints[:, 2:4]
    
    # Pre-calculate all distance matrices between endpoint sets

    dist_start_start = cdist(start_points, start_points)
    dist_start_end = cdist(start_points, end_points)
    dist_end_end = cdist(end_points, end_points)
        
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
        # Find the minimum distance between any endpoint of i and any endpoint of j
            min_dist = min(dist_start_start[i, j],
                           dist_start_end[i, j],
                           dist_start_end[j, i], # This is equivalent to dist_end_start[j,i]
                           dist_end_end[i, j])
            if min_dist <= dist_threshold:
                adj_list[i].append(j)
                adj_list[j].append(i)
                                 
    return adj_list

    
def find_segment_clusters(segment_endpoints, dist_threshold):
    # finds segement clusters based on a minimum threshold distance between endpoirnts
    # uses depth first search (DFS) to traverse points, forming a cluster

    adj_endpoints_list = find_adjacent_segments(segment_endpoints, dist_threshold)
    # list of segments consisting of only those with endpoints within a certain
    #   distance (dist_threshold) of each other

    num_segments = len(segment_endpoints)

    visited = set() # keeps track of segments already visited

    all_clusters = [] # holds the clusters of segements, as determined by connectivity

    for i in range(num_segments):
        if i not in visited:
            # Start of a new cluster
            current_cluster_indices = [] # indices of current cluster segements
            stack = [i] # the "todo" list re: checking threshold distance to the next endpoint
            visited.add(i)

            # DFS
            while stack: # while there are still nodes to travel

                node_idx = stack.pop()
                current_cluster_indices.append(node_idx) # to-do -> to-done

                for neighbor_idx in adj_endpoints_list[node_idx]:

                    if neighbor_idx not in visited: # if you haven't been there before

                        visited.add(neighbor_idx)
                        stack.append(neighbor_idx)

            # Convert indices back to segments data
            cluster_segments = segment_endpoints[current_cluster_indices]
            all_clusters.append(cluster_segments)

    return all_clusters

def define_starting_points(ne_label_img, ne_mask_img, method='skeleton'):
    match method:
        case 'skeleton':
            skeleton_points = skeletonize(ne_label_img, method='lee')
            starting_points = np.column_stack(np.where(skeleton_points))
        case 'max_intensity':
            starting_points = peak_local_max(ne_mask_img, min_distance=1, exclude_border=False, p_norm=2)
    
    return starting_points