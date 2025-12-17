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
from tools.utility_functions import sigmoid, plot_points_on_image, coord_of_crop_box
from tools.geom_tools import bspline_from_tck, find_closest_u_on_spline, u_values_to_ranges, build_curve_bridge


# Initial detection

def detect_npc(img_track_path,
               frame_range,
               NE_model,
               device,
               img_width = 256, # ???: best way/place to set this? default this?
               init_spline_sampling = 1000,
               qc_min_labeled = 75,
               masking_threshold = 0.5,
               bbox_dim = {"width": 75, "height": 75},
               skeleton_peaks = True,
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
    
    npc_init_fit_results = {}
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
        # test by visualizing
        if(plot_test_imgs):
            # fig, axes = plt.subplots(nrows = 1, ncols = 2)
            # axes[0].imshow(current_img_crop, cmap='inferno')
            # axes[0].set_title(f"Mask with Binary Dilation (cropped)\nLabel: {ne_mask_label_str}")
            # axes[1].imshow(npc_img_rel_mean_crop, cmap='inferno', alpha = 0.2)
            
            fig, ax = plt.subplots()
            ax.imshow(np.where(current_img_crop, npc_img_rel_mean_crop, False), cmap='inferno', origin='upper')
            ax.set_title(f"{FoV_id}{id_suffix} Mask with Binary Dilation (cropped)\nLabel: {ne_mask_label_str}")


            fig.savefig(f'output/{FoV_id}{id_suffix}_{ne_mask_label_str}_rel_mean_mask_combo.png')
            
            plt.close(fig)
        try:
            current_ne_spline_points, current_ne_spline_deriv, current_ne_bspline, is_periodic, signal_ranges = \
                single_ne_init_fit(
                    current_img_crop,
                    skeleton_peaks,
                    init_spline_sampling,
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

        #           if so, would need to change the labels re: masked image
        if (plot_test_imgs):
            img_path = str(f'output/{FoV_id}{id_suffix}_{ne_mask_label_str}_ne_init_spline_img.png')
            
            axes.set_title(f"Initial Splines {FoV_id}{id_suffix}_{ne_mask_label_str}")
            masked_mean_crop = np.where(current_img_crop, npc_img_rel_mean_crop, False)
            fig, axes = plot_points_on_image( masked_mean_crop , current_ne_spline_points[0], current_ne_spline_points[1])
            axes.set_title(f"Initial Splines {FoV_id}{id_suffix}_{ne_mask_label_str}")
            fig.savefig(img_path)
            
        npc_init_fit_results.update(
            {f'{ne_mask_label_str}':{
                'init_spline_points': current_ne_spline_points,
                'init_spline_deriv': current_ne_spline_deriv
                }
            })
        indiv_ne_bsplines.update({f'{ne_mask_label_str}': {
            'bspline_object': current_ne_bspline,
            'is_periodic': is_periodic,
            'signal_ranges': signal_ranges
                }
            })

    return npc_init_fit_results, indiv_ne_cropped_imgs, indiv_ne_bsplines
    

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

# !!!: max merge in config!!!

# Replace the existing single_ne_init_fit function with this new debug version.
def single_ne_init_fit(current_ne_img: np.ndarray, skeleton_peaks: bool = True, init_spline_sampling: int = 1000, max_merge_dist: int = 10, min_peaks_for_hull: int = 4, min_vertices_for_spline: int = 4, use_merged_clusters: bool = True, plot_test_imgs = False, FoV_id = 'NNN', ne_mask_label_str = 'NN', id_suffix = ''):
    
    # --- Start of Super-Debug Version ---
    # Helper for plotting points
    def _debug_plot(image, points, title, filename):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='viridis', origin='upper')
        if points is not None and points.shape[0] > 0:
            # Assumes points are (N, 2) with (y, x) columns
            ax.scatter(points[:, 1], points[:, 0], c='cyan', s=15, zorder=2)
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        plt.savefig(f"output/{filename}")
        plt.close(fig)

    # Helper for plotting paths
    def _debug_plot_path(image, points, title, filename):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='viridis', origin='upper')
        if points is not None and points.shape[0] > 1:
            # Assumes points are (N, 2) with (y, x) columns
            ax.plot(points[:, 1], points[:, 0], 'r-', zorder=2, linewidth=1) # Draw lines
            ax.scatter(points[:, 1], points[:, 0], c='cyan', s=15, zorder=3) # Draw points
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        plt.savefig(f"output/{filename}")
        plt.close(fig)

    prefix = f"{FoV_id}{id_suffix}_{ne_mask_label_str}"
    print(f"\n--- RUNNING SUPER-DEBUG FOR {prefix} ---")
    
    # 1. Find key points (peaks)
    if skeleton_peaks:
        skeleton = skeletonize(current_ne_img, method='lee')
        peaks = np.column_stack(np.where(skeleton))
    else:
        peaks = peak_local_max(current_ne_img, min_distance=1, exclude_border=False, p_norm=2)
    
    _debug_plot(current_ne_img, peaks, "Step 1: Detected Peaks", f"{prefix}_01_peaks.png")

    if len(peaks) < min_peaks_for_hull: raise ValueError("Not enough peaks.")

    # 2. Fit an initial, rough spline from the convex hull of the peaks
    try:
        xi, yi, dxi, dyi, good_values = initial_spline_fitting(current_ne_img,
                                                                peaks,
                                                                init_spline_sampling,
                                                                min_vertices_for_spline)
    
    except ValueError as e:
        raise e
        
    initial_spline_coords = np.vstack((yi, xi)).T # Shape (N, 2) with (y, x)

    _debug_plot(current_ne_img, initial_spline_coords, "Step 2: Initial Rough Spline", f"{prefix}_02_initial_spline.png")
    
    # 3. Find contiguous segments that are actually on the mask
    segment_indices = find_segments(good_values, True)

    if not segment_indices:
        raise ValueError("No valid segments found.")

    segment_coords = [np.vstack((yi[start:end+1], xi[start:end+1])) for start, end in segment_indices]
    all_segment_points = np.hstack(segment_coords).T if segment_coords else np.array([])

    _debug_plot(current_ne_img, all_segment_points, "Step 3: Valid Segments on Mask", f"{prefix}_03_segments.png")
    
    # MERGE (completed curve) path logic
    if use_merged_clusters:
        segment_endpoints = np.array([[c[0, 0], c[1, 0], c[0, -1], c[1, -1]] for c in segment_coords])
        endpoint_map = {tuple(ep): sc for ep, sc in zip(segment_endpoints, segment_coords)}
        deriv_endpoint_map = {tuple(ep): np.vstack((dyi[start:end+1], dxi[start:end+1])) for (start, end), ep in zip(segment_indices, segment_endpoints)}
        
        segment_clusters = find_segment_clusters(segment_endpoints, max_merge_dist)
        if not segment_clusters:
            raise ValueError("Could not form clusters.")
        
        largest_cluster = max(segment_clusters, key=len)
        points_in_cluster = [endpoint_map[tuple(ep)] for ep in largest_cluster]
        derivs_in_cluster = [deriv_endpoint_map[tuple(ep)] for ep in largest_cluster]
        
        merged_points = np.hstack(points_in_cluster).T
        merged_derivs = np.hstack(derivs_in_cluster).T

        _debug_plot(current_ne_img, merged_points, "Step 4: Merged Cluster Points", f"{prefix}_04_merged_points.png")

        unique_points, unique_indices = np.unique(merged_points, axis=0, return_index=True)
        unique_derivs = merged_derivs[unique_indices]

        _debug_plot(current_ne_img, unique_points, "Step 5: Unique Points", f"{prefix}_05_unique_points.png")

        if unique_points.shape[0] <= min_vertices_for_spline:
            raise ValueError("Not enough unique points.")

        # 6. Angular Sort
        centroid = np.mean(unique_points, axis=0)
        angles = np.arctan2(unique_points[:, 0] - centroid[0], unique_points[:, 1] - centroid[1])
        sort_indices = np.argsort(angles)
        ordered_path_points = unique_points[sort_indices]
        ordered_path_derivs = unique_derivs[sort_indices]

        _debug_plot_path(current_ne_img, ordered_path_points, "Step 6: Angularly Sorted Path", f"{prefix}_06_angular_sort.png")

        # 7. Bridge the Final Gap
        bezier_bridge = build_curve_bridge( ordered_path_points[-1],
                                            ordered_path_points[0],
                                            ordered_path_derivs[-1],
                                            ordered_path_derivs[0]
                                            )
        final_points_to_fit = np.vstack([ordered_path_points, bezier_bridge[1:-1]]) if bezier_bridge.size > 0 else ordered_path_points

        _debug_plot_path(current_ne_img, final_points_to_fit, "Step 7: Bridged Path", f"{prefix}_07_bridged_path.png")

        # 8. Fit Final Spline - includes bezier bridge points
        tck_final, _ = splprep([final_points_to_fit[:, 1],
                                final_points_to_fit[:, 0]],
                                s=0, k=3, per=True) # splprep needs [x, y]
        final_bspline = bspline_from_tck(tck_final, is_periodic=True)
        
        final_spline_eval_points = np.array(final_bspline(np.linspace(0, 1, 200))).T # Shape (N, 2) with (x, y)

        _debug_plot_path(current_ne_img, np.fliplr(final_spline_eval_points), "Step 8: Final Fitted Spline", f"{prefix}_08_final_spline.png")

        is_periodic = True
        
        u_values_for_signal = [find_closest_u_on_spline(p, final_bspline) for p in unique_points]
        signal_ranges = u_values_to_ranges(u_values_for_signal)

        final_u = np.linspace(0, 1, init_spline_sampling)
        evaluated_points = final_bspline(final_u)
        evaluated_derivatives = final_bspline.derivative(1)(final_u)
        final_spline_points = np.vstack((evaluated_points[:,1], evaluated_points[:,0]))
        final_spline_derivs = np.vstack((evaluated_derivatives[:,1], evaluated_derivatives[:,0]))
    else:
        # Simplified non-merge path for brevity in this example
        segment_lengths = [end - start for start, end in segment_indices]
        if not segment_lengths:
            raise ValueError("No segments found.")
        longest_segment_indices = segment_indices[np.argmax(segment_lengths)]
        points_in_segment = initial_spline_coords[longest_segment_indices[0] : longest_segment_indices[1] + 1].T
        tck_final, _ = splprep([points_in_segment[1, :], points_in_segment[0, :]], s=0, k=3, per=False)
        final_bspline = bspline_from_tck(tck_final, is_periodic=False)
        is_periodic = False
        start_u = longest_segment_indices[0] / init_spline_sampling
        end_u = longest_segment_indices[1] / init_spline_sampling
        signal_ranges = [(start_u, end_u)]
        final_u = np.linspace(0, 1, init_spline_sampling)
        evaluated_points = final_bspline(final_u)
        evaluated_derivatives = final_bspline.derivative(1)(final_u)
        final_spline_points = np.vstack((evaluated_points[:,1], evaluated_points[:,0]))
        final_spline_derivs = np.vstack((evaluated_derivatives[:,1], evaluated_derivatives[:,0]))
        _debug_plot_path(current_ne_img, np.fliplr(evaluated_points), "Final Spline (Non-Merged)", f"{prefix}_final_spline.png")

    return final_spline_points, final_spline_derivs, final_bspline, is_periodic, signal_ranges


def initial_spline_fitting(single_ne_img: np.ndarray,
                           initial_peaks: np.ndarray,
                           init_sampling: int,
                           min_vertices: int):
    
    # ne_img_dim: tuple (height, width) AKA (rows, columns)
    ne_img_dim = (single_ne_img.shape)

    hull = scipy.spatial.ConvexHull(initial_peaks)

    if len(hull.vertices) < min_vertices:
        raise ValueError(f"Fewer than {min_vertices} convex hull vertices found.")
    
    # make_: periodic spline (think: closed shapes)
    #          smoothing factor (s) of 0 forces interpolation through points
    #          expects [x, y]

    tck, _ = splprep(
            [initial_peaks[hull.vertices, 1], initial_peaks[hull.vertices, 0]],
            k = 3,
            s = 0,
            per = True
        )
    init_bspline = bspline_from_tck(tck, is_periodic = True)   
    # evaluate the spline fits for 1000 evenly spaced distance values 
    # (arbitrary value to "smooth" the curve)
    u = np.linspace(0, 1, init_sampling)

    # Evaluate the spline object directly
    evaluated_points = init_bspline(u)
    evaluated_derivatives = init_bspline.derivative(1)(u)

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

# finds contiguous segments (islands) of True values in y.
# returns a list of [start_index, end_index] pairs for these segments
def find_segments(y, trigger_val, stopind_inclusive = True):
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

###                      ###
# -- SEGMENT CLUSTERING -- #
###                      ###

def find_adjacent_segments(segment_endpoints, dist_threshold):
    # finds neighoring segments as defined by a threshold (inclusive)
    #       re: maxmimum distance between the endpoints of segements
   
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
