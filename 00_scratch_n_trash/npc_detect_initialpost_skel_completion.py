"""
@author: jctourtellotte
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
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

# --- Included Modules --- # 
from utils.Neural_networks import Segment_NE
from tools.utility_functions import sigmoid, plot_points_on_image, coord_of_crop_box, two_opt


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
               plot_test_imgs = False,
               FoV_id = 'NNN'):
    
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
    npc_labeled_img, logits = apply_segment_ne_model(NE_model,
                                                     npc_img_relative_mean,
                                                     pad_size,
                                                     masking_threshold,
                                                     architecture,
                                                     encoder,
                                                     device)

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
    if(plot_test_imgs == True):
        fig, axes = plt.subplots()
        axes.imshow(npc_labeled_img, cmap='hot', interpolation='nearest')
        axes.set_title("All NE labeled")
        fig.savefig(f'output/{FoV_id}_all_ne_labeled.png')
    
    npc_init_fit_results = []
    indiv_ne_cropped_imgs = []

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
            coord_of_crop_box(current_img,
                              bbox_dim["width"],
                              bbox_dim["height"]
                              )
        current_img_crop = current_img[final_top:final_bottom,
                                                      final_left:final_right]
        npc_img_rel_mean_crop = npc_img_relative_mean[final_top:final_bottom,
                                                      final_left:final_right]

                
        indiv_ne_cropped_imgs.append({'ne_label': ne_mask_label_str,
                                      'orig_pos': {'y_pos': final_top,
                                                   'x_pos': final_left},
                                      'rel_mean_img_crop': npc_img_rel_mean_crop
                                       })
        # test by visualizing
        if(plot_test_imgs == True):
            fig, axes = plt.subplots(nrows = 1, ncols = 2)
            axes[0].imshow(current_img_crop, cmap='inferno')
            #current_label_crop_plt.savefig(f'output/{FoV_id}_{ne_mask_label_str}_crop.png')
            axes[0].set_title(f"Label: {ne_mask_label_str}")
            
            axes[1].imshow(npc_img_rel_mean_crop, cmap='inferno')
            #rel_mean_crop_plt.savefig(f'output/{FoV_id}_{ne_mask_label_str}_rel_mean_crop.png')
            
            fig.savefig(f'output/{FoV_id}_{ne_mask_label_str}_rel_mean_mask_combo.png')
            
            fig.show()
            
        
        try:
            current_ne_spline_points, current_ne_spline_deriv = \
                single_ne_init_fit(current_img_crop,
                                   skeleton_peaks,
                                   init_spline_sampling,
                                   plot_test_imgs = plot_test_imgs,
                                   FoV_id = FoV_id,
                                   ne_mask_label_str = ne_mask_label_str)
            # exception raised by/within single_ne_init_fit
        except ValueError:
            # skip to the next iteration if there are fewer than 5 zero values 
            #   within the region
            continue
        # ???: Re: ID -- would it be better shift to labels starting with 1?
        #           if so, would need to change the labels re: masked image
        if (plot_test_imgs == True):
            img_path = str(f'output/{FoV_id}_{ne_mask_label_str}_ne_init_spline_img.png')
            fig, axes = plot_points_on_image(npc_img_rel_mean_crop,
                                       current_ne_spline_points[0],
                                       current_ne_spline_points[1])
            axes.set_title(f"Initial Splines {FoV_id}_{ne_mask_label_str}")
            fig.savefig(img_path)
            fig.show()
            
        npc_init_fit_results.append({'ne_label': ne_mask_label_str,
                                'init_spline_points': current_ne_spline_points,
                                'init_spline_deriv': current_ne_spline_deriv
                                })

    return npc_init_fit_results, indiv_ne_cropped_imgs
    

### --- Support Functions: NPC Detection and Fitting --- ###

def apply_segment_ne_model(trained_NE_model,
                           npc_img_relative_mean,
                           padsize,
                           threshold,
                           architecture = "FPN",
                           encoder = "resnet34",
                           device = torch.device('cpu'),
                           plot_test_imgs = False):

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
        npc_img_mean_ = F.pad(npc_img_mean_,
                              (padsize, padsize, padsize, padsize),
                              mode='reflect').type(torch.float32)
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


def single_ne_init_fit(current_ne_img: np.ndarray,
                       skeleton_peaks: bool = True,
                       init_spline_sampling: int = 1000,
                       max_merge_dist: int = 10,
                       min_peaks_for_hull: int = 4,
                       min_vertices_for_spline: int = 4,
                       use_merged_clusters: bool = False,
                       plot_test_imgs = False,
                       FoV_id = 'NNN',
                       ne_mask_label_str = 'NN'):
    
    # Process for Initial Fitting
    # 1. Get points.
    # 2. Get an initial spline.
    # 3. Break it into segments.
    # 4. Cluster the segments.
    # 5. Fit final splines.

    
    # 1. Find key points (peaks) on the NE
    #       determine if applying skeletonization (default) or local maxima to define peaks
    
    if (skeleton_peaks == True):
        skeleton = skeletonize(current_ne_img, method = 'lee')
        peaks = np.column_stack(np.where(skeleton))
    else:
        peaks = peak_local_max(current_ne_img,  min_distance = 1, exclude_border = False, p_norm = 2)

    if(plot_test_imgs == True):
        img_path = str(f'output/{FoV_id}_{ne_mask_label_str}_ne_peaks_img.png')
        fig, ax = plot_points_on_image(current_ne_img,
                                       peaks[:,1],
                                       peaks[:,0])
        ax.set_title(f"Peaks {FoV_id}_{ne_mask_label_str}")
        fig.savefig(img_path)
        plt.close(fig)
    # peaks - paired [row_index, column_index]
    
    # Check conditions for creating a 2D convex hull
    #   dim + 1 -> Need at least 3 non-collinear points to compute a meaningful 2D convex hull
    #   use 4 as a more stringent criteria
    if len(peaks) < min_peaks_for_hull:
        raise ValueError(f"Warning: Not enough peaks ({len(peaks)}) found. Skipping.")

    distances = cdist(peaks, peaks)
    initial_path = list(range(len(peaks)))
    ordered_indices = two_opt(initial_path, distances)
    ordered_points = peaks[ordered_indices] # ordered_points is shape [N, 2] with (y, x)
    tck_final, _ = scipy.interpolate.splprep(
        [ordered_points[:, 1], ordered_points[:, 0]],  # Pass as (x, y)
        s=1.0, k=3, per=True, quiet=2
    )
    
    final_u = np.linspace(0, 1, init_spline_sampling)
    final_xi, final_yi = scipy.interpolate.splev(final_u, tck_final)
    final_dxi, final_dyi = scipy.interpolate.splev(final_u, tck_final, der=1)

    final_spline_points = np.vstack((final_yi, final_xi))
    final_spline_derivs = np.vstack((final_dyi, final_dxi))
    
    # ### --- Spline fitting ---
    # # 2. Fit an initial, rough spline around the entire object
    # xi, yi, dxi, dyi, good_values = None, None, None, None, None
    # try:
    #     xi, yi, dxi, dyi, good_values = initial_spline_fitting(current_ne_img,
    #                                                            peaks,
    #                                                            init_spline_sampling,
    #                                                            min_vertices_for_spline
    #                                                            )

    # except ValueError as e:
    #     print(f"Warning: Initial spline fitting failed. {e}. Skipping.")
    #     return
        
    # initial_spline_coords = np.vstack((xi, yi))

    # # ### --- Island Connecting ---
    # # 3. Find contiguous segments ("islands") where the spline is on the object
    # # [start_index, end_index] pairs for segment endpoints (determined based on y-values)
    # segment_indices = find_islands(good_values, True)

    # if not segment_indices:
    #     raise ValueError("Warning: No valid spline segments found on the object. Skipping.")
   
    # # 4. Extract all coordinates associated with a segment as well as their endpoints

    # segment_coords = [initial_spline_coords[:, start:end+1] for start, end in segment_indices]
   
    # # Format: [start_y, start_x, end_y, end_x]
    # segment_endpoints = np.array([[coords[0, 0], coords[1, 0], coords[0, -1], coords[1, -1]] for coords in segment_coords])

    # final_spline_points = np.zeros((2, init_spline_sampling))
    # final_spline_derivs = np.zeros((2, init_spline_sampling))
    # final_u = np.linspace(0, 1, init_spline_sampling)

    # # Create a mapping from an endpoint tuple to its original full coordinate data
    # endpoint_map = {tuple(end_pt): seg_coords for end_pt, seg_coords in zip(segment_endpoints, segment_coords)}

    # if use_merged_clusters: # clustering and merging 
    #     # 5. Cluster segments with endpoints within a maximum distance from each other
    #     segment_clusters = find_segment_clusters(segment_endpoints, max_merge_dist)
        
    #     if not segment_clusters:
    #         raise ValueError("Could not form any segment clusters. Skipping.")
        
    #     # 6. Merge the points of the largest cluster
    #     largest_cluster = max(segment_clusters, key=len)
    #     points_in_cluster = [endpoint_map[tuple(endpoint_row)] \
    #                          for endpoint_row in largest_cluster]
    #     merged_points = np.hstack(points_in_cluster)
    #     unique_points = np.unique(merged_points.T, axis=0)
    #     # unique_points is shape [N, 2] with (y, x)
        
    #     if unique_points.shape[0] <= min_vertices_for_spline:
    #         raise ValueError("Not enough unique points in the largest cluster.")
        
    #     # 7. Ordering the points into a continuous path
    #     #       1st. calculate pairwise distances
    #     #       2nd. use 2-opt nearest neighbor algorithm to order the points 
    #     distances = cdist(unique_points, unique_points)
    #     initial_path = list(range(unique_points.shape[0]))
    #     ordered_indices = two_opt(initial_path, distances) 
    #     ordered_points = unique_points[ordered_indices]

    #     # 8. Fit a single, CLOSED (periodic) spline to the ordered points
    #     #    Note: splprep expects points as (x, y), so we pass them in that order.
    #     #    per = True creates a closed loop
    #     tck_final, _ = scipy.interpolate.splprep([ordered_points[:, 1],
    #                                               ordered_points[:, 0]],
    #                                              s = 1.0,
    #                                              k = 3,
    #                                              per = True,
    #                                              quiet = 2)

    #     # 9. Evaluate the spline to get points and derivatives for the full circle
    #     final_u = np.linspace(0, 1, init_spline_sampling)
    #     final_xi, final_yi = scipy.interpolate.splev(final_u,
    #                                                  tck_final)
    #     final_dxi, final_dyi = scipy.interpolate.splev(final_u,
    #                                                    tck_final,
    #                                                    der = 1)
        
    #     final_spline_points = np.vstack((final_yi,
    #                                      final_xi))
    #     final_spline_derivs = np.vstack((final_dyi,
    #                                      final_dxi))
    # else:
    #     # use longest spline instead
    #     segment_lengths = np.sqrt((segment_endpoints[:, 0] - \
    #                                segment_endpoints[:, 2])**2 + \
    #                               (segment_endpoints[:, 1] - \
    #                                segment_endpoints[:, 3])**2)
       
    #     points_in_segment = endpoint_map[tuple(segment_endpoints[np.argmax(segment_lengths)])]
        
    #     tck_final, _ = scipy.interpolate.splprep([points_in_segment[1, :],
    #                                               points_in_segment[0, :]],
    #                                              s=1.0,
    #                                              k=3,
    #                                              per=0,
    #                                              quiet=2)
        
    #     final_xi, final_yi = scipy.interpolate.splev(final_u, tck_final)
    #     final_dxi, final_dyi = scipy.interpolate.splev(final_u, tck_final, der=1)
        
    #     # have to flip before appending because will be comparing to an image,
    #     #   where expected order is (y,x)
    #     final_spline_points = np.vstack((final_yi, final_xi))
    #     final_spline_derivs = np.vstack((final_dyi, final_dxi))
        
    return final_spline_points, final_spline_derivs


def initial_spline_fitting(single_ne_img: np.ndarray,
                           initial_peaks: np.ndarray,
                           init_sampling: int,
                           min_vertices: int):
    
    # ne_img_dim - tuple (height, width) AKA (rows, columns)
    ne_img_dim = (single_ne_img.shape)
   
    hull = scipy.spatial.ConvexHull(initial_peaks)

    if len(hull.vertices) < min_vertices:
        raise ValueError(f"Fewer than {min_vertices} convex hull vertices found.")
    
    # splprep - periodic spline (think: closed shapes)
    #           smoothing factor (s) of 0 forces interpolation through points
    #           expects [x, y]

    tck, _ = scipy.interpolate.splprep([initial_peaks[hull.vertices, 1],
                                        initial_peaks[hull.vertices, 0]],
                                       s = 0,
                                       per = True,
                                       quiet = 2)
    
    # evaluate the spline fits for 1000 evenly spaced distance values 
    # (arbitrary value to "smooth" the curve)
    u = np.linspace(0, 1, init_sampling)
    
    xi, yi = scipy.interpolate.splev(u, tck)
    dxi, dyi = scipy.interpolate.splev(u, tck, der=1)
    
    # use image dimensions for clipping
    #   values below 0 -> 0
    #   values above the max for either dimension -> dim's max value
    # good_values - values > 0 = True
    rounded_coords = np.round(np.array([yi, xi])).astype(np.int16)
    
    # reminder: y - rows, x - columns

    good_values = single_ne_img[np.clip(rounded_coords[0, :], 0, ne_img_dim[0] - 1),
                                np.clip(rounded_coords[1, :], 0, ne_img_dim[1] - 1)] > 0
    

    return xi, yi, dxi, dyi, good_values
    


### --- Helper functions --- ###

# finds contiguous segments (islands) of True values in y.
# returns a list of [start_index, end_index] pairs for these segments
def find_islands(y, trigger_val, stopind_inclusive = True):
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

def build_curve_bridge(segment1, segment2, tangent_scale = 0.5):
    """
    Completes a curve using a cubic Bezier bridge.
    
    Args:
        segment1 (np.ndarray): Nx2 array of points for the first segment.
        segment2 (np.ndarray): Mx2 array of points for the second segment.
        tangent_scale (float): Controls how much the tangents influence the curve.
    
    Returns:
        np.ndarray: The new points forming the bridge.
    """
    # 1. Get endpoints
    p0 = segment1[-1]  # Last point of first segment
    p3 = segment2[0]   # First point of second segment

    # 2. Estimate tangents (derivatives) from the last few points
    # You would use your pre-calculated derivatives here.
    # For this example, we estimate them.
    tangent0 = p0 - segment1[-2]
    tangent3 = p3 - segment2[1]
    
    # Normalize tangents
    tangent0 /= np.linalg.norm(tangent0)
    tangent3 /= np.linalg.norm(tangent3)

    # 3. Calculate control points based on tangents
    dist = np.linalg.norm(p3 - p0)
    p1 = p0 + tangent0 * dist * tangent_scale
    p2 = p3 - tangent3 * dist * tangent_scale # Move against the tangent from the endpoint

    # 4. Generate the Bezier curve points
    t = np.linspace(0, 1, 50)[:, np.newaxis]
    bezier_bridge = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    
    return bezier_bridge

# # --- Run and Visualize ---
# geometric_bridge = complete_curve_geometric(segment1, segment2)

# plt.figure(figsize=(8, 8))
# plt.imshow(image, cmap='gray')
# plt.plot(segment1[:, 0], segment1[:, 1], 'g-', lw=2, label='Original Segment 1')
# plt.plot(segment2[:, 0], segment2[:, 1], 'g-', lw=2, label='Original Segment 2')
# plt.plot(geometric_bridge[:, 0], geometric_bridge[:, 1], 'c--', lw=2, label='Geometric Bridge')
# plt.title("Geometric Completion (Bezier Curve)")
# plt.legend()
# plt.axis('off')
# plt.show()