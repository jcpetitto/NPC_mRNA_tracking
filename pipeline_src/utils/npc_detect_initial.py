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
from scipy.interpolate import make_splprep
from scipy.spatial.distance import cdist

# --- Included Modules --- # 
from utils.Neural_networks import Segment_NE
from utils.utility_functions import sigmoid, plot_points_on_image, coord_of_crop_box


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
    npc_labeled_img, _ = apply_segment_ne_model(NE_model,
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
    indiv_ne_bsplines = []

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
        indiv_ne_cropped_imgs.append({'ne_label': ne_mask_label_str,
                                      'orig_pos': {'final_top': final_top,
                                                   'final_left': final_left,
                                                   'final_bottom': final_bottom,
                                                   'final_right': final_right}
                                       })
        # test by visualizing
        if(plot_test_imgs):
            fig, axes = plt.subplots(nrows = 1, ncols = 2)
            axes[0].imshow(current_img_crop, cmap='inferno')
            #current_label_crop_plt.savefig(f'output/{FoV_id}_{ne_mask_label_str}_crop.png')
            axes[0].set_title(f"Label: {ne_mask_label_str}")
            
            axes[1].imshow(npc_img_rel_mean_crop, cmap='inferno')
            #rel_mean_crop_plt.savefig(f'output/{FoV_id}_{ne_mask_label_str}_rel_mean_crop.png')
            
            fig.savefig(f'output/{FoV_id}_{ne_mask_label_str}_rel_mean_mask_combo.png')
            
            plt.close(fig)
        try:
            current_ne_spline_points, current_ne_spline_deriv, current_ne_bspline = \
                single_ne_init_fit(
                    current_img_crop,
                    skeleton_peaks,
                    init_spline_sampling,
                    plot_test_imgs = plot_test_imgs,
                    FoV_id = FoV_id,
                    ne_mask_label_str = ne_mask_label_str,
                    use_merged_clusters = use_merged_clusters
                    )
            # exception raised by/within single_ne_init_fit
        except ValueError:
            # skip to the next iteration if there are fewer than 5 zero values 
            #   within the region
            continue
        # ???: Re: ID -- would it be better shift to labels starting with 1?
        #           if so, would need to change the labels re: masked image
        if (plot_test_imgs):
            img_path = str(f'output/{FoV_id}_{ne_mask_label_str}_ne_init_spline_img.png')
            fig, axes = plot_points_on_image(npc_img_rel_mean_crop, current_ne_spline_points[0], current_ne_spline_points[1])
            axes.set_title(f"Initial Splines {FoV_id}_{ne_mask_label_str}")
            fig.savefig(img_path)
            fig.show()
            
        npc_init_fit_results.append(
            {'ne_label': ne_mask_label_str,
            'init_spline_points': current_ne_spline_points,
            'init_spline_deriv': current_ne_spline_deriv}
            )
        indiv_ne_bsplines.append(
            {'ne_label': ne_mask_label_str,
            'bspline_object': current_ne_bspline}
            )

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
def single_ne_init_fit(current_ne_img: np.ndarray, skeleton_peaks: bool = True, init_spline_sampling: int = 1000, max_merge_dist: int = 10, min_peaks_for_hull: int = 4, min_vertices_for_spline: int = 4, use_merged_clusters: bool = True, plot_test_imgs = False, FoV_id = 'NNN', ne_mask_label_str = 'NN'):
    
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

    ### --- Spline fitting ---
    # 2. Fit an initial, rough spline around the entire object
    xi, yi, dxi, dyi, good_values = None, None, None, None, None
    
    try:
        xi, yi, dxi, dyi, good_values = initial_spline_fitting(current_ne_img,
                                                               peaks,
                                                               init_spline_sampling,
                                                               min_vertices_for_spline
                                                               )

    except ValueError as e:
        print(f"Warning: Initial spline fitting failed. {e}. Skipping.")
        return
        
    initial_spline_coords = np.vstack((xi, yi))
    initial_spline_deriv = np.vstack((dxi, dyi))

    # ### --- Segment Connecting ---
    # 3. Find contiguous segments od spline points
    # [start_index, end_index] pairs for segment endpoints (determined based on y-values)
    segment_indices = find_segments(good_values, True)

    if not segment_indices:
        raise ValueError("Warning: No valid spline segments found on the object. Skipping.")
   
    # 4. Extract all coordinates associated with a segment as well as their endpoints

    segment_coords = [initial_spline_coords[:, start:end+1] for start, end in segment_indices]
    segment_deriv = [initial_spline_deriv[:, start:end+1] for start, end in segment_indices]
   
    # Format: [start_y, start_x, end_y, end_x]
    segment_endpoints = np.array([[coords[0, 0],
                                   coords[1, 0],
                                   coords[0, -1],
                                   coords[1, -1]] for coords in segment_coords])
    
    final_spline_points = np.zeros((2, init_spline_sampling))
    final_spline_derivs = np.zeros((2, init_spline_sampling))
    final_u = np.linspace(0, 1, init_spline_sampling)

    # Create a mapping from an endpoint tuple to its original full coordinate data
    endpoint_map = {tuple(end_pt): seg_coords \
                    for end_pt, seg_coords in zip(segment_endpoints, segment_coords)}
    deriv_endpoint_map = {tuple(end_pt): seg_deriv \
                          for end_pt, seg_deriv in zip(segment_endpoints, segment_deriv)}

    if use_merged_clusters: # clustering and merging 
        # 5. Cluster segments with endpoints within a maximum distance from each other
        segment_clusters = find_segment_clusters(segment_endpoints, max_merge_dist)
        
        if not segment_clusters:
            raise ValueError("Could not form any segment clusters. Skipping.")
        
        # 6. Merge both points AND derivatives from the largest cluster
        largest_cluster = max(segment_clusters, key=len)
        points_in_cluster = [endpoint_map[tuple(endpoint_row)] \
                             for endpoint_row in largest_cluster]
        derivs_in_cluster = [deriv_endpoint_map[tuple(ep)] \
                             for ep in largest_cluster]
        merged_points = np.hstack(points_in_cluster)
        merged_derivs = np.hstack(derivs_in_cluster)
        # Transpose for easier row-wise operations
        merged_points_t = merged_points.T # Shape (N_total, 2) with (y, x)
        merged_derivs_t = merged_derivs.T # Shape (N_total, 2) with (dy, dx)
        
        unique_points, unique_indices = np.unique(merged_points_t,
                                                  axis=0,
                                                  return_index=True)
        unique_derivs = merged_derivs_t[unique_indices]
        
        if unique_points.shape[0] <= min_vertices_for_spline:
            raise ValueError("Not enough unique points in the largest cluster.")

        # !!!: make this a function in the utility_functions module
        
        # 7. --- NEW: Fast Angular Sort (Replaces two_opt) ---
        # Calculate the centroid (center) of the points
        centroid = np.mean(unique_points, axis=0)
        # Calculate the angle of each point relative to the centroid
        angles = np.arctan2(unique_points[:, 0] - centroid[0], unique_points[:, 1] - centroid[1])
        # Sort the points by angle to create a continuous path
        sort_indices = np.argsort(angles)
        ordered_path_points = unique_points[sort_indices]
        ordered_path_derivs = unique_derivs[sort_indices] 

        # 8. --- Bridge the Final Gap ---
        # Combine the sorted path and the bridge to form a complete loop
        # To connect the end of the path to its start, use the path itself 
        #   as both the first and second segment argument.
        # Note: Check if bezier_bridge is not empty before stacking
        if len(ordered_path_points) > 1:

            bezier_bridge = build_curve_bridge(
                p_end = ordered_path_points[-1],
                p_start = ordered_path_points[0],
                tangent_end = ordered_path_derivs[-1],
                tangent_start = ordered_path_derivs[0],
                tangent_scale=0.3
            )
        else:
            bezier_bridge = np.array([])

        # 9. Combine the main path and the bridge, removing duplicate points at the seams
        if bezier_bridge.size > 0:
            # Use bezier_bridge[1:-1] to exclude its start and end points,
            #   which are already in ordered_path_points.
            final_points_to_fit_untransposed = np.vstack([ordered_path_points, bezier_bridge[1:-1]])
        else:
            final_points_to_fit_untransposed = ordered_path_points

        final_points_to_fit = final_points_to_fit_untransposed.T 
        # ???: Transpose to (2, N) for splprep (or (N,2) ?)
        
        if len(final_points_to_fit[0]) > 2: # Need at least 3 points to check rank
            # Check if the points are collinear by checking the rank of their covariance matrix.
            # Rank < 2 means the points have collapsed to a line.
            rank = np.linalg.matrix_rank(np.cov(final_points_to_fit))
            
            if rank < 2:
                print(f"ERROR: Points for NE label {ne_mask_label_str} are collinear!")
                # The debugger will pause here, letting you see the bad points
                breakpoint()
                # Or raise an error to stop
                # raise ValueError(f"Collinear points detected for NE label {ne_mask_label_str}")


        # is_periodic = True # The path is now a guaranteed closed loop
        

        # 10. Fit a single, CLOSED (periodic) spline to the ordered points
        #       Note: splprep expects points as (x, y)
        #       per = True creates a closed loop
        #     Evaluate the spline to get points and derivatives for the closed shape
        tck_final, _ = scipy.interpolate.make_splprep(
            [final_points_to_fit[1,:], final_points_to_fit[0,:]],
            s = 1.0,
            k = 3)

        final_u = np.linspace(0, 1, init_spline_sampling)
        
        # Evaluate the spline object directly
        evaluated_points = tck_final(final_u)
        evaluated_derivatives = tck_final.derivative(1)(final_u)

        # Unpack the results
        final_xi, final_yi = evaluated_points[0,:], evaluated_points[1,:]
        final_dxi, final_dyi  = evaluated_derivatives[0,:], evaluated_derivatives[1,:]

        final_spline_points = np.vstack((final_yi,
                                         final_xi))
        final_spline_derivs = np.vstack((final_dyi,
                                         final_dxi))
    else:
        # Consider all the segements of the nuclear envelope, fit only the the longest one
        # in this case, the spline is not periodic

        # QUESTION if the endpoints are within the acceptable distance for merging segments, is periodic (?!)
        segment_lengths = np.sqrt((segment_endpoints[:, 0] - \
                                   segment_endpoints[:, 2])**2 + \
                                  (segment_endpoints[:, 1] - \
                                   segment_endpoints[:, 3])**2)
       
        points_in_segment = endpoint_map[tuple(segment_endpoints[np.argmax(segment_lengths)])]
        
        tck_final, _ = scipy.interpolate.make_splprep([points_in_segment[1, :],
                                                  points_in_segment[0, :]],
                                                 s=1.0,
                                                 k=3)
        # Evaluate the spline object directly
        evaluated_points = tck_final(final_u)
        evaluated_derivatives = tck_final.derivative(1)(final_u)

        # Unpack the results
        xi, yi = evaluated_points[0,:], evaluated_points[1,:]
        dxi, dyi = evaluated_derivatives[0,:], evaluated_derivatives[1,:]
        
        # have to flip before appending because will be comparing to an image,
        #   where expected order is (y,x)
        final_spline_points = np.vstack((final_yi, final_xi))
        final_spline_derivs = np.vstack((final_dyi, final_dxi))
        
    return final_spline_points, final_spline_derivs, tck_final


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

    tck, _ = scipy.interpolate.make_splprep(
                [initial_peaks[hull.vertices, 1], initial_peaks[hull.vertices, 0]],
                k = 3,
                s = 0
        )
       
    # evaluate the spline fits for 1000 evenly spaced distance values 
    # (arbitrary value to "smooth" the curve)
    u = np.linspace(0, 1, init_sampling)

    # Evaluate the spline object directly
    evaluated_points = tck(u)
    evaluated_derivatives = tck.derivative(1)(u)

    # Unpack the results
    xi, yi = evaluated_points[0,:], evaluated_points[1,:]
    dxi, dyi = evaluated_derivatives[0,:], evaluated_derivatives[1,:]
    
    
    # use image dimensions for clipping
    #   values below 0 -> 0
    #   values above the max for either dimension -> dim's max value
    # good_values: values > 0 = True
    rounded_coords = np.round(np.array([yi, xi])).astype(np.int16)
    
    # reminder: y : rows, x : columns

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

def build_curve_bridge(p_end, p_start, tangent_end, tangent_start, tangent_scale=0.3):
    """
    Completes a curve using a cubic Bezier bridge and pre-calculated tangents.
    
    Args:
        p_end (np.ndarray): 1x2 array for the end of the segment / 
                                starting point of the bridge (y, x).
        p_start (np.ndarray): 1x2 array for the start of the segment / 
                                  ending point of the brdige (y, x).
        tangent_end (np.ndarray): 1x2 array for the derivative at p_end (dy, dx).
        tangent_start (np.ndarray): 1x2 array for the derivative at p_start (dy, dx).
        tangent_scale (float): Controls influence of tangents on the curve.
    
    Returns:
        np.ndarray: The new points forming the bridge.
    """
    # 1. Normalize the provided tangents
    tangent_end = tangent_end / np.linalg.norm(tangent_end)
    tangent_start = tangent_start / np.linalg.norm(tangent_start)

    # 2. Calculate control points based on tangents
    dist = np.linalg.norm(p_start - p_end)
    p1 = p_end + tangent_end * dist * tangent_scale
    # Note: The tangent at the destination point (p_start)
    #       needs to point away from the curve.
    #       Since we are connecting the end of a path to its start,
    #       the tangent at the start point is already pointing in
    #       the correct "away" direction.
    p2 = p_start - tangent_start * dist * tangent_scale

    # 3. Generate the Bezier curve points
    t = np.linspace(0, 1, 50)[:, np.newaxis]
    bezier_bridge = (1-t)**3 * p_end + 3*(1-t)**2 * t * p1 + \
        3*(1-t) * t**2 * p2 + t**3 * p_start
    
    return bezier_bridge