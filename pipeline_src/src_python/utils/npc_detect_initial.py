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
from scipy.interpolate import make_splprep
from scipy.spatial.distance import cdist

# --- Included Modules --- # 
from utils.Neural_networks import Segment_NE
from tools.utility_functions import sigmoid, plot_points_on_image, coord_of_crop_box, find_segments
from tools.geom_tools import find_closest_u_on_spline, u_values_to_ranges, build_curve_bridge, angular_sort_alg, get_spline_arc_length, get_u_range_from_bspline


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
                                                            np.array([   [1, 1, 1],
                                                                        [1, 1, 1],
                                                                        [1, 1, 1]   ])
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
            segment_dictionary = \
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

        indiv_ne_bsplines.update({f'{ne_mask_label_str}': segment_dictionary})

#    return npc_init_fit_results, indiv_ne_cropped_imgs, indiv_ne_bsplines
    return indiv_ne_cropped_imgs, indiv_ne_bsplines, npc_labeled_img

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
def single_ne_init_fit(current_ne_label_img: np.ndarray, current_ne_intensity_mask: np.ndarray, use_skel: bool = True, init_sampling_density: int = 50, bspline_smoothing = 1.6, max_merge_dist: int = 10, min_peaks_for_hull: int = 4, min_vertices_for_spline: int = 4, use_merged_clusters: bool = True, plot_test_imgs = False, FoV_id = 'NNN', ne_mask_label_str = 'NN', id_suffix = ''):
    
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
    
    # --- Find key points (peaks) --- #
    if use_skel:
        skeleton_points = skeletonize(current_ne_label_img, method='lee') # Shape: (N, 2) as [y, x]
        skeleton_points = np.column_stack(np.where(skeleton_points))
        # grabbed the intensity peaks to see how they compare to the skeletonization points
        # could combined and weight, but (likely??) not necessary with spline refinement
        intensity_peaks = peak_local_max(current_ne_intensity_mask, min_distance=1, exclude_border=False, p_norm=2)
        points_for_spline = skeleton_points
    else:
        intensity_peaks = peak_local_max(current_ne_intensity_mask, min_distance=1, exclude_border=False, p_norm=2)
        points_for_spline = intensity_peaks
    
    # print(f"{prefix}_01_skel_peaks.png")
    # _debug_plot(current_ne_intensity_mask, skeleton_points, "Step 1: Skeletonize (compared to max intensity peaks)", f"{prefix}_01_skel_peaks.png", intensity_peaks)

    if (len(points_for_spline) < min_peaks_for_hull):
        raise ValueError("Not enough peaks to create hull.")
    try:
        # good_values - are within the boundaries of the mask
        xi, yi, good_values, u_range = initial_spline_fitting(current_ne_label_img, points_for_spline, init_sampling_density, bspline_smoothing, min_vertices_for_spline, prefix)
    
    except ValueError as e:
        raise e
    
    # --- Find contiguous segments that are actually on/within the mask --- #
    segment_indices = find_segments(good_values, True)

    if not segment_indices:
        raise ValueError("No valid segments found.")
    
    segment_spline_dict = {}
    # --- Fitting splines to individual segments without bridging --- #
    for i, (start, end) in enumerate(segment_indices):
        segment_label = f"segment_{i:0{2}}" # add leading zero if necessary
        start = int(start)
        end = int(end)

        seg_coords_yx = np.vstack((yi[start:end+1], xi[start:end+1])) # pull the coordinates of the segement using the start and end indices; (2, N)
        u_segment = u_range[start:end+1] # Get the *original* u-values for this segment

        # Need at least k+1 points to fit a spline (k=3, so 4 points)
        num_points = len(seg_coords_yx[1, :])
        if num_points < 4:
            print(f"  Skipping {segment_label}, not enough points (n={num_points}).")
            continue
            
        try:
            # Fit a non-periodic spline to *this segment only*
            # s=0 to interpolate (it was smoothed during its initial creation)
            bspline_obj, _ = make_splprep([seg_coords_yx[1, :], seg_coords_yx[0, :]], u=u_segment, s=0, k=3)

            # Store this bspline object in the spline segment dictionary
            segment_spline_dict.update({f'{segment_label}': bspline_obj})
            
            if plot_test_imgs:
                fig, ax = plt.subplots()
                ax.imshow(current_ne_label_img, cmap='inferno', origin='upper')
                # u_plot = np.linspace(0, 1, 100)
                plot_points = bspline_obj(u_segment) # because bspline_obj was created using (y,x), the returned y values will be in the 1st index (re: the image coordinate system)
                ax.plot(plot_points[0], plot_points[1], 'c-', linewidth=2)
                ax.scatter(seg_coords_yx[1, :], seg_coords_yx[0, :], c='cyan', s=10)
                ax.set_title(f"Initial Fit: {segment_label}")
                plt.savefig(f"output/{prefix}_{segment_label}_initial_fit.png")
                plt.close(fig)

        except Exception as e:
            print(f"  Warning: Spline fit failed for {segment_label}. {e}")
            continue
        if not segment_spline_dict:
            raise ValueError("No valid spline segments could be fitted.")
    return segment_spline_dict

def initial_spline_fitting(single_ne_img: np.ndarray, initial_peaks_yx: np.ndarray, init_sampling_density: int, bspline_smoothing, min_vertices: int, prefix = '', segment_label = ''):
    
    def plot_hull_on_peaks(image, initial_peaks_yx, sorted_hull_points_yx, title="Convex Hull on Initial Peaks", save_path=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='inferno', origin='upper', alpha=0.7)
        
        # 1. Plot ALL initial peaks in one color
        #    (Assumes initial_peaks_yx is (y, x))
        ax.scatter(initial_peaks_yx[:, 1], initial_peaks_yx[:, 0], c='blue', s=10, alpha=0.3, label='All Peaks')
        
        # 2. Get the hull points and plot them in a different color
        #    hull.vertices are indices *into* initial_peaks_yx

        ax.scatter(sorted_hull_points_yx[:, 1], sorted_hull_points_yx[:, 0], c='cyan', s=20, label='Hull Vertices')
        
        # 3. (Optional) Draw the lines connecting the hull vertices in order
        #    We must add the first point to the end to close the loop
        for i in range(len(sorted_hull_points_yx)):
            p1 = sorted_hull_points_yx[i]
            p2 = sorted_hull_points_yx[(i + 1) % len(sorted_hull_points_yx)]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-', linewidth=1) # Plot (x, y)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    # ne_img_dim: tuple (height, width) AKA (rows, columns)
    ne_img_dim = (single_ne_img.shape)

    hull = scipy.spatial.ConvexHull(initial_peaks_yx) # initial_peaks is [y, x] -> ConvexHull is as well
    
    if len(hull.vertices) < min_vertices:
        raise ValueError(f"Fewer than {min_vertices} convex hull vertices found.")
    
    hull_points_yx = initial_peaks_yx[hull.vertices]
    sorted_indices = angular_sort_alg(hull_points_yx)
    sorted_hull_points_yx = hull_points_yx[sorted_indices]
    # make_: periodic spline (think: closed shapes)
    #       - smoothing factor (s) of 0 forces interpolation through points 
    #       - expects [x, y]
    # ??? Why are points lost somewhere re: that pesky broken-jaw of a PacMan (s=0 still loses these spline points)
    # if(plot_test_imgs):
    #     plot_hull_on_peaks( single_ne_img, initial_peaks_yx, sorted_hull_points_yx, \
    #             title=f"Hull Fit: {prefix}", save_path=f"output/{prefix}__hull_fit.png")

    num_hull_points = len(sorted_hull_points_yx)
    u_parameter_vector = np.linspace(0, 1, num=num_hull_points)

    # stored [y (row), x (col)], need to use [x (col), y (row)]
    init_bspline, _ = make_splprep(
            [sorted_hull_points_yx[:, 1], sorted_hull_points_yx[:, 0]],
            u = u_parameter_vector,
            k = 3,
            s = bspline_smoothing
        )
    # SELF - note use of density and arc length via integration rather than depending on equidistance in parameter space
    u_range = get_u_range_from_bspline(init_bspline, init_sampling_density)

    # Evaluate the spline object directly
    evaluated_points_xy = init_bspline(u_range)

    # Unpack the results
    xi, yi = evaluated_points_xy[0,:], evaluated_points_xy[1,:]    
    
    # use image dimensions for clipping
    #   values below 0 -> 0
    #   values above the max for either dimension -> dim's max value
    # good_values: values > 0 = True
    rounded_coords_yx = np.round(np.array([yi, xi])).astype(np.int16)
    
    # reminder: y : rows, x : columns
    # This is how rounded_coords_yx is formatted, making index 0 y and index 1 x for indexing re: assigning good_values (below)

    # if spline evaluates to points beyond the img limits,
    #   np.clip forces them back into range by assigning such points
    #   the value at the boundary
    good_values = single_ne_img[np.clip(rounded_coords_yx[0, :], 0, ne_img_dim[0] - 1),
                                np.clip(rounded_coords_yx[1, :], 0, ne_img_dim[1] - 1)] > 0
    

    return xi, yi, good_values, u_range
    


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