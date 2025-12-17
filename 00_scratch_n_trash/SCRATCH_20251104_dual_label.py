import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_splprep, make_interp_spline, BSpline 
from scipy.optimize import minimize_scalar, minimize

from image_processor import ImageProcessor
from utils.npc_detect_initial import detect_npc
from utils.img_registration import image_registration
from tools.output_handling import load_json_experiments_data
from tools.utility_functions import config_from_file
from tools.geom_tools import points_by_arc_len_parm_robust, bspline_transformation

from utils.ne_dual_labels import match_ne_labels_by_iou

proc_config = config_from_file("config_options.json", ["pipe globals", "image processor"])
dual_img_exper_dict = load_json_experiments_data("yeast_output/dual_label/")

img_proc_instances = {}

for key, value in dual_img_exper_dict.items():
    img_proc_instances[key] = ImageProcessor(proc_config, value)
    img_proc_instances[key]._set_FoV_collection_dict(dual_img_exper_dict[key]) # these two dictionaries have the same keys
    # img_proc_instances[key].juxtapose_dual_labels(img_proc_instances[key]._get_FoV_collection_dict())

# ??? I need to revamp a bit whether the config contains this
# Error: Directory 'example_data/Example_raw_data/BMY823_7_25_23_aqsettings1_batchC/' not found.

# img_proc_instances['BMY1408_12_14_2023'].juxtapose_dual_labels(img_proc_instances['BMY1408_12_14_2023']._get_FoV_collection_dict())

FoV_dict = img_proc_instances['BMY1408_12_14_2023']._get_FoV_collection_dict()

frame_range = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['frame_range']
ne_trained_model = os.path.join(
    img_proc_instances['BMY1408_12_14_2023']._cfg['directories']['model root'],
    img_proc_instances['BMY1408_12_14_2023']._cfg['model_NE'])
current_device = img_proc_instances['BMY1408_12_14_2023']._current_device
masking_threshold = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['masking_threshold']
bbox_dim = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['bbox_dim']
use_merged_clusters = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['use_merged_clusters']
max_merge_dist = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['max_merge_dist']
plot_test_imgs = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['plot_test_imgs']
bspline_smoothing = img_proc_instances['BMY1408_12_14_2023']._cfg['ne_fit']['bspline_smoothing']

# ??? Future question: how could this be set up for more than 2 channels?
# PER EXPERIMENT
ch1_init_fit_dict = {}
ch2_init_fit_dict = {}
ch1_img_crop_dict = {}
ch2_img_crop_dict = {}
ch1_bspline_dict = {}
ch2_bspline_dict = {}


track_1_str = 'fn_track_ch1'
track_2_str = 'fn_track_ch2'


## TODO - do this BEFORE refining splines; pass culled FoV dictionaries that only containing matching label pairs; also, save label pairs as part of the img_proc
###
# Applying `find_all_best_matches` to the ch1 and ch2 dictionaries for every FoV in an experiment
##
crop_box_matches = {}

for FoV_id, crop_box in ch1_img_crop_dict.items():
    if FoV_id in ch2_img_crop_dict.keys():
        best_matches = match_ne_labels_by_iou(crop_box, ch2_img_crop_dict[FoV_id], 0.90)
        crop_box_matches.update({f'{FoV_id}': best_matches})

# ABOVE: returns the pairs of ne_label matches as a pair alongside the containing FoV_id
# REMEMBER: the numerical labels are NOT the same between channels (ex. 0191_05_ch1 != 0191_05_ch2); if they were, we wouldn't have to find matches by comparing crop boxes

###
# For an identified match, applying functions to the pair of associated bsplines
###

# TODO plot after adjusting re: registration
def plot_bspline_pair(ch1_bspline, ch2_bspline, N):
    u_values = np.linspace(0, 1, N) # equidistant parameters (NOT geometrically equidistant)

    # Evaluate each B-spline at the same x-values. This will return a (2, N) array.
    ch1_bspline_eval = test_ch1_ne_bspline(u_values)
    ch2_bspline_eval = test_ch2_ne_bspline(u_values)

    # Transpose the output and plot (y, x) because imaging
    y_ch1, x_ch1 = ch1_bspline_eval
    y_ch2, x_ch2 = ch2_bspline_eval

    # Create the plot
    plt.figure(figsize=(6, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the first B-spline
    plt.plot(x_ch1, y_ch1, color='blue', linestyle='-', label='Channel 1 B-spline')

    # Plot the second B-spline
    plt.plot(x_ch2, y_ch2, color='red', linestyle='--', label='Channel 2 B-spline')

    # Add plot labels, title, and a legend
    plt.title(f'B-spline Plot for FoV {test_FoV}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)


test_FoV = '0191'
test_pairs = crop_box_matches[f'{test_FoV}']
test_ch1_ne, test_ch2_ne = next(iter(test_pairs.items()))

test_ch1_ne_bspline = ch1_bspline_dict[f'{test_FoV}'][f'{test_ch1_ne}']

test_ch2_ne_bspline = ch2_bspline_dict[f'{test_FoV}'][f'{test_ch2_ne}']


N = 50
u_values = np.linspace(0, 1, N) 
equidist_params_ch1 = test_ch1_ne_bspline(u_values)
equidist_params_ch2 = test_ch2_ne_bspline(u_values)

equidist_points_ch1 = points_by_arc_len_parm_robust(test_ch1_ne_bspline, N)
equidist_points_ch2 = points_by_arc_len_parm_robust(test_ch2_ne_bspline, N)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(equidist_params_ch1[:,1], equidist_params_ch1[:,0], color='blue', marker='o')
ax1.scatter(equidist_params_ch2[:,1], equidist_params_ch2[:,0], color='red', marker='o')

ax2.scatter(equidist_points_ch1[:,1], equidist_points_ch1[:,0], color='blue', marker='o')
ax2.scatter(equidist_points_ch2[:,1], equidist_points_ch2[:,0], color='red', marker='o')
# PLOTTING - 2 spline objects
# TODO confirm this is the variable used for sampling in the initial fitting process
# ??? has this number been optimzied?

###
# Worked through example of registration and transformation to be turned into a function
###

# ??? FoV_id: 191, ne_id pair: 12 & 13 - will registering and aligning fix the obvious issue with alignment of the splines?
# TODO plot the images for the mentioned pair, calc reg etc.


ch1_crop_dim_dict = {key: value for key, value in ch1_img_crop_dict[test_FoV].items() if key in test_pairs.keys()}
ch2_crop_dim_dict = {key: value for key, value in ch2_img_crop_dict[test_FoV].items() if key in test_pairs.values()}

test_reg_mode1 = image_registration(FoV_dict[test_FoV], ch1_crop_dim_dict, ch2_crop_dim_dict, [0,250], 25, 5, 1000, 1)
test_reg_mode2 = image_registration(FoV_dict[test_FoV], ch1_crop_dim_dict, ch2_crop_dim_dict, [0,250], 25, 5, 1000, 1, reg_mode = 2)

# first member of the tuple contains the FoV registration
# second member of the tuple has the results for each individual ne label,
#   in this case those that have a "match" in channel 2)
# test_reg_mode1[1]

def make_matched_reg_dict(orig_reg_dict):
    # The set of keys we want to keep for the registration values
    registration_keys = {'scale', 'angle', 'shift_vector'}
    matched_reg_dict = {}

    for ne_id, ne_dict in orig_reg_dict[1].items(): # NOTE tuple
        new_dict = {key: value for key, value in ne_dict.items() if key in registration_keys}
        matched_reg_dict.update({f'{ne_id}': new_dict})
    
    return matched_reg_dict

matched_reg_mode1_dict = make_matched_reg_dict(test_reg_mode1)

matched_reg_mode2_dict = make_matched_reg_dict(test_reg_mode2)


# Calculate the difference in registration between the first and second sets of images
#   QC Check - filter via (difference in translation vectors between the first and second sets of brightfield images distance) < threshold

def calc_vtec_diff(vtec1, vtec2):
    vtech_diff = np.sqrt(abs(vtec1[0] - vtec2[0]) ** 2 + abs(vtec1[0] - vtec2[0]) ** 2)
    return vtech_diff

for key, value in matched_reg_mode1_dict.items():
    vtec_diff = calc_vtec_diff(
                        vtec1 = value['shift_vector'],
                        vtec2 = matched_reg_mode2_dict[key]['shift_vector']
                        )
    print(vtec_diff)

# matching is based on the crop boxes; however, if a bspline was not successfully fit to the cropboxes, it will not be included in the bspline dictionary
# want to apply the transformation to ch2 bsplines
test_bsplines_ch1 = {key: value for key, value in ch1_bspline_dict[test_FoV].items() if key in test_pairs.keys()}

test_bsplines_ch2 = {key: value for key, value in ch2_bspline_dict[test_FoV].items() if key in test_pairs.values()}

box_w, box_h = [75, 75]
center_of_rotation = [75/2, 75/2]

transformed_bspline_dict = {}
for ne_label, reg_values in matched_reg_mode1_dict.items():
    print(f"Transforming ch2 spline for ne label: {ne_label}")
    current_bspline = test_bsplines_ch2[test_pairs[ne_label]]
    # current_bspline_knots = current_bspline.t
    # new_ctrl_pts = transform_coordinates(current_bspline.c, reg_values['scale'], reg_values['angle'], reg_values['shift_vector'], [box_w/2, box_h/2])
    # current_bspline_degree = current_bspline.k
    # transformed_bspline = BSpline(current_bspline_knots, new_ctrl_pts, current_bspline_degree, extrapolate = 'periodic')
    transformed_bspline = bspline_transformation(current_bspline, reg_values, center_of_rotation)
    transformed_bspline_dict.update({f'{ne_label}': transformed_bspline})

    # GRAPH FOR TESTING
    N = 50
    u_values = np.linspace(0, 1, N) 

    equidist_points_orig = points_by_arc_len_parm_robust(current_bspline, N)
    equidist_points_transf = points_by_arc_len_parm_robust(transformed_bspline, N)

    fig, ax = plt.subplots()

    ax.scatter(equidist_points_orig[:,1], equidist_points_orig[:,0], color='blue', marker='o', alpha=0.5)
    ax.scatter(equidist_points_transf[:,1], equidist_points_transf[:,0], color='red', marker='o', alpha=0.5)


###
# END OF REGISTRATION TESTING
###


test_ne_bspline_ch1 = test_bsplines_ch1[test_ch1_ne]
test_ne_bspline_ch2 = transformed_bspline_dict[test_ch1_ne]

# Calculate the distance between curves
def find_min_distance_from_point_to_spline(point, bspline_obj):
    def objective_func(u):
        # 'splev' evaluates the spline at parameter 'u'
        curve_pt = bspline_obj(u)
        eucl_sq_dist = np.sum((np.array(point) - np.array(curve_pt))**2)
        return eucl_sq_dist
    opt_result = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')
    return opt_result

# Evaluate points on the first spline to sample from
num_points_to_sample = 50
sample_params = np.linspace(0, 1, num_points_to_sample)
points_on_spline1 = test_ne_bspline_ch1(sample_params)

points_on_spline2 = np.ones(points_on_spline1.shape)
distances = np.ones(points_on_spline1.shape[0])

for index, pt in enumerate(points_on_spline1):
    min_dist_opt_result = find_min_distance_from_point_to_spline(pt, test_ne_bspline_ch2)
    # 'x' is the optimal parameter 'u' from the result
    pt_from_opt_param = test_ne_bspline_ch2(min_dist_opt_result.x)
    # place the coord assoc w the opt param in the same index position in the points_on_spline2 array as the pt was located in the points_on_spline1 array
    points_on_spline2[index, :] = pt_from_opt_param
    distances[index] = np.sqrt(min_dist_opt_result.fun)

print("--- Distance Calculation Results ---")
print(f"Average distance between splines: {np.mean(distances):.4f}")
print(f"Maximum distance between splines: {np.max(distances):.4f}")
print("-" * 35)


fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(points_on_spline1[:, 0], points_on_spline1[:, 1], facecolors='none', edgecolors='blue', label='Channel 1 Points (Final Pos)', s=80)
ax.scatter(points_on_spline2[:, 0], points_on_spline2[:, 1], facecolors='none', edgecolors='red', label='Channel 2 Points (Final Pos)', s=80)

# Evaluate points for plotting
eval_params = np.linspace(0, 1, 200)
spline1_points_eval = test_ne_bspline_ch1(eval_params)
spline2_points_eval = test_ne_bspline_ch2(eval_params)

# Plot the stable SciPy splines
ax.plot(spline1_points_eval[:, 0], spline1_points_eval[:,1], color='blue', linewidth=2, label='Fitted Spline 1 (Final)')
ax.plot(spline2_points_eval[:,0], spline2_points_eval[:,1], color='red', linewidth=2, label='Fitted Spline 2 (Final)')

###
# PLOT INDIVIDUAL BSPLINE PAIR DISTANCES
###

# To avoid clutter, label every Nth point
label_every_n_points = 10

for i in range(num_points_to_sample):
    # Get the pair of corresponding points
    pt1 = points_on_spline1[i]
    pt2 = points_on_spline2[i]
    
    # Plot the connecting line between the pair of points
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='gray', linestyle='--', linewidth=0.8)

    # Add a text label for the distance at the line's midpoint
    if i % label_every_n_points == 0:
        mid_point = (pt1 + pt2) / 2
        ax.text(mid_point[0], mid_point[1], f'{distances[i]:.2f}', 
                color='black', 
                fontsize=8, 
                ha='center', 
                va='center',
                # Add a small white box behind the text for readability
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))


# Formatting
ax.set_title('B-Spline Fitting with SciPy (Stable)')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
ax.axis('equal')
plt.show()