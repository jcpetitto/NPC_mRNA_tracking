import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Your pipeline imports
from utils.npc_spline_refinement import extract_profile_along_norm
from tools.geom_tools import calc_tangent_endpts, get_u_range_from_bspline

# 1. Load checkpoint
checkpoint_path = Path("/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/checkpoints/BMY9999_99_99_9999/state_after_refinement.pkl")  # or partial_ch1_refine.pkl
with open(checkpoint_path, 'rb') as f:
    img_proc = pickle.load(f)

# 2. Get the data you need
ne_label = 12  # Your FoV label from the plot title
segment_key = "segment_0"  # Which segment

# Get refined splines and cropped images
refined_splines = img_proc._get_ch1_refined_bsplines()  # or ch2
cropped_imgs = img_proc._get_ch1_cropped_imgs()  # or ch2

# 3. Extract the specific segment
bspline_obj = refined_splines[ne_label][segment_key]
mean_ne_img = cropped_imgs[ne_label]

# 4. Pull intensity profiles
line_length = 10  # Your normal line length (pixels)
sampling_density = 0.1  # Your sampling density
n_samples_along_normal = 10  # Number of samples along each normal

u_values = get_u_range_from_bspline(bspline_obj, sampling_density)
segment_points_xy = bspline_obj(u_values)
segment_derivs_xy = bspline_obj(u_values, nu=1)

_, normal_endpoints = calc_tangent_endpts(
    segment_points_xy, segment_derivs_xy, line_length, True
)

intensity_profiles, dist_along_norm, validity_mask = extract_profile_along_norm(
    mean_ne_img, 
    normal_endpoints, 
    len(u_values), 
    n_samples_along_normal
)

# 5. Plot a specific profile (e.g., index 100 like your image)
profile_idx = 100
plt.figure(figsize=(10, 6))
plt.plot(dist_along_norm[profile_idx], intensity_profiles[profile_idx], 'ko', 
         label='Raw Data', markersize=4)
plt.xlabel("Distance along norm (pixels)")
plt.ylabel("Intensity (ADU)")
plt.title(f"Profile for NE {ne_label}, {segment_key}, Index {profile_idx}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()