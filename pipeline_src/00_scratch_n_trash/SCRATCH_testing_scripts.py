#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 08:49:40 2025

@author: jctourtellotte
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk


from utils.npc_detection import single_ne_init_fit

# --- Create a Sample Image for Testing ---
# A hollow circle with a break in it, to test segment clustering
image = np.zeros((300, 300), dtype=np.uint8)
rr, cc = disk((150, 150), 100, shape=image.shape)
image[rr, cc] = 255
rr, cc = disk((150, 150), 85, shape=image.shape)
image[rr, cc] = 0
image[140:160, 240:] = 0 # Create a break in the ring

# --- Run the Pipeline ---
final_splines, final_derivs = single_ne_init_fit(
    image,
    max_merge_dist=30, # A smaller distance for this example
    init_spline_sampling=500
)

# --- Visualize the Results ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(image, cmap='gray', origin='lower')

# Plot the final splines
if final_splines:
    for i, spline_pts in enumerate(final_splines):
        ax.plot(spline_pts[0, :], spline_pts[1, :], '-', lw=2, label=f'Final Spline Cluster {i+1}')
    ax.legend()
else:
    ax.set_title("No final splines were fitted.")

# For more detailed debugging, you would plot intermediate steps:
# - The `peaks` from skeletonization
# - The `initial_spline_coords`
# - Each identified `segment_coords` in a different color

ax.set_aspect('equal')
plt.show()


# -- 2025-07-09 -- #
refined_fit_list = img_proc._get_ne_refined_fit() # list of dictionaries
refined_fit_list[0].keys() # dict_keys(['FoV_id', 'refined_fit'])
refined_fit = refined_fit_list[0]['refined_fit'] # entry for 1st FoV
single_ne_refined_fit = refined_fit[0] # entry for 1st mask label
single_ne_refined_fit.keys() # dict_keys(['ne_mask_label', 'parameters', 'param_history'])
single_ne_params = single_ne_refined_fit['parameters']
single_ne_param_history = single_ne_refined_fit['param_history']

import torch
batch_size = 1000
iterations = 300
device = img_proc._current_device

# finding the iteration(s) for which each individual parameters equal zero
#   (rather than when they all equal zero)
param_history_permuted = single_ne_param_history.permute(1, 0, 2)

zero_mask = (param_history_permuted[:, :, 0] == 0)
# Find first True in each row (batch)
first_zero_idx = zero_mask.float().argmax(dim=1)
# Check if there actually was a zero (vs argmax returning 0 for all False)
has_zero = zero_mask.any(dim=1)
# Set iterations_vector
iterations_vector = torch.where(has_zero, first_zero_idx.float(), torch.tensor(iterations, dtype=torch.float32, device=device))

# conversions for testing
params_tensor_np = params_tensor.detach().numpy()


# %%
# 2025-07-09
ne_refined_fit = img_proc._get_ne_refined_fit()
ne_refined_fit[0]['refined_fit'][0].keys()

# %%
# 2025-07-10
# goal: dynamic registration precision -> threshold based on registration data

# need: to be able to do this on the cluster (jupyter?) in the lab folder before pipeline is complete
#       upside - will speak to modularity
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor

pipeline_for_reg = ImagingPipeline()
    
pipeline_for_reg.load_config_file('config_options.json')

img_proc_for_reg = ImageProcessor(config_dict = pipeline_for_reg.get_config(),
                          device = pipeline_for_reg.get_device())

img_proc_for_reg.determine_responsivity()

img_proc_for_reg.register_images()
