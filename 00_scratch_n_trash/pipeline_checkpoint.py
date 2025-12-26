"""
PIPELINE CHECKPOINT FILE
========================
This file restores full context for the yeast imaging pipeline.
Includes:
1. Full import paths and function signatures.
2. Minimal runnable skeleton.
3. Config templates for GLRT, refinement, bridging.
4. Restart logic for refinement and bridging.
5. README-style summary of pipeline stages.
"""

# =============================
# IMPORTS
# =============================
import os
import torch
import numpy as np

# Core pipeline modules
from main_cluster import run_main_cluster
from imaging_pipeline import ImagingPipeline
from image_processor import ImageProcessor

# Detection & Segmentation
from npc_detect_initial import detect_initial_ne_labels
from Neural_networks import Segment_NE

# Refinement
from npc_spline_refinement import NESplineRefiner
from ne_fit_utils import npcfit_class
from MLEstimation import richards_curve_gaussian_vectorized

# Bridging & Geometry
from spline_bridging import bridge_refined_splines
from geom_tools import calc_bspline_curvature, reconstruct_periodic_spline

# Dual-label Analysis
from ne_dual_labels import match_ne_labels_by_iou, calc_dual_distances

# Calibration & Registration
from responsivity import compute_responsivity
from img_registration import image_registration

# Particle Detection & GLRT
from particle_detection import detect_spots_with_glrt
from GLRT import glrtfunction

# PSF Utilities
from psf_fit_utils import Gaussian2D_IandBg

# Utilities
from utility_functions import config_from_file, dict_to_json

# =============================
# CONFIG TEMPLATES
# =============================
CONFIG_TEMPLATE = {
    "glrt_multichannel": {
        "roi_size": 16,
        "time_points_per_roi": 20,
        "frame_batch_size": 100,
        "spatial_batch_size": 256,
        "alpha": 0.05,
        "sigma": 1.5,
        "iterations": 50
    },
    "spline_refinement": {
        "model": "richards_gaussian",
        "sampling_density": 0.5,
        "noise_multiplier": 0.01,
        "timeout": 120
    },
    "bridging": {
        "run_bezier_bridging": True,
        "bspline_smoothing": 1.6,
        "max_merge_dist": 10
    },
    "registration": {
        "precision_threshold": 0.8,
        "max_iterations": 100
    }
}

# =============================
# PIPELINE SKELETON
# =============================
def run_pipeline(config_path: str):
    """Minimal runnable skeleton for pipeline execution."""
    config = config_from_file(config_path)
    pipeline = ImagingPipeline(config)

    for experiment in pipeline.experiments:
        processor = ImageProcessor(experiment, config)

        # Stage 1: Initial NE detection
        detect_initial_ne_labels(processor)

        # Stage 2: Registration
        image_registration(processor)

        # Stage 3: Refinement (with restart logic)
        for attempt in range(3):
            try:
                NESplineRefiner(processor).refine()
                break
            except TimeoutError:
                print(f"Refinement attempt {attempt+1} failed. Retrying...")

        # Stage 4: Bridging (with restart logic)
        for attempt in range(2):
            try:
                bridge_refined_splines(processor)
                break
            except Exception as e:
                print(f"Bridging attempt {attempt+1} failed: {e}")

        # Stage 5: Dual-label analysis
        calc_dual_distances(processor)

        # Stage 6: GLRT particle detection
        detect_spots_with_glrt(processor.image_stack, processor.background_model, torch.device('cuda'), config)

        print(f"Experiment {experiment} completed.")

# =============================
# README SUMMARY
# =============================
"""
PIPELINE FLOW:
1. Initial NE detection → npc_detect_initial.py
2. Registration → img_registration.py
3. Spline refinement → npc_spline_refinement.py
4. Bridging → spline_bridging.py
5. Dual-label analysis → ne_dual_labels.py
6. GLRT particle detection → particle_detection.py

RESTART POINTS:
- Refinement: 3 attempts with timeout handling.
- Bridging: 2 attempts with exception handling.

QC HOOKS:
- Visualization functions in spline_bridging.py and geom_tools.py.
- Registration QC in img_registration.py.
- GLRT QC via likelihood ratio plots.
"""
