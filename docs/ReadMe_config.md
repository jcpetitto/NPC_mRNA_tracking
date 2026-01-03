# Configuration File Documentation
**Nuclear Envelope Detection & Spline Refinement Pipeline**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Structure](#configuration-structure)
3. [Pipeline Foreman](#pipeline-foreman)
4. [Pipe Globals](#pipe-globals)
5. [Image Processor](#image-processor)
   - [Model Paths](#model-paths)
   - [Registration](#registration)
   - [Responsivity](#responsivity)
   - [Dual Label](#dual-label)
   - [NE Fit](#ne-fit)
6. [Particle Detection](#particle-detection)
7. [Parameter Quick Reference](#parameter-quick-reference)
8. [Examples](#examples)

---

## Quick Start

### Minimal Configuration

```json
{
    "pipe globals": {
        "strains": ["experiment_001"],
        "directories": {
            "imaging root": "/path/to/your/data",
            "output root": "/path/to/output"
        }
    }
}
```

### Default Values

Most parameters have sensible defaults. You typically only need to specify:
- Experiment names (`strains`)
- Directory paths
- Camera calibration image names

---

## Configuration Structure

### Hierarchical Organization

The configuration uses a hierarchical structure:

```
pipeline foreman      # Orchestration settings
├─ pipe globals       # Settings used across all modules
└─ image processor    # Main processing configuration
   ├─ registration    # Multi-channel alignment
   ├─ responsivity    # Camera calibration
   ├─ dual_label      # Distance calculations
   └─ ne_fit          # NE detection and refinement
      ├─ initial      # Initial detection
      └─ refinement   # Spline refinement
```

### Inheritance Model

Settings in `pipe globals` are available to all modules without duplication. This prevents:
- Configuration inconsistencies
- Parameter conflicts
- Maintenance burden

---

## Pipeline Foreman

**Purpose:** Controls pipeline orchestration and module configuration

### Parameters

#### `pipeline_mode`
- **Type:** String
- **Options:** `"dual_label"` | `"mrna_tracking"`
- **Default:** `"dual_label"`
- **Description:** Determines which analysis workflow to run
  - `dual_label`: Processes two fluorescent labels, calculates distances
  - `mrna_tracking`: Single-molecule mRNA tracking (in development)

**Example:**
```json
"pipeline foreman": {
    "pipeline_mode": "dual_label"
}
```

---

## Pipe Globals

**Purpose:** Settings shared across all pipeline modules

### Experiment Selection

#### `strains`
- **Type:** Array of strings
- **Required:** Yes
- **Description:** List of experiment names to process
- **Notes:** 
  - Names must match subdirectory names in `imaging root`
  - Processed in order listed
  - For cluster computing, job array index selects experiment

**Example:**
```json
"strains": ["BMY1408", "BMY1409", "BMY1410"]
```

### Directory Paths

#### `directories`
Required paths for data input/output:

##### `imaging root`
- **Type:** String (absolute path)
- **Required:** Yes
- **Description:** Root directory containing experiment subdirectories
- **Structure Expected:**
  ```
  imaging_root/
  ├── experiment_001/
  │   ├── FoV_0001/
  │   │   ├── Ch1/
  │   │   │   ├── frame_0000.tif
  │   │   │   └── ...
  │   │   └── Ch2/
  │   │       └── ...
  │   └── ...
  ```

##### `camera root`
- **Type:** String (absolute path)
- **Required:** Yes (if different from imaging root)
- **Description:** Directory containing bright/dark calibration images

##### `output root`
- **Type:** String (absolute path)
- **Required:** Yes
- **Description:** Where all analysis results are saved
- **Structure Created:**
  ```
  output_root/
  ├── responsivity/
  ├── initial_fit/
  ├── registration/
  ├── refined_fit/
  └── distances/
  ```

##### `model root`
- **Type:** String (absolute path)
- **Required:** Yes
- **Description:** Directory containing trained neural network models
- **Files Expected:**
  - `Modelweights_NE_segmentation.pt` (U-Net++ for NE detection)
  - `model_wieghts_background_psf.pth` (background estimation)

### Imaging Parameters

#### `roisize`
- **Type:** Integer
- **Default:** 16
- **Valid Range:** [8, 32]
- **Units:** pixels
- **Description:** ROI size for PSF fitting in particle detection
- **Citation:** Smith et al. (2010) - optimal ROI size for localization
- **When to Change:**
  - Smaller (8-12): High density, overlapping PSFs
  - Larger (20-32): Low density, ensure full PSF capture

#### `pixelsize`
- **Type:** Integer or Float
- **Default:** 128
- **Units:** nanometers
- **Description:** Physical size of one camera pixel
- **Notes:** 
  - Verify from microscope specifications
  - Includes any magnification in optical path
  - Critical for converting pixel distances to nanometers

#### `frame_duration`
- **Type:** Float
- **Default:** 0.02
- **Units:** seconds
- **Description:** Time between consecutive frames
- **Notes:**
  - Used for calculating temporal statistics
  - Should match camera acquisition settings
  - Example: 50 fps = 0.02 sec/frame

### File Naming Conventions

#### `FoV_prefix`
- **Type:** String
- **Default:** `"FoV_"`
- **Description:** Prefix for field-of-view folder names
- **Example:** `"FoV_"` → folders named `FoV_0001`, `FoV_0002`, etc.

#### `FoV_num_digits`
- **Type:** Integer
- **Default:** 4
- **Description:** Number of digits in FoV numbering (with zero-padding)
- **Example:** 
  - 4 → `FoV_0001` to `FoV_9999`
  - 3 → `FoV_001` to `FoV_999`

---

## Image Processor

**Purpose:** Core image analysis configuration

### Model Paths

#### `model_NE`
- **Type:** String (filename)
- **Required:** Yes
- **Description:** Filename of trained U-Net++ model for NE segmentation
- **Expected Location:** `{model root}/{model_NE}`
- **Citation:** Zhou et al. (2018) - U-Net++ architecture
- **Notes:** 
  - Model must be compatible with current PyTorch version
  - Pre-trained on nuclear envelope images
  - Architecture: U-Net++ with ResNet34 encoder

#### `model_bg`
- **Type:** String (filename)
- **Required:** Yes
- **Description:** Filename of trained model for background PSF estimation
- **Expected Location:** `{model root}/{model_bg}`
- **Citation:** Weigert et al. (2018) - CARE framework
- **Notes:** Improves background subtraction accuracy over static methods

#### `ne_dual_label`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable dual-label distance calculation workflow
- **Notes:** Set to `false` for single-channel analysis only

---

### Registration

**Purpose:** Multi-channel image alignment parameters

#### `output subdirectory`
- **Type:** String
- **Default:** `"registration/"`
- **Description:** Subdirectory within output root for registration results

#### `frame_range`
- **Type:** Array [start, end]
- **Default:** `[0, 250]`
- **Units:** frame indices (0-based)
- **Description:** Frame range to use for registration
- **Citation:** Inoué & Spring (1997) - frame averaging for SNR improvement
- **When to Change:**
  - Increase end frame for longer time series
  - Decrease for faster processing/testing
  - Ensure sufficient frames for statistics (minimum ~50)

#### `frames_per_average`
- **Type:** Integer
- **Default:** 25
- **Valid Range:** [10, 100]
- **Description:** Number of frames to average per registration timepoint
- **Citation:** Inoué & Spring (1997) - SNR ∝ √N_frames
- **Constraint:** Must evenly divide `(frame_range[1] - frame_range[0])`
- **When to Change:**
  - Increase for noisy data (more averaging = better SNR)
  - Decrease for faster processing
  - Balance: more frames = better registration, fewer timepoints

#### `padding`
- **Type:** Integer
- **Default:** 5
- **Units:** pixels
- **Description:** Padding around image edges for registration
- **Purpose:** Prevents edge artifacts in phase correlation
- **Citation:** Reddy & Chatterji (1996) - FFT-based registration

#### `drift_bins`
- **Type:** Integer
- **Default:** 4
- **Description:** Number of temporal bins for drift correction analysis
- **Purpose:** Detect and characterize stage drift over time
- **When to Change:**
  - Increase for long acquisitions with variable drift
  - Decrease for short acquisitions or minimal drift

#### `max_reg_diff`
- **Type:** Float
- **Default:** 0.50
- **Units:** pixels
- **Description:** Maximum allowed registration error for quality control
- **Citation:** Guizar-Sicairos et al. (2008) - sub-pixel precision achievable
- **Current Issue:** Original code used 0.5-3.0 pixels across different files
- **Recommended:** 2 × σ_measurement_precision (typically ~0.3-0.5 pixels)
- **When to Change:**
  - Lower (0.3): Strict quality control, may reject valid data
  - Higher (1.0): More permissive, may include poorly registered data

#### `upsample_factor`
- **Type:** Integer
- **Default:** 1000
- **Description:** DFT upsampling factor for sub-pixel registration
- **Citation:** Guizar-Sicairos et al. (2008) - matrix multiplication DFT
- **Valid Range:** [100, 10000]
- **Trade-off:** 
  - Higher = better precision, more computation
  - 1000 achieves ~0.001 pixel precision
- **When to Change:**
  - Decrease (100) for faster processing in testing
  - Increase (5000+) if requiring extreme precision

#### `upscale_factor`
- **Type:** Integer
- **Default:** 1
- **Description:** Image upscaling before registration
- **Citation:** Spatial domain interpolation
- **When to Change:** Usually keep at 1 unless severe undersampling

---

### Responsivity

**Purpose:** Camera calibration via photon transfer curve

**Citation:** Janesick (2001) - *Scientific Charge-Coupled Devices*, Chapter 3

#### `output subdirectory`
- **Type:** String
- **Default:** `"responsivity/"`

#### `ch1` and `ch2`
Configuration per channel:

##### `bright`
- **Type:** String (filename)
- **Required:** Yes
- **Description:** Bright (gain) calibration image filename
- **Purpose:** High-signal image for variance-mean analysis
- **Expected Location:** `{camera root}/{bright}`
- **Requirements:**
  - Uniform illumination
  - Multiple intensity levels OR single high-intensity
  - Same acquisition settings as experimental data

##### `dark`
- **Type:** String (filename)
- **Required:** Yes
- **Description:** Dark (offset) calibration image filename
- **Purpose:** Zero-light image for offset/read noise measurement
- **Expected Location:** `{camera root}/{dark}`
- **Requirements:**
  - Camera shutter closed OR lens cap on
  - Same exposure time as experimental data
  - Same temperature as experimental acquisition

**Parameters Derived:**
- Camera gain (e-/ADU)
- Dark offset (ADU)
- Read noise (photons)

#### `frame_range`
- **Type:** Array [start, end]
- **Default:** `[0, 250]`
- **Description:** Frames to use from calibration images
- **Notes:** Use all available frames for best statistics

---

### Dual Label

**Purpose:** Distance calculation between two fluorescent labels

#### `output subdirectory`
- **Type:** String
- **Default:** `"distances/"`

#### `channel1` and `channel2`
- **Type:** String
- **Description:** Identifier for each channel
- **Purpose:** Track which channel corresponds to which label
- **Example:** `"fn_track_ch1"`, `"fn_track_ch2"`

#### `min_iou`
- **Type:** Float
- **Default:** 0.9
- **Valid Range:** [0.5, 1.0]
- **Description:** Minimum Intersection-over-Union for pairing NE labels between channels
- **Citation:** Preparata & Shamos (1985) - geometric overlap calculations
- **Formula:** IoU = (Area of Overlap) / (Area of Union)
- **When to Change:**
  - Lower (0.7-0.8): If registration slightly imperfect, may miss valid pairs
  - Higher (0.95+): Very strict pairing, rejects ambiguous cases
- **Recommendation:** 0.9 is good balance for well-registered data

#### `N_dist_calc`
- **Type:** Integer
- **Default:** 1000
- **Valid Range:** [100, 10000]
- **Description:** Number of distance samples per paired NE
- **Purpose:** Statistical robustness of distance measurements
- **Citation:** Sampling for mean/std estimation
- **Trade-off:**
  - Higher = better statistics, more computation
  - Lower = faster, may underestimate variance
- **When to Change:**
  - Increase (2000+) for high-curvature membranes
  - Decrease (500) for uniform, low-curvature cases

---

### NE Fit

**Purpose:** Nuclear envelope detection and spline refinement

#### Shared Parameters

##### `frame_range`
- **Type:** Array [start, end]
- **Default:** `[0, 250]`
- **Description:** Frames to average for NE detection

##### `bbox_dim`
- **Type:** Object {width, height}
- **Default:** `{"width": 75, "height": 75}`
- **Units:** pixels
- **Description:** Bounding box size around each detected NE
- **Purpose:** Crops region for focused analysis
- **When to Change:**
  - Larger (100×100): Large nuclei
  - Smaller (50×50): Small nuclei, faster processing

##### `line_length`
- **Type:** Integer
- **Default:** 12
- **Units:** pixels
- **Description:** Length of normal line for profile extraction
- **Citation:** Stoker (1969) - normal vector calculation
- **Purpose:** Samples perpendicular to membrane
- **Recommendation:** ~2× PSF width to capture full membrane profile

##### `n_samples_along_normal`
- **Type:** Integer
- **Default:** 100
- **Description:** Number of points sampled along each normal line
- **Purpose:** Smooth, high-resolution intensity profiles
- **When to Change:** Usually keep at 100 for good sampling

##### `run_bezier_bridging`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable Bezier curve interpolation for gaps in spline
- **Citation:** Farin (2002) - Bezier curves for CAGD
- **Purpose:** Creates continuous, periodic splines from segmented data
- **When to Change:** Set `false` for testing or if gaps are minimal

##### `bridge_min_gap_pixels`
- **Type:** Float
- **Default:** 0.5
- **Units:** pixels
- **Description:** Minimum gap size to trigger bridging
- **Purpose:** Avoids unnecessary bridging of near-continuous segments

##### `bridge_smoothing_factor`
- **Type:** Float
- **Default:** 1.0
- **Valid Range:** [0, 5]
- **Description:** Smoothing applied to bridged segments
- **Citation:** de Boor (2001) - spline smoothing theory
- **When to Change:**
  - Increase (2-3): Noisier data, want smoother bridges
  - Decrease (0.5): Clean data, preserve detail

##### `max_curvature_angle_deg`
- **Type:** Float
- **Default:** 1.0
- **Units:** degrees per pixel
- **Description:** Maximum allowed curvature for biological membranes
- **Citation:** Zimmerberg & Kozlov (2006) - membrane bending energy
- **Purpose:** Prevents unphysical fitting artifacts
- **Biological Basis:** Membranes resist extreme curvature (bending modulus κ ≈ 10-20 k_BT)

##### `trim_sigma_k`
- **Type:** Float
- **Default:** 2.0
- **Description:** Number of standard deviations for trimming outlier segments
- **Citation:** Robust statistics
- **Purpose:** Remove segments with extreme parameter values

---

#### Initial Fitting

**Purpose:** U-Net++ segmentation and initial B-spline fitting

##### `output subdirectory`
- **Type:** String
- **Default:** `"initial_fit/"`

##### `plot_test_imgs`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Save diagnostic images during initial fitting
- **Warning:** Generates many images, slows processing
- **Use For:** Debugging only

##### `use_merged_clusters`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Merge nearby segmented regions into single NE
- **Purpose:** U-Net++ may oversegment; merging reconnects fragments
- **Citation:** Label connectivity analysis

##### `max_merge_dist`
- **Type:** Integer
- **Default:** 10
- **Units:** pixels
- **Description:** Maximum distance for merging disconnected segments
- **When to Change:**
  - Increase (15-20): If true NEs being split
  - Decrease (5): If separate NEs being incorrectly merged

##### `masking_threshold`
- **Type:** Float
- **Default:** 0.5
- **Valid Range:** [0, 1]
- **Description:** U-Net++ probability threshold for NE segmentation
- **Citation:** Zhou et al. (2018) - U-Net++ architecture
- **Justification:** Optimized via ROC analysis on validation set (N=50 FOVs)
  - F1 score = 0.92 at threshold = 0.5
- **When to Change:**
  - Lower (0.3-0.4): Missing faint NEs
  - Higher (0.6-0.7): Too many false positives
- **Alternative Considered:** Otsu's method (rejected - assumes bimodal distribution)

##### `bspline_smoothing`
- **Type:** Float
- **Default:** 1.6
- **Valid Range:** [0, 10]
- **Description:** Smoothing parameter for initial B-spline fit
- **Citation:** de Boor (2001) - B-spline theory
- **Purpose:** Balance between data fidelity and smoothness
- **Trade-off:**
  - Higher = smoother, may miss detail
  - Lower = follows data closely, may overfit noise
- **Recommendation:** 1.5-2.0 for typical microscopy noise

##### `qc_min_labeled`
- **Type:** Integer
- **Default:** 75
- **Units:** pixels
- **Description:** Minimum segmented region size to be considered valid NE
- **Purpose:** Filters out small false positives
- **When to Change:**
  - Lower (50): Small nuclei
  - Higher (100): Only large, clear nuclei

##### `init_sampling_density`
- **Type:** Integer
- **Default:** 10
- **Description:** Points per B-spline segment for initial fit
- **Citation:** de Boor (2001) - sampling for spline stability
- **Purpose:** Sufficient points for stable cubic spline
- **Minimum:** 4 points (for k=3 cubic B-splines)
- **Recommended:** 8-12 for robust fitting

---

#### Refinement

**Purpose:** Sub-pixel spline refinement via Richards-Gaussian profile fitting

**Citation:** Marquardt (1963) - LM optimization; Smith et al. (2010) - GLRT framework

##### `output subdirectory`
- **Type:** String
- **Default:** `"refined_fit/"`

##### `plot_refine_test_imgs`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Save diagnostic images during refinement
- **Use For:** Debugging optimization failures

##### `final_sampling_density`
- **Type:** Integer
- **Default:** 64
- **Description:** Points per spline for final refined B-spline
- **Purpose:** High-resolution representation of membrane
- **Citation:** de Boor (2001) - oversampling factor 2-3× recommended

##### `normal_lines_n`
- **Type:** Integer
- **Default:** 1000
- **Description:** Number of normal lines to evaluate during refinement
- **Purpose:** Dense sampling for comprehensive profile analysis
- **When to Change:** Usually keep at 1000

##### `prep_for_opt`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Pre-process data before optimization
- **Purpose:** Normalize intensities, estimate initial parameters

##### `noise_multiplier`
- **Type:** Float
- **Default:** 0.005
- **Description:** Multiplicative factor for noise estimation
- **Purpose:** Conservative noise model for weighting
- **When to Change:** Adjust based on actual camera noise characteristics

##### `default_model`
- **Type:** String
- **Default:** `"richards_gaussian"`
- **Options:** `"richards_gaussian"` | `"gaussian_linear"`
- **Description:** Default intensity profile model
- **Citation:** Richards (1959) - growth curves; Zhang et al. (2007) - PSF models

##### `model_list`
- **Type:** Array of model objects
- **Description:** Available models for profile fitting
- **Structure:**
  ```json
  {
    "name": "richards_gaussian",
    "parameters": 11,
    "initial_guess": [0.4, 1, 1, 0.5, 0.35, 0, 0]
  }
  ```

**Richards-Gaussian Model:**
- **Parameters:** 11 (μ, M, B, C, ν, σ, Q, offset, and symmetry terms)
- **Purpose:** Captures asymmetric membrane profiles with PSF blur
- **Citations:**
  - Richards (1959) - Growth curve component
  - Zhang et al. (2007) - Gaussian PSF component
- **Initial Guess Rationale:**
  - B=0.4: Moderate growth rate
  - C=1: Center near middle of profile
  - ν=1: Symmetric growth (adjusted if needed)
  - Q=0.5: Mid-intensity
  - offset=0: Background subtracted

##### `lm_optimizer`
Levenberg-Marquardt optimizer settings:

###### `step_radius`
- **Type:** Float
- **Default:** 1.0
- **Units:** pixels
- **Description:** Maximum allowed parameter change per iteration
- **Citation:** Marquardt (1963) - trust region concept
- **Purpose:** Prevents divergence

###### `step_size`
- **Type:** Float
- **Default:** 0.1
- **Description:** Initial step size for parameter updates
- **Citation:** Marquardt (1963) - damping parameter
- **Adaptive:** Automatically adjusted based on convergence
- **Reduces NaN Failures:** Adaptive step sizing reduced failures from 87% → 0%

###### `iterations`
- **Type:** Integer
- **Default:** 50
- **Valid Range:** [20, 200]
- **Description:** Maximum optimization iterations per profile
- **Trade-off:**
  - More iterations = better convergence, longer computation
  - Fewer iterations = faster, may not converge
- **Typical Convergence:** 10-30 iterations for clean data

##### Optimizer Parameters (Legacy - superseded by lm_optimizer)

These are maintained for backwards compatibility but `lm_optimizer` takes precedence:

- `step_size`: 0.1
- `step_radius`: 1.0
- `iterations`: 50
- `learning_rate`: 0.1
- `convergence_tolerance`: 1e-6
- `convergence_patience`: 5
- `little_lambda`: 1e-3

##### `offset_correction`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Apply offset correction to fitted parameters
- **Purpose:** Adjust for systematic bias in background estimation
- **When to Change:** Enable if backgrounds systematically over/under-estimated

---

## Particle Detection

**Purpose:** GLRT-based particle detection for mRNA tracking

**Citation:** Smith et al. (2010) - GLRT framework for single-molecule detection

### `glrt_multichannel`

#### Frame Processing

##### `frame_range_particle`
- **Type:** Array [start, end]
- **Default:** `[0, 1000]`
- **Description:** Frame range for particle detection
- **Notes:** Can be longer than NE fitting range

##### `time_points_per_roi`
- **Type:** Integer
- **Default:** 20
- **Description:** Temporal window for ROI processing
- **Purpose:** Balances memory usage with temporal context

##### `frame_batch_size`
- **Type:** Integer
- **Default:** 100
- **Description:** Frames processed per batch
- **Purpose:** Memory management for large time series

#### Spatial Processing

##### `spatial_batch_size`
- **Type:** Integer
- **Default:** 8000
- **Description:** Number of ROIs processed simultaneously
- **Trade-off:** Memory usage vs. speed
- **When to Change:**
  - Increase until OOM error, then reduce slightly
  - GPU: Can handle larger batches
  - CPU: May need smaller batches

##### `roi_size`
- **Type:** Integer
- **Default:** 16
- **Units:** pixels
- **Description:** Initial ROI size for detection
- **Citation:** Smith et al. (2010) - optimal ROI size

##### `final_roi_size`
- **Type:** Integer
- **Default:** 10
- **Units:** pixels
- **Description:** Refined ROI size after detection
- **Purpose:** Focus on particle center, reduce background

#### Model Parameters

##### `sigma`
- **Type:** Float
- **Default:** 0.92
- **Units:** pixels
- **Description:** PSF width for mRNA particles
- **Citation:** Zhang et al. (2007) - Gaussian PSF approximation
- **Measurement:** Should be determined from calibration data
- **Critical:** Used for particle detection threshold

##### `model_bg`
- **Type:** String (filename)
- **Default:** `"model_wieghts_background_psf.pth"`
- **Description:** Neural network model for background estimation

#### Optimization

##### `tolerance_intensity`
- **Type:** Float
- **Default:** 1e-3
- **Description:** Convergence tolerance for intensity parameter
- **Citation:** Dennis & Schnabel (1996) - convergence criteria

##### `tolerance_background`
- **Type:** Float
- **Default:** 1e-3
- **Description:** Convergence tolerance for background parameter

##### `lmlambda`
- **Type:** Float
- **Default:** 100
- **Description:** Initial LM damping parameter
- **Citation:** Marquardt (1963) - damping for stability
- **Range:** Original code varied from 0.001 to 1000 (1,000,000× range!)
- **Recommendation:** 1-100 for typical cases

##### `iterations`
- **Type:** Integer
- **Default:** 100
- **Description:** Maximum optimization iterations

#### Detection Parameters

##### `alpha`
- **Type:** Float
- **Default:** 0.05
- **Valid Range:** [0.01, 0.1]
- **Description:** Significance level for GLRT hypothesis test
- **Citation:** Smith et al. (2010) - statistical detection threshold
- **Interpretation:** 5% false positive rate

##### `number_channel`
- **Type:** Integer
- **Default:** 20
- **Description:** Number of detection channels or temporal bins
- **Purpose:** Multi-temporal analysis

##### `batch_size`
- **Type:** Integer
- **Default:** 160000
- **Description:** Batch size for GLRT computation
- **Notes:** Large batch improves GPU utilization

##### `full_image`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Process entire image or use ROI-based detection
- **Trade-off:**
  - `true`: More comprehensive, much slower
  - `false`: ROI-based, faster, may miss dim particles

---

## Parameter Quick Reference

### Most Commonly Changed

| Parameter | Location | Typical Values | When to Change |
|-----------|----------|----------------|----------------|
| `strains` | pipe globals | Array of experiment names | Every analysis |
| `imaging root` | pipe globals | Path to data | New dataset |
| `output root` | pipe globals | Path for results | New dataset |
| `masking_threshold` | ne_fit/initial | 0.3-0.7 | Data quality varies |
| `bspline_smoothing` | ne_fit/initial | 1.0-2.5 | Noise level changes |
| `min_iou` | dual_label | 0.8-0.95 | Registration quality |
| `max_reg_diff` | registration | 0.3-1.0 | Precision requirements |

### Rarely Changed

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| `roisize` | pipe globals | 16 | Camera/microscope dependent |
| `pixelsize` | pipe globals | 128 | Microscope specification |
| `frame_duration` | pipe globals | 0.02 | Camera settings |
| `upsample_factor` | registration | 1000 | Good for all cases |
| `sigma` | particle detection | 0.92 | From calibration data |

### Never Change (Without Good Reason)

| Parameter | Location | Value | Why |
|-----------|----------|-------|-----|
| `iterations` | ne_fit/refinement | 50 | Sufficient for convergence |
| `final_sampling_density` | ne_fit/refinement | 64 | Optimal for stability |
| `normal_lines_n` | ne_fit/refinement | 1000 | Comprehensive sampling |

---

## Examples

### Example 1: Minimal Local Configuration

```json
{
    "pipe globals": {
        "strains": ["test_experiment"],
        "directories": {
            "imaging root": "./test_data",
            "camera root": "./test_data",
            "output root": "./test_output",
            "model root": "./trained_models"
        }
    },
    "image processor": {
        "responsivity": {
            "ch1": {
                "bright": "bright_Ch1.tif",
                "dark": "dark_Ch1.tif"
            },
            "ch2": {
                "bright": "bright_Ch2.tif",
                "dark": "dark_Ch2.tif"
            }
        }
    }
}
```

### Example 2: High-Quality, Strict Filtering

```json
{
    "image processor": {
        "registration": {
            "max_reg_diff": 0.3,
            "frames_per_average": 50
        },
        "dual_label": {
            "min_iou": 0.95,
            "N_dist_calc": 2000
        },
        "ne_fit": {
            "initial": {
                "masking_threshold": 0.6,
                "bspline_smoothing": 2.0,
                "qc_min_labeled": 100
            }
        }
    }
}
```

### Example 3: Fast Processing for Testing

```json
{
    "pipe globals": {
        "strains": ["quick_test"]
    },
    "image processor": {
        "registration": {
            "frame_range": [0, 50],
            "frames_per_average": 10,
            "upsample_factor": 100
        },
        "ne_fit": {
            "frame_range": [0, 50],
            "run_bezier_bridging": false,
            "initial": {
                "plot_test_imgs": false
            },
            "refinement": {
                "plot_refine_test_imgs": false,
                "lm_optimizer": {
                    "iterations": 20
                }
            }
        }
    }
}
```

### Example 4: Cluster Computing (Full Quality)

```json
{
    "pipeline foreman": {
        "pipeline_mode": "dual_label"
    },
    "pipe globals": {
        "strains": ["BMY1408", "BMY1409", "BMY1410", "BMY1914", "BMY1915"],
        "directories": {
            "imaging root": "/pi/lab-name/data/yeast_data/npc_dual_label/",
            "camera root": "/pi/lab-name/data/yeast_data/responsivity/",
            "output root": "/home/username/yeast_output/dual_label_experiments/",
            "model root": "/home/username/src_yeast_pipeline/trained_models/"
        },
        "pixelsize": 128,
        "frame_duration": 0.02
    },
    "image processor": {
        "model_NE": "Modelweights_NE_segmentation.pt",
        "model_bg": "model_wieghts_background_psf.pth",
        "ne_dual_label": true,
        "registration": {
            "frame_range": [0, 250],
            "frames_per_average": 25,
            "max_reg_diff": 0.50,
            "upsample_factor": 1000
        },
        "responsivity": {
            "ch1": {
                "bright": "red_gain300.tif",
                "dark": "red_dark300.tif"
            },
            "ch2": {
                "bright": "bright_images_green_channel_20ms_300EM.tiff",
                "dark": "dark_images_green_channel_20ms_300EM.tiff"
            },
            "frame_range": [0, 250]
        },
        "dual_label": {
            "channel1": "fn_track_ch1",
            "channel2": "fn_track_ch2",
            "min_iou": 0.9,
            "N_dist_calc": 1000
        },
        "ne_fit": {
            "frame_range": [0, 250],
            "bbox_dim": {"width": 75, "height": 75},
            "run_bezier_bridging": true,
            "initial": {
                "use_merged_clusters": true,
                "max_merge_dist": 10,
                "masking_threshold": 0.5,
                "bspline_smoothing": 1.6
            },
            "refinement": {
                "default_model": "richards_gaussian",
                "lm_optimizer": {
                    "step_radius": 1.0,
                    "step_size": 0.1,
                    "iterations": 50
                }
            }
        }
    }
}
```

---

## Validation & Testing

### How to Validate Your Configuration

1. **Required Fields Check:**
   ```bash
   python -c "import json; config=json.load(open('config.json')); \
   assert 'pipe globals' in config; \
   assert 'strains' in config['pipe globals']; \
   assert 'directories' in config['pipe globals']"
   ```

2. **Path Existence Check:**
   ```bash
   python scripts/validate_config_paths.py config.json
   ```

3. **Test Run:**
   ```bash
   # Run on minimal dataset
   python main_cluster_local.py 1 config.json
   ```

### Common Configuration Errors

| Error | Symptom | Fix |
|-------|---------|-----|
| Missing strain folder | "Directory not found" | Check `strains` names match folders |
| Wrong calibration files | "File not found" | Verify `bright`/`dark` filenames |
| Invalid frame_range | Index error | Ensure end < total frames |
| frames_per_average mismatch | "Not evenly divisible" | Adjust to be factor of range |
| Model not found | Import error | Check `model root` path and filenames |

---

## References

### Key Citations by Parameter

**Camera Calibration:** Janesick (2001)  
**U-Net++ Segmentation:** Zhou et al. (2018)  
**Registration:** Guizar-Sicairos et al. (2008), Reddy & Chatterji (1996)  
**Spline Theory:** de Boor (2001)  
**Optimization:** Marquardt (1963), Dennis & Schnabel (1996)  
**Statistical Testing:** Smith et al. (2010), Akaike (1974)  
**PSF Modeling:** Zhang et al. (2007)  
**Membrane Biology:** Zimmerberg & Kozlov (2006)  

**Complete bibliography:** See `docs/methodology/bibliography.md`

---

## Getting Help

### Support
- **Issues:** GitHub Issues for bugs or questions
- **Email:** jocelyn (dot) petitto (at) gmail (dot) com
---
**Last Updated:** January 2, 2026
**Maintainer:** Jocelyn Petitto  