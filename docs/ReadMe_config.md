# Configuration File Documentation

## Fields

Note: field names do not repeat at the top level within each parent node of the
JSON file This is allows the `general pipeline` node to be combined updated with
module specific nodes (eg. `image processor`) for module configuration without
increasing the complexity of accessing settings within the module.

## General Pipeline

`pipeline_mode`: "dual_label", "mrna_tracking"

### Initial Image Processing

-   `path`: directory containing set of images to be analyzed; set per
    directory; Note: the root directory of images sent to "single_folder" is
    sent via the CLI *assumes a specific subdirectory organization*

-   `roisize`: ROI size for fitting PSFs `sigma`: PSF width for mRNA (in pixels)

-   `frames`: frames to consider for processing

-   `frames_npcfit`: frames for NE fitting

-   `drift_bins`: number of drift bins

-   `pixelsize`: pixel size of the camera (nm) *assumed to be the same as in
    initial image processing stage*

-   `cell_line_id`

-   `resultsdir`: directory to save individual FoV results

-   `bright`: path to bright images (gain) for mRNA channel

-   `dark`: path to dark images (offset) for mRNA channel

-   `model_NE`: path to trained weights for NE segmentation NN

-   `model_bg`: path to trained weights for background estimation

-   `pixelsize`: pixel size of the camera (nm)

### Registration

-   `registration`
    -   `frame_range`: (default = \[0, 250\])
    -   `frames_per_average`: (default = 50), used to create averaged images for
        each channel for registration; should be a factor of the number of
        frames in the frame range `drift_bins`: (default = 4)
        `max_reg_diff_mode1`: (default = 0.50) `upsample_factor`: (default
        = 1000) upsampling factor used with
        skimage.registration.phase_cross_correlation `upscale_factor`: (default
        = 1) scaling factor for use with scipy.ndimage.zoom

`responsivity`: `frame_range`: (default = \[0, 250\])

### NE Detection

`trim_calculation_mode`: Options: "ne", "fov", "compare"

#### Initial Fitting

#### Refined Fitting

#### Distance between Nup Labels

### Drift

### Particle Tracking

### GLRT

-   `spatial_batch_size` - controls the trade-off between speed and memory usage
    for the core GLRT related computations; can be increased until memory limit
    is reached, then reduced to below that threshold