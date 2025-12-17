import pickle
import torch
import tqdm
import numpy as np
from scipy.special import erf

from utils.GLRT import glrtfunction
from tools.utility_functions import get_centered_crop_indices

def detect_spots_with_glrt(image_stack, background_model, device, config, ne_label=""):

    # def batch_frames(input_array, batch_size):
    #     num_frames = input_array.shape[0]
    #     num_batches = int(np.ceil(num_frames / batch_size))
    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = min(start_idx + batch_size, num_frames)
    #         yield input_array[start_idx:end_idx]
            
    # def concatenate_images(images_list):
    #         return np.concatenate(images_list, axis=0)

    glrt_config = config['glrt_multichannel']
    print(f"Running multichannel GLRT detector for NE: {ne_label}...")

    # --- Config Parameters ---
    time_chunk_size = glrt_config['time_points_per_roi'] # e.g., 20 frames
    roi_size = glrt_config['roi_size']               # e.g., 16 pixels
    central_frame_idx = time_chunk_size // 2
    alpha = glrt_config['alpha']
    spatial_batch_size = glrt_config['spatial_batch_size']

    # Pre-allocate the final mask
    num_frames, height, width = image_stack.shape
    final_mask = np.zeros_like(image_stack, dtype=bool)

    # Calculate number of valid temporal chunks by sliding a window 1 frame at a time
    num_temporal_chunks = num_frames - time_chunk_size + 1
    if num_temporal_chunks <= 0:
        print("Warning: Image stack is shorter than the time window. No detection possible.")
        return final_mask
    # extract all ROI and temporal (frame) stacks at once; leverages "unroll" capability of tensors via pytorch

    rois, positions = extract_spatio_temporal_rois(
        image_stack, 
        time_chunk_size=time_chunk_size, 
        roi_size=roi_size
    )

    if rois.size == 0:
        print("No valid ROIs could be extracted.")
        return final_mask
    print(f"Extracted {len(rois)} total ROIs.")

    # background prediction using U-Net++ & specified time range (20)
    predicted_bgs = predict_backgrounds(rois, background_model, device, batch_size=glrt_config['spatial_batch_size'])
    
    # grab central frame for use in the GLRT
    # GLRT runs on the 2D frame corresponding with the U-Net predicted bg
    rois_2d_central = rois[:, central_frame_idx, :, :]

    
    ratios = calculate_likelihood_ratios(rois_2d_central, predicted_bgs, device, glrt_config)

    print("Step 4: Applying statistical correction to find significant spots...")

    significant_mask_1d = apply_fdr_correction(ratios, alpha=glrt_config['alpha'])

    # !!! Remove after testing!
    with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/{ne_label}_significant_mask_1d_{i}.pkl', 'wb') as file:
        pickle.dump(significant_mask_1d, file)

    print("Step 5: Reconstructing the final binary mask from the results...")
    # Reconstruct the mask for just this batch of frames
    batch_mask = reconstruct_mask_from_rois(significant_mask_1d, positions, frame_batch.shape, glrt_config['roisize'])

    # !!! Remove after testing!
    with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/{ne_label}_batch_mask_{i}.pkl', 'wb') as file:
        pickle.dump(batch_mask, file)

    all_partial_masks.append(batch_mask)

    print("All batches processed. Concatenating results...")
    
    # Stitch the partial masks from each batch together
    if all_partial_masks:
        final_mask = np.concatenate(all_partial_masks, axis=0)
    else:
        final_mask = np.zeros(image_stack.shape, dtype=bool)

    print("Detection complete.")

    return final_mask


def extract_spatio_temporal_rois(image_stack: np.ndarray, time_chunk_size: int, roi_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts all 3D spatio-temporal ROIs using torch.unfold.
    """
    num_frames, height, width = image_stack.shape
    
    # Check for valid dimensions; return empty arrays if conditions not met
    if num_frames < time_chunk_size or height < roi_size or width < roi_size:
        return np.array([]), np.array([])

    # Convert to tensor for 3 step unfolding, then reshape in the 4th step
    # TODO sketch a visual of this process
    stack_tensor = torch.from_numpy(image_stack).float()

    # 1. Unfold in time (dim 0): (F, H, W) -> (H, W, F', T)
    stack_unfolded = stack_tensor.unfold(0, time_chunk_size, 1).permute(1, 2, 0, 3)
    
    # 2. Unfold in height (dim 1): (H, W, F', T) -> (W, F', T, H', R)
    stack_unfolded = stack_unfolded.unfold(0, roi_size, 1).permute(1, 2, 3, 0, 4)
    
    # 3. Unfold in width (dim 2): (W, F', T, H', R) -> (F', T, H', R, W', R)
    stack_unfolded = stack_unfolded.unfold(0, roi_size, 1).permute(1, 2, 3, 4, 0, 5)

    # 4. Reshape into a clean batch of ROIs
    # (F', H', W', T, R, R)
    stack_unfolded = stack_unfolded.permute(0, 2, 4, 1, 3, 5)
    
    valid_frames = num_frames - time_chunk_size + 1
    valid_height = height - roi_size + 1
    valid_width = width - roi_size + 1

    # (N, T, R, R) where N = F' * H' * W'
    all_rois = stack_unfolded.reshape(-1, time_chunk_size, roi_size, roi_size)

    # Create array that maps each ROI back to its *central* pixel location
    center_f = time_chunk_size // 2
    center_y = roi_size // 2
    center_x = roi_size // 2
    
    # Create 3D grids of the *central* pixel coordinates
    f_coords, y_coords, x_coords = np.meshgrid(
        np.arange(valid_frames) + center_f, # Frame indices
        np.arange(valid_height) + center_y, # Y indices
        np.arange(valid_width) + center_x,  # X indices
        indexing='ij'
    )
    
    # Flatten and stack the ROI position grids; dim: (N, 3)
    all_positions = np.stack(
        (f_coords.ravel(), y_coords.ravel(), x_coords.ravel()), 
        axis=-1
    )

    return all_rois.numpy(), all_positions


def predict_backgrounds(rois: np.ndarray, model: torch.nn.Module, device: torch.device, batch_size: int) -> np.ndarray:
    #Takes the ROIs (N, T, R, R), runs them through the U-Net model in batches, and returns an array (N, R, R) of the predicted background for each ROI.
    model.to(device)
    model.eval()
    all_predicted_bgs = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(rois), batch_size), desc="Predicting background"):
            batch_rois = rois[i:i + batch_size]
            rois_tensor = torch.from_numpy(batch_rois).float().to(device)
        # INPUT: [batch_size, 20, 16, 16]
            
            # Normalize the batch; get max intensity across 20 frames
            norm = torch.amax(rois_tensor, dim=(1, 2, 3), keepdim=True) # keep the dimensions so that broadcasting is possible

            rois_for_nn = rois_tensor / (norm + 1e-6)
            
            predicted_bg_batch = model(rois_for_nn)
        # OUPUT: [batch_size, 1, 16, 16]

            # De-normalize the model output
            predicted_bg_batch = predicted_bg_batch * norm
            all_predicted_bgs.append(predicted_bg_batch.squeeze(1).cpu().numpy())
            
    return np.concatenate(all_predicted_bgs, axis=0)


#Takes the original ROIs and their predicted backgrounds and runs the glrtfunction to get the statistical score for each one.
def calculate_likelihood_ratios(rois_2d: np.ndarray, backgrounds: np.ndarray, device: torch.device, glrt_config: dict) -> np.ndarray:
    """Performs the GLRT for each ROI against its predicted background."""
    all_ratios = []
    
    batch_size = glrt_config['spatial_batch_size']
    roi_size = glrt_config['roi_size']
    final_roi_size = glrt_config['final_roi_size']
    # ???: why are there two different ROIs (it is like this in the original code, though I made it more flexible)
    # ANSWER: The first size (e.g., 16) is likely what the background U-Net expects.
    # The second, smaller size (e.g., 10 or 8) is what the PSF fitting/GLRT focuses on,
    # potentially for speed or to avoid edge effects from the background prediction.
    
    # bounds_glrt =  [
    #                     [0, roi_size - 1],  # x bounds
    #                     [0, roi_size],      # y bounds
    #                     [0, 1e9],   # Photons bounds
    #                     [0, 1e6]    # Background bounds
    #                 ]
    initial_template = [
                        final_roi_size / 2, # x initial guess
                        final_roi_size / 2, # y initial guess
                        0,  # Photons initial guess
                        60  # Background initial guess
                    ]
    # bounds = torch.tensor(bounds_glrt, device=device, dtype=torch.float32)
    sigma = glrt_config['sigma']
    iterations = glrt_config['iterations']
    
    tol_intensity = glrt_config['tolerance_intensity']
    tol_background = glrt_config['tolerance_background']
    tol_tensor = torch.tensor([tol_intensity, tol_background], device=device)

    crop_s, crop_e = get_centered_crop_indices(roi_size, final_roi_size)
    if crop_s == -1: # Handle invalid crop case
            raise ValueError(f"Cannot crop from {roi_size} to {final_roi_size}")

    # Define bounds SPECIFICALLY for the cropped size
    # Positional bounds are [0, final_roi_size - 1]
    cropped_bounds_glrt = [
        [0, final_roi_size - 1], # x bounds for the 10x10 crop
        [0, final_roi_size - 1], # y bounds for the 10x10 crop
        [0, 1e9],                  # Photons bounds (likely unchanged)
        [0, 1e6]                   # Background bounds (likely unchanged)
    ]
    bounds = torch.tensor(cropped_bounds_glrt, device=device, dtype=torch.float32)
    
    for i in tqdm.tqdm(range(0, len(rois_2d), batch_size), desc="Calculating GLRT"):
        batch_rois_2d = rois_2d[i:i + batch_size]
        batch_backgrounds = backgrounds[i:i + batch_size]
        
        # Prepare initial guess for the batch
        initial_guess = np.zeros((len(batch_rois_2d), 4), dtype = np.float32)
        initial_guess[:, 0] = initial_template[0] # Centered x in cropped ROI
        initial_guess[:, 1] = initial_template[1] # Centered y in cropped ROI
        initial_guess[:, 2] = initial_template[2] # Photon guess
        initial_guess[:, 3] = np.mean(batch_rois_2d[:, crop_s:crop_e, crop_s:crop_e], axis = (1, 2)) # BG from CROPPED ROI mean

        # Convert to tensors
        rois_tensor = torch.from_numpy(batch_rois_2d).to(device)
        backgrounds_tensor = torch.from_numpy(batch_backgrounds).to(device)
        initial_tensor = torch.from_numpy(initial_guess).to(device)

        # Run the GLRT function with the 2D data
        ratio_batch, _, _, _, _, _, _ = glrtfunction(
            smp_arr = rois_tensor[:, crop_s:crop_e, crop_s:crop_e],
            batch_size = len(rois_tensor),
            bounds = bounds,
            initial_arr = initial_tensor,
            roisize = final_roi_size,
            sigma = sigma,
            tol = tol_tensor,
            iterations = iterations,
            bg_constant = backgrounds_tensor[:, crop_s:crop_e, crop_s:crop_e] # must also be cropped to smp_arr dimensions
        )
        all_ratios.append(ratio_batch.cpu().numpy())
            
    return np.concatenate(all_ratios)

#Takes the likelihood ratios, converts them to p-values, and performs the Benjamini–Hochberg procedure to return a final boolean array of which spots are statistically significant.
def apply_fdr_correction(likelihood_ratios: np.ndarray, alpha: float) -> np.ndarray:
    """
    Converts likelihood ratios to p-values and applies the Benjamini-Hochberg
    FDR correction to find significant results.
    """
    # Define a helper for the cumulative distribution function
    def normcdf(x):
        return 0.5 * (1 + erf(x / np.sqrt(2)))
        
    # Define a helper for the harmonic number approximation
    def fast_harmonic(n):
        gamma = 0.57721566490153286 # Euler-Mascheroni constant
        return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

    # Convert likelihood ratios to p-values (probability of false alarm)
    p_values = 2 * normcdf(-np.sqrt(np.clip(likelihood_ratios, 0, np.inf)))
    
    # Benjamini-Hochberg procedure
    num_tests = len(p_values)
    if num_tests == 0:
        return np.array([], dtype=bool)
        
    sorted_indices = np.argsort(p_values)
    
    p_values_sorted = p_values[sorted_indices]
    
    # Calculate the critical value for each p-value based on its rank
    c_m = fast_harmonic(num_tests)
    fdr_thresholds = (np.arange(1, num_tests + 1) / (num_tests * c_m)) * alpha
    
    # Find the p-values that are below their critical value
    significant_sorted = p_values_sorted <= fdr_thresholds
    
    # Unsort the boolean results to match the original order of the ROIs
    unsorted_indices = np.argsort(sorted_indices)
    significant_mask = significant_sorted[unsorted_indices]
    
    return significant_mask

def reconstruct_mask_from_rois(significant_mask: np.ndarray, positions: np.ndarray, final_shape: tuple, roi_size: int) -> np.ndarray:
    """
    Creates a full-size boolean mask from the 1D list of significant results.
    The positions are assumed to be the *center* pixel of the ROI.
    """
    # Create an empty boolean array with the final desired shape
    final_mask = np.zeros(final_shape, dtype=bool)
    
    # Get only the positions of the ROIs that were significant
    significant_positions = positions[significant_mask]
    
    # Use advanced integer indexing to set the corresponding pixels to True
    frames = significant_positions[:, 0]
    y_coords = significant_positions[:, 1]
    x_coords = significant_positions[:, 2]

    final_mask[frames, y_coords, x_coords] = True
    
    return final_mask
