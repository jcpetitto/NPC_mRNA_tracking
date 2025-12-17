
import pickle
import torch
import tqdm
import numpy as np
from scipy.special import erf

from utils.GLRT import glrtfunction

def detect_spots_with_glrt(image_stack, background_model, device, config, ne_label=""):

    def batch_frames(input_array, batch_size):
        num_frames = input_array.shape[0]
        num_batches = int(np.ceil(num_frames / batch_size))
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            yield input_array[start_idx:end_idx]
            
    def concatenate_images(images_list):
            return np.concatenate(images_list, axis=0)

    glrt_config = config['glrt_multichannel']
    print('multichannel GLRT detector')
    print("Step 1: Extracting all ROIs from the image stack...")

    all_partial_masks = []
    batch_size = glrt_config['frame_batch_size']

    i = 0
    # Loop through the image stack in manageable batches
    for frame_batch in tqdm.tqdm(batch_frames(image_stack, batch_size), desc="Processing Frame Batches"):
        
        i = i + 1
        # --- Run the pipeline on the current batch ---
        
        rois, positions = extract_rois(frame_batch, glrt_config)
        
        # !!! Remove after testing!
        with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/{ne_label}_rois_{i}.pkl', 'wb') as file:
            pickle.dump(rois, file)

        if rois.size == 0:
            continue # Skip empty frames if they occur

        print("Step 2: Predicting background for each ROI using the DL model...")
        predicted_bgs = predict_backgrounds(rois, background_model, device, batch_size=glrt_config['spatial_batch_size'])
        
        # !!! Remove after testing!
        with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/{ne_label}_predicted_bgs_{i}.pkl', 'wb') as file:
            pickle.dump(predicted_bgs, file)

        print("Step 3: Calculating likelihood ratios for each ROI...")
        
        ratios = calculate_likelihood_ratios(rois, predicted_bgs, device, glrt_config)
        

        # !!! Remove after testing!
        with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/{ne_label}_ratios_{i}.pkl', 'wb') as file:
            pickle.dump(ratios, file)

        print("Step 4: Applying statistical correction to find significant spots...")

        significant_mask_1d = apply_fdr_correction(ratios, alpha=glrt_config['alpha'])

        # !!! Remove after testing!
        with open(f'/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/output/FoV0120_GLRT_Testing/{ne_label}_significant_mask_1d_{i}.pkl', 'wb') as file:
            pickle.dump(significant_mask_1d, file)

        print("Step 5: Reconstructing the final binary mask from the results...")
        # Reconstruct the mask for just this batch of frames
        batch_mask = reconstruct_mask_from_rois(significant_mask_1d, positions, frame_batch.shape, glrt_config['roi_size'])

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


def extract_rois(image_stack: np.ndarray, glrt_config) -> tuple[np.ndarray, np.ndarray]:
    roi_size = glrt_config['roi_size']
    time_points_per_roi = glrt_config['time_points_per_roi']
    num_frames, height, width = image_stack.shape

    valid_frames = num_frames - time_points_per_roi + 1
    valid_height = height - roi_size + 1
    valid_width = width - roi_size + 1

    # Calculate the total number of ROIs that will be extracted
    num_rois = valid_frames * valid_height * valid_width
    if num_rois <= 0:
        return np.array([]), np.array([])

    # Pre-allocate arrays for efficiency
    all_rois = np.zeros((num_rois, time_points_per_roi, roi_size, roi_size), dtype=image_stack.dtype)
    all_positions = np.zeros((num_rois, 3), dtype=int)

    # Iterate through all possible top-left starting positions of the 3D window
    idx = 0
    for f in range(valid_frames):
        for r in range(valid_height):
            for c in range(valid_width):
                # Extract the 3D stack (the mini-movie)
                all_rois[idx] = image_stack[f:f + time_points_per_roi, r:r + roi_size, c:c + roi_size]
                
                # Store the starting (frame, y, x) coordinate
                all_positions[idx] = [f, r, c]
                idx += 1

    return all_rois, all_positions



def predict_backgrounds(rois: np.ndarray, model: torch.nn.Module, device: torch.device, batch_size: int) -> np.ndarray:
    #Takes the ROIs, runs them through the U-Net model in batches, and returns an array of the predicted background for each ROI.
    model.to(device)
    model.eval()
    all_predicted_bgs = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(rois), batch_size), desc="Predicting background"):
            batch_rois = rois[i:i + batch_size]
            rois_tensor = torch.from_numpy(batch_rois).float().to(device)
            # [batch_size, 20, 16, 16]
            
            # Normalize the batch; get max intensity across all 20 frames within the batch
            norm = torch.amax(rois_tensor, dim=(1, 2, 3), keepdim=True) # keep the dimensions so that broadcasting is possible

            rois_for_nn = rois_tensor / (norm + 1e-6)
            
            predicted_bg_batch = model(rois_for_nn)
            # [batch_size, 1, 16, 16]

            # De-normalize the model output
            predicted_bg_batch = predicted_bg_batch * norm
            all_predicted_bgs.append(predicted_bg_batch.squeeze(1).cpu().numpy())
            
    return np.concatenate(all_predicted_bgs, axis=0)


#Takes the original ROIs and their predicted backgrounds and runs the glrtfunction to get the statistical score for each one.
def calculate_likelihood_ratios(rois: np.ndarray, backgrounds: np.ndarray, device: torch.device, glrt_config: dict) -> np.ndarray:
    """Performs the GLRT for each ROI against its predicted background."""
    all_ratios = []
    batch_size = glrt_config['spatial_batch_size']
    
    # GLRT parameters derived from glrt_config
    bounds_glrt =  [
                        [0, glrt_config['roi_size'] - 1],  # x bounds
                        [0, glrt_config['roi_size']],      # y bounds
                        [0, 1e9],   # Photons bounds
                        [0, 1e6]    # Background bounds
                    ]
    initial_guess = [
                        glrt_config['roi_size'] / 2, # x initial guess
                        glrt_config['roi_size'] / 2, # y initial guess
                        0,  # Photons initial guess
                        60  # Background initial guess
                    ]
    bounds = torch.tensor(bounds_glrt, device=device, dtype=torch.float32)
    initial_guess_template = initial_guess
    
    for i in tqdm.tqdm(range(0, len(rois), batch_size), desc="Calculating GLRT"):
        batch_rois_3d = rois[i:i + batch_size]
        batch_backgrounds = backgrounds[i:i + batch_size]
        
        # The GLRT needs a 2D image, not a 20-frame stack.
        # We select the central frame from the time-series ROI.
        time_points_per_roi = glrt_config['time_points_per_roi']
        central_frame_index = time_points_per_roi // 2
        batch_rois_2d = batch_rois_3d[:, central_frame_index, :, :]
        
        # Prepare initial guess for the batch
        initial_guess = np.zeros((len(batch_rois_2d), 4), dtype=np.float32)
        initial_guess[:, :2] = glrt_config['roi_size'] / 2
        initial_guess[:, 2] = initial_guess_template[2]
        initial_guess[:, 3] = np.mean(batch_rois_2d, axis=(1, 2))
        
        # Convert to tensors
        rois_tensor = torch.from_numpy(batch_rois_2d).to(device)
        backgrounds_tensor = torch.from_numpy(batch_backgrounds).to(device)
        initial_tensor = torch.from_numpy(initial_guess).to(device)
        
        # Run the GLRT function with the 2D data
        ratio_batch, _, _, _, _, _, _ = glrtfunction(
            smp_arr=rois_tensor,
            batch_size=len(rois_tensor),
            bounds=bounds,
            initial_arr=initial_tensor,
            roi_size=glrt_config['roi_size'],
            sigma=glrt_config['sigma'],
            tol=torch.tensor([1e-3, 1e-3], device=device),
            iterations=glrt_config['iterations'],
            bg_constant=backgrounds_tensor
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

# #Takes the final boolean results and uses the original positions to build the full-size output mask image.
def reconstruct_mask_from_rois(significant_mask: np.ndarray, positions: np.ndarray, final_shape: tuple, roi_size: int) -> np.ndarray:
    """
    Creates a full-size boolean mask from the 1D list of significant results.
    """
    # Create an empty boolean array with the final desired shape
    final_mask = np.zeros(final_shape, dtype=bool)
    
    # Get only the positions of the ROIs that were significant
    significant_positions = positions[significant_mask]
    
    # ??? --- SUGGESTED CHANGE ---
    # !!! remove change to stick w/ convention + making it easier to stitch back
    # Add half the ROI size to the top-left corner coordinates to get the center
    center_offset = roi_size // 2
    # Use advanced integer indexing to set the corresponding pixels to True
    frames = significant_positions[:, 0]
    y_coords = significant_positions[:, 1] + center_offset
    x_coords = significant_positions[:, 2] + center_offset
    
    # Ensure coordinates are still within bounds after the offset
    y_coords = np.clip(y_coords, 0, final_shape[1] - 1)
    x_coords = np.clip(x_coords, 0, final_shape[2] - 1)

    final_mask[frames, y_coords, x_coords] = True
    
    return final_mask
