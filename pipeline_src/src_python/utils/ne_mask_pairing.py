"""
NE pairing using actual segmentation masks from detection.
"""

import numpy as np
from scipy.ndimage import center_of_mass


def pair_nes_by_masks(
    ch1_masks_dict,
    ch2_masks_dict,
    min_iou=0.7,
    max_centroid_distance_pixels=10
):
    """
    Pair NEs between channels using actual segmentation masks.
    
    Args:
        ch1_masks_dict: Dict {label: binary_mask} for channel 1
        ch2_masks_dict: Dict {label: binary_mask} for channel 2
        min_iou: Minimum intersection-over-union for valid pair
        max_centroid_distance_pixels: Maximum centroid distance for valid pair
        
    Returns:
        pairs: Dict mapping ch1_label -> ch2_label
    """
    
    def calculate_iou(mask1, mask2):
        """Calculate intersection over union of two binary masks."""
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        return intersection / union if union > 0 else 0.0
    
    def get_mask_centroid(mask):
        """Get centroid of binary mask."""
        if np.sum(mask) == 0:
            return None
        try:
            centroid_y, centroid_x = center_of_mass(mask)
            return (centroid_y, centroid_x)
        except:
            return None
    
    # Extract centroids for all masks
    ch1_info = {}
    for label, mask in ch1_masks_dict.items():
        centroid = get_mask_centroid(mask)
        if centroid is not None:
            ch1_info[label] = {
                'mask': mask,
                'centroid': centroid,
                'area': np.sum(mask)
            }
    
    ch2_info = {}
    for label, mask in ch2_masks_dict.items():
        centroid = get_mask_centroid(mask)
        if centroid is not None:
            ch2_info[label] = {
                'mask': mask,
                'centroid': centroid,
                'area': np.sum(mask)
            }
    
    # Pairing algorithm: greedy best-match
    pairs = {}
    used_ch2_labels = set()
    
    for ch1_label, ch1_data in ch1_info.items():
        best_match = None
        best_score = 0
        
        for ch2_label, ch2_data in ch2_info.items():
            if ch2_label in used_ch2_labels:
                continue
            
            # Calculate IOU of actual segmentation masks
            iou = calculate_iou(ch1_data['mask'], ch2_data['mask'])
            
            # Calculate centroid distance
            dist = np.sqrt(
                (ch1_data['centroid'][0] - ch2_data['centroid'][0])**2 +
                (ch1_data['centroid'][1] - ch2_data['centroid'][1])**2
            )
            
            # Both criteria must be satisfied
            if iou >= min_iou and dist <= max_centroid_distance_pixels:
                # Combined score: higher IOU and closer centroids = better
                score = iou * (1.0 / (1.0 + dist))
                
                if score > best_score:
                    best_score = score
                    best_match = ch2_label
        
        if best_match:
            pairs[ch1_label] = best_match
            used_ch2_labels.add(best_match)
    
    return pairs