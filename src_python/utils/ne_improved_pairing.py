"""
Improved NE pairing using crop box overlap and centroid distance.
"""

import numpy as np
from scipy.ndimage import center_of_mass

def improved_ne_crop_and_pair(
    ch1_labeled_img, 
    ch2_labeled_img,
    ch1_existing_crops,
    ch2_existing_crops,
    min_iou_tight=0.7,
    max_centroid_distance_pixels=10
):
    """
    Improved NE pairing using CROP BOX overlap and centroid distance.
    
    Args:
        ch1_labeled_img: Labeled mask image for channel 1 (used only for centroids)
        ch2_labeled_img: Labeled mask image for channel 2 (used only for centroids)
        ch1_existing_crops: Dict of existing crop info for ch1
        ch2_existing_crops: Dict of existing crop info for ch2
        min_iou_tight: Minimum IOU for crop box overlap
        max_centroid_distance_pixels: Max distance between centroids
        
    Returns:
        pairs: Dict mapping ch1_label -> ch2_label (only valid pairs)
    """
    
    def get_centroid_from_mask(labeled_img, label_value):
        """Extract centroid from labeled mask."""
        label_int = int(label_value)
        mask = (labeled_img == label_int).astype(np.uint8)
        
        if np.sum(mask) == 0:
            return None
        
        try:
            centroid_y, centroid_x = center_of_mass(mask)
            return (centroid_y, centroid_x)
        except:
            return None
    
    def calculate_crop_box_iou(crop1, crop2):
        """
        Calculate IOU of two CROP BOXES (bounding boxes).
        
        This is the CORRECT approach - using the crop box coordinates directly,
        not extracting masks from the labeled images.
        """
        # Get intersection coordinates
        x_left = max(crop1['final_left'], crop2['final_left'])
        y_top = max(crop1['final_top'], crop2['final_top'])
        x_right = min(crop1['final_right'], crop2['final_right'])
        y_bottom = min(crop1['final_bottom'], crop2['final_bottom'])
        
        # No overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        crop1_area = (crop1['final_right'] - crop1['final_left']) * \
                     (crop1['final_bottom'] - crop1['final_top'])
        crop2_area = (crop2['final_right'] - crop2['final_left']) * \
                     (crop2['final_bottom'] - crop2['final_top'])
        
        union_area = crop1_area + crop2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    # Build info dict for Ch1
    ch1_info = {}
    for label_str in ch1_existing_crops.keys():
        centroid = get_centroid_from_mask(ch1_labeled_img, label_str)
        if centroid:
            ch1_info[label_str] = {
                'crop': ch1_existing_crops[label_str],
                'centroid': centroid
            }
    
    # Build info dict for Ch2
    ch2_info = {}
    for label_str in ch2_existing_crops.keys():
        centroid = get_centroid_from_mask(ch2_labeled_img, label_str)
        if centroid:
            ch2_info[label_str] = {
                'crop': ch2_existing_crops[label_str],
                'centroid': centroid
            }
    
    # --- PAIRING ALGORITHM ---
    pairs = {}
    used_ch2_labels = set()
    
    for ch1_label, ch1_data in ch1_info.items():
        best_match = None
        best_score = 0
        
        for ch2_label, ch2_data in ch2_info.items():
            if ch2_label in used_ch2_labels:
                continue
            
            # Calculate crop box IOU (using bounding boxes)
            iou = calculate_crop_box_iou(ch1_data['crop'], ch2_data['crop'])
            
            # Calculate centroid distance
            dist = np.sqrt(
                (ch1_data['centroid'][0] - ch2_data['centroid'][0])**2 +
                (ch1_data['centroid'][1] - ch2_data['centroid'][1])**2
            )
            
            # Must pass BOTH filters (original logic)
            if iou >= min_iou_tight and dist <= max_centroid_distance_pixels:
                # Score combines IOU and inverse distance
                score = iou * (1.0 / (1.0 + dist))
                
                if score > best_score:
                    best_score = score
                    best_match = ch2_label
        
        if best_match:
            pairs[ch1_label] = best_match
            used_ch2_labels.add(best_match)
    
    return pairs