import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import BSpline, splprep

from tools.geom_tools import bspline_transformation, calculate_signed_distances


def match_ne_labels_by_iou(list_A, list_B, min_overlap_percent):
    """
    For each rectangle in list_A, finds the single best-matching rectangle
    in list_B that meets a minimum overlap threshold.
    """
    
    # Helper function to calculate the Intersection over Union (IoU) of two rectangles
    def calculate_iou(rect1, rect2):
        # Determine the coordinates of the intersection rectangle
        x_left = max(rect1['final_left'], rect2['final_left'])
        y_top = max(rect1['final_top'], rect2['final_top'])
        x_right = min(rect1['final_right'], rect2['final_right'])
        y_bottom = min(rect1['final_bottom'], rect2['final_bottom'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate the area of both rectangles
        rect1_area = (rect1['final_right'] - rect1['final_left']) * (rect1['final_bottom'] - rect1['final_top'])
        rect2_area = (rect2['final_right'] - rect2['final_left']) * (rect2['final_bottom'] - rect2['final_top'])
        
        union_area = rect1_area + rect2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

    all_matches = {}
    min_threshold = min_overlap_percent

    # Iterate through each rectangle in list_A to find its best match in list_B
    for rect_A in list_A:
        max_iou_for_A = -1.0
        best_match_B = None

        # Iterate through list_B to find the best match for the current rect_A
        for rect_B in list_B:
            iou = calculate_iou(list_A[rect_A], list_B[rect_B])
            
            # Update the best match if the current IoU is higher and meets the threshold
            if iou > max_iou_for_A and iou >= min_threshold:
                max_iou_for_A = iou
                best_match_B = rect_B
        
        # If a valid match was found, add it to the results list
        if best_match_B is not None:
            all_matches.update({f'{rect_A}': f'{best_match_B}'})
            
    return all_matches





def calc_dual_distances(ne_label_pairs, bsplines_A_dict, bsplines_B_dict, FoV_reg_values, bbox_dim_array, N_dist_samples):
    registration_keys = {'scale', 'angle', 'shift_vector'}

    dist_bt_label_pairs = {}
    for key, value in ne_label_pairs.items():
        # slim down the # of dual labels based on the pairing information
        if key in bsplines_A_dict.keys() and value in bsplines_B_dict.keys():
            # if bsplines exist for both channels:
            #   transform the bsplines of ch2 to based on the registration values
            #   calculate the min distance @ N points between the two bsplines
            current_bspline_A = bsplines_A_dict[key]
            current_bspline_B = bsplines_B_dict[value]

            current_reg_values = {key: value for key, value in FoV_reg_values['ne_label_registration'][key].items() if key in registration_keys}

            transformed_bspline_B = bspline_transformation(current_bspline_B, current_reg_values, center_coords = bbox_dim_array / 2)
            dist_bt_bsplines = calculate_signed_distances(current_bspline_A, transformed_bspline_B, N_dist_samples)
            dist_bt_label_pairs.update({f'{key}':{f'{value}': dist_bt_bsplines}})

    return dist_bt_label_pairs