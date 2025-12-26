import numpy as np
from tools.geom_tools import calculate_signed_distances
from utils.spline_bridging import identify_segment_type_at_u

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

def calc_dual_distances(FoV_dict, ch1_bsplines, ch2_bsplines, ne_pairs_map, 
                        ch1_key, ch2_key, min_iou, N_dist_calc, drift_data=None, 
                        pixel_size_nm=128, distance_logger=None):
    """
    Calculates SIGNED distances using polygon containment test.
    
    Args:
        FoV_dict: Dictionary of FoV metadata
        ch1_bsplines: Channel 1 bridged splines dict (from bridge_refined_splines)
        ch2_bsplines: Channel 2 bridged splines dict
        ne_pairs_map: Dictionary mapping ch1 labels to ch2 labels per FoV
        ch1_key: Config key for channel 1
        ch2_key: Config key for channel 2
        min_iou: Minimum IOU threshold (not used but kept for API consistency)
        N_dist_calc: Number of distance samples to calculate
        drift_data: Optional drift correction data
        pixel_size_nm: Pixel size in nanometers (default 128)
        distance_logger: Optional DistanceCalculationLogger for comprehensive tracking

    Returns:
        dict: {fov_id: {ne_pair_label: {distances, stats, ...}}}
    """
    
    distances_by_FoV = {}
    matched_FoV_ids = set(ch1_bsplines.keys()) & set(ch2_bsplines.keys()) & set(ne_pairs_map.keys())
    
    for fov_id in matched_FoV_ids:
        dist_by_ne_pair = {}
        
        for ch1_ne_label, ch2_ne_label in ne_pairs_map[fov_id].items():
            
            try:
                ch1_data = ch1_bsplines[fov_id][ch1_ne_label]
                ch2_data = ch2_bsplines[fov_id][ch2_ne_label]
            except KeyError:
                print(f"  --> Missing spline data for {fov_id}/{ch1_ne_label}/{ch2_ne_label}. Skipping.")
                continue
            
            # Extract the full periodic splines
            spline_ch1 = ch1_data['full_periodic_spline']
            spline_ch2 = ch2_data['full_periodic_spline']
            u_ranges_ch1 = ch1_data['u_ranges']
            u_ranges_ch2 = ch2_data['u_ranges']
            
            # Use the existing signed distance function
            signed_distances_pixels = calculate_signed_distances(spline_ch1, spline_ch2, N_dist_calc)
            
            # Convert to nm
            signed_distances_nm = signed_distances_pixels * pixel_size_nm
            
            # Get u values for segment type identification
            u_values_ch1 = np.linspace(0, 1, N_dist_calc, endpoint=False)
            
            # Get segment types for each point
            ch1_segment_types = []
            ch2_segment_types = []
            pair_types = []

            for u_ch1 in u_values_ch1:
                type_ch1 = identify_segment_type_at_u(u_ch1, u_ranges_ch1)
                type_ch2 = identify_segment_type_at_u(u_ch1, u_ranges_ch2)
                
                ch1_segment_types.append(type_ch1)
                ch2_segment_types.append(type_ch2)
                
                # Proper classification - KEEPING THEM SEPARATE
                if type_ch1 == 'data' and type_ch2 == 'data':
                    pair_types.append('data_data')
                elif type_ch1 == 'bridge' and type_ch2 == 'bridge':
                    pair_types.append('bridge_bridge')
                elif type_ch1 == 'data' and type_ch2 == 'bridge':
                    pair_types.append('data_bridge')  # Ch1 real, Ch2 interpolated
                elif type_ch1 == 'bridge' and type_ch2 == 'data':
                    pair_types.append('bridge_data')  # Ch1 interpolated, Ch2 real
                else:
                    pair_types.append('unknown')

            # Calculate statistics - SEPARATE CATEGORIES
            data_data_dists = [d for d, pt in zip(signed_distances_nm, pair_types) if pt == 'data_data']
            bridge_bridge_dists = [d for d, pt in zip(signed_distances_nm, pair_types) if pt == 'bridge_bridge']
            data_bridge_dists = [d for d, pt in zip(signed_distances_nm, pair_types) if pt == 'data_bridge']
            bridge_data_dists = [d for d, pt in zip(signed_distances_nm, pair_types) if pt == 'bridge_data']

            # Build result dictionary
            result = {
                'distances': signed_distances_nm.tolist(),
                'ch1_u_values': u_values_ch1.tolist(),
                'ch1_segment_types': ch1_segment_types,
                'ch2_segment_types': ch2_segment_types,  # ADDED
                'pair_types': pair_types,
                'mean_distance': float(np.mean(signed_distances_nm)),
                'std_distance': float(np.std(signed_distances_nm)),
                'median_distance': float(np.median(signed_distances_nm)),
                'min_distance': float(np.min(signed_distances_nm)),
                'max_distance': float(np.max(signed_distances_nm)),
                
                # Statistics by segment type pairing
                'data_data_mean': float(np.mean(data_data_dists)) if data_data_dists else None,
                'data_data_std': float(np.std(data_data_dists)) if len(data_data_dists) > 1 else None,
                'bridge_bridge_mean': float(np.mean(bridge_bridge_dists)) if bridge_bridge_dists else None,
                'bridge_bridge_std': float(np.std(bridge_bridge_dists)) if len(bridge_bridge_dists) > 1 else None,
                'data_bridge_mean': float(np.mean(data_bridge_dists)) if data_bridge_dists else None,
                'data_bridge_std': float(np.std(data_bridge_dists)) if len(data_bridge_dists) > 1 else None,
                'bridge_data_mean': float(np.mean(bridge_data_dists)) if bridge_data_dists else None,
                'bridge_data_std': float(np.std(bridge_data_dists)) if len(bridge_data_dists) > 1 else None,
                
                # Counts
                'n_data_data_points': len(data_data_dists),
                'n_bridge_bridge_points': len(bridge_bridge_dists),
                'n_data_bridge_points': len(data_bridge_dists),
                'n_bridge_data_points': len(bridge_data_dists),
                'n_points': N_dist_calc,
                'pixel_size_nm': pixel_size_nm
            }

            dist_by_ne_pair[f'{ch1_ne_label}_vs_{ch2_ne_label}'] = result

            # === LOG DISTANCE CALCULATION ===
            if distance_logger is not None:
                # Get quality metrics from spline data
                ch1_quality = ch1_data.get('quality_stats', {})
                ch2_quality = ch2_data.get('quality_stats', {})
                
                # Log segment quality for Ch1
                if ch1_quality:
                    distance_logger.log_segment_quality(
                        fov_id=fov_id,
                        channel=1,
                        ne_label=ch1_ne_label,
                        segment_label='full_periodic',
                        n_total=ch1_quality.get('n_total', 0),
                        n_success=ch1_quality.get('n_success', 0),
                        n_curvature_fail=ch1_quality.get('n_curvature_fail', 0),
                        n_likelihood_fail=ch1_quality.get('n_likelihood_fail', 0),
                        n_optimization_fail=ch1_quality.get('n_optimization_fail', 0)
                    )
                
                # Log segment quality for Ch2
                if ch2_quality:
                    distance_logger.log_segment_quality(
                        fov_id=fov_id,
                        channel=2,
                        ne_label=ch2_ne_label,
                        segment_label='full_periodic',
                        n_total=ch2_quality.get('n_total', 0),
                        n_success=ch2_quality.get('n_success', 0),
                        n_curvature_fail=ch2_quality.get('n_curvature_fail', 0),
                        n_likelihood_fail=ch2_quality.get('n_likelihood_fail', 0),
                        n_optimization_fail=ch2_quality.get('n_optimization_fail', 0)
                    )
                
                # Get other quality metrics from spline data if available
                ch1_n_refined = len(ch1_data.get('data_segments', []))
                ch2_n_refined = len(ch2_data.get('data_segments', []))
                
                # Estimate success rates from data vs bridge ratios
                # (Ideally these would come from refinement metadata)
                ch1_success_rate = len(data_data_dists) / N_dist_calc if N_dist_calc > 0 else 0
                ch2_success_rate = ch1_success_rate  # Simplified - would need actual data
                
                distance_result = {
                    'mean_distance': result['mean_distance'],
                    'median_distance': result['median_distance'],
                    'std_distance': result['std_distance'],
                    'min_distance': result['min_distance'],
                    'max_distance': result['max_distance'],
                    'n_points': N_dist_calc,
                    'ch1_n_refined': ch1_n_refined,
                    'ch2_n_refined': ch2_n_refined,
                    'ch1_success_rate': ch1_success_rate,
                    'ch2_success_rate': ch2_success_rate,
                }
                
                distance_logger.log_distance_calculation(
                    fov_id=fov_id,
                    ne_ch1=ch1_ne_label,
                    ne_ch2=ch2_ne_label,
                    ch1_segment='full_periodic',  # Since we're using bridged splines
                    ch2_segment='full_periodic',
                    distance_result=distance_result
                )
            
            if dist_by_ne_pair:
                distances_by_FoV[fov_id] = dist_by_ne_pair
    
    return distances_by_FoV