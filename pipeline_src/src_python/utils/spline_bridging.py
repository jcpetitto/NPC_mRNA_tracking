"""
Contains the logic for bridging disjoint refined spline segments into
a single, continuous, periodic spline.
"""

import numpy as np
from scipy.interpolate import make_splprep, splev
from tools.geom_tools import build_curve_bridge, get_spline_arc_length
from tools.utility_functions import sample_bspline
import logging
import traceback

logger = logging.getLogger(__name__)

# === REQUIRED IMPORTS (add to top of file if not present) ===
# from scipy.interpolate import BSpline
# import numpy as np
# === END IMPORTS ===

def validate_bridge_curvature(bridge_spline, max_angle_change_deg=1.0, n_check_points=50):
    """
    Validate that a bridge spline doesn't have extreme curvature using the
    same method as refinement: checking tangent angle changes.
    
    Uses the nuclear envelope physics constraint: at ~20nm sampling, expected
    angular deviation is ~0.3°. We use 1° threshold (3× safety margin).
    
    Args:
        bridge_spline: BSpline object for the bridge
        max_angle_change_deg: Maximum allowed angle change between consecutive points (default 1.0°)
        n_check_points: Number of points to check along bridge
        
    Returns:
        (is_valid, max_angle_found): Tuple of (bool, float in degrees)
    """
    u_values = np.linspace(0, 1, n_check_points)
    
    # Get points and tangents along bridge
    points = np.array([bridge_spline(u) for u in u_values]).T  # (2, N)
    
    # Calculate tangent vectors via derivatives
    derivs = np.zeros_like(points)
    derivs[:, 1:-1] = (points[:, 2:] - points[:, :-2]) / 2.0
    derivs[:, 0] = points[:, 1] - points[:, 0]
    derivs[:, -1] = points[:, -1] - points[:, -2]
    
    # Calculate tangent angles
    angles = np.arctan2(derivs[1, :], derivs[0, :])
    
    # Angle differences (normalized to [-π, π])
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    angle_changes_deg = np.abs(np.rad2deg(angle_diffs))
    
    # Check if any angle change exceeds threshold
    max_angle = np.max(angle_changes_deg) if len(angle_changes_deg) > 0 else 0.0
    
    if np.isnan(max_angle) or np.isinf(max_angle):
        return False, np.inf
    
    is_valid = max_angle <= max_angle_change_deg

    return is_valid, max_angle


def fit_parametric_spline(points_yx, smoothing=1.0, periodic=False):
    """
    Fit a parametric B-spline to 2D points.
    
    Args:
        points_yx: (N, 2) array of [y, x] coordinates
        smoothing: smoothing parameter
        periodic: whether to create periodic spline
        
    Returns:
        BSpline object
    """
    from scipy.interpolate import splprep, BSpline
    
    # splprep has the 'per' parameter (make_splprep doesn't)
    per_value = 1 if periodic else 0
    tck, u = splprep([points_yx[:, 0], points_yx[:, 1]], s=smoothing, k=3, per=per_value)
    
    # tck is (t, c, k) where:
    # t = knot vector
    # c = [coeffs_for_x, coeffs_for_y] (list of two arrays)
    # k = degree
    t, c, k = tck
    
    # For parametric BSpline, coefficients must be shape (n_knots, ndim)
    # c from splprep is a list [c_x, c_y], need to transpose
    c_array = np.column_stack(c)  # Stack into (n, 2) array
    
    return BSpline(t, c_array, k)

def bridge_refined_splines(refined_splines_dict, config_dict):
    """
    Connects disjoint refined spline segments for each NE label into a 
    single, continuous, periodic spline.
    
    Args:
        refined_splines_dict (dict): {fov_id: {ne_label: {seg_label: BSpline_object}}}
        config_dict (dict): The 'ne_fit' section of the config.

    Returns:
        bridged_splines_output: Dict with structure:
            {fov_id: {ne_label: {
                'data_segments': [BSpline, ...],
                'bridge_segments': [BSpline, ...],
                'full_periodic_spline': BSpline,
                'u_ranges': {
                    'data': [(u_start, u_end), ...],
                    'bridge': [(u_start, u_end), ...]
                }
            }}}
    """
    
    bridged_splines_output = {}
    
    # Get bridging parameters from config
    ne_fit_cfg = config_dict.get('ne_fit', {})
    final_smoothing = ne_fit_cfg.get('bridge_smoothing_factor', 1.0)
    sampling_density = ne_fit_cfg.get('refinement', {}).get('final_sampling_density', 64)
    # Bridge validation parameters
    max_angle_change_deg = ne_fit_cfg.get('max_curvature_angle_deg', 1.0)  # Same as refinement threshold (based on NE biophysics)
    min_gap_threshold = ne_fit_cfg.get('bridge_min_gap_pixels', 1.0) # Minimum gap distance (pixels) to bridge

    logger.info(f"Bridging parameters: smoothing={final_smoothing}, max_angle={max_angle_change_deg}°, min_gap={min_gap_threshold}px")
    if not refined_splines_dict:
        logger.warning("Refined splines dictionary is None or empty.")
        return bridged_splines_output

    for fov_id, ne_labels_dict in refined_splines_dict.items():
        if not ne_labels_dict:
            logger.warning(f"No NE labels found for FoV {fov_id}.")
            continue
            
        bridged_splines_output[fov_id] = {}
        
        for ne_label, segments in ne_labels_dict.items():
            if not segments:
                logger.warning(f"No segments found for {fov_id} / {ne_label}.")
                continue
                
            num_segments = len(segments)
            
            if num_segments == 0:
                continue
            
            # --- CASE 1: Single segment ---
            if num_segments == 1:
                # Get the only segment, regardless of label format
                seg_key = list(segments.keys())[0]
                seg_spline = segments[seg_key]
                
                if seg_spline is None:
                    logger.warning(f"No data for {fov_id}/{ne_label}/{seg_key}.")
                    continue
                
                # seg_spline is already a BSpline object
                try:
                    # For single segment, create closure bridge
                    p_end = seg_spline(1.0)
                    p_start = seg_spline(0.0)
                    tan_end = seg_spline.derivative(1)(1.0)
                    tan_start = seg_spline.derivative(1)(0.0)
                    
                    gap_distance = np.linalg.norm(p_start - p_end)
                    
                    if gap_distance < min_gap_threshold:
                        logger.info(f"{fov_id}/{ne_label}: Gap too small ({gap_distance:.2f}px), treating as already closed")
                        # Just use the data segment as periodic
                        seg_points, _ = sample_bspline(seg_spline, sampling_density)
                        seg_points = np.array(seg_points)
                        if seg_points.ndim == 3:
                            seg_points = seg_points.squeeze()
                        if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                            seg_points = seg_points.T
                        
                        periodic_spline = fit_parametric_spline(seg_points, smoothing=final_smoothing, periodic=True)
                        
                        bridged_splines_output[fov_id][ne_label] = {
                            'data_segments': [seg_spline],
                            'bridge_segments': [],
                            'full_periodic_spline': periodic_spline,
                            'u_ranges': {
                                'data': [(0.0, 1.0)],
                                'bridge': []
                            }
                        }
                        continue
                    
                    bridge_points = build_curve_bridge(p_end, p_start, tan_end, tan_start)
                    bridge_spline = fit_parametric_spline(bridge_points, smoothing=0, periodic=False)
                    
                    # VALIDATE BRIDGE CURVATURE
                    is_valid, max_angle = validate_bridge_curvature(bridge_spline, max_angle_change_deg)
                    if not is_valid:
                        logger.warning(f"{fov_id}/{ne_label}: Bridge failed curvature check (max_angle={max_angle:.3f}°), skipping bridge")
                        # Use data segment only
                        seg_points, _ = sample_bspline(seg_spline, sampling_density)
                        seg_points = np.array(seg_points)
                        if seg_points.ndim == 3:
                            seg_points = seg_points.squeeze()
                        if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                            seg_points = seg_points.T
                        
                        periodic_spline = fit_parametric_spline(seg_points, smoothing=final_smoothing, periodic=True)
                        
                        bridged_splines_output[fov_id][ne_label] = {
                            'data_segments': [seg_spline],
                            'bridge_segments': [],
                            'full_periodic_spline': periodic_spline,
                            'u_ranges': {
                                'data': [(0.0, 1.0)],
                                'bridge': []
                            }
                        }
                        continue
                    
                    logger.info(f"{fov_id}/{ne_label}: Bridge validated (gap={gap_distance:.2f}px, max_angle={max_angle:.3f}°)")
                    
                    # Build periodic spline with validated bridge
                    seg_points, _ = sample_bspline(seg_spline, sampling_density)
                    seg_points = np.array(seg_points)
                    if seg_points.ndim == 3:
                        seg_points = seg_points.squeeze()
                    if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                        seg_points = seg_points.T  # Transpose from (2, N) to (N, 2)

                    all_points = np.concatenate((seg_points[:-1], bridge_points[1:-1]), axis=0)
                    periodic_spline = fit_parametric_spline(all_points, smoothing=final_smoothing, periodic=True)
                    
                    seg_len = get_spline_arc_length(seg_spline)
                    bridge_len = get_spline_arc_length(bridge_spline)
                    total_len = seg_len + bridge_len
                    u_data_end = seg_len / total_len
                    
                    bridged_splines_output[fov_id][ne_label] = {
                        'data_segments': [seg_spline],
                        'bridge_segments': [bridge_spline],
                        'full_periodic_spline': periodic_spline,
                        'u_ranges': {
                            'data': [(0.0, u_data_end)],
                            'bridge': [(u_data_end, 1.0)]
                        }
                    }
                except Exception as e:
                    logger.error(f"Failed to bridge single segment {fov_id}/{ne_label}: {e}")
                    logger.error(traceback.format_exc())
                continue

            # --- CASE 2: Multiple segments ---
            data_segments = []
            bridge_segments = []
            all_points_for_periodic_fit = []
            segment_lengths = []
            skipped_bridges = []
            
            try:
                # Filter to only segment keys (exclude any metadata)
                seg_keys_only = [k for k in segments.keys() if k.startswith('segment_')]
                sorted_seg_keys = sorted(seg_keys_only, key=lambda x: int(x.split('_')[-1]))

                for i in range(len(sorted_seg_keys)):
                    current_seg_key = sorted_seg_keys[i]
                    next_seg_key = sorted_seg_keys[(i + 1) % len(sorted_seg_keys)]
                    
                    # segments[key] is already a BSpline object
                    current_spline = segments[current_seg_key]
                    next_spline = segments[next_seg_key]
                    
                    data_segments.append(current_spline)
                    
                    data_seg_len = get_spline_arc_length(current_spline)
                    segment_lengths.append(('data', data_seg_len))
                    
                    p_end = current_spline(1.0)
                    p_start = next_spline(0.0)
                    tan_end = current_spline.derivative(1)(1.0)
                    tan_start = next_spline.derivative(1)(0.0)
                    
                    gap_distance = np.linalg.norm(p_start - p_end)
                    
                    # Check if gap is too small to bridge
                    if gap_distance < min_gap_threshold:
                        logger.info(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Gap too small ({gap_distance:.2f}px), skipping bridge")
                        skipped_bridges.append(i)
                        
                        # Add data segment points only
                        seg_points, _ = sample_bspline(current_spline, sampling_density)
                        seg_points = np.array(seg_points)
                        if seg_points.ndim == 3:
                            seg_points = seg_points.squeeze()
                        if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                            seg_points = seg_points.T
                        all_points_for_periodic_fit.append(seg_points[:-1])
                        continue
                    
                    bridge_points = build_curve_bridge(p_end, p_start, tan_end, tan_start)
                    
                    if bridge_points.size == 0:
                        logger.warning(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Bridge generation failed")
                        skipped_bridges.append(i)
                        seg_points, _ = sample_bspline(current_spline, sampling_density)
                        seg_points = np.array(seg_points)
                        if seg_points.ndim == 3:
                            seg_points = seg_points.squeeze()
                        if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                            seg_points = seg_points.T

                        all_points_for_periodic_fit.append(seg_points[:-1])
                        continue
                    
                    bridge_spline = fit_parametric_spline(bridge_points, smoothing=0, periodic=False)
                    
                    # VALIDATE BRIDGE CURVATURE
                    is_valid, max_angle = validate_bridge_curvature(bridge_spline, max_angle_change_deg)
                    if not is_valid:
                        logger.warning(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Bridge failed curvature (max_angle={max_angle:.3f}°), skipping")
                        skipped_bridges.append(i)
                        seg_points, _ = sample_bspline(current_spline, sampling_density)
                        seg_points = np.array(seg_points)
                        if seg_points.ndim == 3:
                            seg_points = seg_points.squeeze()
                        if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                            seg_points = seg_points.T
                        all_points_for_periodic_fit.append(seg_points[:-1])
                        continue
                    
                    logger.info(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Bridge validated (gap={gap_distance:.2f}px, max_angle={max_angle:.3f}°)")
                    
                    # Bridge is valid, add it
                    bridge_segments.append(bridge_spline)
                    bridge_len = get_spline_arc_length(bridge_spline)
                    segment_lengths.append(('bridge', bridge_len))
                    
                    seg_points, _ = sample_bspline(current_spline, sampling_density)
                    seg_points = np.array(seg_points)
                    if seg_points.ndim == 3:
                        seg_points = seg_points.squeeze()
                    if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                        seg_points = seg_points.T

                    all_points_for_periodic_fit.append(seg_points[:-1])
                    all_points_for_periodic_fit.append(bridge_points[1:-1])

                if not all_points_for_periodic_fit:
                    logger.error(f"No points for periodic spline {fov_id}/{ne_label}.")
                    continue
                
                if skipped_bridges:
                    logger.info(f"{fov_id}/{ne_label}: Skipped {len(skipped_bridges)} bridges due to validation failures")
                    
                all_points = np.concatenate(all_points_for_periodic_fit, axis=0)
                periodic_spline = fit_parametric_spline(all_points, smoothing=final_smoothing, periodic=True)
                
                # Calculate u-ranges
                total_length = sum(length for _, length in segment_lengths)
                u_ranges_data = []
                u_ranges_bridge = []
                
                current_u = 0.0
                for seg_type, seg_length in segment_lengths:
                    u_start = current_u
                    u_end = current_u + (seg_length / total_length)
                    
                    if seg_type == 'data':
                        u_ranges_data.append((u_start, u_end))
                    else:
                        u_ranges_bridge.append((u_start, u_end))
                    
                    current_u = u_end
                
                bridged_splines_output[fov_id][ne_label] = {
                    'data_segments': data_segments,
                    'bridge_segments': bridge_segments,
                    'full_periodic_spline': periodic_spline,
                    'u_ranges': {
                        'data': u_ranges_data,
                        'bridge': u_ranges_bridge
                    }
                }
            except Exception as e:
                logger.error(f"Failed to bridge multi-segment {fov_id}/{ne_label}: {e}")
                logger.error(traceback.format_exc())

    return bridged_splines_output


def identify_segment_type_at_u(u_value, u_ranges_dict):
    """
    Given a u parameter value, returns 'data', 'bridge', or None.
    """
    u_value = u_value % 1.0
    
    for u_start, u_end in u_ranges_dict['data']:
        if u_start <= u_value <= u_end:
            return 'data'
    
    for u_start, u_end in u_ranges_dict['bridge']:
        if u_start <= u_value <= u_end:
            return 'bridge'
    
    return None