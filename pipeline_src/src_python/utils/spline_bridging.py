"""
Contains the logic for bridging disjoint refined spline segments into
a single, continuous, periodic spline.
"""
import logging
import traceback
import numpy as np
from scipy.interpolate import make_splprep, splprep, BSpline
from scipy.integrate import quad
from scipy.optimize import brentq # Root finding to map Length -> t

from tools.geom_tools import build_curve_bridge, get_spline_arc_length
from tools.utility_functions import sample_bspline


logger = logging.getLogger(__name__)

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

def validate_segment_orientation(spline_obj, nucleus_center, threshold=0.7):
    """
    detects 'Spokes' (lines pointing at the center) vs 'Arcs' (lines curving around).
    
    Args:
        threshold: Dot product threshold. 
                   0.0 = Perfectly Perpendicular (Good)
                   1.0 = Perfectly Parallel (Bad/Spike)
                   > 0.7 usually indicates a spike.
    """
    try:
        # Sample the midpoint of the segment
        mid_point = spline_obj(0.5)
        
        # 1. Calculate Radial Vector (Center -> Point)
        radial_vec = mid_point - nucleus_center
        radial_vec /= (np.linalg.norm(radial_vec) + 1e-6)
        
        # 2. Calculate Tangent Vector (Direction of line)
        tangent = spline_obj.derivative(1)(0.5)
        tangent /= (np.linalg.norm(tangent) + 1e-6)
        
        # 3. Check Alignment (Dot Product)
        # If abs(dot) is high (~1), the line points AT the center (BAD)
        # If abs(dot) is low (~0), the line goes AROUND the center (GOOD)
        alignment = np.abs(np.dot(radial_vec, tangent))
        
        is_valid = alignment < threshold
        return is_valid, alignment
    except Exception as e:
        return False, 1.0

def get_segment_centroid_and_angle(spline, center_ref):
    """Calculates the mean position and angular position of a spline segment."""
    u_vals = np.linspace(0.1, 0.9, 10)
    pts = np.array([spline(u) for u in u_vals])
    centroid = np.mean(pts, axis=0)
    angle = np.arctan2(centroid[0] - center_ref[0], centroid[1] - center_ref[1])
    return centroid, angle

def fit_parametric_spline(points_yx, smoothing=1.0, periodic=False):
    """
    Detects 'Spokes' (lines pointing at the center) vs 'Arcs' (lines curving around).
    Used to filter out collapsed or spiky segments from the refinement step.
    
    Args:
        points_yx: (N, 2) array of [y, x] coordinates
        smoothing: smoothing parameter
        periodic: whether to create periodic spline
        
    Returns:
        BSpline object
    """    
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

def precise_trim_spline(spline_obj, trim_length_pixels):
    """
    Trims a specific arc-length off BOTH ends of a spline.
    
    Args:
        spline_obj: The scipy BSpline object.
        trim_length_pixels: Length to remove from each end (float).
        
    Returns:
        BSpline: A new, shorter BSpline object, or None if the segment 
                 is shorter than 2x the trim length.
    """
    if trim_length_pixels <= 0:
        return spline_obj

    # 1. Define Speed Function (derivative magnitude)
    def speed(t):
        # derivative(1) returns the velocity vector (dx/dt, dy/dt)
        d = spline_obj.derivative(1)(t)
        return np.sqrt(d[0]**2 + d[1]**2)

    # 2. Calculate Total Arc Length
    total_length, _ = quad(speed, 0, 1)

    # 3. Safety Check: Don't delete the whole segment
    # We require at least some "core" segment to remain.
    if total_length <= (2.1 * trim_length_pixels):
        # Segment is entirely noise/end-effects. Kill it.
        return None

    # 4. Find new t_start and t_end
    # We want to find t such that Integral(0 to t) = trim_length
    
    # Target lengths
    target_start = trim_length_pixels
    target_end = total_length - trim_length_pixels

    # Function to minimize: Integral(0 to t) - Target
    def length_residual(t, target):
        L, _ = quad(speed, 0, t)
        return L - target

    try:
        # Find t where length matches trim_length
        t_start = brentq(length_residual, 0, 1, args=(target_start,))
        t_end = brentq(length_residual, 0, 1, args=(target_end,))
    except ValueError:
        # Fallback if solver fails (rare on smooth splines)
        # Linear approximation: t ~ L / Total_L
        t_start = target_start / total_length
        t_end = target_end / total_length

    # 5. Create new Spline (Reparameterize)
    # We can't just slice a BSpline. We sample the "good" region and refit.
    # This preserves the geometric shape without the artifacts.
    
    # High density sample of the VALID region
    u_new = np.linspace(t_start, t_end, 50) 
    new_points = spline_obj(u_new) # Shape (2, 50)
    
    # Refit non-periodic, passing through these points
    # We use s=0 to force it to stick to the original curve geometry
    # Transpose new_points to match fit_parametric_spline expectations (N, 2)
    new_spline = fit_parametric_spline(new_points.T, smoothing=0, periodic=False)
    
    return new_spline

def bridge_refined_splines(refined_splines_dict, config_dict, reg_prec_map = None):
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


    # Bridge Validation
    # Same as refinement threshold (based on NE biophysics)
    max_angle_change_deg = ne_fit_cfg.get('max_curvature_angle_deg', 1.0)
    # Minimum gap distance (pixels) to bridge
    min_gap_threshold = ne_fit_cfg.get('bridge_min_gap_pixels', 1.0)
    # tangent orientation
    # If dot product > 0.6, the line is pointing too much at the center.
    orientation_threshold = 0.6
    # Confidence factor for trimming (default 2.0 sigma)
    trim_sigma_k = config_dict.get('ne_fit', {}).get('trim_sigma_k', 2.0)

    logger.info(f"Bridging parameters: smoothing={final_smoothing}, max_angle={max_angle_change_deg}°, min_gap={min_gap_threshold}px")
    
    if not refined_splines_dict:
        logger.warning("Refined splines dictionary is None or empty.")
        return bridged_splines_output

    for fov_id, ne_labels_dict in refined_splines_dict.items():
        if not ne_labels_dict:
            logger.warning(f"No NE labels found for FoV {fov_id}.")
            continue
            
        bridged_splines_output[fov_id] = {}

        # Determine registration precision for this FoV if available
        fov_reg_prec = {}
        if reg_prec_map and fov_id in reg_prec_map:
            fov_reg_prec = reg_prec_map[fov_id]

        for ne_label, segments in ne_labels_dict.items():
            if not segments:
                logger.warning(f"No segments found for {fov_id} / {ne_label}.")
                continue
            
            # --- PRE-BRIDGING: TRIM AND FILTER SEGMENTS ---
            num_segments = len(segments)
            
            if num_segments == 0:
                continue
            
            # --- CALCULATE APPROXIMATE CENTER ---
            # finds a reference point to define "radial" vs "tangential"; proxy - the centroid of ALL control points from ALL segments
            all_control_points = []
            for spline in segments.values():
                # Extract control points (handles typical scipy shape variations)
                pts = spline.c
                if pts.shape[0] == 2 and pts.shape[1] > 2: 
                    pts = pts.T
                all_control_points.append(pts)
            
            if not all_control_points:
                continue
            
            all_cp_stacked = np.vstack(all_control_points)
            nucleus_center = np.mean(all_cp_stacked, axis=0) # [y, x]

            # --- PRE-PROCESS SEGMENTS (Trim & Filter) ---
            processed_segments = {}
            seg_keys_only = [k for k in segments.keys() if k.startswith('segment_')]

            # ne_label level sigma
            sigma = 0.5 # default (0.5 px)
            if isinstance(fov_reg_prec, dict) and ne_label in fov_reg_prec:
                # Check if the map has per-label structure
                if isinstance(fov_reg_prec[ne_label], dict) and 'sigma_reg' in fov_reg_prec[ne_label]:
                    sigma = fov_reg_prec[ne_label]['sigma_reg']
                elif isinstance(fov_reg_prec[ne_label], float):
                    sigma = fov_reg_prec[ne_label]
            elif isinstance(fov_reg_prec, dict) and 'sigma_reg' in fov_reg_prec:
                # FoV-level fallback
                sigma = fov_reg_prec['sigma_reg']

            trim_px = sigma * trim_sigma_k

            for key in seg_keys_only:
                spline = segments[key]
                
                # PHYSICS FILTER: Precision trim ends based on registration uncertainty
                if trim_px > 0:
                    trimmed_spline = precise_trim_spline(spline, trim_px)
                    if trimmed_spline is None:
                        logger.info(f"{fov_id}/{ne_label}: Segment {key} removed by trim (shorter than noise floor)")
                        continue
                    spline = trimmed_spline

                # GEOMETRIC FILTER: Orientation Check to remove "spokes" that point toward the center
                is_valid_orient, dot_prod = validate_segment_orientation(
                    spline, 
                    nucleus_center, 
                    threshold=orientation_threshold
                )

                if not is_valid_orient:
                    logger.warning(f"{fov_id}/{ne_label}: Segment {key} removed (Orientation: Spoke-like, dot={dot_prod:.2f})")
                    continue

                processed_segments[key] = spline
            
            if not processed_segments:
                logger.warning(f"{fov_id}/{ne_label}: All segments removed after filtering.")
                continue

            # Update segments list for the bridging logic
            segments = processed_segments
            num_segments = len(segments)

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
                    
                    # Add data segment points
                    seg_points, _ = sample_bspline(current_spline, sampling_density)
                    seg_points = np.array(seg_points)
                    if seg_points.ndim == 3:
                        seg_points = seg_points.squeeze()
                    if seg_points.shape[0] == 2 and seg_points.shape[1] > 2:
                        seg_points = seg_points.T
                    all_points_for_periodic_fit.append(seg_points[:-1])

                    # Check if gap is too small to bridge
                    if gap_distance < min_gap_threshold:
                        logger.info(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Gap too small ({gap_distance:.2f}px), skipping bridge")
                        skipped_bridges.append(i)
                        continue
                    
                    bridge_points = build_curve_bridge(p_end, p_start, tan_end, tan_start)
                    
                    if bridge_points.size == 0:
                        logger.warning(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Bridge generation failed")
                        skipped_bridges.append(i)
                        continue
                    
                    bridge_spline = fit_parametric_spline(bridge_points, smoothing=0, periodic=False)
                    
                    # VALIDATE BRIDGE CURVATURE
                    is_valid, max_angle = validate_bridge_curvature(bridge_spline, max_angle_change_deg)
                    if not is_valid:
                        logger.warning(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Bridge failed curvature (max_angle={max_angle:.3f}°), skipping")
                        skipped_bridges.append(i)
                        continue
                    
                    logger.info(f"{fov_id}/{ne_label} {current_seg_key}→{next_seg_key}: Bridge validated (gap={gap_distance:.2f}px, max_angle={max_angle:.3f}°)")
                    
                    # Bridge is valid, add it
                    bridge_segments.append(bridge_spline)
                    bridge_len = get_spline_arc_length(bridge_spline)
                    segment_lengths.append(('bridge', bridge_len))
                    
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
                traceback.format_exc()

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