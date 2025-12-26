import math
import numpy as np
import skimage.measure

from scipy.interpolate import BSpline, splprep
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def get_u_range_from_bspline(bspline_obj, u_density):
    bspline_length = get_spline_arc_length(bspline_obj)

    if bspline_length < 1e-6:
        raise ValueError('Length of bspline is approaching zero.')
        return

    # number of points needed to cover the given arc length (in pixels)
    n_target_samples = np.round(bspline_length * u_density).astype(int)

    if n_target_samples < 2:
        n_target_samples = 2
    
    # equidistant in pixel space
    target_distances = np.linspace(0, bspline_length, n_target_samples)

    # high-res lookup table
    n_lut_samples = max(1000, n_target_samples * 10)
    # divide parameter space into the number of points desired
    u_dense = np.linspace(0, 1, n_lut_samples)
    # find where those points evaluate on the bspline object
    points_dense = np.array(bspline_obj(u_dense))

    if points_dense.shape[0] < points_dense.shape[1]:
        points_dense = points_dense.T

    segment_lengths = np.linalg.norm(np.diff(points_dense, axis=0), axis=1)
    cumulative_arc_lengths_lut = np.insert(np.cumsum(segment_lengths), 0, 0)
    cumulative_arc_lengths_lut = (
        cumulative_arc_lengths_lut * (bspline_length / cumulative_arc_lengths_lut[-1])
    )

    # interpolate using desired distance between points, distances from the lookup table, and the u values that produced the lookup table distances
    target_parameters = np.interp(target_distances, cumulative_arc_lengths_lut, u_dense)

    # Ensure start and end points are correct
    target_parameters[0] = 0.0
    target_parameters[-1] = 1.0

    return target_parameters

def get_spline_arc_length(bspline_obj):
    def speed(t):
        dxdy = bspline_obj(t, nu=1)
        # return with a tiny epsilon to avoid subsequent division by zero
        return np.hypot(dxdy[0], dxdy[1]) + 1e-9
    
    knots = bspline_obj.t
    interior_knots = np.unique(knots[(knots > 0) & (knots < 1)])
    # calc number of knots + a buffer versus the default max of the quad function
    required_limit = max(50, len(interior_knots) + 10)
    # Integrate the speed function from t=0 to t=1; use knots as boundaries 
    # The [0] at the end gets the result, [1] would be the error estimate
    pixel_length, _ = quad(speed, 0, 1, points=interior_knots, limit=required_limit)
    return pixel_length

def calc_bspline_curvature(bspline_obj, u_param_to_test):
    eval_bspline_obj_d1 = bspline_obj(u_param_to_test, nu=1)
    eval_bspline_obj_d2 = bspline_obj(u_param_to_test, nu=2)

    # assign to variables for readability
    dy = eval_bspline_obj_d1[:,1]
    dx = eval_bspline_obj_d1[:,0]

    ddy = eval_bspline_obj_d2[:,1]
    ddx = eval_bspline_obj_d2[:,0]

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    return curvature

# !!!: this is done better in get_u_range_from_bsplines re: calc arc length -- uses scipy quad
def points_by_arc_len_parm_robust(bspline_obj: BSpline, n_points: int, n_samples: int = 1000):
    u_dense = np.linspace(0, 1, n_samples)
    points_dense = np.array(bspline_obj(u_dense))

    if points_dense.shape[0] < points_dense.shape[1]:
        points_dense = points_dense.T

    segment_lengths = np.linalg.norm(np.diff(points_dense, axis=0), axis=1)
    cumulative_arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative_arc_lengths[-1]

    # Keep the zero-length check, as this is a valid edge case
    if total_length < 1e-6:
        print("Warning: B-spline has zero length...")
        start_point = bspline_obj(0.0).reshape(-1, 1)
        return np.tile(start_point, n_points)

    target_distances = np.linspace(0, total_length, n_points)
    target_parameters = np.interp(target_distances, cumulative_arc_lengths, u_dense)
    equidistant_points = bspline_obj(target_parameters)

    return np.array(equidistant_points)


def bspline_from_tck(tck, is_periodic = False):
    bspline_knots = tck[0]
    bspline_ctrlpts = np.array(tck[1])
    if np.array(tck[1]).shape[0] < np.array(tck[1]).shape[1]:
        bspline_ctrlpts = bspline_ctrlpts.T
    bspline_degree = tck[2]
    bspline_obj = BSpline(bspline_knots, bspline_ctrlpts, bspline_degree, extrapolate = is_periodic)

    return bspline_obj

def transform_coordinates(coordinates, scale_factor, angle, translation_vector, center, angle_units = 'radians'):
    # Scaling matrix
    scale_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

    # Rotation matrix
    angle_radians = math.radians(angle) if angle_units == 'degrees' else angle
    # Down → Left is Clockwise
    # rotation_matrix = np.array([[math.cos(angle_radians), -math.sin(angle_radians)],
    #                             [math.sin(angle_radians), math.cos(angle_radians)]])

    # Rotation matrix for Top-Left Origin (Y-Down) with (Row, Col) inputs
    #    Invert the sine terms to achieve Visual Counter-Clockwise rotation
    #    matching standard image processing libraries.
    angle_radians = math.radians(angle) if angle_units == 'degrees' else angle
    rotation_matrix = np.array([
        [ math.cos(angle_radians), math.sin(angle_radians)],  
        [-math.sin(angle_radians), math.cos(angle_radians)]   
    ])
    # Down → Right is Counter-Clockwise
    rotation_matrix = np.array([
        [ math.cos(angle_radians), math.sin(angle_radians)],  # Row maps to Row*cos + Col*sin
        [-math.sin(angle_radians), math.cos(angle_radians)]   # Col maps to -Row*sin + Col*cos
    ])

    # Combine scaling and rotation matrices
    transformation_matrix = np.dot(rotation_matrix, scale_matrix)

    # Apply the scaling and rotation
    scaled_rotated_coords = np.dot(coordinates - center, transformation_matrix) + center

    # Apply the translation
    transformed_coords = scaled_rotated_coords + translation_vector

    return transformed_coords

def bspline_transformation(orig_bspline, reg_values, center_coords = [0,0]):
    orig_bspline_knots = orig_bspline.t
    new_ctrl_pts = transform_coordinates(orig_bspline.c, reg_values['scale'], reg_values['angle'], reg_values['shift_vector'], center_coords)
    orig_bspline_degree = orig_bspline.k
    transformed_bspline = BSpline(orig_bspline_knots, new_ctrl_pts, orig_bspline_degree, extrapolate = 'periodic')
    return transformed_bspline

def find_min_distance_from_point_to_spline(point, bspline_obj):
    def objective_func(u):
        # 'splev' evaluates the spline at parameter 'u'
        curve_pt = bspline_obj(u)
        eucl_sq_dist = np.sum((np.array(point) - np.array(curve_pt))**2)
        return eucl_sq_dist
    opt_result = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')
    return opt_result

def calc_dist_bt_bsplines(bspline_objA, bspline_objB, N_sample_points):
    sample_params = np.linspace(0, 1, N_sample_points)
    points_on_spline1 = bspline_objA(sample_params)

    points_on_spline2 = np.ones(points_on_spline1.shape)
    distances = np.ones(points_on_spline1.shape[0])

    for index, pt in enumerate(points_on_spline1):
        min_dist_opt_result = find_min_distance_from_point_to_spline(pt, bspline_objB)
        # 'x' is the optimal parameter 'u' from the result
        pt_from_opt_param = bspline_objB(min_dist_opt_result.x)
        # place the coord assoc w the opt param in the same index position in the points_on_spline2 array as the pt was located in the points_on_spline1 array
        points_on_spline2[index, :] = pt_from_opt_param
        distances[index] = np.sqrt(min_dist_opt_result.fun)

    print("--- Distance Calculation Results ---")
    print(f"Average distance between splines: {np.mean(distances):.4f}")
    print(f"Maximum distance between splines: {np.max(distances):.4f}")
    print("-" * 35)

    return distances

def shift_bspline(bspline_obj, offset_vector):
    """
    Translates a B-spline by a fixed vector.
    Used for moving from Local Crop coordinates to Global Image coordinates.
    
    Args:
        bspline_obj: scipy.interpolate.BSpline
        offset_vector: List or array of offsets matching the spline dimensionality 
                        (e.g., [row_offset, col_offset])
    """
    # Create copy of control points to avoid modifying original
    new_ctrlpts = bspline_obj.c.copy()
    
    # Broadcast addition
    new_ctrlpts += np.array(offset_vector)
    
    # Return new BSpline object
    return BSpline(bspline_obj.t, new_ctrlpts, bspline_obj.k, extrapolate=bspline_obj.extrapolate)


def mirror_bspline(bspline_obj, boundary_dim, axis=1):
    """
    Reflects a B-spline across a boundary (e.g., width or height).
    CRITICAL: Also reverses the control point array to maintain curve winding 
    (prevents the spline from twisting into a figure-8).

    Args:
        bspline_obj: scipy.interpolate.BSpline
        boundary_dim: The dimension of the box to flip within (e.g. image width)
        axis: The column index to modify. 
                0 = Row/Y-flip (Vertical)
                1 = Col/X-flip (Horizontal)
    """
    new_ctrlpts = bspline_obj.c.copy()
    
    # 1. Flip the coordinate: new = width - old
    new_ctrlpts[:, axis] = boundary_dim - new_ctrlpts[:, axis]
    
    # 2. Reverse the sequence of control points
    # When you mirror a shape, Clockwise becomes Counter-Clockwise. 
    # We must reverse the array to preserve the original winding direction.
    new_ctrlpts = np.flip(new_ctrlpts, axis=0)
    
    return BSpline(bspline_obj.t, new_ctrlpts, bspline_obj.k, extrapolate=bspline_obj.extrapolate)

def calculate_signed_distances(bspline_A, bspline_B, N_samples):

    u_poly = np.linspace(0, 1, 2000) # High resolution for accuracy
    poly_points = bspline_A(u_poly)
    #polygon_A = np.fliplr(poly_points) # Convert from (x,y) to (y,x) for points_in_poly
    polygon_A = poly_points
    # 2. Sample N points along the reference spline (A) where we will measure the distance
    u_samples = np.linspace(0, 1, N_samples)
    points_A = bspline_A(u_samples)

    # 3. For each point on A, find the closest point on B and calculate the distance
    distances = []
    closest_points_B = []
    for p_A in points_A:
        # Define a function that returns the squared distance from p_A to a point on bspline_B(u)
        dist_sq_func = lambda u: np.sum((p_A - bspline_B(u))**2)
        
        # Find the parameter 'u' on spline B that minimizes this distance
        res = minimize_scalar(dist_sq_func, bounds=(0, 1), method='bounded')
        
        # Get the coordinate of the closest point on B
        p_B_closest = bspline_B(res.x)
        closest_points_B.append(p_B_closest)
        
        # Calculate the Euclidean distance
        dist = np.linalg.norm(p_A - p_B_closest)
        distances.append(dist)

    closest_points_B = np.array(closest_points_B)
    distances = np.array(distances)
    
    # 4. Determine the sign for each distance
    #    Convert closest_points_B from (x,y) to (y,x) for the check
    # points_to_check = np.fliplr(closest_points_B)
    points_to_check = closest_points_B
    in_out_flags = skimage.measure.points_in_poly(points_to_check, polygon_A)

    # Create a sign array: +1 for inside (True), -1 for outside (False)
    signs = np.where(in_out_flags, 1, -1)

    # 5. Return the final signed distances
    return distances * signs

def find_closest_u_on_spline(point, bspline):
    """Finds the parameter u on a B-spline closest to a given (y, x) point."""
    # Note: Scipy splines expect (x, y) order, so we flip the point.
    point_xy = np.flip(point)
    
    def dist_sq_func(u):
        spline_point_xy = bspline(u)
        return np.sum((point_xy - spline_point_xy)**2)
        
    res = minimize_scalar(dist_sq_func, bounds=(0, 1), method='bounded')
    return res.x

def u_values_to_ranges(u_values, gap_threshold=0.05):
    """
    Converts a list of u-values on a periodic spline into continuous ranges.
    A gap is defined as a jump in sorted u-values larger than the threshold.
    """
    if len(u_values) < 2:
        return []

    # Sort the u values
    u_sorted = np.sort(np.array(u_values))
    
    # Calculate the difference between adjacent sorted values
    diffs = np.diff(u_sorted)
    
    # Find indices where the jump is larger than the threshold (these are gaps)
    gap_indices = np.where(diffs > gap_threshold)[0]
    
    # If there are no gaps, it's one continuous range
    if len(gap_indices) == 0:
        return [(u_sorted[0], u_sorted[-1])]
        
    # Create ranges based on the gap locations
    ranges = []
    start_idx = 0
    for gap_idx in gap_indices:
        ranges.append((u_sorted[start_idx], u_sorted[gap_idx]))
        start_idx = gap_idx + 1
    ranges.append((u_sorted[start_idx], u_sorted[-1]))
    
    # Handle the wrap-around case for periodic splines
    # If the distance between the last and first point is not a gap, merge them.
    if (1.0 - u_sorted[-1] + u_sorted[0]) <= gap_threshold and len(ranges) > 1:
        # Merge the last range with the first range
        last_range_start, _ = ranges.pop()
        first_range_start, first_range_end = ranges.pop(0)
        # The new range starts at the last segment and ends at the first
        ranges.append((last_range_start, first_range_end))
        
    return ranges

def calculate_range_overlap(range1, range2):
    """Calculates the overlap length between two ranges (tuples)."""
    start1, end1 = range1
    start2, end2 = range2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    return 0

def build_curve_bridge(p_end, p_start, tangent_end, tangent_start, tangent_scale=0.3):
    """
    Completes a curve using a cubic Bezier bridge and pre-calculated tangents.
    
    Args:
        p_end (np.ndarray): 1x2 array for the end of the segment / 
                                starting point of the bridge (y, x).
        p_start (np.ndarray): 1x2 array for the start of the segment / 
                                  ending point of the brdige (y, x).
        tangent_end (np.ndarray): 1x2 array for the derivative at p_end (dy, dx).
        tangent_start (np.ndarray): 1x2 array for the derivative at p_start (dy, dx).
        tangent_scale (float): Controls influence of tangents on the curve.
    
    Returns:
        np.ndarray: The new points forming the bridge.
    """
    # 1. Normalize the provided tangents
    tangent_end = tangent_end / np.linalg.norm(tangent_end)
    tangent_start = tangent_start / np.linalg.norm(tangent_start)

    # 2. Calculate control points based on tangents
    dist = np.linalg.norm(p_start - p_end)
    p1 = p_end + tangent_end * dist * tangent_scale
    # Note: The tangent at the destination point (p_start)
    #       needs to point away from the curve.
    #       Since we are connecting the end of a path to its start,
    #       the tangent at the start point is already pointing in
    #       the correct "away" direction.
    p2 = p_start - tangent_start * dist * tangent_scale

    # 3. Generate the Bezier curve points
    t = np.linspace(0, 1, 50)[:, np.newaxis]
    bezier_bridge = (1-t)**3 * p_end + 3*(1-t)**2 * t * p1 + \
        3*(1-t) * t**2 * p2 + t**3 * p_start
    
    return bezier_bridge

def reconstruct_periodic_spline(segments_dict):
    """
    Takes a dictionary of refined, non-periodic spline segments and stitches them
    together to form a single, periodic B-spline.
    """
    if not segments_dict or len(segments_dict) < 2:
        # If there's only one segment, we can't make a periodic loop
        # You might return the single segment or handle as needed
        if segments_dict:
            return list(segments_dict.values())[0]['bspline_object']
        return None

    # 1. Sort the segments based on their original position ('original_u_range')
    sorted_segments = sorted(segments_dict.values(), key=lambda s: s['original_u_range'][0])
    
    all_points = []
    
    # 2. Iterate through segments and bridge the gaps
    for i in range(len(sorted_segments)):
        current_segment = sorted_segments[i]
        next_segment = sorted_segments[(i + 1) % len(sorted_segments)] # Wrap around to the first

        # Get the point cloud for the current segment
        current_points = points_by_arc_len_parm_robust(current_segment['bspline_object'], n_points=100)
        all_points.append(current_points)
        
        # Get the end of the current segment and start of the next
        p_end = current_points[-1]
        p_start = points_by_arc_len_parm_robust(next_segment['bspline_object'], n_points=2)[0]
        
        # Get the tangents at the endpoints
        tangent_end = current_segment['bspline_object'].derivative(1)(1.0)
        tangent_start = next_segment['bspline_object'].derivative(1)(0.0)

        # 3. Use your existing bridge function to fill the gap
        bridge_points = build_curve_bridge(p_end, p_start, tangent_end, tangent_start)
        if bridge_points.size > 0:
            all_points.append(bridge_points[1:-1]) # Exclude endpoints to avoid duplication

    # 4. Fit a new, single, periodic spline to all the points
    full_contour = np.concatenate(all_points, axis=0)
    
    # splprep expects (x, y) coordinates
    tck_final, _ = splprep([full_contour[:, 0], full_contour[:, 1]], s=1.0, k=3, per=True)
    
    final_periodic_spline = bspline_from_tck(tck_final, is_periodic=True)
    
    return final_periodic_spline

# ??? should this throw a warning re: potential for wonky results if the max angle is smaller than some value? Like less than pi or...?
def angular_sort_alg(points):
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 0] - centroid[0], points[:, 1] - centroid[1])
    sort_indices = np.argsort(angles)
    return sort_indices