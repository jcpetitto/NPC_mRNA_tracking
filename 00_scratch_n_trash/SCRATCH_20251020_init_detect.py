import numpy as np
from scipy.interpolate import splprep
# Assuming these functions are available in your environment,
# as seen in your npc_detect_initial.py
from tools.geom_tools import bspline_from_tck, find_closest_u_on_spline

def analyze_peak_locations( image: np.ndarray, skel_points: np.ndarray, peak_points: np.ndarray):

    # --- 1. Get intensity at each skeleton point ---
    # We index the image at all (y, x) locations provided by skel_points
    skel_intensities = image[skel_points[:, 0], skel_points[:, 1]]

    # --- 2. Make B-spline from skeleton points ---

    # CRITICAL: Skeleton points from np.where are not ordered.
    # We must sort them to form a continuous, non-self-intersecting path.
    # We use the same angular sorting method from your main script.
    centroid = np.mean(skel_points, axis=0)
    angles = np.arctan2(skel_points[:, 0] - centroid[0],
                       skel_points[:, 1] - centroid[1])
    sort_indices = np.argsort(angles)
    ordered_skel_points = skel_points[sort_indices]

    try:
        # Fit a periodic (closed) spline, interpolating through the points (s=0)
        # splprep expects points in [x, y] format
        tck_skel, _ = splprep(
            [ordered_skel_points[:, 1], ordered_skel_points[:, 0]],
            s=0, k=3, per=True
        )
        skel_bspline = bspline_from_tck(tck_skel, is_periodic=True)
    except Exception as e:
        print(f"Error: Could not fit skeleton spline - {e}")
        return [], None, None

    # --- 3. Determine distance and location for each peak ---
    peak_analysis_results = []

    for peak_yx in peak_points:
        # Convert (y, x) to (x, y) for spline calculations
        peak_xy = np.array([peak_yx[1], peak_yx[0]])

        # Get the intensity value at the peak
        peak_intensity = image[peak_yx[0], peak_yx[1]]

        # Find the parameter 'u' of the closest point on the spline
        u_closest = find_closest_u_on_spline(peak_xy, skel_bspline)

        # Get the (x, y) coordinates of that closest point
        spline_xy = skel_bspline(u_closest)

        # Vector from the spline to the peak
        vec_spline_to_peak = peak_xy - spline_xy

        # A) Calculate the Euclidean distance
        distance = np.linalg.norm(vec_spline_to_peak)

        # B) Determine "inside" vs. "outside"
        # Get the derivative (tangent) vector at that point
        deriv_xy = skel_bspline.derivative(1)(u_closest)  # [dx/du, dy/du]

        # Get the "outward" normal vector by rotating the tangent +90 deg
        # (This assumes a counter-clockwise path from angular sorting)
        normal_xy = np.array([deriv_xy[1], -deriv_xy[0]])

        # The dot product of the (spline->peak) vector and the normal vector
        # tells us if they point in the same (outside) or opposite (inside)
        # direction.
        dot_product = np.dot(vec_spline_to_peak, normal_xy)

        if abs(distance) < 1e-6:
            location = "on_spline"
        elif dot_product > 0:
            location = "outside"  # Convex side
        else:
            location = "inside"   # Concave side

        peak_analysis_results.append({
            'peak_yx': peak_yx,
            'peak_intensity': peak_intensity,
            'distance_to_spline': distance,
            'location': location,
            'dot_product_sign': np.sign(dot_product),
            'closest_spline_u': u_closest,
            'closest_spline_xy': spline_xy
        })

    return peak_analysis_results, skel_intensities, skel_bspline