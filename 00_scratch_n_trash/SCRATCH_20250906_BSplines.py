

# --- NURB
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.optimize import minimize_scalar
    # from geomdl import BSpline
    # from geomdl.fitting import approximate_curve
    # from geomdl import operations
    # from geomdl.visualization import VisMPL
    # ch1_points = np.array([
    #     [80.0, 60.0], [79.2, 67.8], [77.5, 75.4], [74.3, 82.5], [69.8, 88.9],
    #     [64.5, 94.4], [58.2, 98.6], [51.3, 101.3], [44.1, 102.4], [37.0, 101.8],
    #     [30.4, 99.6], [24.5, 95.8], [19.7, 90.7], [16.2, 84.6], [14.3, 77.8],
    #     [14.1, 70.6], [15.5, 63.3], [18.4, 56.4], [22.6, 50.1], [27.8, 44.8],
    #     [33.8, 40.8], [40.2, 38.3], [46.9, 37.4], [53.6, 38.1], [60.0, 40.2],
    #     [65.9, 43.6], [71.1, 48.1], [75.4, 53.5], [78.5, 59.6], [80.1, 66.0]
    # ])
    # ch2_points = np.array([
    #     [78.5, 62.1], [78.1, 68.9], [76.5, 75.5], [73.8, 81.7], [69.9, 87.2],
    #     [65.2, 91.8], [59.6, 95.3], [53.5, 97.4], [47.2, 98.1], [41.1, 97.2],
    #     [35.4, 94.8], [30.4, 91.1], [26.3, 86.3], [23.3, 80.7], [21.6, 74.6],
    #     [21.5, 68.1], [22.8, 61.6], [25.5, 55.6], [29.4, 50.4], [34.2, 46.3],
    #     [39.7, 43.5], [45.6, 42.1], [51.7, 42.2], [57.7, 43.8], [63.3, 46.7],
    #     [68.3, 50.7], [72.5, 55.6], [75.8, 61.1], [78.0, 67.0], [78.7, 72.8]
    # ])



    # # --- 1. Fit B-Splines ---
    # ch1_points_closed = np.vstack([ch1_points, ch1_points[0]])
    # ch2_points_closed = np.vstack([ch2_points, ch2_points[0]])
    # curve1 = approximate_curve(ch1_points_closed.tolist(), degree=3)
    # curve2 = approximate_curve(ch2_points_closed.tolist(), degree=3)

    # # --- 2. & 3. Translate B-Splines ---
    # dy, dx = 2.5, -3.0
    # y_orig, x_orig = 15, 10
    # registration_vector = (dy, dx)
    # crop_offset_vector = (y_orig, x_orig) # as if aligning to coordinate system of pre-crop image
    # rotation_angle = 15  # degrees
    # scale_factor = 1.1   # 110% uniform scaling

    # bbox = curve1.bbox
    # transformation_center = [(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2]

    # # First, move curves from crop coordinates to global coordinates
    # curve1_final = operations.translate(curve1, crop_offset_vector)
    # curve2_translated = operations.translate(curve2, crop_offset_vector)

    # # The transformation center must also be moved to the global coordinate system
    # final_center = [
    #     transformation_center[0] + crop_offset_vector[0],
    #     transformation_center[1] + crop_offset_vector[1]
    # ]

    # # Apply registration, scaling, and rotation to the second curve
    # curve2_registered = operations.translate(curve2_translated, registration_vector)
    # curve2_scaled = operations.scale(curve2_registered, scale_factor, center=final_center)
    # curve2_final = operations.rotate(curve2_scaled, rotation_angle, center=final_center)

    # # --- 4. & 5. (Part 1) Calculate Distances Between Curves ---
    # # summary stats re: original points making up the shape
    # # as these are based on the distance between max and min points, the values
    # #   are unaffected by translation or rotation, but are affected by the scale factor
    # ch1_max_dims = np.max(ch1_points, axis = 0) - np.min(ch1_points, axis = 0)
    # ch2_max_dims = np.max(ch2_points, axis = 0) - np.min(ch2_points, axis = 0)
    # overall_max_dim = np.max([ch1_max_dims, ch2_max_dims])*scale_factor

    # def find_min_distance_from_point_to_curve(point, curve):
    #     def objective_func(u):
    #         curve_pt = curve.evaluate_single(u)
    #         eucl_sq_dist = np.sum((np.array(point) - np.array(curve_pt))**2)
    #         return eucl_sq_dist
    #     opt_result = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')
    #     sqrt_dist = np.sqrt(opt_result.fun)
    #     return sqrt_dist

    # num_points_to_sample = 50
    # sample_params = np.linspace(0, 1, num_points_to_sample)
    # points_on_curve1 = curve1_final.evaluate_list(sample_params)
    # distances = [find_min_distance_from_point_to_curve(pt, curve2_final) for pt in points_on_curve1]

    # print("--- Distance Calculation Results ---")
    # print(f"Average distance between splines: {np.mean(distances):.4f}")
    # print(f"Maximum distance between splines: {np.max(distances):.4f}")
    # print("-" * 35)

    # outlier_indices = np.where(distances > overall_max_dim)[0]

    # if outlier_indices.size > 0:
    #     print(f"\nFound {len(outlier_indices)} points where distance > {overall_max_dim}:")
    #     # Loop through the outlier indices and print their values
    #     for index in outlier_indices:
    #         # points_on_spline1 is the Nx2 array of points we sampled
    #         outlier_point = points_on_curve1[index]
    #         outlier_dist = distances[index]
    #         print(f"  - Point at index {index} ({np.round(outlier_point, 2)}) has a large distance of {outlier_dist:.4f}")
    # else:
    #     print(f"\nAll distances are within the threshold of {overall_max_dim}.")


    # # # --- 5. (Part 2) Line-Spline Intersection via Optimization ---

    #     # y1, x1 = 50, 50
    #     # y2, x2 = 100, 100
    #     # point_a = np.array([y1, x1])
    #     # point_b = np.array([y2, x2])

    #     # def find_intersection_by_minimization(curve, line_start, line_end):
    #     #     """
    #     #     Finds the intersection of a curve and a line segment by minimizing the
    #     #     distance between them.
            
    #     #     Args:
    #     #         curve (geomdl.BSpline.Curve): The B-spline curve.
    #     #         line_start (np.ndarray): The starting point of the line segment.
    #     #         line_end (np.ndarray): The ending point of the line segment.
                
    #     #     Returns:
    #     #         tuple or None: The coordinate of the intersection, or None if no
    #     #                        intersection is found.
    #     #     """
    #     #     # This objective function calculates the minimum distance from a point
    #     #     # on the line segment to the B-spline curve.
    #     #     def objective_func(line_param):
    #     #         # 'line_param' (v) goes from 0 to 1 to define a point along the line
    #     #         point_on_line = line_start + line_param * (line_end - line_start)
                
    #     #         # Inner function to find min distance from this point_on_line to the curve
    #     #         def dist_to_curve_func(curve_param):
    #     #             point_on_curve = curve.evaluate_single(curve_param)
    #     #             return np.sum((point_on_line - np.array(point_on_curve))**2)
                    
    #     #         res = minimize_scalar(dist_to_curve_func, bounds=(0, 1), method='bounded')
    #     #         return res.fun # Return the squared minimum distance
            
    #     #     # Minimize the objective function to find the point on the line that is
    #     #     # closest to the curve.
    #     #     opt_result = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')
            
    #     #     # If the minimum distance is very small, we have an intersection.
    #     #     min_dist_sq = opt_result.fun
    #     #     if np.sqrt(min_dist_sq) < 1e-3: # Using a small tolerance
    #     #         line_param_at_intersect = opt_result.x
    #     #         intersection_point = line_start + line_param_at_intersect * (line_end - line_start)
    #     #         return intersection_point.tolist()
                
    #     #     return None

    #     # # --- Find the intersection ---
    #     # intersection_coord = find_intersection_by_minimization(curve1_final, point_a, point_b)


    #     # print("--- Intersection Test Results (via Optimization) ---")
    #     # if intersection_coord:
    #     #     print(f"Intersection found on the segment at: {np.round(intersection_coord, 2)}")
    #     # else:
    #     #     print("No intersection found.")
    #     # print("-" * 50)

    # # --- 6. Final Visualization ---
    # fig, ax = plt.subplots(figsize=(10, 10))

    # ax.scatter(ch1_points[:, 0], ch1_points[:, 1], facecolors='none', edgecolors='green', label='Channel 1 Points (Original Pos)', s=80)
    # ax.scatter(ch2_points[:, 0], ch2_points[:, 1], facecolors='none', edgecolors='orange', label='Channel 2 Points (Original Pos)', s=80)

    # # Plot the final positions of the original data points for context
    # final_ch1_points = ch1_points + crop_offset_vector
    # final_ch2_points = ch2_points + crop_offset_vector + registration_vector
    # ax.scatter(final_ch1_points[:, 0], final_ch1_points[:, 1], facecolors='none', edgecolors='blue', label='Channel 1 Points (Final Pos)', s=80)
    # ax.scatter(final_ch2_points[:, 0], final_ch2_points[:, 1], facecolors='none', edgecolors='red', label='Channel 2 Points (Final Pos)', s=80)

    # # Check if any outliers were found
    # if outlier_indices.size > 0:
    #     # Use fancy indexing to get the coordinates of the outlier points
    #     point1_to_highlight = ch1_points[outlier_indices]
    #     point2_to_highlight = ch2_points[outlier_indices]

    #     point1_to_highlight = point1_to_highlight + crop_offset_vector
    #     point2_to_highlight = point2_to_highlight + crop_offset_vector + registration_vector
        
    #     ax.scatter(point1_to_highlight[:, 0], point1_to_highlight[:, 1], marker='*',  s=250, c='gold', edgecolors='black', zorder=5, label=f'Ch1 (d >{overall_max_dim})')
    #     ax.scatter(point2_to_highlight[:, 0], point2_to_highlight[:, 1], marker='*',  s=250, c='silver', edgecolors='black', zorder=5, label=f'Ch2 (d > {overall_max_dim})')
        
    #     ax.legend()

    # Evaluate 200 points along each curve to draw a smooth line
    # eval_params = np.linspace(0, 1, 200)
    # curve1_points_eval = np.array(curve1_final.evaluate_list(eval_params))
    # curve2_points_eval = np.array(curve2_final.evaluate_list(eval_params))

    # # Use ax.plot() and set the label directly
    # ax.plot(curve1_points_eval[:, 0], curve1_points_eval[:, 1], color='blue', linewidth=2, label='Fitted Spline 1 (Final)')
    # ax.plot(curve2_points_eval[:, 0], curve2_points_eval[:, 1], color='red', linewidth=2, label='Fitted Spline 2 (Final)')

    # # Plot the intersection test line and result
    # ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'g--', label='Intersection Line')
    # if intersection_coord is not None:
    #     ax.plot(intersection_coord[0], intersection_coord[1], 'gx', markersize=15, markeredgewidth=3, label='Intersection Point') 

    # Formatting
    # ax.set_title('B-Spline Fitting, Transformation, and Analysis')
    # ax.set_xlabel('X-coordinate')
    # ax.set_ylabel('Y-coordinate')
    # ax.legend()
    # ax.grid(True, linestyle='--', alpha=0.6)
    # ax.axis('equal')
    # plt.show()


# --- USING SCIPY --- #
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import BSpline, splprep
from scipy.optimize import minimize

# --- DATA: Example Irregularly Shaped Circles ---
ch1_points = np.array([
    [80.0, 60.0], [79.2, 67.8], [77.5, 75.4], [74.3, 82.5], [69.8, 88.9],
    [64.5, 94.4], [58.2, 98.6], [51.3, 101.3], [44.1, 102.4], [37.0, 101.8],
    [30.4, 99.6], [24.5, 95.8], [19.7, 90.7], [16.2, 84.6], [14.3, 77.8],
    [14.1, 70.6], [15.5, 63.3], [18.4, 56.4], [22.6, 50.1], [27.8, 44.8],
    [33.8, 40.8], [40.2, 38.3], [46.9, 37.4], [53.6, 38.1], [60.0, 40.2],
    [65.9, 43.6], [71.1, 48.1], [75.4, 53.5], [78.5, 59.6], [80.1, 66.0]
])
ch2_points = np.array([
    [78.5, 62.1], [78.1, 68.9], [76.5, 75.5], [73.8, 81.7], [69.9, 87.2],
    [65.2, 91.8], [59.6, 95.3], [53.5, 97.4], [47.2, 98.1], [41.1, 97.2],
    [35.4, 94.8], [30.4, 91.1], [26.3, 86.3], [23.3, 80.7], [21.6, 74.6],
    [21.5, 68.1], [22.8, 61.6], [25.5, 55.6], [29.4, 50.4], [34.2, 46.3],
    [39.7, 43.5], [45.6, 42.1], [51.7, 42.2], [57.7, 43.8], [63.3, 46.7],
    [68.3, 50.7], [72.5, 55.6], [75.8, 61.1], [78.0, 67.0], [78.7, 72.8]
])

line_start = np.array([20, 40])
line_end = np.array([40,60])

SMOOTHING_FACTOR = 0
PERIODIC = True
SPLINE_DEGREE = 3

# Fit BSpline to data points
ch1_bspline_obj, u1 = splprep([ch1_points[:, 0], ch1_points[:, 1]], s=SMOOTHING_FACTOR, k=SPLINE_DEGREE, per=PERIODIC)
ch2_bspline_obj, u2 = splprep([ch2_points[:, 0], ch2_points[:, 1]], s=SMOOTHING_FACTOR, k=SPLINE_DEGREE, per=PERIODIC)

# Transform bsplines by transforming the control points and creating new, translated BSpline objects
dy, dx = 2.5, -3.0
y_orig, x_orig = 15, 10
registration_vector = np.array([dy, dx])
crop_offset_vector = np.array([y_orig, x_orig])
# !!! NEED to check that the points do not "double back" if you will

# To translate a SciPy spline, we translate its control points.
# The control points are in tck[1].
ch1_ctrlpts_final = ch1_bspline_obj[1] # ch1_bspline_obj.c
ch1_ctrlpts_final[1][0] += crop_offset_vector[0]
ch1_ctrlpts_final[1][1] += crop_offset_vector[1]
ch1_knots_final = ch1_bspline_obj[0] # = ch1_bspline_obj.t
ch1_degree_final = ch1_bspline_obj[2]  # = ch1_bspline_obj.k

# ch1_bspline_transp = BSpline(ch1_knots_final, ch1_ctrlpts_final, ch1_degree_final, extrapolate = "periodic")
ch1_bspline_transp = BSpline(ch1_knots_final, np.array(ch1_ctrlpts_final).T, ch1_degree_final, extrapolate = "periodic")

ch2_ctrlpts_final = ch2_bspline_obj[1] # = ch2_bspline_obj.c
ch2_ctrlpts_final[1][0] += crop_offset_vector[0] + registration_vector[0]
ch2_ctrlpts_final[1][1] += crop_offset_vector[1] + registration_vector[1]
ch2_knots_final  = ch2_bspline_obj[0] # = ch2_bspline_obj.t
ch2_degree_final = ch1_bspline_obj[2] # = ch2_bspline_obj.k

# ch2_bspline_transp = BSpline(ch2_knots_final, ch2_ctrlpts_final, ch2_degree_final, extrapolate = "periodic")
ch2_bspline_transp = BSpline(ch2_knots_final, np.array(ch2_ctrlpts_final).T, ch2_degree_final, extrapolate = "periodic")


# Calculate the distance between curves
def find_min_distance_from_point_to_spline(point, bspline_obj):
    def objective_func(u):
        # 'splev' evaluates the spline at parameter 'u'
        curve_pt = bspline_obj(u)
        eucl_sq_dist = np.sum((np.array(point) - np.array(curve_pt))**2)
        return eucl_sq_dist
    opt_result = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')
    return opt_result

# Evaluate points on the first spline to sample from
num_points_to_sample = 50
sample_params = np.linspace(0, 1, num_points_to_sample)
points_on_spline1 = ch1_bspline_transp(sample_params)

points_on_spline2 = np.ones(points_on_spline1.shape)
distances = np.ones(points_on_spline1.shape[0])

for index, pt in enumerate(points_on_spline1):
    min_dist_opt_result = find_min_distance_from_point_to_spline(pt, ch2_bspline_transp)
    # 'x' is the optimal parameter 'u' from the result
    pt_from_opt_param = ch2_bspline_transp(min_dist_opt_result.x)
    # place the coord assoc w the opt param in the same index position in the points_on_spline2 array as the pt was located in the points_on_spline1 array
    points_on_spline2[index, :] = pt_from_opt_param
    distances[index] = np.sqrt(min_dist_opt_result.fun)

print("--- Distance Calculation Results ---")
print(f"Average distance between splines: {np.mean(distances):.4f}")
print(f"Maximum distance between splines: {np.max(distances):.4f}")
print("-" * 35)

# --- 6. Final Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(points_on_spline1[:, 0], points_on_spline1[:, 1], facecolors='none', edgecolors='blue', label='Channel 1 Points (Final Pos)', s=80)
ax.scatter(points_on_spline2[:, 0], points_on_spline2[:, 1], facecolors='none', edgecolors='red', label='Channel 2 Points (Final Pos)', s=80)

# Evaluate points for plotting
eval_params = np.linspace(0, 1, 200)
spline1_points_eval = ch1_bspline_transp(eval_params)
spline2_points_eval = ch2_bspline_transp(eval_params)

# Plot the stable SciPy splines
ax.plot(spline1_points_eval[:, 0], spline1_points_eval[:,1], color='blue', linewidth=2, label='Fitted Spline 1 (Final)')
ax.plot(spline2_points_eval[:,0], spline2_points_eval[:,1], color='red', linewidth=2, label='Fitted Spline 2 (Final)')

# Formatting
ax.set_title('B-Spline Fitting with SciPy (Stable)')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
ax.axis('equal')
plt.show()

def find_intersection(bspline_obj, point1, point2):
    """
        Finds the intersection of a SciPy BSpline object and a line segment.

        Args:
            bspline_obj (scipy.interpolate.BSpline): The B-spline curve.
            point1 (np.ndarray): The starting point of the line segment.
            point2 (np.ndarray): The ending point of the line segment.

        Returns:
            np.ndarray or None: The coordinate of the intersection, or None if no
                                intersection is found.
    """
    def objective_func(params):
        u, v = params  # Unpack the two parameters
        
        point_on_spline = bspline_obj(u)
        point_on_line = point1 + v * (point2 - point1)
        
        return np.sum((point_on_spline - point_on_line)**2)

    # Use a multivariate optimizer to find the (u, v) pair that minimizes the distance.
    # We provide an initial guess [0.5, 0.5] and bounds for u and v (0 to 1).
    result = minimize(
        objective_func,
        x0=[0.5, 0.5],  # Initial guess for (u, v)
        bounds=[(0, 1), (0, 1)]
    )

    # Check if the optimization was successful and the minimum distance is near zero.
    if result.success and result.fun < 1e-6:
        # The intersection point can be calculated from the optimal 'v' on the line.
        intersection_point = point1 + result.x[1] * (point2 - point1)
        return intersection_point

    return None

intersection = find_intersection(ch1_bspline_transp, line_start, line_end)

