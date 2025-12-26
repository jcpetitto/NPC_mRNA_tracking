# ========================================================================
# FIX: Constrain Spline Control Points to Crop Box Bounds
# ========================================================================
#
# Location: npc_spline_refinement.py, around line 256-270
# Problem: Refined splines have control points far outside crop box (e.g., -726 to 741 instead of 0-75)
# Solution: Clip refined points + validate control points + retry with tighter smoothing if needed
# ========================================================================

# REPLACE lines 256-270 with:

                # 7. Re-assemble and fit the final spline WITH BOUNDS CONSTRAINTS
                try:
                    refined_points = adjust_spline_points(
                        optimized_mu_full,        # Full (N_total,) mu array
                        line_length,
                        segment_points_xy_all,    # Full (2, N_total) points
                        segment_derivs_xy_all     # Full (2, N_total) derivatives
                    )
                    
                    # STEP 1: Clip refined points to crop box bounds (with small margin)
                    # Crop boxes are typically 75x75, allow small buffer for edge effects
                    crop_margin = 10  # pixels
                    x_min, x_max = -crop_margin, 75 + crop_margin
                    y_min, y_max = -crop_margin, 75 + crop_margin
                    
                    n_clipped_x = np.sum((refined_points[0, :] < x_min) | (refined_points[0, :] > x_max))
                    n_clipped_y = np.sum((refined_points[1, :] < y_min) | (refined_points[1, :] > y_max))
                    
                    if n_clipped_x > 0 or n_clipped_y > 0:
                        print(f"    CLIPPING: {seg_label} had {n_clipped_x} X and {n_clipped_y} Y points out of bounds")
                    
                    refined_points[0, :] = np.clip(refined_points[0, :], x_min, x_max)
                    refined_points[1, :] = np.clip(refined_points[1, :], y_min, y_max)
                    
                    # STEP 2: Fit spline with initial smoothing
                    initial_smoothing = 1.0
                    bspline_refined, _ = make_splprep(
                        [refined_points[0, :], refined_points[1, :]], 
                        s=initial_smoothing, 
                        k=3
                    )
                    
                    # STEP 3: Validate control points are within acceptable bounds
                    bspline_obj = bspline_from_tck(bspline_refined)
                    
                    # Check control points with slightly tighter bounds
                    ctrl_x_min, ctrl_x_max = bspline_obj.c[:, 0].min(), bspline_obj.c[:, 0].max()
                    ctrl_y_min, ctrl_y_max = bspline_obj.c[:, 1].min(), bspline_obj.c[:, 1].max()
                    
                    # Allow slightly larger margin for control points (smoothing can extrapolate a bit)
                    ctrl_margin = 20  # pixels
                    ctrl_bounds_ok = (
                        ctrl_x_min >= -ctrl_margin and ctrl_x_max <= 75 + ctrl_margin and
                        ctrl_y_min >= -ctrl_margin and ctrl_y_max <= 75 + ctrl_margin
                    )
                    
                    if not ctrl_bounds_ok:
                        print(f"    WARNING: {seg_label} control points escaped bounds:")
                        print(f"      X: [{ctrl_x_min:.1f}, {ctrl_x_max:.1f}] (should be ~[0, 75])")
                        print(f"      Y: [{ctrl_y_min:.1f}, {ctrl_y_max:.1f}] (should be ~[0, 75])")
                        print(f"      Retrying with tighter smoothing...")
                        
                        # STEP 4: Retry with progressively tighter smoothing
                        smoothing_attempts = [0.5, 0.1, 0.01]
                        for attempt_smoothing in smoothing_attempts:
                            bspline_refined, _ = make_splprep(
                                [refined_points[0, :], refined_points[1, :]], 
                                s=attempt_smoothing, 
                                k=3
                            )
                            
                            bspline_obj = bspline_from_tck(bspline_refined)
                            ctrl_x_min, ctrl_x_max = bspline_obj.c[:, 0].min(), bspline_obj.c[:, 0].max()
                            ctrl_y_min, ctrl_y_max = bspline_obj.c[:, 1].min(), bspline_obj.c[:, 1].max()
                            
                            ctrl_bounds_ok = (
                                ctrl_x_min >= -ctrl_margin and ctrl_x_max <= 75 + ctrl_margin and
                                ctrl_y_min >= -ctrl_margin and ctrl_y_max <= 75 + ctrl_margin
                            )
                            
                            if ctrl_bounds_ok:
                                print(f"      SUCCESS with smoothing={attempt_smoothing}")
                                print(f"        X: [{ctrl_x_min:.1f}, {ctrl_x_max:.1f}]")
                                print(f"        Y: [{ctrl_y_min:.1f}, {ctrl_y_max:.1f}]")
                                break
                        
                        if not ctrl_bounds_ok:
                            print(f"      FAILED: Even with s=0.01, control points still out of bounds")
                            print(f"      Skipping {seg_label} (refinement produced invalid geometry)")
                            continue  # Don't add this segment to results
                    
                    # STEP 5: Final validation - check for extreme control points
                    max_allowed_extent = 150  # Absolute maximum (2× crop box size)
                    if (abs(ctrl_x_min) > max_allowed_extent or abs(ctrl_x_max) > max_allowed_extent or
                        abs(ctrl_y_min) > max_allowed_extent or abs(ctrl_y_max) > max_allowed_extent):
                        print(f"    REJECTING: {seg_label} has extreme control points (>150 pixels)")
                        print(f"      X: [{ctrl_x_min:.1f}, {ctrl_x_max:.1f}]")
                        print(f"      Y: [{ctrl_y_min:.1f}, {ctrl_y_max:.1f}]")
                        continue
                    
                    # Success! Add to results
                    refined_segments_for_this_ne[seg_label] = bspline_refined
                
                except Exception as e:
                    print(f"    --> ERROR: Spline fitting failed for {seg_label} after refinement.")
                    print(f"        Error: {e}")
                    import traceback
                    traceback.print_exc()

            if refined_segments_for_this_ne:
                final_refined_splines_dict[ne_label] = refined_segments_for_this_ne


# ========================================================================
# REQUIRED IMPORT (add at top of file if not present)
# ========================================================================
# from tools.geom_tools import bspline_from_tck


# ========================================================================
# EXPECTED BEHAVIOR:
# ========================================================================
# - Clips refined points to [-10, 85] range (crop box with margin)
# - Fits spline with s=1.0
# - Checks if control points escaped bounds
# - If yes: retries with s=0.5, 0.1, 0.01 until control points are within [-20, 95]
# - If still fails: rejects segment (logs warning, doesn't crash)
# - Final safety check: rejects segments with control points > 150 pixels
#
# This should fix:
# - Ch1 segment_01: X=[31, 165], Y=[36, 424] → will retry with tighter smoothing
# - Ch2 segment_01: X=[-726, 46], Y=[35, 741] → will be rejected or heavily constrained
# ========================================================================


# ========================================================================
# TESTING THE FIX:
# ========================================================================
# After implementing, check the refined splines:
#
import pickle
with open('local_yeast_output/dual_label/refined_fit/refine_results_ch2_BMY9999_99_99_9999.pkl', 'rb') as f:
    ch2 = pickle.load(f)

seg = ch2['0083']['09']['segment_01']
print(f"Segment_01 X: [{seg.c[:, 0].min():.1f}, {seg.c[:, 0].max():.1f}]")
print(f"Segment_01 Y: [{seg.c[:, 1].min():.1f}, {seg.c[:, 1].max():.1f}]")
#
# Expected: X and Y both within roughly [-20, 95] instead of [-726, 741]
# ========================================================================
