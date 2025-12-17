# """
# Diagnostic visualization for dual-label distance calculations.
# Shows exactly what points are being compared and where distances are coming from.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import BSpline
# import pickle
# from pathlib import Path

# def plot_distance_calculation_diagnostic(ch1_splines, ch2_splines, fov_id, 
#                                          ch1_label, ch2_label, n_samples=200):
#     """
#     Create diagnostic plot showing:
#     1. Ch1 spline (blue)
#     2. Ch2 spline (red)  
#     3. Sample points on Ch1 (blue dots)
#     4. Closest points on Ch2 (red dots)
#     5. Distance vectors (arrows)
#     """
    
#     # Get splines
#     ch1_data = ch1_splines[fov_id][ch1_label]
#     ch2_data = ch2_splines[fov_id][ch2_label]
    
#     spline_ch1 = ch1_data['full_periodic_spline']
#     spline_ch2 = ch2_data['full_periodic_spline']
    
#     # Sample Ch1 spline
#     u_ch1 = np.linspace(0, 1, n_samples)
#     points_ch1 = np.array([spline_ch1(u) for u in u_ch1])
    
#     # Sample Ch2 spline for visualization
#     u_ch2 = np.linspace(0, 1, 1000)
#     points_ch2 = np.array([spline_ch2(u) for u in u_ch2])
    
#     # For each Ch1 point, find closest Ch2 point
#     from scipy.optimize import minimize_scalar
#     closest_ch2_points = []
#     distances = []
    
#     for p_ch1 in points_ch1:
#         def dist_sq(u):
#             return np.sum((p_ch1 - spline_ch2(u))**2)
        
#         res = minimize_scalar(dist_sq, bounds=(0, 1), method='bounded')
#         p_ch2_closest = spline_ch2(res.x)
#         closest_ch2_points.append(p_ch2_closest)
#         distances.append(np.linalg.norm(p_ch1 - p_ch2_closest))
    
#     closest_ch2_points = np.array(closest_ch2_points)
#     distances = np.array(distances)
    
#     # Create figure
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # === PLOT 1: Full view with all distances ===
#     ax = axes[0]
    
#     # Plot splines
#     ax.plot(points_ch1[:, 0], points_ch1[:, 1], 'b-', linewidth=2, label=f'Ch1 NE {ch1_label}')
#     ax.plot(points_ch2[:, 0], points_ch2[:, 1], 'r-', linewidth=2, label=f'Ch2 NE {ch2_label}')
    
#     # Plot sample points
#     ax.scatter(points_ch1[::10, 0], points_ch1[::10, 1], c='blue', s=30, zorder=5, alpha=0.5)
#     ax.scatter(closest_ch2_points[::10, 0], closest_ch2_points[::10, 1], c='red', s=30, zorder=5, alpha=0.5)
    
#     # Plot distance vectors (every 10th to avoid clutter)
#     for i in range(0, len(points_ch1), 10):
#         ax.arrow(points_ch1[i, 0], points_ch1[i, 1],
#                 closest_ch2_points[i, 0] - points_ch1[i, 0],
#                 closest_ch2_points[i, 1] - points_ch1[i, 1],
#                 head_width=2, head_length=2, fc='gray', ec='gray', alpha=0.3)
    
#     ax.set_xlabel('X (pixels)')
#     ax.set_ylabel('Y (pixels)')
#     ax.set_title(f'FoV {fov_id}: Ch1 {ch1_label} ↔ Ch2 {ch2_label}\nFull View')
#     ax.legend()
#     ax.axis('equal')
#     ax.grid(True, alpha=0.3)
    
#     # === PLOT 2: Zoomed view of one section ===
#     ax = axes[1]
    
#     # Pick a section (indices 50-70)
#     idx_start, idx_end = 50, 70
    
#     ax.plot(points_ch1[:, 0], points_ch1[:, 1], 'b-', linewidth=1, alpha=0.3)
#     ax.plot(points_ch2[:, 0], points_ch2[:, 1], 'r-', linewidth=1, alpha=0.3)
    
#     # Highlight section
#     ax.plot(points_ch1[idx_start:idx_end, 0], points_ch1[idx_start:idx_end, 1], 
#             'b-', linewidth=3, label='Ch1 section')
    
#     # Plot corresponding Ch2 points
#     ax.scatter(closest_ch2_points[idx_start:idx_end, 0], 
#               closest_ch2_points[idx_start:idx_end, 1], 
#               c='red', s=50, zorder=5, label='Ch2 closest')
    
#     # Plot distance vectors
#     for i in range(idx_start, idx_end):
#         ax.arrow(points_ch1[i, 0], points_ch1[i, 1],
#                 closest_ch2_points[i, 0] - points_ch1[i, 0],
#                 closest_ch2_points[i, 1] - points_ch1[i, 1],
#                 head_width=1, head_length=1, fc='green', ec='green', alpha=0.6)
    
#     ax.set_xlabel('X (pixels)')
#     ax.set_ylabel('Y (pixels)')
#     ax.set_title(f'Zoomed: Points {idx_start}-{idx_end}')
#     ax.legend()
#     ax.axis('equal')
#     ax.grid(True, alpha=0.3)
    
#     # === PLOT 3: Distance histogram ===
#     ax = axes[2]
    
#     ax.hist(distances, bins=50, edgecolor='black', alpha=0.7)
#     ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, 
#                label=f'Mean: {np.mean(distances):.1f} px')
#     ax.axvline(np.median(distances), color='green', linestyle='--', linewidth=2,
#                label=f'Median: {np.median(distances):.1f} px')
    
#     ax.set_xlabel('Distance (pixels)')
#     ax.set_ylabel('Count')
#     ax.set_title('Distance Distribution')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Add text with statistics
#     stats_text = f'''Statistics:
# Mean: {np.mean(distances):.2f} px
# Std: {np.std(distances):.2f} px
# Min: {np.min(distances):.2f} px
# Max: {np.max(distances):.2f} px
# Range: {np.max(distances) - np.min(distances):.2f} px'''
    
#     ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
#             verticalalignment='top', horizontalalignment='right',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
#             fontsize=9, family='monospace')
    
#     plt.tight_layout()
    
#     # Print coordinate ranges to check for issues
#     print(f"\n=== COORDINATE RANGES ===")
#     print(f"Ch1 X: [{points_ch1[:, 0].min():.1f}, {points_ch1[:, 0].max():.1f}]")
#     print(f"Ch1 Y: [{points_ch1[:, 1].min():.1f}, {points_ch1[:, 1].max():.1f}]")
#     print(f"Ch2 X: [{points_ch2[:, 0].min():.1f}, {points_ch2[:, 0].max():.1f}]")
#     print(f"Ch2 Y: [{points_ch2[:, 1].min():.1f}, {points_ch2[:, 1].max():.1f}]")
#     print(f"\n=== DISTANCE STATS ===")
#     print(f"Mean: {np.mean(distances):.2f} px")
#     print(f"Std: {np.std(distances):.2f} px")
#     print(f"Min: {np.min(distances):.2f} px")
#     print(f"Max: {np.max(distances):.2f} px")
    
#     return fig


# def diagnose_problematic_pair(fov_id='0083', ch1_label='08', ch2_label='09'):
#     """
#     Diagnose NE 08↔09 which shows -17,374 pixel distances (impossible!)
#     """
    
#     # Load splines
#     ch1_path = Path('local_yeast_output/dual_label/refined_fit/bridged_splines_ch1_BMY9999_99_99_9999.pkl')
#     ch2_path = Path('local_yeast_output/dual_label/refined_fit/bridged_splines_ch2_BMY9999_99_99_9999.pkl')
    
#     with open(ch1_path, 'rb') as f:
#         ch1_splines = pickle.load(f)
    
#     with open(ch2_path, 'rb') as f:
#         ch2_splines = pickle.load(f)
    
#     print(f"=== DIAGNOSING FoV {fov_id}: Ch1 {ch1_label} ↔ Ch2 {ch2_label} ===")
    
#     # Check structure
#     ch1_data = ch1_splines[fov_id][ch1_label]
#     ch2_data = ch2_splines[fov_id][ch2_label]
    
#     print(f"\nCh1 NE {ch1_label}:")
#     print(f"  Data segments: {len(ch1_data['data_segments'])}")
#     print(f"  Bridge segments: {len(ch1_data['bridge_segments'])}")
#     print(f"  U-ranges (data): {ch1_data['u_ranges']['data']}")
#     print(f"  U-ranges (bridge): {ch1_data['u_ranges']['bridge']}")
    
#     print(f"\nCh2 NE {ch2_label}:")
#     print(f"  Data segments: {len(ch2_data['data_segments'])}")
#     print(f"  Bridge segments: {len(ch2_data['bridge_segments'])}")
#     print(f"  U-ranges (data): {ch2_data['u_ranges']['data']}")
#     print(f"  U-ranges (bridge): {ch2_data['u_ranges']['bridge']}")
    
#     # Plot diagnostic
#     fig = plot_distance_calculation_diagnostic(
#         ch1_splines, ch2_splines, fov_id, ch1_label, ch2_label
#     )
    
#     plt.show()
    
#     return fig


# if __name__ == "__main__":
#     # Diagnose the problematic pair
#     diagnose_problematic_pair(fov_id='0083', ch1_label='08', ch2_label='09')

# /Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output

"""
Diagnostic: Check if splines are in crop-box local coordinates vs global coordinates
"""

import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt

def check_spline_coordinate_system():
    """
    Verify whether splines are stored in local crop-box coordinates or global image coordinates.
    """
    
    print("="*80)
    print("SPLINE COORDINATE SYSTEM DIAGNOSTIC")
    print("="*80)
    
    # Load splines
    ch1_spline_path = Path('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/refined_fit/bridged_splines_ch1_BMY9999_99_99_9999.pkl')
    ch2_spline_path = Path('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/refined_fit/bridged_splines_ch2_BMY9999_99_99_9999.pkl')
    
    with open(ch1_spline_path, 'rb') as f:
        ch1_splines = pickle.load(f)
    
    with open(ch2_spline_path, 'rb') as f:
        ch2_splines = pickle.load(f)
    
    # Load crop boxes
    with open('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/initial_fit/ch1_crop_BMY9999_99_99_9999.json', 'r') as f:
        ch1_crops = json.load(f)
    
    with open('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/initial_fit/ch2_crop_BMY9999_99_99_9999.json', 'r') as f:
        ch2_crops = json.load(f)
    
    # Test case: FoV 0083, NE 08 (Ch1) and NE 09 (Ch2)
    fov_id = '0083'
    ch1_label = '08'
    ch2_label = '09'
    
    print(f"\nTest case: FoV {fov_id}, Ch1 NE {ch1_label} ↔ Ch2 NE {ch2_label}")
    print("-"*80)
    
    # Get crop box positions
    ch1_crop = ch1_crops[fov_id][ch1_label]
    ch2_crop = ch2_crops[fov_id][ch2_label]
    
    print("\nCROP BOX POSITIONS (global image coordinates):")
    print(f"  Ch1 NE {ch1_label}: top={ch1_crop['final_top']}, left={ch1_crop['final_left']}, "
          f"bottom={ch1_crop['final_bottom']}, right={ch1_crop['final_right']}")
    print(f"  Ch2 NE {ch2_label}: top={ch2_crop['final_top']}, left={ch2_crop['final_left']}, "
          f"bottom={ch2_crop['final_bottom']}, right={ch2_crop['final_right']}")
    
    print(f"\n  Crop box offset difference:")
    print(f"    ΔX = {ch2_crop['final_left'] - ch1_crop['final_left']} pixels")
    print(f"    ΔY = {ch2_crop['final_top'] - ch1_crop['final_top']} pixels")
    print(f"    → These crop boxes are nearly IDENTICAL in global space!")
    
    # Sample spline points
    ch1_spline = ch1_splines[fov_id][ch1_label]['full_periodic_spline']
    ch2_spline = ch2_splines[fov_id][ch2_label]['full_periodic_spline']
    
    u_sample = np.linspace(0, 1, 100)
    ch1_points = np.array([ch1_spline(u) for u in u_sample])
    ch2_points = np.array([ch2_spline(u) for u in u_sample])
    
    print("\nSPLINE COORDINATE RANGES:")
    print(f"  Ch1 NE {ch1_label}:")
    print(f"    X: [{ch1_points[:, 0].min():.1f}, {ch1_points[:, 0].max():.1f}]")
    print(f"    Y: [{ch1_points[:, 1].min():.1f}, {ch1_points[:, 1].max():.1f}]")
    
    print(f"  Ch2 NE {ch2_label}:")
    print(f"    X: [{ch2_points[:, 0].min():.1f}, {ch2_points[:, 0].max():.1f}]")
    print(f"    Y: [{ch2_points[:, 1].min():.1f}, {ch2_points[:, 1].max():.1f}]")
    
    # Check if splines are in local coordinates (0-75 range)
    ch1_in_local = (ch1_points[:, 0].max() < 100 and ch1_points[:, 1].max() < 100)
    ch2_in_local = (ch2_points[:, 0].max() < 100 and ch2_points[:, 1].max() < 100)
    
    print("\nCOORDINATE SYSTEM DETERMINATION:")
    if ch1_in_local and ch2_in_local:
        print("  ✓ CONFIRMED: Splines are in LOCAL crop-box coordinates (0-75 pixel range)")
        print("  ✗ PROBLEM: Distance calculation treats them as if in SAME coordinate system")
        print("  ✗ RESULT: Distances are calculated correctly in local space, but meaningless!")
        print()
        print("  → Ch1 and Ch2 splines SHOULD be ~1 pixel apart (crop boxes overlap)")
        print("  → But they appear in different local coordinate frames")
        print("  → Need to add crop box offsets to align them in global space!")
    else:
        print("  ✓ Splines appear to be in GLOBAL coordinates")
        print("  ✓ Distance calculation should be valid")
    
    # Calculate what distances SHOULD be if we add offsets
    print("\n" + "="*80)
    print("WHAT DISTANCES SHOULD BE (with crop box offsets):")
    print("="*80)
    
    # Add crop box offsets
    ch1_global = ch1_points + np.array([ch1_crop['final_left'], ch1_crop['final_top']])
    ch2_global = ch2_points + np.array([ch2_crop['final_left'], ch2_crop['final_top']])
    
    print(f"\nAfter adding crop box offsets:")
    print(f"  Ch1 global X: [{ch1_global[:, 0].min():.1f}, {ch1_global[:, 0].max():.1f}]")
    print(f"  Ch1 global Y: [{ch1_global[:, 1].min():.1f}, {ch1_global[:, 1].max():.1f}]")
    print(f"  Ch2 global X: [{ch2_global[:, 0].min():.1f}, {ch2_global[:, 0].max():.1f}]")
    print(f"  Ch2 global Y: [{ch2_global[:, 1].min():.1f}, {ch2_global[:, 1].max():.1f}]")
    
    # Calculate distances between corresponding points
    distances = np.linalg.norm(ch1_global - ch2_global, axis=1)
    
    print(f"\nCorrected distances (after crop box alignment):")
    print(f"  Mean: {np.mean(distances):.2f} pixels")
    print(f"  Median: {np.median(distances):.2f} pixels")
    print(f"  Std: {np.std(distances):.2f} pixels")
    print(f"  Min: {np.min(distances):.2f} pixels")
    print(f"  Max: {np.max(distances):.2f} pixels")
    
    # Visual comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Local coordinates (current/wrong)
    ax = axes[0]
    ax.plot(ch1_points[:, 0], ch1_points[:, 1], 'b-', linewidth=2, label=f'Ch1 NE {ch1_label}')
    ax.plot(ch2_points[:, 0], ch2_points[:, 1], 'r-', linewidth=2, label=f'Ch2 NE {ch2_label}')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('WRONG: Local Crop-Box Coordinates\n(Current implementation)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Global coordinates (correct)
    ax = axes[1]
    ax.plot(ch1_global[:, 0], ch1_global[:, 1], 'b-', linewidth=2, label=f'Ch1 NE {ch1_label}')
    ax.plot(ch2_global[:, 0], ch2_global[:, 1], 'r-', linewidth=2, label=f'Ch2 NE {ch2_label}')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('CORRECT: Global Image Coordinates\n(After adding crop box offsets)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/coordinate_system_diagnostic.png', dpi=150)
    print(f"\n✓ Saved diagnostic plot: /Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/coordinate_system_diagnostic.png")
    plt.show()
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    if ch1_in_local and ch2_in_local:
        print("Splines ARE in local crop-box coordinates.")
        print("You MUST add crop box offsets before calculating distances!")
        print(f"\nExpected distance after fix: ~{np.mean(distances):.1f} pixels (not 162 pixels!)")
    else:
        print("Splines appear to be in global coordinates already.")
        print("The distance calculation might be correct.")
    print("="*80)


if __name__ == "__main__":
    check_spline_coordinate_system()