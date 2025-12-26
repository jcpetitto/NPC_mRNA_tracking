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
    ch1_label = '01'
    ch2_label = '03'
    
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