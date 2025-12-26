"""
Test script to validate bridge curvature fixes on problematic NE segments.
Tests NE 08 (Ch1) which had extreme distance values due to bad bridging.
"""

import pickle
import numpy as np
from pathlib import Path
import sys

# Add src path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import the FIXED bridging module
from utils.spline_bridging import bridge_refined_splines, validate_bridge_curvature, fit_parametric_spline
from tools.geom_tools import build_curve_bridge


def test_bridge_validation():
    """Test bridge validation on existing problematic data."""
    
    print("="*80)
    print("BRIDGE CURVATURE VALIDATION TEST")
    print("="*80)
    
    # Load the refined splines (before bridging)
    refined_path = Path('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/refined_fit/refine_results_ch1_BMY9999_99_99_9999.pkl')
    
    if not refined_path.exists():
        print(f"ERROR: Cannot find refined splines at {refined_path}")
        return
    
    with open(refined_path, 'rb') as f:
        ch1_refined = pickle.load(f)
    
    print(f"\nLoaded refined splines from: {refined_path}")
    print(f"FoVs: {list(ch1_refined.keys())}")
    
    # Focus on FoV 0083, NE 08 (the problematic one)
    fov_id = '0083'
    ne_label = '08'
    
    if fov_id not in ch1_refined or ne_label not in ch1_refined[fov_id]:
        print(f"\nERROR: Cannot find {fov_id}/{ne_label} in refined data")
        return
    
    segments = ch1_refined[fov_id][ne_label]
    print(f"\n{fov_id} NE {ne_label} has {len(segments)} segments:")
    for seg_key in sorted(segments.keys()):
        print(f"  {seg_key}")
    
    # Create minimal config
    config = {
        'ne_fit': {
            'bridge_smoothing_factor': 1.0,
            'refinement': {
                'final_sampling_density': 64
            }
        }
    }
    
    # Test OLD bridging (manually recreate the bad bridge)
    print("\n" + "="*80)
    print("TESTING OLD BRIDGING LOGIC (no validation)")
    print("="*80)
    

    
    seg_keys = sorted([k for k in segments.keys() if k.startswith('segment_')], 
                      key=lambda x: int(x.split('_')[-1]))
    
    # Build bridges between all segments (old way)
    for i in range(len(seg_keys)):
        current_key = seg_keys[i]
        next_key = seg_keys[(i + 1) % len(seg_keys)]
        
        current_spline = segments[current_key]
        next_spline = segments[next_key]
        
        p_end = current_spline(1.0)
        p_start = next_spline(0.0)
        tan_end = current_spline.derivative(1)(1.0)
        tan_start = next_spline.derivative(1)(0.0)
        
        gap_distance = np.linalg.norm(p_start - p_end)
        
        print(f"\nBridge {i}: {current_key} → {next_key}")
        print(f"  Gap distance: {gap_distance:.2f} pixels")
        
        bridge_points = build_curve_bridge(p_end, p_start, tan_end, tan_start)
        bridge_spline = fit_parametric_spline(bridge_points, smoothing=0, periodic=False)
        
        # Check angle changes WITHOUT validation (same method as refinement)
        u_check = np.linspace(0, 1, 50)
        points = np.array([bridge_spline(u) for u in u_check]).T  # (2, N)
        
        # Calculate tangent vectors
        derivs = np.zeros_like(points)
        derivs[:, 1:-1] = (points[:, 2:] - points[:, :-2]) / 2.0
        derivs[:, 0] = points[:, 1] - points[:, 0]
        derivs[:, -1] = points[:, -1] - points[:, -2]
        
        # Calculate tangent angles
        angles = np.arctan2(derivs[1, :], derivs[0, :])
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        angle_changes_deg = np.abs(np.rad2deg(angle_diffs))
        
        max_angle = np.max(angle_changes_deg) if len(angle_changes_deg) > 0 else 0.0
        mean_angle = np.mean(angle_changes_deg) if len(angle_changes_deg) > 0 else 0.0
        
        print(f"  Max angle change: {max_angle:.4f}°")
        print(f"  Mean angle change: {mean_angle:.4f}°")
        
        if max_angle > 1.0:
            print(f"  ⚠️  EXCEEDS threshold (1.0°)!")
        if gap_distance < 3.0:
            print(f"  ⚠️  Gap too small (<3px)!")
    
    # Test NEW bridging with validation
    print("\n" + "="*80)
    print("TESTING NEW BRIDGING LOGIC (with validation)")
    print("="*80)
    
    bridged = bridge_refined_splines(ch1_refined, config)
    
    if fov_id not in bridged or ne_label not in bridged[fov_id]:
        print(f"\nERROR: Bridging failed for {fov_id}/{ne_label}")
        return
    
    result = bridged[fov_id][ne_label]
    
    print(f"\nBridging results for {fov_id} NE {ne_label}:")
    print(f"  Data segments: {len(result['data_segments'])}")
    print(f"  Bridge segments: {len(result['bridge_segments'])}")
    print(f"  Data u-ranges: {result['u_ranges']['data']}")
    print(f"  Bridge u-ranges: {result['u_ranges']['bridge']}")
    
    # Validate each bridge that was created
    print("\nValidated bridges:")
    for i, bridge_spline in enumerate(result['bridge_segments']):
        is_valid, max_angle = validate_bridge_curvature(bridge_spline, max_angle_change_deg=1.0)
        u_range = result['u_ranges']['bridge'][i]
        range_size = u_range[1] - u_range[0]
        print(f"  Bridge {i}: u={u_range[0]:.4f} to {u_range[1]:.4f} (Δu={range_size:.4f})")
        print(f"    Valid: {is_valid}, Max angle change: {max_angle:.4f}°")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    n_bridges_old = len(seg_keys)  # Old method created one per segment
    n_bridges_new = len(result['bridge_segments'])
    
    print(f"Bridges created:")
    print(f"  OLD method: {n_bridges_old} (no validation)")
    print(f"  NEW method: {n_bridges_new} (validated)")
    print(f"  Rejected: {n_bridges_old - n_bridges_new}")
    
    if n_bridges_new < n_bridges_old:
        print(f"\n✅ SUCCESS: {n_bridges_old - n_bridges_new} problematic bridge(s) rejected!")
    else:
        print(f"\n⚠️  WARNING: All bridges passed validation")
    
    print("\n" + "="*80)
    print("Test complete. Replace spline_bridging.py with spline_bridging_FIXED.py")
    print("and re-run the pipeline from bridging stage.")
    print("="*80)

if __name__ == "__main__":
    test_bridge_validation()