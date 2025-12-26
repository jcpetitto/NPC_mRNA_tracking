import json
import numpy as np
import logging
from pathlib import Path

# Setup basic logging to see output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneReg")

# --- 1. The Math Helpers (From your img_registration.py) ---

def calculate_rss(vec_y, vec_x, angle_rad, scale_val, radius_px, is_sigma=False):
    """
    Calculates the Root Sum Squared error in 'Effective Pixels'.
    
    Args:
        is_sigma (bool): 
            If True, treats scale_val as a standard deviation (centered at 0).
            If False, treats scale_val as a raw ratio (centered at 1.0).
    """
    # 1. Translation Component (Pixels)
    var_trans = vec_y**2 + vec_x**2 
    
    # 2. Angle Component (Radians -> Arc Length pixels)
    var_angle = (angle_rad * radius_px)**2 
    
    # 3. Scale Component (Ratio -> Pixel Expansion)
    # If this is a sigma (std dev), the spread is around 0.
    # If this is a raw value (drift), the delta is (scale_2 - scale_1).
    # Since we pass the Delta or Sigma directly, we just square it.
    # If passing raw scale factor (e.g. 1.002), we'd subtract 1.
    
    if not is_sigma and abs(scale_val - 1.0) < 0.1:
        # It looks like a raw scale factor (e.g. 0.998), convert to delta
        scale_term = scale_val - 1.0
    else:
        # It looks like a sigma or a pre-calculated delta
        scale_term = scale_val
        
    var_scale = (scale_term * radius_px)**2 

    return np.sqrt(var_trans + var_angle + var_scale)

def get_slice_sigma(ne_data, radius_px):
    """Calculates the internal precision (sigma) of a single nucleus from its slices."""
    slice_metrics = []
    
    # Extract slice data
    for k, v in ne_data.items():
        if k.startswith('slice_') and 'shift_vector' in v:
            slice_metrics.append([
                v['shift_vector'][0], 
                v['shift_vector'][1],
                v.get('angle', 0.0), 
                v.get('scale', 1.0)
            ])
            
    if len(slice_metrics) < 2:
        return 0.5 # Default fallback if no history

    # Calculate Std Devs
    arr = np.array(slice_metrics)
    std_y = np.std(arr[:, 0])
    std_x = np.std(arr[:, 1])
    std_angle = np.std(arr[:, 2])
    std_scale = np.std(arr[:, 3])
    
    # RSS of the Sigmas
    return calculate_rss(std_y, std_x, std_angle, std_scale, radius_px, is_sigma=True)


# --- 2. The Main Standalone Function ---

def run_stability_analysis_standalone(path_mode1, path_mode2, radius_px=37.5):
    """
    Parses JSON registration files and filters nuclei based on stability.
    
    Criteria:
        Precision (Sigma): Calculated from Mode 1 slices.
        Drift: Calculated as (Mode 1 - Mode 2).
        Filter: Fail if Drift > 2 * FoV_Sigma.
    """
    # A. Load Data
    logger.info(f"Loading Mode 1: {path_mode1}")
    with open(path_mode1, 'r') as f:
        reg_m1 = json.load(f)
        
    logger.info(f"Loading Mode 2: {path_mode2}")
    with open(path_mode2, 'r') as f:
        reg_m2 = json.load(f)
        
    report = {}
    fov_stats = {}
    
    # B. Iterate FoVs
    for fov_id, m1_data in reg_m1.items():
        if fov_id not in report: report[fov_id] = {}
        
        # 1. Calculate Precision (Noise Floor) for this FoV
        # We collect all sigmas to find the Pooled Sigma for the FoV
        fov_sigmas = []
        
        ne_keys = [k for k in m1_data.keys() 
                   if isinstance(m1_data[k], dict) and 'shift_vector' in m1_data[k]]
        
        for label in ne_keys:
            sigma = get_slice_sigma(m1_data[label], radius_px)
            fov_sigmas.append(sigma)
            
        # Pooled Sigma (RMS)
        if fov_sigmas:
            pooled_fov_sigma = np.sqrt(np.mean(np.array(fov_sigmas)**2))
        else:
            pooled_fov_sigma = 0.5
            
        fov_stats[fov_id] = pooled_fov_sigma
        threshold = 2.0 * pooled_fov_sigma
        
        # 2. Calculate Drift & Filter
        m2_data = reg_m2.get(fov_id, {})
        
        for label in ne_keys:
            d1 = m1_data[label]
            
            # Default Status
            status = "passed"
            reason = None
            drift_rss = -1.0
            
            if label not in m2_data:
                status = "failed"
                reason = "missing_in_mode2"
            else:
                d2 = m2_data[label]
                
                # --- CALCULATE DRIFT VECTORS ---
                # Shift Delta
                dy = d1['shift_vector'][0] - d2['shift_vector'][0]
                dx = d1['shift_vector'][1] - d2['shift_vector'][1]
                # Angle Delta (Radians)
                da = d1.get('angle', 0.0) - d2.get('angle', 0.0)
                # Scale Delta
                ds = d1.get('scale', 1.0) - d2.get('scale', 1.0)
                
                # Combined Drift Metric (RSS)
                drift_rss = calculate_rss(dy, dx, da, ds, radius_px, is_sigma=True)
                
                # --- THE DECISION ---
                if drift_rss > threshold:
                    status = "failed"
                    reason = f"unstable ({drift_rss:.3f} > {threshold:.3f})"
            
            # 3. Add to Report
            report[fov_id][label] = {
                'status': status,
                'drift': drift_rss,
                'sigma_fov': pooled_fov_sigma,
                'threshold': threshold,
                'reason': reason
            }
            
    return report, fov_stats

# --- 3. Example Usage Block ---
if __name__ == "__main__":
    # Example paths - change these to your actual files
    m1_path = 'reg_results_mode1_BMY9999_99_99_9999.json'
    m2_path = 'reg_results_mode2_BMY9999_99_99_9999.json'
    
    # Run
    try:
        report, stats = run_stability_analysis_standalone(m1_path, m2_path)
        
        # Print Summary
        print("\n--- STABILITY REPORT ---")
        for fov, data in report.items():
            print(f"FoV {fov} (Threshold: {stats[fov]:.4f} px):")
            for ne, res in data.items():
                if res['status'] == 'failed':
                    print(f"  X Nucleus {ne}: FAILED - {res['reason']}")
                else:
                    print(f"  OK Nucleus {ne}: Drift {res['drift']:.4f}")
                    
    except FileNotFoundError:
        print("Could not find example files. Please set m1_path/m2_path correctly.")