"""
Create publication-quality spline fitting visualizations showing:
- Original image
- Initial detection
- Refined splines  
- Bridges
- Final result
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from pathlib import Path


def visualize_spline_pipeline(
    fov_dict,
    fov_id,
    ne_ch1_label,
    ne_ch2_label,
    init_ch1_path,
    init_ch2_path,
    bridged_ch1_path,
    bridged_ch2_path,
    ch1_crop_path,
    ch2_crop_path,
    output_dir='reports_for_boss'
):
    """
    Create comprehensive spline fitting visualization.
    """
    
    print(f"Creating spline visualization for FoV {fov_id}: {ne_ch1_label} (Ch1) vs {ne_ch2_label} (Ch2)")
    
    # Load data
    with open(init_ch1_path, 'rb') as f:
        init_ch1 = pickle.load(f)
    with open(init_ch2_path, 'rb') as f:
        init_ch2 = pickle.load(f)
    with open(bridged_ch1_path, 'rb') as f:
        bridged_ch1 = pickle.load(f)
    with open(bridged_ch2_path, 'rb') as f:
        bridged_ch2 = pickle.load(f)
    with open(ch1_crop_path, 'r') as f:
        ch1_crops = json.load(f)
    with open(ch2_crop_path, 'r') as f:
        ch2_crops = json.load(f)
    
    # Load images
    fov_entry = fov_dict[fov_id]
    ch1_img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs']['fn_track_ch1'])
    ch2_img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs']['fn_track_ch2'])
    
    ch1_stack = tifffile.imread(ch1_img_path)
    ch2_stack = tifffile.imread(ch2_img_path)
    ch1_mean = np.mean(ch1_stack[:250], axis=0)
    ch2_mean = np.mean(ch2_stack[:250], axis=0)
    
    # Get crop regions
    ch1_crop_info = ch1_crops[fov_id][ne_ch1_label]
    ch2_crop_info = ch2_crops[fov_id][ne_ch2_label]
    
    y1_ch1 = ch1_crop_info['final_top']
    y2_ch1 = ch1_crop_info['final_bottom']
    x1_ch1 = ch1_crop_info['final_left']
    x2_ch1 = ch1_crop_info['final_right']
    ch1_crop_img = ch1_mean[y1_ch1:y2_ch1, x1_ch1:x2_ch1]
    
    y1_ch2 = ch2_crop_info['final_top']
    y2_ch2 = ch2_crop_info['final_bottom']
    x1_ch2 = ch2_crop_info['final_left']
    x2_ch2 = ch2_crop_info['final_right']
    ch2_crop_img = ch2_mean[y1_ch2:y2_ch2, x1_ch2:x2_ch2]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # --- CHANNEL 1 ROW ---
    
    # Ch1: Initial detection
    ax = axes[0, 0]
    ax.imshow(ch1_crop_img, cmap='gray', origin='upper', alpha=0.8)
    if fov_id in init_ch1 and ne_ch1_label in init_ch1[fov_id]:
        init_segments = init_ch1[fov_id][ne_ch1_label]
        for seg_label, seg_spline in init_segments.items():
            u_vals = np.linspace(0, 1, 500)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 1], pts[:, 0], 'cyan', linewidth=2, alpha=0.4)
    ax.set_title('Ch1: Initial Splines', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch1: Data segments (from bridged)
    ax = axes[0, 1]
    ax.imshow(ch1_crop_img, cmap='gray', origin='upper', alpha=0.8)
    if fov_id in bridged_ch1 and ne_ch1_label in bridged_ch1[fov_id]:
        for seg in bridged_ch1[fov_id][ne_ch1_label]['data_segments']:
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], 'lime', linewidth=3, alpha=0.4, label='Data')
    ax.set_title('Ch1: Data Driven Spline Refinement', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch1: Full with bridges
    ax = axes[0, 2]
    ax.imshow(ch1_crop_img, cmap='gray', origin='upper', alpha=0.8)
    if fov_id in bridged_ch1 and ne_ch1_label in bridged_ch1[fov_id]:
        # Data segments
        for seg in bridged_ch1[fov_id][ne_ch1_label]['data_segments']:
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], 'lime', linewidth=3, alpha=0.4, label='Data')
        # Bridge segments
        if 'bridge_segments' in bridged_ch1[fov_id][ne_ch1_label]:
            for seg in bridged_ch1[fov_id][ne_ch1_label]['bridge_segments']:
                u_vals = np.linspace(0, 1, 100)
                pts = np.array([seg(u) for u in u_vals])
                ax.plot(pts[:, 1], pts[:, 0], 'yellow', linewidth=2, alpha=0.4, 
                       linestyle='--', label='Bridge')
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    ax.set_title('Ch1: Complete (Refined + Bridges)', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # --- CHANNEL 2 ROW ---
    
    # Ch2: Initial detection
    ax = axes[1, 0]
    ax.imshow(ch2_crop_img, cmap='gray', origin='upper', alpha=0.8)
    if fov_id in init_ch2 and ne_ch2_label in init_ch2[fov_id]:
        init_segments = init_ch2[fov_id][ne_ch2_label]
        for seg_label, seg_spline in init_segments.items():
            u_vals = np.linspace(0, 1, 500)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 1], pts[:, 0], 'yellow', linewidth=2, alpha=0.9)
    ax.set_title('Ch2: Initial Splines', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch2: Data segments
    ax = axes[1, 1]
    ax.imshow(ch2_crop_img, cmap='gray', origin='upper', alpha=0.8)
    if fov_id in bridged_ch2 and ne_ch2_label in bridged_ch2[fov_id]:
        for seg in bridged_ch2[fov_id][ne_ch2_label]['data_segments']:
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg(u) for u in u_vals])
            ax.plot(pts[:, 1], pts[:, 0], 'orange', linewidth=3, alpha=0.9, label='Data')
    ax.set_title('Ch2: Data Driven Spline Refinement', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch2: Full with bridges
    ax = axes[1, 2]
    ax.imshow(ch2_crop_img, cmap='gray', origin='upper', alpha=0.8)
    if fov_id in bridged_ch2 and ne_ch2_label in bridged_ch2[fov_id]:
        # Data segments
        for seg in bridged_ch2[fov_id][ne_ch2_label]['data_segments']:
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg(u) for u in u_vals])
            ax.plot(pts[:, 1], pts[:, 0], 'orange', linewidth=3, alpha=0.9, label='Data')
        # Bridge segments
        if 'bridge_segments' in bridged_ch2[fov_id][ne_ch2_label]:
            for seg in bridged_ch2[fov_id][ne_ch2_label]['bridge_segments']:
                u_vals = np.linspace(0, 1, 100)
                pts = np.array([seg(u) for u in u_vals])
                ax.plot(pts[:, 0], pts[:, 1], 'cyan', linewidth=2, alpha=0.9,
                       linestyle='--', label='Bridge')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    ax.set_title('Ch2: Complete (Data + Bridges)', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'Spline Fitting Pipeline: FoV {fov_id}, NE {ne_ch1_label} (Ch1) vs {ne_ch2_label} (Ch2)',
                fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'spline_pipeline_{fov_id}_{ne_ch1_label}_vs_{ne_ch2_label}.png',
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path / f'spline_pipeline_{fov_id}_{ne_ch1_label}_vs_{ne_ch2_label}.png'}")


# Run for all valid pairs
if __name__ == "__main__":
    from imaging_pipeline import ImagingPipeline
    import pickle
    
    pipeline = ImagingPipeline("config_local_dual.json")
    fov_dict = pipeline.get_experiments()['BMY9999_99_99_9999']
    
    # Load pairs
    with open('local_yeast_output/dual_label/checkpoints/BMY9999_99_99_9999/state_after_filtering.pkl', 'rb') as f:
        img_proc = pickle.load(f)
        ne_pairs = img_proc.get_ne_pairs_by_FoV()
    
    # Generate for each valid pair
    for fov_id, pairs in ne_pairs.items():
        for ch1_label, ch2_label in pairs.items():
            try:
                visualize_spline_pipeline(
                    fov_dict=fov_dict,
                    fov_id=fov_id,
                    ne_ch1_label=ch1_label,
                    ne_ch2_label=ch2_label,
                    init_ch1_path='local_yeast_output/dual_label/init_fit/ch1_bsplines_BMY9999_99_99_9999.pkl',
                    init_ch2_path='local_yeast_output/dual_label/init_fit/ch2_bsplines_BMY9999_99_99_9999.pkl',
                    bridged_ch1_path='local_yeast_output/dual_label/merged_splines/bridged_splines_ch1_BMY9999_99_99_9999.pkl',
                    bridged_ch2_path='local_yeast_output/dual_label/merged_splines/bridged_splines_ch2_BMY9999_99_99_9999.pkl',
                    ch1_crop_path='local_yeast_output/dual_label/init_fit/ch1_crop_BMY9999_99_99_9999.json',
                    ch2_crop_path='local_yeast_output/dual_label/init_fit/ch2_crop_BMY9999_99_99_9999.json',
                    output_dir='reports_for_boss/BMY9999_99_99_9999'
                )
            except Exception as e:
                print(f"✗ Failed for {fov_id}/{ch1_label} vs {ch2_label}: {e}")