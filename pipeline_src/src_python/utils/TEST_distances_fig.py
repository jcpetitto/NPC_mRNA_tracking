# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import BSpline
# import pickle
# import json

# def visualize_dual_label_distances(
#     bridged_ch1_path,
#     bridged_ch2_path,
#     distances_path,
#     fov_id,
#     ne_ch1_label,
#     ne_ch2_label,
#     output_path=None,
#     n_sample_points=500,
#     n_arrows=20
# ):
#     """
#     Creates a beautiful visualization of dual-label NE distances.
#     Plots data segments and bridges as distinct, smooth curves.
#     """
    
#     # Load data
#     with open(bridged_ch1_path, 'rb') as f:
#         ch1_data = pickle.load(f)
#     with open(bridged_ch2_path, 'rb') as f:
#         ch2_data = pickle.load(f)
#     with open(distances_path, 'r') as f:
#         dist_data = json.load(f)
    
#     # Extract splines and distance info
#     ch1_spline_data = ch1_data[fov_id][ne_ch1_label]
#     ch2_spline_data = ch2_data[fov_id][ne_ch2_label]
    
#     # Get the individual segment splines
#     ch1_data_segs = ch1_spline_data['data_segments']
#     ch1_bridge_segs = ch1_spline_data['bridge_segments']
#     ch2_data_segs = ch2_spline_data['data_segments']
#     ch2_bridge_segs = ch2_spline_data['bridge_segments']
    
#     # Get distance data
#     pair_key = f"{ne_ch1_label}_vs_{ne_ch2_label}"
#     distances_info = dist_data[fov_id][pair_key]
    
#     dists = np.array(distances_info['distances'])
#     ch1_u = np.array(distances_info['ch1_u_values'])
#     ch2_u = np.array(distances_info['ch2_u_values'])
    
#     # Get the full periodic splines for sampling distance points
#     full_spline_ch1 = ch1_spline_data['full_periodic_spline']
#     full_spline_ch2 = ch2_spline_data['full_periodic_spline']
    
#     # Create figure
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
#     # --- PLOT 1: Spline overlay ---
#     ax1 = axes[0]
    
#     # Plot Ch1 DATA segments (dark blue, thick, solid)
#     for seg_spline in ch1_data_segs:
#         u_seg = np.linspace(0, 1, 200)
#         points = np.array([seg_spline(u) for u in u_seg])
#         ax1.plot(points[:, 0], points[:, 1], color='#0d47a1', linewidth=3.5, 
#                 solid_capstyle='round', zorder=3, alpha=0.9)
    
#     # Plot Ch1 BRIDGE segments (light blue, thick, with glow)
#     for seg_spline in ch1_bridge_segs:
#         u_seg = np.linspace(0, 1, 100)
#         points = np.array([seg_spline(u) for u in u_seg])
#         # Glow effect
#         ax1.plot(points[:, 0], points[:, 1], color='#64b5f6', linewidth=8, 
#                 alpha=0.3, solid_capstyle='round', zorder=2)
#         # Main line
#         ax1.plot(points[:, 0], points[:, 1], color='#42a5f5', linewidth=3, 
#                 linestyle='--', dashes=(5, 3), solid_capstyle='round', zorder=3)
    
#     # Plot Ch2 DATA segments (dark orange, thick, solid)
#     for seg_spline in ch2_data_segs:
#         u_seg = np.linspace(0, 1, 200)
#         points = np.array([seg_spline(u) for u in u_seg])
#         ax1.plot(points[:, 0], points[:, 1], color='#e65100', linewidth=3.5, 
#                 solid_capstyle='round', zorder=3, alpha=0.9)
    
#     # Plot Ch2 BRIDGE segments (light orange, thick, with glow)
#     for seg_spline in ch2_bridge_segs:
#         u_seg = np.linspace(0, 1, 100)
#         points = np.array([seg_spline(u) for u in u_seg])
#         # Glow effect
#         ax1.plot(points[:, 0], points[:, 1], color='#ffcc80', linewidth=8, 
#                 alpha=0.3, solid_capstyle='round', zorder=2)
#         # Main line
#         ax1.plot(points[:, 0], points[:, 1], color='#ff9800', linewidth=3, 
#                 linestyle='--', dashes=(5, 3), solid_capstyle='round', zorder=3)
    
#     # Draw sample distance vectors
#     arrow_indices = np.linspace(0, len(dists) - 1, n_arrows, dtype=int)
    
#     for idx in arrow_indices:
#         u1 = ch1_u[idx]
#         u2 = ch2_u[idx]
#         p1 = full_spline_ch1(u1)
#         p2 = full_spline_ch2(u2)
#         dist = dists[idx]
        
#         # Color by distance magnitude
#         color = plt.cm.RdYlGn_r(np.clip(abs(dist) / 100, 0, 1))
#         ax1.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
#                  head_width=0.8, head_length=0.8, fc=color, ec=color, 
#                  alpha=0.6, linewidth=1.5, zorder=1)
    
#     # Legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color='#0d47a1', linewidth=3.5, label='Ch1 Data'),
#         Line2D([0], [0], color='#42a5f5', linewidth=3, linestyle='--', label='Ch1 Bridge'),
#         Line2D([0], [0], color='#e65100', linewidth=3.5, label='Ch2 Data'),
#         Line2D([0], [0], color='#ff9800', linewidth=3, linestyle='--', label='Ch2 Bridge')
#     ]
#     ax1.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
#     ax1.set_aspect('equal')
#     ax1.set_xlabel('X (pixels)', fontsize=13, fontweight='bold')
#     ax1.set_ylabel('Y (pixels)', fontsize=13, fontweight='bold')
#     ax1.set_title(f'FoV {fov_id}: {ne_ch1_label} (Ch1) vs {ne_ch2_label} (Ch2)', 
#                  fontsize=15, fontweight='bold', pad=15)
#     ax1.grid(True, alpha=0.25, linewidth=0.5)
#     ax1.set_facecolor('#f8f9fa')
    
#     # --- PLOT 2: Distance distribution ---
#     ax2 = axes[1]
    
#     # Get segment types from the distance data
#     pair_types = distances_info.get('pair_types', [])
    
#     # If pair_types exists, use it; otherwise infer from ch1_segment_types
#     if pair_types:
#         data_mask = np.array([pt == 'data_data' for pt in pair_types])
#         mixed_mask = np.array([pt in ['data_bridge', 'bridge_data'] for pt in pair_types])
#         bridge_mask = np.array([pt == 'bridge_bridge' for pt in pair_types])
#     else:
#         ch1_types = distances_info.get('ch1_segment_types', [])
#         data_mask = np.array([t == 'data' for t in ch1_types])
#         bridge_mask = ~data_mask
#         mixed_mask = np.zeros(len(data_mask), dtype=bool)
    
#     # Scatter plot colored by segment type
#     if np.any(data_mask):
#         ax2.scatter(ch1_u[data_mask], dists[data_mask], 
#                    c='#0d47a1', s=30, alpha=0.7, label='Data-Data', zorder=3)
#     if np.any(mixed_mask):
#         ax2.scatter(ch1_u[mixed_mask], dists[mixed_mask], 
#                    c='#9c27b0', s=30, alpha=0.7, label='Data-Bridge', zorder=2)
#     if np.any(bridge_mask):
#         ax2.scatter(ch1_u[bridge_mask], dists[bridge_mask], 
#                    c='#42a5f5', s=30, alpha=0.5, label='Bridge-Bridge', zorder=1)
    
#     # Statistics lines
#     ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
#     mean_dist = distances_info['mean_distance']
#     std_dist = distances_info['std_distance']
    
#     ax2.axhline(mean_dist, color='red', linestyle='-', linewidth=2.5, 
#                label=f'Overall Mean = {mean_dist:.2f} nm', zorder=4)
#     ax2.fill_between([0, 1], mean_dist - std_dist, mean_dist + std_dist, 
#                      color='red', alpha=0.15, label=f'±1 SD = {std_dist:.2f} nm', zorder=0)
    
#     # Add data-data mean if available
#     if 'data_data_mean' in distances_info and distances_info['data_data_mean'] is not None:
#         dd_mean = distances_info['data_data_mean']
#         ax2.axhline(dd_mean, color='#0d47a1', linestyle='--', linewidth=2, 
#                    label=f'Data-Data Mean = {dd_mean:.2f} nm', alpha=0.8)
    
#     ax2.set_xlabel('Position along Ch1 membrane (u parameter)', fontsize=13, fontweight='bold')
#     ax2.set_ylabel('Distance (nm)', fontsize=13, fontweight='bold')
#     ax2.set_title('Distance Profile', fontsize=15, fontweight='bold', pad=15)
#     ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
#     ax2.grid(True, alpha=0.25, linewidth=0.5)
#     ax2.set_xlim(0, 1)
#     ax2.set_facecolor('#f8f9fa')
    
#     # Add statistics text box
#     stats_text = (
#         f"N points: {len(dists)}\n"
#         f"Mean: {mean_dist:.2f} nm\n"
#         f"Median: {np.median(dists):.2f} nm\n"
#         f"Std Dev: {std_dist:.2f} nm\n"
#         f"Min: {np.min(dists):.2f} nm\n"
#         f"Max: {np.max(dists):.2f} nm"
#     )
    
#     if 'data_data_mean' in distances_info and distances_info['data_data_mean'] is not None:
#         stats_text += f"\n\nData-Data Mean: {distances_info['data_data_mean']:.2f} nm"
#     if 'data_bridge_mean' in distances_info and distances_info['data_bridge_mean'] is not None:
#         stats_text += f"\nData-Bridge Mean: {distances_info['data_bridge_mean']:.2f} nm"
    
#     ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
#             fontsize=10, verticalalignment='top', family='monospace',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Saved figure to {output_path}")
    
#     plt.show()
    
#     return fig


# # --- USAGE EXAMPLE ---
# if __name__ == "__main__":
#     visualize_dual_label_distances(
#         bridged_ch1_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/bridged_splines_ch1_BMY9999_99_99_9999.pkl',
#         bridged_ch2_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/bridged_splines_ch2_BMY9999_99_99_9999.pkl',
#         distances_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/distances/dual_dist_result_BMY9999_99_99_9999.json',
#         fov_id='0083',
#         ne_ch1_label='01',
#         ne_ch2_label='03',
#         output_path='figures/dual_label_example.png'
#     )
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import tifffile
from scipy.interpolate import BSpline

def diagnostic_spline_comparison(
    fov_dict,
    init_ch1_path,
    init_ch2_path,
    refined_ch1_path,
    refined_ch2_path,
    bridged_ch1_path,
    bridged_ch2_path,
    ch1_crop_path,
    ch2_crop_path,
    fov_id,
    ne_ch1_label,
    ne_ch2_label,
    output_path=None
):
    """
    Creates a diagnostic plot showing the evolution from initial detection
    to final bridged splines, overlaid on the original images.
    """
    
    # Load all the data
    with open(init_ch1_path, 'rb') as f:
        init_ch1 = pickle.load(f)
    with open(init_ch2_path, 'rb') as f:
        init_ch2 = pickle.load(f)
    
    with open(refined_ch1_path, 'rb') as f:
        refined_ch1 = pickle.load(f)
    with open(refined_ch2_path, 'rb') as f:
        refined_ch2 = pickle.load(f)
    
    with open(bridged_ch1_path, 'rb') as f:
        bridged_ch1 = pickle.load(f)
    with open(bridged_ch2_path, 'rb') as f:
        bridged_ch2 = pickle.load(f)
    
    with open(ch1_crop_path, 'r') as f:
        ch1_crops = json.load(f)
    with open(ch2_crop_path, 'r') as f:
        ch2_crops = json.load(f)
    
    # Get crop box info and load images
    fov_entry = fov_dict[fov_id] 
    ch1_img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs']['fn_track_ch1'])
    ch2_img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs']['fn_track_ch2'])
    
    # Load full images and extract mean
    ch1_stack = tifffile.imread(ch1_img_path)
    ch2_stack = tifffile.imread(ch2_img_path)
    ch1_mean = np.mean(ch1_stack[:250], axis=0)  # Use first 250 frames
    ch2_mean = np.mean(ch2_stack[:250], axis=0)
    
    # Get crop boxes
    ch1_crop_info = ch1_crops[fov_id][ne_ch1_label]
    ch2_crop_info = ch2_crops[fov_id][ne_ch2_label]
    
    # Extract cropped regions
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
    
    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f'Spline Evolution: FoV {fov_id}, {ne_ch1_label} (Ch1) vs {ne_ch2_label} (Ch2)', 
                 fontsize=16, fontweight='bold')
    
    # --- ROW 1: Channel 1 ---
    
    # Ch1: Initial spline
    ax = axes[0, 0]
    ax.imshow(ch1_crop_img, cmap='gray', origin='upper')
    if fov_id in init_ch1 and ne_ch1_label in init_ch1[fov_id]:
        init_segments_ch1 = init_ch1[fov_id][ne_ch1_label]
        # Loop through segments
        for seg_label, seg_spline in init_segments_ch1.items():
            u_vals = np.linspace(0, 1, 500)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], 'cyan', linewidth=2, alpha=0.8, label=seg_label)
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    ax.set_title('Ch1: Initial Detection', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch1: Refined segments
    ax = axes[0, 1]
    ax.imshow(ch1_crop_img, cmap='gray', origin='upper')
    if fov_id in refined_ch1 and ne_ch1_label in refined_ch1[fov_id]:
        for seg_label, seg_spline in refined_ch1[fov_id][ne_ch1_label].items():
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], linewidth=3, label=seg_label, alpha=0.8)
        ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Ch1: Refined Segments', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch1: Bridged (final)
    ax = axes[0, 2]
    ax.imshow(ch1_crop_img, cmap='gray', origin='upper')
    if fov_id in bridged_ch1 and ne_ch1_label in bridged_ch1[fov_id]:
        bridged_data = bridged_ch1[fov_id][ne_ch1_label]
        
        # Plot data segments
        for seg_spline in bridged_data['data_segments']:
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], '#0d47a1', linewidth=3, label='Data')
        
        # Plot bridges
        for seg_spline in bridged_data['bridge_segments']:
            u_vals = np.linspace(0, 1, 100)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], '#42a5f5', linewidth=2.5, 
                   linestyle='--', label='Bridge', alpha=0.8)
        
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.set_title('Ch1: Bridged (Final)', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # --- ROW 2: Channel 2 ---
    
    # Ch2: Initial spline
    ax = axes[1, 0]
    ax.imshow(ch2_crop_img, cmap='gray', origin='upper')
    if fov_id in init_ch2 and ne_ch2_label in init_ch2[fov_id]:
        init_segments_ch2 = init_ch2[fov_id][ne_ch2_label]
        # Loop through segments
        for seg_label, seg_spline in init_segments_ch2.items():
            u_vals = np.linspace(0, 1, 500)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], 'yellow', linewidth=2, alpha=0.8, label=seg_label)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    ax.set_title('Ch2: Initial Detection', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch2: Refined segments
    ax = axes[1, 1]
    ax.imshow(ch2_crop_img, cmap='gray', origin='upper')
    if fov_id in refined_ch2 and ne_ch2_label in refined_ch2[fov_id]:
        for seg_label, seg_spline in refined_ch2[fov_id][ne_ch2_label].items():
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], linewidth=3, label=seg_label, alpha=0.8)
        ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Ch2: Refined Segments', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Ch2: Bridged (final)
    ax = axes[1, 2]
    ax.imshow(ch2_crop_img, cmap='gray', origin='upper')
    if fov_id in bridged_ch2 and ne_ch2_label in bridged_ch2[fov_id]:
        bridged_data = bridged_ch2[fov_id][ne_ch2_label]
        
        # Plot data segments
        for seg_spline in bridged_data['data_segments']:
            u_vals = np.linspace(0, 1, 200)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], '#e65100', linewidth=3, label='Data')
        
        # Plot bridges
        for seg_spline in bridged_data['bridge_segments']:
            u_vals = np.linspace(0, 1, 100)
            pts = np.array([seg_spline(u) for u in u_vals])
            ax.plot(pts[:, 0], pts[:, 1], '#ff9800', linewidth=2.5, 
                   linestyle='--', label='Bridge', alpha=0.8)
        
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.set_title('Ch2: Bridged (Final)', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    output_path = Path(output_path)
    if output_path:
        plt.savefig(output_path / "diagnostic_spline_evolution.png", dpi=300, bbox_inches='tight')
        print(f"Saved diagnostic figure to {output_path}")
    
    plt.show()
    
    return fig


# --- USAGE ---
if __name__ == "__main__":
    # You'll need to load FoV_dict from somewhere
    from imaging_pipeline import ImagingPipeline
    
    pipeline = ImagingPipeline("config_local_dual.json")
    fov_dict = pipeline.get_experiments()['BMY9999_99_99_9999']
    
    diagnostic_spline_comparison(
        fov_dict=fov_dict,
        init_ch1_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/initial_fit/ch1_bsplines_BMY9999_99_99_9999.pkl',
        init_ch2_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/initial_fit/ch2_bsplines_BMY9999_99_99_9999.pkl',
        refined_ch1_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/refine_results_ch1_BMY9999_99_99_9999.pkl',
        refined_ch2_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/refine_results_ch2_BMY9999_99_99_9999.pkl',
        bridged_ch1_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/bridged_splines_ch1_BMY9999_99_99_9999.pkl',
        bridged_ch2_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/bridged_splines_ch2_BMY9999_99_99_9999.pkl',
        ch1_crop_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/initial_fit/ch1_crop_BMY9999_99_99_9999.json',
        ch2_crop_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/initial_fit/ch2_crop_BMY9999_99_99_9999.json',
        fov_id='0083',
        ne_ch1_label='01',
        ne_ch2_label='03',
        output_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/figures/'
    )