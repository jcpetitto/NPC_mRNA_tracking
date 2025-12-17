import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import tifffile
import os

def visualize_ne_pairing(
    fov_dict,
    ch1_crop_path,
    ch2_crop_path,
    ne_pairs_map,
    distances_dict,
    fov_id,
    distance_threshold=500
):
    """
    Visualizes NE pairing by showing crop boxes on the full FoV images.
    Colors pairs based on whether distance is reasonable.
    """
    
    # Load crop boxes
    with open(ch1_crop_path, 'r') as f:
        ch1_crops = json.load(f)
    with open(ch2_crop_path, 'r') as f:
        ch2_crops = json.load(f)
    
    # Load full FoV images - FIXED PATHS
    fov_entry = fov_dict[fov_id]
    ch1_img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs']['fn_track_ch1'])
    ch2_img_path = os.path.join(fov_entry['FoV_collection_path'], fov_entry['imgs']['fn_track_ch2'])
    
    ch1_stack = tifffile.imread(ch1_img_path)
    ch2_stack = tifffile.imread(ch2_img_path)
    ch1_mean = np.mean(ch1_stack[:250], axis=0)
    ch2_mean = np.mean(ch2_stack[:250], axis=0)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Plot Ch1 ---
    ax1 = axes[0]
    ax1.imshow(ch1_mean, cmap='gray', origin='upper')
    ax1.set_title(f'Channel 1 (FoV {fov_id})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # --- Plot Ch2 ---
    ax2 = axes[1]
    ax2.imshow(ch2_mean, cmap='gray', origin='upper')
    ax2.set_title(f'Channel 2 (FoV {fov_id})', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Get pairs for this FoV
    pairs = ne_pairs_map[fov_id]
    distances_fov = distances_dict.get(fov_id, {})
    
    # Color scheme
    colors_good = plt.cm.Set2(np.linspace(0, 1, len(pairs)))
    
    # Plot each pair
    for idx, (ch1_label, ch2_label) in enumerate(pairs.items()):
        pair_key = f"{ch1_label}_vs_{ch2_label}"
        
        # Determine if this is a good or bad pair
        if pair_key in distances_fov:
            mean_dist = abs(distances_fov[pair_key]['mean_distance'])
            is_good = mean_dist < distance_threshold
            color = colors_good[idx] if is_good else 'red'
            status = "GOOD" if is_good else "BAD"
            dist_text = f"{distances_fov[pair_key]['mean_distance']:.1f} nm"
        else:
            color = 'orange'
            status = "NO DIST"
            dist_text = "N/A"
        
        # Get Ch1 crop box - FIXED KEYS
        ch1_crop = ch1_crops[fov_id][ch1_label]
        x1 = ch1_crop['final_left']
        y1 = ch1_crop['final_top']
        x2 = ch1_crop['final_right']
        y2 = ch1_crop['final_bottom']
        w1 = x2 - x1
        h1 = y2 - y1
        
        # Get Ch2 crop box - FIXED KEYS
        ch2_crop = ch2_crops[fov_id][ch2_label]
        x2_box = ch2_crop['final_left']
        y2_box = ch2_crop['final_top']
        x2_right = ch2_crop['final_right']
        y2_bottom = ch2_crop['final_bottom']
        w2 = x2_right - x2_box
        h2 = y2_bottom - y2_box
        
        # Draw Ch1 box
        rect1 = patches.Rectangle((x1, y1), w1, h1, linewidth=3, 
                                  edgecolor=color, facecolor='none', linestyle='-')
        ax1.add_patch(rect1)
        ax1.text(x1 + w1/2, y1 + h1/2, ch1_label, color=color, 
                fontsize=14, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Draw Ch2 box
        rect2 = patches.Rectangle((x2_box, y2_box), w2, h2, linewidth=3, 
                                  edgecolor=color, facecolor='none', linestyle='-')
        ax2.add_patch(rect2)
        ax2.text(x2_box + w2/2, y2_box + h2/2, ch2_label, color=color, 
                fontsize=14, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Calculate centers for distance measurement
        center1_x = x1 + w1/2
        center1_y = y1 + h1/2
        center2_x = x2_box + w2/2
        center2_y = y2_box + h2/2
        
        # Calculate pixel distance between centers
        pixel_dist = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
        
        # Add status annotation above Ch1 box
        ax1.text(x1 + w1/2, y1 - 5, f"{status}\n{dist_text}\nΔpx={pixel_dist:.1f}", 
                color=color, fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2))
        
        # Add pairing info on Ch2
        ax2.text(x2_box + w2/2, y2_box - 5, f"↔ Ch1:{ch1_label}", 
                color=color, fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2))
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label=f'Good pair (< {distance_threshold} nm)'),
        Line2D([0], [0], color='red', linewidth=3, label='Bad pair (wrong match)'),
        Line2D([0], [0], color='orange', linewidth=3, label='No distance calculated')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add statistics box
    n_good = sum(1 for p in distances_fov.values() if abs(p['mean_distance']) < distance_threshold)
    n_bad = len(distances_fov) - n_good
    n_total = len(pairs)
    
    stats_text = f"Total pairs: {n_total}\nGood: {n_good}\nBad: {n_bad}\nNo dist: {n_total - len(distances_fov)}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1))
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/ne_pairing_diagnostic_{fov_id}.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return fig


# --- Run for both FoVs ---
import pickle
from imaging_pipeline import ImagingPipeline

pipeline = ImagingPipeline("config_local_dual.json")
fov_dict = pipeline.get_experiments()['BMY9999_99_99_9999']

# Load NE pairs
with open('local_yeast_output/dual_label/checkpoints/BMY9999_99_99_9999/state_after_filtering.pkl', 'rb') as f:
    img_proc = pickle.load(f)
    ne_pairs = img_proc.get_ne_pairs_by_FoV()

for fov_id in ['0083', '0191']:
    print(f"\n=== Visualizing FoV {fov_id} ===")
    visualize_ne_pairing(
        fov_dict=fov_dict,
        ch1_crop_path='local_yeast_output/dual_label/init_fit/ch1_crop_BMY9999_99_99_9999.json',
        ch2_crop_path='local_yeast_output/dual_label/init_fit/ch2_crop_BMY9999_99_99_9999.json',
        ne_pairs_map=ne_pairs,
        distances_dict=filtered_distances,
        fov_id=fov_id
    )