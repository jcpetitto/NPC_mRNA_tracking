"""
Spline Visualization Tool - 3-Panel Diagnostic Plots
Shows splines in global coordinates with data/bridge differentiation and underlying images.
"""

import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Rectangle

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def identify_segment_type_at_u(u_value, u_ranges_dict):
    """Helper to identify if u is in data or bridge range."""
    u_value = u_value % 1.0
    
    for u_start, u_end in u_ranges_dict['data']:
        if u_start <= u_value <= u_end:
            return 'data'
    
    for u_start, u_end in u_ranges_dict['bridge']:
        if u_start <= u_value <= u_end:
            return 'bridge'
    
    return 'unknown'


def sample_spline_by_segments(spline_data, n_samples=500):
    """
    Sample spline and return separate arrays for data and bridge segments.
    
    Returns:
        data_points: (N, 2) array of points from data segments
        bridge_points: (N, 2) array of points from bridge segments (may be empty)
    """
    spline = spline_data['full_periodic_spline']
    u_ranges = spline_data['u_ranges']
    
    # Sample uniformly along u
    u_samples = np.linspace(0, 1, n_samples, endpoint=False)
    
    data_points_list = []
    bridge_points_list = []
    
    for u in u_samples:
        point = spline(u)
        seg_type = identify_segment_type_at_u(u, u_ranges)
        
        if seg_type == 'data':
            data_points_list.append(point)
        elif seg_type == 'bridge':
            bridge_points_list.append(point)
    
    data_points = np.array(data_points_list) if data_points_list else np.empty((0, 2))
    bridge_points = np.array(bridge_points_list) if bridge_points_list else np.empty((0, 2))
    
    return data_points, bridge_points


def load_cropped_fluorescence_image(fov_path, ne_crop, channel_key, frame_range=(0, 10)):
    """
    Load and crop fluorescence image for a specific NE.
    
    Args:
        fov_path: Path to FoV directory
        ne_crop: Crop box dict with final_left, final_top, etc.
        channel_key: 'ch1' or 'ch2'
        frame_range: Tuple of (start, end) frames to average
        
    Returns:
        cropped_img: 2D array of cropped image
        offset: (left, top) offset in global coordinates
    """
    
    # Construct fluorescence image path
    # Adjust pattern based on your naming convention
    img_files = list(Path(fov_path).glob(f'*{channel_key}*.tif'))
    
    if not img_files:
        raise FileNotFoundError(f"No {channel_key} image found in {fov_path}")
    
    # Load first matching file
    img_path = img_files[0]
    img_stack = tifffile.imread(img_path)
    
    # Average across frames
    if len(img_stack.shape) == 3:
        start, end = frame_range
        img_mean = np.mean(img_stack[start:end], axis=0)
    else:
        img_mean = img_stack
    
    # Crop to NE region
    top = ne_crop['final_top']
    left = ne_crop['final_left']
    bottom = ne_crop['final_bottom']
    right = ne_crop['final_right']
    
    cropped = img_mean[top:bottom, left:right]
    
    return cropped, (left, top)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_all_visualization_data(output_dir='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label'):
    """
    Load all data needed for visualization.
    
    Returns:
        dict with keys: ch1_splines, ch2_splines, ch1_crops, ch2_crops, ne_pairs
    """
    output_path = Path(output_dir)
    
    # Load splines
    ch1_spline_path = output_path / 'refined_fit' / 'bridged_splines_ch1_BMY9999_99_99_9999.pkl'
    ch2_spline_path = output_path / 'refined_fit' / 'bridged_splines_ch2_BMY9999_99_99_9999.pkl'
    
    with open(ch1_spline_path, 'rb') as f:
        ch1_splines = pickle.load(f)
    with open(ch2_spline_path, 'rb') as f:
        ch2_splines = pickle.load(f)
    
    # Load crop boxes
    ch1_crop_path = output_path / 'initial_fit' / 'ch1_crop_BMY9999_99_99_9999.json'
    ch2_crop_path = output_path / 'initial_fit' / 'ch2_crop_BMY9999_99_99_9999.json'
    
    with open(ch1_crop_path, 'r') as f:
        ch1_crops = json.load(f)
    with open(ch2_crop_path, 'r') as f:
        ch2_crops = json.load(f)
    
    # Load NE pairs
    pairs_path = output_path / 'initial_fit' / 'ne_label_pairs_BMY9999_99_99_9999.json'
    with open(pairs_path, 'r') as f:
        ne_pairs = json.load(f)
    
    return {
        'ch1_splines': ch1_splines,
        'ch2_splines': ch2_splines,
        'ch1_crops': ch1_crops,
        'ch2_crops': ch2_crops,
        'ne_pairs': ne_pairs
    }


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plot_ne_pair_3panel(fov_id, ch1_label, ch2_label, data_dict, 
                        fov_path=None, n_samples=500, save_path=None):
    """
    Create 3-panel diagnostic plot for a single NE pair.
    
    Args:
        fov_id: FoV identifier (e.g., '0083')
        ch1_label: Ch1 NE label (e.g., '08')
        ch2_label: Ch2 NE label (e.g., '09')
        data_dict: Dict with keys: ch1_splines, ch2_splines, ch1_crops, ch2_crops
        fov_path: Optional path to FoV directory for loading images
        n_samples: Number of points to sample per spline
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure object
    """
    
    # Extract data
    ch1_splines = data_dict['ch1_splines']
    ch2_splines = data_dict['ch2_splines']
    ch1_crops = data_dict['ch1_crops']
    ch2_crops = data_dict['ch2_crops']
    
    # Get spline and crop data
    ch1_data = ch1_splines[fov_id][ch1_label]
    ch2_data = ch2_splines[fov_id][ch2_label]
    ch1_crop = ch1_crops[fov_id][ch1_label]
    ch2_crop = ch2_crops[fov_id][ch2_label]
    
    # Sample splines (already in global coordinates based on image_processor updates)
    ch1_data_pts, ch1_bridge_pts = sample_spline_by_segments(ch1_data, n_samples)
    ch2_data_pts, ch2_bridge_pts = sample_spline_by_segments(ch2_data, n_samples)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ========== PANEL 1: Overlay of both channels ==========
    ax = axes[0]
    
    # Plot Ch1 data segments
    if len(ch1_data_pts) > 0:
        ax.plot(ch1_data_pts[:, 0], ch1_data_pts[:, 1], 'b-', linewidth=2.5, 
                label=f'Ch1 NE {ch1_label} (data)', zorder=4, alpha=0.9)
    
    # Plot Ch1 bridge segments
    if len(ch1_bridge_pts) > 0:
        ax.plot(ch1_bridge_pts[:, 0], ch1_bridge_pts[:, 1], 'b--', linewidth=2, 
                alpha=0.4, label=f'Ch1 NE {ch1_label} (bridge)', zorder=2)
    
    # Plot Ch2 data segments
    if len(ch2_data_pts) > 0:
        ax.plot(ch2_data_pts[:, 0], ch2_data_pts[:, 1], 'r-', linewidth=2.5, 
                label=f'Ch2 NE {ch2_label} (data)', zorder=3, alpha=0.9)
    
    # Plot Ch2 bridge segments
    if len(ch2_bridge_pts) > 0:
        ax.plot(ch2_bridge_pts[:, 0], ch2_bridge_pts[:, 1], 'r--', linewidth=2, 
                alpha=0.4, label=f'Ch2 NE {ch2_label} (bridge)', zorder=1)
    
    # Add crop box rectangles
    ch1_rect = Rectangle((ch1_crop['final_left'], ch1_crop['final_top']),
                          ch1_crop['width'], ch1_crop['height'],
                          linewidth=1.5, edgecolor='blue', facecolor='none', 
                          linestyle=':', alpha=0.6, zorder=0, label='Ch1 crop box')
    ax.add_patch(ch1_rect)
    
    ch2_rect = Rectangle((ch2_crop['final_left'], ch2_crop['final_top']),
                          ch2_crop['width'], ch2_crop['height'],
                          linewidth=1.5, edgecolor='red', facecolor='none', 
                          linestyle=':', alpha=0.6, zorder=0, label='Ch2 crop box')
    ax.add_patch(ch2_rect)
    
    ax.set_xlabel('X (pixels, global)', fontsize=11)
    ax.set_ylabel('Y (pixels, global)', fontsize=11)
    ax.set_title(f'FoV {fov_id}: Ch1 {ch1_label} ↔ Ch2 {ch2_label}\nBoth Channels (Global Coordinates)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========== PANEL 2: Ch1 with image ==========
    ax = axes[1]
    
    # Try to load and display image
    if fov_path is not None:
        try:
            ch1_img, (offset_x, offset_y) = load_cropped_fluorescence_image(
                fov_path, ch1_crop, 'ch1'
            )
            
            # Display image
            extent = [offset_x, offset_x + ch1_crop['width'],
                     offset_y + ch1_crop['height'], offset_y]
            ax.imshow(ch1_img, cmap='gray', origin='upper',
                     extent=extent, alpha=0.6, zorder=1)
            
        except Exception as e:
            print(f"  Warning: Could not load Ch1 image: {e}")
            # Draw crop box as placeholder
            ch1_rect = Rectangle((ch1_crop['final_left'], ch1_crop['final_top']),
                                ch1_crop['width'], ch1_crop['height'],
                                linewidth=2, edgecolor='blue', facecolor='lightblue', 
                                alpha=0.2, zorder=0)
            ax.add_patch(ch1_rect)
    
    # Plot spline on top
    if len(ch1_data_pts) > 0:
        ax.plot(ch1_data_pts[:, 0], ch1_data_pts[:, 1], 'cyan', linewidth=3, 
                label='Data', zorder=3)
    if len(ch1_bridge_pts) > 0:
        ax.plot(ch1_bridge_pts[:, 0], ch1_bridge_pts[:, 1], 'yellow', linewidth=2.5, 
                alpha=0.7, linestyle='--', label='Bridge', zorder=2)
    
    ax.set_xlabel('X (pixels, global)', fontsize=11)
    ax.set_ylabel('Y (pixels, global)', fontsize=11)
    ax.set_title(f'Ch1 NE {ch1_label} with Fluorescence Image', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # ========== PANEL 3: Ch2 with image ==========
    ax = axes[2]
    
    # Try to load and display image
    if fov_path is not None:
        try:
            ch2_img, (offset_x, offset_y) = load_cropped_fluorescence_image(
                fov_path, ch2_crop, 'ch2'
            )
            
            # Display image
            extent = [offset_x, offset_x + ch2_crop['width'],
                     offset_y + ch2_crop['height'], offset_y]
            ax.imshow(ch2_img, cmap='gray', origin='upper',
                     extent=extent, alpha=0.6, zorder=1)
            
        except Exception as e:
            print(f"  Warning: Could not load Ch2 image: {e}")
            # Draw crop box as placeholder
            ch2_rect = Rectangle((ch2_crop['final_left'], ch2_crop['final_top']),
                                ch2_crop['width'], ch2_crop['height'],
                                linewidth=2, edgecolor='red', facecolor='lightcoral', 
                                alpha=0.2, zorder=0)
            ax.add_patch(ch2_rect)
    
    # Plot spline on top
    if len(ch2_data_pts) > 0:
        ax.plot(ch2_data_pts[:, 0], ch2_data_pts[:, 1], 'cyan', linewidth=3, 
                label='Data', zorder=3)
    if len(ch2_bridge_pts) > 0:
        ax.plot(ch2_bridge_pts[:, 0], ch2_bridge_pts[:, 1], 'yellow', linewidth=2.5, 
                alpha=0.7, linestyle='--', label='Bridge', zorder=2)
    
    ax.set_xlabel('X (pixels, global)', fontsize=11)
    ax.set_ylabel('Y (pixels, global)', fontsize=11)
    ax.set_title(f'Ch2 NE {ch2_label} with Fluorescence Image', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add overall title
    fig.suptitle(f'Spline Diagnostic: FoV {fov_id}, Ch1 {ch1_label} ↔ Ch2 {ch2_label}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


# ============================================================================
# BATCH PLOTTING FUNCTIONS
# ============================================================================

def plot_all_pairs_in_fov(fov_id, data_dict, fov_path=None, 
                          output_dir='local_yeast_output/dual_label',
                          n_samples=500, show_plots=False):
    """
    Create diagnostic plots for all NE pairs in a FoV.
    
    Args:
        fov_id: FoV identifier (e.g., '0083')
        data_dict: Dict with splines, crops, and pairs
        fov_path: Optional path to FoV directory for images
        output_dir: Directory to save plots
        n_samples: Points per spline
        show_plots: Whether to display plots (default False for batch processing)
    """
    
    ne_pairs = data_dict['ne_pairs']
    
    if fov_id not in ne_pairs:
        print(f"No NE pairs found for FoV {fov_id}")
        return
    
    pairs = ne_pairs[fov_id]
    
    print(f"\nPlotting {len(pairs)} NE pairs for FoV {fov_id}:")
    print("="*60)
    
    for ch1_label, ch2_label in pairs.items():
        print(f"  Ch1 {ch1_label} ↔ Ch2 {ch2_label}...")
        
        # Create save path
        save_dir = Path(output_dir) / 'diagnostics'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'FoV{fov_id}_Ch1-{ch1_label}_Ch2-{ch2_label}_3panel.png'
        
        try:
            fig = plot_ne_pair_3panel(
                fov_id, ch1_label, ch2_label, data_dict,
                fov_path=fov_path, n_samples=n_samples, save_path=save_path
            )
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)  # Close to save memory
                
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print(f"✓ Completed all plots for FoV {fov_id}")
    print(f"  Saved to: {save_dir}")


# ============================================================================
# CONVENIENCE WRAPPER FUNCTIONS
# ============================================================================

def quick_plot_single_pair(fov_id, ch1_label, ch2_label,
                           output_dir='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label',
                           fov_path=None):
    """
    Quick plot of a single NE pair (loads all data automatically).
    
    Usage:
        quick_plot_single_pair('0083', '08', '09')
    
    Args:
        fov_id: FoV identifier
        ch1_label: Ch1 NE label
        ch2_label: Ch2 NE label
        output_dir: Path to pipeline output directory
        fov_path: Optional path to FoV images
    """
    
    print(f"Loading data from {output_dir}...")
    data_dict = load_all_visualization_data(output_dir)
    
    print(f"Plotting FoV {fov_id}: Ch1 {ch1_label} ↔ Ch2 {ch2_label}...")
    fig = plot_ne_pair_3panel(fov_id, ch1_label, ch2_label, data_dict, fov_path=fov_path)
    
    plt.show()
    
    return fig


def quick_plot_all_in_fov(fov_id, output_dir='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label',
                          fov_path=None, show_plots=False):
    """
    Quick plot of all NE pairs in a FoV (loads all data automatically).
    
    Usage:
        quick_plot_all_in_fov('0083')
    
    Args:
        fov_id: FoV identifier
        output_dir: Path to pipeline output directory
        fov_path: Optional path to FoV images
        show_plots: Whether to display plots interactively
    """
    
    print(f"Loading data from {output_dir}...")
    data_dict = load_all_visualization_data(output_dir)
    
    plot_all_pairs_in_fov(fov_id, data_dict, fov_path=fov_path, 
                         output_dir=output_dir, show_plots=show_plots)


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SPLINE VISUALIZATION TOOL - 3-Panel Diagnostics")
    print("="*80)
    
    # Example 1: Plot a single NE pair
    print("\n1. Plotting single NE pair: FoV 0083, Ch1 01 ↔ Ch2 03")
    quick_plot_single_pair('0083', '01', '03')
    
    # Example 2: Plot all pairs in a FoV (saves to files, doesn't display)
    # print("\n2. Plotting all pairs in FoV 0083...")
    # quick_plot_all_in_fov('0083', show_plots=False)
    
    # Example 3: With image overlay (if you have the FoV path)
    # fov_path = Path('path/to/your/fov/images')
    # quick_plot_single_pair('0083', '08', '09', fov_path=fov_path)
