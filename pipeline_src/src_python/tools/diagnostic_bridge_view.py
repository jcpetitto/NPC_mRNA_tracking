import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import BSpline

def inspect_spline_components(spline_pickle_path, fov_id, ne_label, crop_image_path=None):
    """
    Visualizes the exact composition of a bridged spline:
    - Data Segments: BLUE (Solid)
    - Bridge Segments: RED (Dashed)
    - Junctions: GREEN Dots (Where Data ends and Bridge begins)
    """
    # 1. Load Data
    with open(spline_pickle_path, 'rb') as f:
        spline_dict = pickle.load(f)
    
    if fov_id not in spline_dict or ne_label not in spline_dict[fov_id]:
        print(f"Error: Could not find {fov_id} / {ne_label} in the provided file.")
        return

    ne_data = spline_dict[fov_id][ne_label]
    data_segs = ne_data.get('data_segments', [])
    bridge_segs = ne_data.get('bridge_segments', [])
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('black') # High contrast for fluorescent-style viewing
    
    # Optional: Load background image if provided (requires your crop JSON and image paths)
    # (Skipping here to keep this function purely geometry-focused and standalone)

    print(f"--- Components for {fov_id} / {ne_label} ---")
    print(f"Data Segments: {len(data_segs)}")
    print(f"Bridge Segments: {len(bridge_segs)}")

    # 3. Plot Data Segments (The "Real" Physics)
    for i, spline in enumerate(data_segs):
        u = np.linspace(0, 1, 100)
        pts = np.array([spline(x) for x in u])
        
        # Plot Line
        ax.plot(pts[:, 0], pts[:, 1], color='cyan', linewidth=3, alpha=0.8, 
                label='Data' if i == 0 else "")
        
        # Plot Endpoints (The "Hooks")
        # Start = Circle, End = X
        ax.scatter(pts[0, 0], pts[0, 1], color='cyan', marker='o', s=50, zorder=10)
        ax.scatter(pts[-1, 0], pts[-1, 1], color='cyan', marker='x', s=50, zorder=10)

    # 4. Plot Bridge Segments (The "Fake" Math)
    for i, spline in enumerate(bridge_segs):
        u = np.linspace(0, 1, 100)
        pts = np.array([spline(x) for x in u])
        
        # Plot Line
        ax.plot(pts[:, 0], pts[:, 1], color='red', linewidth=2, linestyle='--', 
                label='Bridge' if i == 0 else "")

    # 5. Legend and Labels
    ax.legend(loc='upper right', frameon=True, facecolor='gray')
    ax.set_title(f"Spline Topology: FoV {fov_id} Label {ne_label}\nCyan=Data (o=Start, x=End) | Red=Bridge", 
                 color='white', fontsize=14)
    ax.tick_params(colors='white')
    
    # Invert Y axis to match image coordinates (Images usually have 0,0 at top-left)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Replace with your actual path
    pkl_path = "/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/refined_fit/bridged_splines_ch2_BMY9999_99_99_9999.pkl"
    
    # Run the inspector
    inspect_spline_components(pkl_path, fov_id='0083', ne_label='03')
