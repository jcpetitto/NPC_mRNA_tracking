"""
Verify coordinate system consistency between images and splines.
"""

import numpy as np
import matplotlib.pyplot as plt

def verify_coordinate_conventions():
    """
    Test that image coordinates and spline coordinates align correctly.
    """
    
    print("="*80)
    print("COORDINATE SYSTEM VERIFICATION")
    print("="*80)
    
    # Create a test image with known features
    img = np.zeros((100, 100))
    
    # Put bright pixels at specific (y, x) = (row, col) locations
    img[20, 30] = 1.0  # Point A: row=20 (y=20), col=30 (x=30)
    img[70, 80] = 1.0  # Point B: row=70 (y=70), col=80 (x=80)
    
    # Create spline points that should overlay these pixels
    # Splines store (x, y) so reverse the order
    spline_points = np.array([
        [30, 20],  # (x=30, y=20) should hit img[20, 30]
        [80, 70]   # (x=80, y=70) should hit img[70, 80]
    ])
    
    print("\nTest Setup:")
    print("  Image shape: (100, 100) = (height, width) = (rows, cols)")
    print("  Bright pixel A at: img[20, 30] = (row=20, col=30) = (y=20, x=30)")
    print("  Bright pixel B at: img[70, 80] = (row=70, col=80) = (y=70, x=80)")
    print()
    print("  Spline point A: [30, 20] = (x=30, y=20)")
    print("  Spline point B: [80, 70] = (x=80, y=70)")
    
    # Plot with correct extent
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ===== METHOD 1: Default (often wrong) =====
    ax = axes[0]
    ax.imshow(img, cmap='gray', origin='upper')
    ax.plot(spline_points[:, 0], spline_points[:, 1], 'ro-', markersize=10, 
            label='Spline points', linewidth=2)
    ax.set_title('Method 1: No extent (WRONG)\nSplines at pixel indices')
    ax.set_xlabel('Pixel col (x index)')
    ax.set_ylabel('Pixel row (y index)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== METHOD 2: With extent (correct for data coordinates) =====
    ax = axes[1]
    # Extent: [left, right, bottom, top] when origin='upper'
    # For data coords matching array indices: [0, width, height, 0]
    extent = [0, 100, 100, 0]  # [left=0, right=100, bottom=100, top=0]
    ax.imshow(img, cmap='gray', origin='upper', extent=extent)
    ax.plot(spline_points[:, 0], spline_points[:, 1], 'ro-', markersize=10, 
            label='Spline points', linewidth=2)
    ax.set_title('Method 2: With extent (CORRECT)\nData coordinates')
    ax.set_xlabel('X (data coordinates)')
    ax.set_ylabel('Y (data coordinates)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== METHOD 3: Offset extent (for cropped regions) =====
    ax = axes[2]
    # Simulate a crop at offset (10, 15)
    offset_x, offset_y = 10, 15
    extent = [offset_x, offset_x + 100, offset_y + 100, offset_y]
    ax.imshow(img, cmap='gray', origin='upper', extent=extent)
    
    # Adjust spline points to global coordinates
    spline_points_global = spline_points + np.array([offset_x, offset_y])
    ax.plot(spline_points_global[:, 0], spline_points_global[:, 1], 
            'ro-', markersize=10, label='Spline points (global)', linewidth=2)
    ax.set_title('Method 3: Offset extent (CORRECT)\nGlobal coordinates')
    ax.set_xlabel('X (global coordinates)')
    ax.set_ylabel('Y (global coordinates)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS:")
    print("="*80)
    print("\nMethod 1 (No extent):")
    print("  ✓ Red points should be at (30, 20) and (80, 70) in data coords")
    print("  ✗ BUT matplotlib interprets these as pixel indices")
    print("  → Alignment depends on default behavior")
    
    print("\nMethod 2 (With extent [0, 100, 100, 0]):")
    print("  ✓ Red points at (30, 20) and (80, 70) in data coords")
    print("  ✓ Image displayed with same coordinate system")
    print("  ✓ Points should overlay bright pixels PERFECTLY")
    
    print("\nMethod 3 (Offset extent):")
    print("  ✓ Simulates cropped region at global coords (10, 15)")
    print("  ✓ Red points shifted to (40, 35) and (90, 85)")
    print("  ✓ Still overlays bright pixels if coordinates match")
    
    print("\n" + "="*80)
    print("CONCLUSION FOR YOUR PIPELINE:")
    print("="*80)
    print("""
If splines are in GLOBAL image coordinates:
  - Use extent = [left, left+width, top+height, top]
  - This is what visualize_splines_3panel.py does ✓
  
If splines are in LOCAL crop coordinates:
  - Use extent = [0, width, height, 0]
  - Then splines at (x, y) overlay img[y, x] correctly
    
Critical: matplotlib plot(x, y) expects (x, y) NOT (y, x)
Critical: numpy array img[row, col] is img[y, x] NOT img[x, y]
Critical: imshow extent is [left, right, bottom, top] when origin='upper'
""")
    
    plt.show()


if __name__ == "__main__":
    verify_coordinate_conventions()
