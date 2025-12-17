import numpy as np
import matplotlib.pyplot as plt

def visualize_transformation(transform_func, scale, angle, shift, center, title="Transformation Debug"):
    """
    Visualizes a transformation to verify rotation direction and translation.
    
    Args:
        transform_func: The function to test (e.g., transform_coordinates)
        scale, angle, shift, center: Parameters to pass to the function.
        title: Title for the plot.
    """
    # 1. Define an Asymmetric Test Shape (The "L" / Arrow)
    #    (Row, Col) format.
    #    Points: Origin, Down 20px, Right 10px (at the bottom)
    original_points = np.array([
        [0, 0],    # Top-Left (Center/Origin)
        [40, 0],   # Down
        [40, 20]   # Right (The "foot" of the L)
    ])
    
    # 2. Apply the Transformation
    #    We assume the function signature matches your current implementation
    transformed_points = transform_func(original_points, scale, angle, shift, center)

    # 3. Setup Plot
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # --- THE CRITICAL PART: Image Coordinates ---
    # Invert Y axis so "Down" is positive Y (Image style)
    ax.invert_yaxis()
    
    # 4. Plot Original (Blue)
    plt.plot(original_points[:, 1], original_points[:, 0], 'o-', 
             label='Original (Input)', color='blue', linewidth=2, markersize=8)
    # Label the "Start" to see translation clearly
    plt.text(original_points[0, 1], original_points[0, 0], ' Origin', 
             color='blue', verticalalignment='bottom')
    
    # 5. Plot Transformed (Red)
    plt.plot(transformed_points[:, 1], transformed_points[:, 0], 'o-', 
             label=f'Transformed\n(Angle: {angle}°, Shift: {shift})', 
             color='red', linewidth=2, markersize=8)
    # Label the "Start" to see where the origin moved
    plt.text(transformed_points[0, 1], transformed_points[0, 0], ' New Origin', 
             color='red', verticalalignment='bottom')

    # 6. Formatting to make it readable
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.axvline(0, color='black', linewidth=1, alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel('Column (X)')
    plt.ylabel('Row (Y)')
    plt.title(title)
    
    # Force aspect ratio to be equal so circles don't look like ovals
    ax.set_aspect('equal', adjustable='box')
    
    # Show it
    plt.show()

# --- USAGE EXAMPLE ---
# from geom_tools import transform_coordinates

# Test a simple 90 degree rotation
# visualize_transformation(
#     transform_func=transform_coordinates, 
#     scale=1.0, 
#     angle=90, 
#     shift=np.array([0,0]), 
#     center=np.array([0,0]),
#     title="Check: Does +90 go Left (CW) or Right (CCW)?"
# )