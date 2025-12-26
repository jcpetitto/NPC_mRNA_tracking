import numpy as np
import matplotlib.pyplot as plt

class YeastTester:

    # --- Primary Image Processing Functions --- #
    ch1_points = np.array([
        [80.0, 60.0], [79.2, 67.8], [77.5, 75.4], [74.3, 82.5], [69.8, 88.9],
        [64.5, 94.4], [58.2, 98.6], [51.3, 101.3], [44.1, 102.4], [37.0, 101.8],
        [30.4, 99.6], [24.5, 95.8], [19.7, 90.7], [16.2, 84.6], [14.3, 77.8],
        [14.1, 70.6], [15.5, 63.3], [18.4, 56.4], [22.6, 50.1], [27.8, 44.8],
        [33.8, 40.8], [40.2, 38.3], [46.9, 37.4], [53.6, 38.1], [60.0, 40.2],
        [65.9, 43.6], [71.1, 48.1], [75.4, 53.5], [78.5, 59.6], [80.1, 66.0]
    ])
    ch2_points = np.array([
        [78.5, 62.1], [78.1, 68.9], [76.5, 75.5], [73.8, 81.7], [69.9, 87.2],
        [65.2, 91.8], [59.6, 95.3], [53.5, 97.4], [47.2, 98.1], [41.1, 97.2],
        [35.4, 94.8], [30.4, 91.1], [26.3, 86.3], [23.3, 80.7], [21.6, 74.6],
        [21.5, 68.1], [22.8, 61.6], [25.5, 55.6], [29.4, 50.4], [34.2, 46.3],
        [39.7, 43.5], [45.6, 42.1], [51.7, 42.2], [57.7, 43.8], [63.3, 46.7],
        [68.3, 50.7], [72.5, 55.6], [75.8, 61.1], [78.0, 67.0], [78.7, 72.8]
    ])
    # noise - decimal representation of % noise
    # ch2 point defaults alightly smaller radius with small spatial shift
    sample_parameters = { "center1" : (100, 100), "radii1" : (50, 45), "noise1" : 0.2, "center2" : (102, 98), "radii2" : (48, 43), "noise2" : 0.2}

    @staticmethod
    def generate_irregular_shapes(ch_def_dict = None, num_points=40):
        """
            Generates two sets of 2D points forming irregular circular/elliptical shapes.
            Args:
                num_points (int): The number of points to generate for each shape.
            Returns:
                tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays,
                                            ch1_points and ch2_points, each with
                                            shape (num_points, 2).
        """
        if ch_def_dict is None:
            ch_def_dict = YeastTester.sample_parameters
        else:
            try: 
                ch_keys = ["center1", "radii1", "noise1", "center2", "radii2", "noise2"]
                missing_or_none_keys = [key for key in ch_keys if params.get(key) is None]
                missing_or_falsy_keys = [key for key in ch_keys if not params.get(key)]

                if missing_or_none_keys or missing_or_falsy_keys:
                    raise ValueError(print(f"Error: The following keys are missing or have no value: {missing_or_none_keys}"))

                print("Success: All required keys are present and have values.")
            except ValueError as e:
                print(f"Point generation halted. \n{e}")
                return

        # --- Generate the two shapes with slightly different properties ---
        
        # Shape 1 (e.g., outer envelope)
        
        ch1_points = YeastTester.create_single_ellipse(num_points, ch_def_dict["center1"], ch_def_dict["radii1"], ch_def_dict["noise1"])
        
        # Shape 2 (e.g., inner envelope, slightly shifted)
        ch2_points = YeastTester.create_single_ellipse(num_points, ch_def_dict["center2"], ch_def_dict["radii2"], ch_def_dict["noise2"])
        
        return ch1_points, ch2_points

# Used in generate_irregular_ellipse
    @staticmethod
    def create_single_ellipse(num_points, center, radii, noise_level):
        """Generates points that form a single noisy ellipse."""
        # Generate evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Create points for a perfect ellipse
        perfect_x = center[0] + radii[0] * np.cos(angles)
        perfect_y = center[1] + radii[1] * np.sin(angles)
        
        # Generate random noise to make the shape irregular
        # The noise is scaled by the radius to keep it proportional
        noise_x = (np.random.rand(num_points) - 0.5) * noise_level * radii[0]
        noise_y = (np.random.rand(num_points) - 0.5) * noise_level * radii[1]
        
        # Combine into a (num_points, 2) array
        points = np.column_stack((perfect_x + noise_x, perfect_y + noise_y))
        
        return points

ch1_random, ch2_random = YeastTester.generate_irregular_shapes(num_points=50)

print(f"Generated Channel 1 with shape: {ch1_random.shape}")
print(f"Generated Channel 2 with shape: {ch2_random.shape}")


# # --- 2. Visualize the generated shapes ---
    # plt.figure(figsize=(8, 8))

    # # Scatter plot for the first shape
    # plt.scatter(ch1_random[:, 0], ch1_random[:, 1], c='dodgerblue', label='Random Shape 1')

    # # Scatter plot for the second shape
    # plt.scatter(ch2_random[:, 0], ch2_random[:, 1], c='crimson', label='Random Shape 2')

    # # Formatting
    # plt.title(f'Randomly Generated Shapes ({len(ch1_random)} points each)')
    # plt.xlabel('X-coordinate')
    # plt.ylabel('Y-coordinate')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.axis('equal') # Ensures shapes are not distorted
    # plt.show()