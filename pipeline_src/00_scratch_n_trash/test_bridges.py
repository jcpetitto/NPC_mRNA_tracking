import numpy as np
from utils.spline_bridging import bridge_refined_splines, fit_parametric_spline
from scipy.interpolate import splprep

def create_noisy_segments(num_segments=3, radius=50, noise_level=0.5):
    angles = np.linspace(0, 2*np.pi, num_segments+1)
    segments = {}
    for i in range(num_segments):
        start_angle, end_angle = angles[i], angles[i+1]
        theta = np.linspace(start_angle, end_angle, 30)
        x = radius * np.cos(theta) + np.random.normal(0, noise_level, size=theta.shape)
        y = radius * np.sin(theta) + np.random.normal(0, noise_level, size=theta.shape)
        tck, _ = splprep([x, y], s=0, k=3)
        segments[f"segment_{i}"] = tck
    return segments

def create_mock_fov_data(num_fovs=3):
    fov_dict = {}
    for f in range(num_fovs):
        ne_label = f"ne_label_{f+1}"
        segments = create_noisy_segments()
        fov_dict[f"FoV_{f+1:04d}"] = {ne_label: segments}
    return fov_dict

if __name__ == "__main__":
    # Create mock data
    mock_data = create_mock_fov_data(num_fovs=3)

    # Config for bridging
    config = {'final_sampling_density': 64, 'bridge_smoothing_factor': 1.0}

    # Run bridging
    bridged_results = bridge_refined_splines(mock_data, config)

    # Print summary
    for fov_id, ne_data in bridged_results.items():
        for ne_label, result in ne_data.items():
            print(f"{fov_id}/{ne_label}:")
            print(f"  Data segments: {len(result['data_segments'])}")
            print(f"  Bridge segments: {len(result['bridge_segments'])}")
            print(f"  Visualization saved as: {fov_id}_{ne_label}_bridging.png")