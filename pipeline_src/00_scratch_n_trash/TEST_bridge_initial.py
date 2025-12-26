from utils.spline_bridging import bridge_refined_splines
import pickle
import json

# Load config for bridging parameters
with open('config_local_dual.json', 'r') as f:
    config = json.load(f)
ne_fit_config = config['image processor']['ne_fit']

# Load initial splines
with open('local_yeast_output/dual_label/init_fit/ch1_bsplines_BMY9999_99_99_9999.pkl', 'rb') as f:
    init_ch1 = pickle.load(f)
with open('local_yeast_output/dual_label/init_fit/ch2_bsplines_BMY9999_99_99_9999.pkl', 'rb') as f:
    init_ch2 = pickle.load(f)

# Bridge them!
print("Bridging initial Ch1 splines...")
bridged_ch1 = bridge_refined_splines(init_ch1, ne_fit_config)

print("Bridging initial Ch2 splines...")
bridged_ch2 = bridge_refined_splines(init_ch2, ne_fit_config)

# Save for distance calculation
with open('local_yeast_output/dual_label/merged_splines/bridged_initial_ch1.pkl', 'wb') as f:
    pickle.dump(bridged_ch1, f)
with open('local_yeast_output/dual_label/merged_splines/bridged_initial_ch2.pkl', 'wb') as f:
    pickle.dump(bridged_ch2, f)

print(f"Ch1 bridged FoVs: {len(bridged_ch1)}")
print(f"Ch2 bridged FoVs: {len(bridged_ch2)}")