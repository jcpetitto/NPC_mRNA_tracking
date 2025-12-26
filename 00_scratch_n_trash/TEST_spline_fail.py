import pickle
import pandas as pd

# Load the segment log (it gets saved with your ImageProcessor checkpoint)
with open('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/checkpoints/BMY9999_99_99_9999/state_after_refinement.pkl', 'rb') as f:
    img_proc = pickle.load(f)

segment_log = img_proc.get_segment_logs()
df = pd.DataFrame(segment_log)

# Analyze NaN failures
nan_failures = df[df['status'] == 'fail:nan']
successes = df[df['status'] == 'success']

print("=== NaN Failures vs Successes ===")
print("\nNaN failures intensity stats:")
print(nan_failures[['intensity_max', 'intensity_min', 'intensity_range', 'intensity_std']].describe())

print("\nSuccesses intensity stats:")
print(successes[['intensity_max', 'intensity_min', 'intensity_range', 'intensity_std']].describe())

# Look for patterns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(nan_failures['intensity_range'], bins=50, alpha=0.5, label='NaN failures')
axes[0, 0].hist(successes['intensity_range'], bins=50, alpha=0.5, label='Successes')
axes[0, 0].set_xlabel('Intensity Range')
axes[0, 0].legend()

axes[0, 1].hist(nan_failures['intensity_mean'], bins=50, alpha=0.5, label='NaN failures')
axes[0, 1].hist(successes['intensity_mean'], bins=50, alpha=0.5, label='Successes')
axes[0, 1].set_xlabel('Mean Intensity')
axes[0, 1].legend()

axes[1, 0].hist(nan_failures['intensity_std'], bins=50, alpha=0.5, label='NaN failures')
axes[1, 0].hist(successes['intensity_std'], bins=50, alpha=0.5, label='Successes')
axes[1, 0].set_xlabel('Intensity Std Dev')
axes[1, 0].legend()

axes[1, 1].scatter(nan_failures['intensity_mean'], nan_failures['intensity_std'], alpha=0.3, s=1, label='NaN failures')
axes[1, 1].scatter(successes['intensity_mean'], successes['intensity_std'], alpha=0.3, s=1, label='Successes')
axes[1, 1].set_xlabel('Mean Intensity')
axes[1, 1].set_ylabel('Std Dev')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('failure_analysis.png')
plt.show()