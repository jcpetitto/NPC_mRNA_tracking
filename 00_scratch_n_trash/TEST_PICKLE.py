import pickle
import numpy as np

# Load the in-progress checkpoint
with open('/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/local_yeast_output/dual_label/merged_splines/refine_results_ch1_BMY9999_99_99_9999.pkl', 'rb') as f:
    data = pickle.load(f)

print("=" * 70)
print("IN-PROGRESS CHECKPOINT STRUCTURE:")
print("=" * 70)

# Top level
print(f"\nTop-level type: {type(data)}")
if isinstance(data, dict):
    print(f"Top-level keys (FoVs): {list(data.keys())}")
    print(f"Number of FoVs: {len(data)}")
    
    # Drill into first FoV
    if data:
        first_fov = list(data.keys())[0]
        print(f"\n--- First FoV: {first_fov} ---")
        print(f"  Type: {type(data[first_fov])}")
        
        if isinstance(data[first_fov], dict):
            print(f"  Keys (NE labels): {list(data[first_fov].keys())}")
            
            # Drill into first NE
            if data[first_fov]:
                first_ne = list(data[first_fov].keys())[0]
                print(f"\n  --- First NE: {first_ne} ---")
                print(f"    Type: {type(data[first_fov][first_ne])}")
                
                if isinstance(data[first_fov][first_ne], dict):
                    print(f"    Keys (segments): {list(data[first_fov][first_ne].keys())}")
                    
                    # Drill into first segment
                    if data[first_fov][first_ne]:
                        first_seg = list(data[first_fov][first_ne].keys())[0]
                        seg_data = data[first_fov][first_ne][first_seg]
                        
                        print(f"\n    --- First Segment: {first_seg} ---")
                        print(f"      Type: {type(seg_data)}")
                        
                        # Check what it actually is
                        if isinstance(seg_data, tuple):
                            print(f"      ✓ IT'S A TUPLE (tck format)")
                            print(f"        Length: {len(seg_data)}")
                            if len(seg_data) == 3:
                                t, c, k = seg_data
                                print(f"        t (knots): type={type(t)}, shape={t.shape if hasattr(t, 'shape') else len(t)}")
                                print(f"        c (coeffs): type={type(c)}, len={len(c) if isinstance(c, (list, tuple)) else 'N/A'}")
                                print(f"        k (degree): {k}")
                        
                        elif isinstance(seg_data, dict):
                            print(f"      ✗ IT'S A DICT (unexpected wrapper!)")
                            print(f"        Keys: {list(seg_data.keys())}")
                            for key, value in seg_data.items():
                                print(f"          '{key}': {type(value)}")
                        
                        elif hasattr(seg_data, 't'):
                            print(f"      ✓ IT'S A BSPLINE OBJECT")
                            print(f"        .t (knots): shape={seg_data.t.shape}")
                            print(f"        .c (coeffs): shape={seg_data.c.shape}")
                            print(f"        .k (degree): {seg_data.k}")
                        
                        else:
                            print(f"      ✗ UNKNOWN TYPE!")
                            print(f"        First 10 attributes: {dir(seg_data)[:10]}")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)

# Count total segments
total_segments = 0
for fov_id, ne_labels in data.items():
    for ne_label, segments in ne_labels.items():
        total_segments += len(segments)

print(f"Total FoVs processed: {len(data)}")
print(f"Total segments saved: {total_segments}")