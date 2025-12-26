"""
Extract Single NE Pair as Test Dataset

Creates a minimal test dataset containing just one NE pair from both channels,
with properly cropped images and updated metadata for pipeline testing.
"""

import numpy as np
import pickle
import json
import shutil
from pathlib import Path
import tifffile
from datetime import datetime


def create_test_dataset_single_ne_pair(
    fov_id, 
    ch1_label, 
    ch2_label,
    source_fov_path,
    output_dir='test_single_ne',
    ch1_img_pattern='*ch1*.tif',
    ch2_img_pattern='*ch2*.tif',
    ch1_reg_pattern='*ch1*reg*.tif',
    ch2_reg_pattern='*ch2*reg*.tif',
    pipeline_output_dir='local_yeast_output/dual_label',
    expand_crop=10
):
    """
    Extract a single NE pair and create a minimal test dataset.
    
    Args:
        fov_id: FoV identifier (e.g., '0083')
        ch1_label: Ch1 NE label (e.g., '08')
        ch2_label: Ch2 NE label (e.g., '09')
        source_fov_path: Path to original FoV images
        output_dir: Directory to create test dataset
        ch1_img_pattern: Glob pattern for Ch1 fluorescence images
        ch2_img_pattern: Glob pattern for Ch2 fluorescence images
        ch1_reg_pattern: Glob pattern for Ch1 registration (brightfield) images
        ch2_reg_pattern: Glob pattern for Ch2 registration (brightfield) images
        pipeline_output_dir: Path to existing pipeline output with crop boxes
        expand_crop: Pixels to expand crop box on each side (for context)
        
    Returns:
        dict with paths to created test dataset
    """
    
    print("="*80)
    print(f"CREATING TEST DATASET: FoV {fov_id}, Ch1 {ch1_label} ↔ Ch2 {ch2_label}")
    print("="*80)
    
    # Load crop boxes
    print("\n1. Loading crop box information...")
    ch1_crop_path = Path(pipeline_output_dir) / 'initial_fit' / 'ch1_crop_BMY9999_99_99_9999.json'
    ch2_crop_path = Path(pipeline_output_dir) / 'initial_fit' / 'ch2_crop_BMY9999_99_99_9999.json'
    
    with open(ch1_crop_path, 'r') as f:
        ch1_crops = json.load(f)
    with open(ch2_crop_path, 'r') as f:
        ch2_crops = json.load(f)
    
    ch1_crop = ch1_crops[fov_id][ch1_label]
    ch2_crop = ch2_crops[fov_id][ch2_label]
    
    print(f"  Ch1 crop: top={ch1_crop['final_top']}, left={ch1_crop['final_left']}, "
          f"size={ch1_crop['height']}x{ch1_crop['width']}")
    print(f"  Ch2 crop: top={ch2_crop['final_top']}, left={ch2_crop['final_left']}, "
          f"size={ch2_crop['height']}x{ch2_crop['width']}")
    
    # Calculate expanded bounding box that covers both NEs
    print(f"\n2. Calculating expanded bounding box (expand={expand_crop}px)...")
    
    min_left = min(ch1_crop['final_left'], ch2_crop['final_left']) - expand_crop
    min_top = min(ch1_crop['final_top'], ch2_crop['final_top']) - expand_crop
    max_right = max(ch1_crop['final_right'], ch2_crop['final_right']) + expand_crop
    max_bottom = max(ch1_crop['final_bottom'], ch2_crop['final_bottom']) + expand_crop
    
    # Ensure non-negative
    min_left = max(0, min_left)
    min_top = max(0, min_top)
    
    new_width = max_right - min_left
    new_height = max_bottom - min_top
    
    print(f"  Combined crop: top={min_top}, left={min_left}, "
          f"size={new_height}x{new_width}")
    
    # Create output directory structure
    print(f"\n3. Creating output directory: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory
    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Crop and save images
    print("\n4. Cropping and saving images...")
    
    created_files = {
        'ch1_fluor': None,
        'ch2_fluor': None,
        'ch1_reg': None,
        'ch2_reg': None,
        'metadata': None,
        'crop_info': None
    }
    
    source_path = Path(source_fov_path)
    
    # Process Ch1 fluorescence
    print("  Processing Ch1 fluorescence...")
    ch1_files = list(source_path.glob(ch1_img_pattern))
    if ch1_files:
        ch1_stack = tifffile.imread(ch1_files[0])
        ch1_cropped = ch1_stack[:, min_top:max_bottom, min_left:max_right]
        
        ch1_output = images_dir / f'test_fov_ch1.tif'
        tifffile.imwrite(ch1_output, ch1_cropped)
        created_files['ch1_fluor'] = ch1_output
        print(f"    Saved: {ch1_output} (shape: {ch1_cropped.shape})")
    else:
        print(f"    WARNING: No Ch1 images found with pattern {ch1_img_pattern}")
    
    # Process Ch2 fluorescence
    print("  Processing Ch2 fluorescence...")
    ch2_files = list(source_path.glob(ch2_img_pattern))
    if ch2_files:
        ch2_stack = tifffile.imread(ch2_files[0])
        ch2_cropped = ch2_stack[:, min_top:max_bottom, min_left:max_right]
        
        ch2_output = images_dir / f'test_fov_ch2.tif'
        tifffile.imwrite(ch2_output, ch2_cropped)
        created_files['ch2_fluor'] = ch2_output
        print(f"    Saved: {ch2_output} (shape: {ch2_cropped.shape})")
    else:
        print(f"    WARNING: No Ch2 images found with pattern {ch2_img_pattern}")
    
    # Process Ch1 registration (brightfield)
    print("  Processing Ch1 registration images...")
    ch1_reg_files = list(source_path.glob(ch1_reg_pattern))
    if ch1_reg_files:
        ch1_reg_stack = tifffile.imread(ch1_reg_files[0])
        ch1_reg_cropped = ch1_reg_stack[:, min_top:max_bottom, min_left:max_right]
        
        ch1_reg_output = images_dir / f'test_fov_ch1_reg1.tif'
        tifffile.imwrite(ch1_reg_output, ch1_reg_cropped)
        created_files['ch1_reg'] = ch1_reg_output
        print(f"    Saved: {ch1_reg_output} (shape: {ch1_reg_cropped.shape})")
    else:
        print(f"    WARNING: No Ch1 reg images found with pattern {ch1_reg_pattern}")
    
    # Process Ch2 registration (brightfield)
    print("  Processing Ch2 registration images...")
    ch2_reg_files = list(source_path.glob(ch2_reg_pattern))
    if ch2_reg_files:
        ch2_reg_stack = tifffile.imread(ch2_reg_files[0])
        ch2_reg_cropped = ch2_reg_stack[:, min_top:max_bottom, min_left:max_right]
        
        ch2_reg_output = images_dir / f'test_fov_ch2_reg1.tif'
        tifffile.imwrite(ch2_reg_output, ch2_reg_cropped)
        created_files['ch2_reg'] = ch2_reg_output
        print(f"    Saved: {ch2_reg_output} (shape: {ch2_reg_cropped.shape})")
    else:
        print(f"    WARNING: No Ch2 reg images found with pattern {ch2_reg_pattern}")
    
    # Update crop box coordinates to new coordinate system
    print("\n5. Creating updated crop box metadata...")
    
    # Adjust crop coordinates to new coordinate system
    ch1_crop_adjusted = {
        'height': ch1_crop['height'],
        'width': ch1_crop['width'],
        'final_top': ch1_crop['final_top'] - min_top,
        'final_left': ch1_crop['final_left'] - min_left,
        'final_bottom': ch1_crop['final_bottom'] - min_top,
        'final_right': ch1_crop['final_right'] - min_left,
        'original_label': ch1_label,
        'original_fov': fov_id
    }
    
    ch2_crop_adjusted = {
        'height': ch2_crop['height'],
        'width': ch2_crop['width'],
        'final_top': ch2_crop['final_top'] - min_top,
        'final_left': ch2_crop['final_left'] - min_left,
        'final_bottom': ch2_crop['final_bottom'] - min_top,
        'final_right': ch2_crop['final_right'] - min_left,
        'original_label': ch2_label,
        'original_fov': fov_id
    }
    
    # Create new crop box files with original NE labels preserved
    ch1_crop_new = {
        'test_fov': {
            ch1_label: ch1_crop_adjusted
        }
    }
    
    ch2_crop_new = {
        'test_fov': {
            ch2_label: ch2_crop_adjusted
        }
    }
    
    # Save crop boxes
    ch1_crop_file = output_path / 'ch1_crop_test.json'
    ch2_crop_file = output_path / 'ch2_crop_test.json'
    
    with open(ch1_crop_file, 'w') as f:
        json.dump(ch1_crop_new, f, indent=2)
    with open(ch2_crop_file, 'w') as f:
        json.dump(ch2_crop_new, f, indent=2)
    
    print(f"  Saved: {ch1_crop_file}")
    print(f"  Saved: {ch2_crop_file}")
    
    created_files['crop_info'] = {
        'ch1': str(ch1_crop_file),
        'ch2': str(ch2_crop_file)
    }
    
    # Create NE pairing file with original labels
    print("\n6. Creating NE pairing metadata...")
    
    ne_pairs = {
        'test_fov': {
            ch1_label: ch2_label  # Preserve original labels
        }
    }
    
    pairs_file = output_path / 'ne_pairs_test.json'
    with open(pairs_file, 'w') as f:
        json.dump(ne_pairs, f, indent=2)
    
    print(f"  Saved: {pairs_file}")
    print(f"  Pairing: Ch1 {ch1_label} ↔ Ch2 {ch2_label}")
    
    # Create FoV metadata file
    print("\n7. Creating FoV metadata...")
    
    fov_metadata = {
        'test_fov': {
            'FoV_collection_path': str(images_dir),
            'imgs': {
                'fn_ch1': 'test_fov_ch1.tif',
                'fn_ch2': 'test_fov_ch2.tif',
                'fn_ch1_reg1': 'test_fov_ch1_reg1.tif',
                'fn_ch2_reg1': 'test_fov_ch2_reg1.tif'
            },
            'original_fov_id': fov_id,
            'original_ch1_label': ch1_label,
            'original_ch2_label': ch2_label,
            'crop_offset': {
                'left': min_left,
                'top': min_top
            },
            'image_shape': [new_height, new_width],
            'created': datetime.now().isoformat()
        }
    }
    
    metadata_file = output_path / 'fov_metadata_test.json'
    with open(metadata_file, 'w') as f:
        json.dump(fov_metadata, f, indent=2)
    
    print(f"  Saved: {metadata_file}")
    created_files['metadata'] = str(metadata_file)
    
    # Create README
    print("\n8. Creating README...")
    
    readme_content = f"""
TEST DATASET - Single NE Pair
=============================

Created: {datetime.now().isoformat()}
Source: FoV {fov_id}, Ch1 NE {ch1_label} ↔ Ch2 NE {ch2_label}

DATASET STRUCTURE:
- images/
  - test_fov_ch1.tif          Ch1 fluorescence (cropped)
  - test_fov_ch2.tif          Ch2 fluorescence (cropped)
  - test_fov_ch1_reg1.tif     Ch1 registration/brightfield (cropped)
  - test_fov_ch2_reg1.tif     Ch2 registration/brightfield (cropped)

- ch1_crop_test.json          Ch1 crop box (in new coordinate system)
- ch2_crop_test.json          Ch2 crop box (in new coordinate system)
- ne_pairs_test.json          NE pairing (Ch1 {ch1_label} ↔ Ch2 {ch2_label})
- fov_metadata_test.json      Complete FoV metadata for pipeline

COORDINATE TRANSFORMATION:
Original crop boxes were adjusted to new coordinate system:
  - Original bounding box: top={min_top}, left={min_left}
  - All coordinates shifted by: (-{min_left}, -{min_top})
  - NE labels preserved: Ch1 {ch1_label} ↔ Ch2 {ch2_label}

USAGE:
To run pipeline on this test dataset, use fov_metadata_test.json as input.
The single NE pair should be detectable and processable through the full pipeline.

NE LABELS:
  - Ch1: {ch1_label}
  - Ch2: {ch2_label}
  - Pairing: {ch1_label} ↔ {ch2_label}

ORIGINAL CROP BOXES:
Ch1 NE {ch1_label}:
  top={ch1_crop['final_top']}, left={ch1_crop['final_left']}
  bottom={ch1_crop['final_bottom']}, right={ch1_crop['final_right']}
  size={ch1_crop['height']}x{ch1_crop['width']}

Ch2 NE {ch2_label}:
  top={ch2_crop['final_top']}, left={ch2_crop['final_left']}
  bottom={ch2_crop['final_bottom']}, right={ch2_crop['final_right']}
  size={ch2_crop['height']}x{ch2_crop['width']}

ADJUSTED CROP BOXES (in test dataset):
Ch1 NE {ch1_label}:
  top={ch1_crop_adjusted['final_top']}, left={ch1_crop_adjusted['final_left']}
  bottom={ch1_crop_adjusted['final_bottom']}, right={ch1_crop_adjusted['final_right']}
  
Ch2 NE {ch2_label}:
  top={ch2_crop_adjusted['final_top']}, left={ch2_crop_adjusted['final_left']}
  bottom={ch2_crop_adjusted['final_bottom']}, right={ch2_crop_adjusted['final_right']}
"""
    
    readme_file = output_path / 'README.txt'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"  Saved: {readme_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST DATASET CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nLocation: {output_path.absolute()}")
    print(f"\nFiles created:")
    for key, value in created_files.items():
        if value:
            print(f"  {key}: {value}")
    
    print(f"\nTo run pipeline on this test dataset:")
    print(f"  1. Load FoV metadata: {metadata_file}")
    print(f"  2. Use FoV ID: 'test_fov'")
    print(f"  3. NE pair: Ch1 {ch1_label} ↔ Ch2 {ch2_label}")
    
    return {
        'output_dir': str(output_path.absolute()),
        'fov_id': 'test_fov',
        'ne_pair': {'ch1': ch1_label, 'ch2': ch2_label},
        'metadata_file': str(metadata_file),
        'files': created_files,
        'coordinate_offset': {'left': min_left, 'top': min_top}
    }


# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

def quick_create_test_ne(fov_id, ch1_label, ch2_label, source_fov_path,
                         output_dir='test_single_ne'):
    """
    Quick wrapper to create test dataset with minimal arguments.
    
    Usage:
        quick_create_test_ne('0083', '08', '09', '/path/to/fov/images')
    """
    
    return create_test_dataset_single_ne_pair(
        fov_id=fov_id,
        ch1_label=ch1_label,
        ch2_label=ch2_label,
        source_fov_path=source_fov_path,
        output_dir=output_dir
    )


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    # print("TEST DATASET CREATOR - Single NE Pair Extraction")
    # print()
    # print("Usage example:")
    # print("  from create_test_dataset import quick_create_test_ne")
    # print("  result = quick_create_test_ne('0083', '08', '09', '/path/to/fov/images')")
    # print()
    # print("Or with full control:")
    # print("  from create_test_dataset import create_test_dataset_single_ne_pair")
    # print("  result = create_test_dataset_single_ne_pair(")
    # print("      fov_id='0083',")
    # print("      ch1_label='08',")
    # print("      ch2_label='09',")
    # print("      source_fov_path='/path/to/images',")
    # print("      output_dir='my_test_dataset',")
    # print("      expand_crop=15")
    # print("  )")

    result = create_test_dataset_single_ne_pair(
      fov_id='0083',
      ch1_label='08',
      ch2_label='09',
      source_fov_path='/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/example_data/dual_strain_label/BMY9999/99_99_9999',
      output_dir='my_test_dataset',
      expand_crop=15
  )
