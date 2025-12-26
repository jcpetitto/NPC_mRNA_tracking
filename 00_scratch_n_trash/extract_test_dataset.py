"""
Extract Single NE Pair as Test Dataset - Using Existing Pipeline Metadata

Works backwards from pipeline metadata to extract a test dataset.
Points to original FoV directory and uses existing crop boxes/pairing.
"""

import numpy as np
import pickle
import json
import shutil
from pathlib import Path
import tifffile
from datetime import datetime


def extract_test_ne_from_metadata(
    fov_id,
    ch1_label,
    ch2_label,
    fov_dict,
    ch1_crops,
    ch2_crops,
    output_dir='test_single_ne',
    expand_crop=10
):
    """
    Extract a single NE pair using existing pipeline metadata.
    
    Args:
        fov_id: FoV identifier (e.g., '0083')
        ch1_label: Ch1 NE label (e.g., '08')
        ch2_label: Ch2 NE label (e.g., '09')
        fov_dict: FoV metadata dict with image paths (from pipeline)
        ch1_crops: Ch1 crop boxes dict (from pipeline output)
        ch2_crops: Ch2 crop boxes dict (from pipeline output)
        output_dir: Directory to create test dataset
        expand_crop: Pixels to expand crop box on each side
        
    Returns:
        dict with paths to created test dataset
    """
    
    print("="*80)
    print(f"EXTRACTING TEST DATASET: FoV {fov_id}, Ch1 {ch1_label} ↔ Ch2 {ch2_label}")
    print("="*80)
    
    # Get FoV metadata
    if fov_id not in fov_dict:
        raise ValueError(f"FoV {fov_id} not found in fov_dict")
    
    fov_meta = fov_dict[fov_id]
    fov_path = Path(fov_meta['FoV_collection_path'])
    
    print(f"\nSource FoV directory: {fov_path}")
    
    # Get crop boxes
    print("\n1. Loading crop box information...")
    
    if fov_id not in ch1_crops or ch1_label not in ch1_crops[fov_id]:
        raise ValueError(f"Ch1 NE {ch1_label} not found in FoV {fov_id}")
    if fov_id not in ch2_crops or ch2_label not in ch2_crops[fov_id]:
        raise ValueError(f"Ch2 NE {ch2_label} not found in FoV {fov_id}")
    
    ch1_crop = ch1_crops[fov_id][ch1_label]
    ch2_crop = ch2_crops[fov_id][ch2_label]
    
    print(f"  Ch1 NE {ch1_label}: top={ch1_crop['final_top']}, left={ch1_crop['final_left']}, "
          f"size={ch1_crop['height']}x{ch1_crop['width']}")
    print(f"  Ch2 NE {ch2_label}: top={ch2_crop['final_top']}, left={ch2_crop['final_left']}, "
          f"size={ch2_crop['height']}x{ch2_crop['width']}")
    
    # Calculate expanded bounding box
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
    
    # Create output directory
    print(f"\n3. Creating output directory: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Crop and save images
    print("\n4. Cropping and saving images...")
    
    created_files = []
    
    # Process Ch1 fluorescence
    print("  Processing Ch1 fluorescence...")
    ch1_img_name = fov_meta['imgs']['fn_ch1']
    ch1_img_path = fov_path / ch1_img_name
    
    if ch1_img_path.exists():
        ch1_stack = tifffile.imread(ch1_img_path)
        ch1_cropped = ch1_stack[:, min_top:max_bottom, min_left:max_right]
        
        ch1_output = output_path / f'FoV{fov_id}_NE{ch1_label}-{ch2_label}_ch1.tif'
        tifffile.imwrite(ch1_output, ch1_cropped)
        created_files.append(ch1_output)
        print(f"    Saved: {ch1_output.name} (shape: {ch1_cropped.shape})")
    else:
        print(f"    WARNING: {ch1_img_path} not found")
    
    # Process Ch2 fluorescence
    print("  Processing Ch2 fluorescence...")
    ch2_img_name = fov_meta['imgs']['fn_ch2']
    ch2_img_path = fov_path / ch2_img_name
    
    if ch2_img_path.exists():
        ch2_stack = tifffile.imread(ch2_img_path)
        ch2_cropped = ch2_stack[:, min_top:max_bottom, min_left:max_right]
        
        ch2_output = output_path / f'FoV{fov_id}_NE{ch1_label}-{ch2_label}_ch2.tif'
        tifffile.imwrite(ch2_output, ch2_cropped)
        created_files.append(ch2_output)
        print(f"    Saved: {ch2_output.name} (shape: {ch2_cropped.shape})")
    else:
        print(f"    WARNING: {ch2_img_path} not found")
    
    # Process Ch1 registration (if exists)
    print("  Processing Ch1 registration...")
    if 'fn_ch1_reg1' in fov_meta['imgs']:
        ch1_reg_name = fov_meta['imgs']['fn_ch1_reg1']
        ch1_reg_path = fov_path / ch1_reg_name
        
        if ch1_reg_path.exists():
            ch1_reg_stack = tifffile.imread(ch1_reg_path)
            ch1_reg_cropped = ch1_reg_stack[:, min_top:max_bottom, min_left:max_right]
            
            ch1_reg_output = output_path / f'FoV{fov_id}_NE{ch1_label}-{ch2_label}_ch1_reg1.tif'
            tifffile.imwrite(ch1_reg_output, ch1_reg_cropped)
            created_files.append(ch1_reg_output)
            print(f"    Saved: {ch1_reg_output.name} (shape: {ch1_reg_cropped.shape})")
        else:
            print(f"    WARNING: {ch1_reg_path} not found")
    
    # Process Ch2 registration (if exists)
    print("  Processing Ch2 registration...")
    if 'fn_ch2_reg1' in fov_meta['imgs']:
        ch2_reg_name = fov_meta['imgs']['fn_ch2_reg1']
        ch2_reg_path = fov_path / ch2_reg_name
        
        if ch2_reg_path.exists():
            ch2_reg_stack = tifffile.imread(ch2_reg_path)
            ch2_reg_cropped = ch2_reg_stack[:, min_top:max_bottom, min_left:max_right]
            
            ch2_reg_output = output_path / f'FoV{fov_id}_NE{ch1_label}-{ch2_label}_ch2_reg1.tif'
            tifffile.imwrite(ch2_reg_output, ch2_reg_cropped)
            created_files.append(ch2_reg_output)
            print(f"    Saved: {ch2_reg_output.name} (shape: {ch2_reg_cropped.shape})")
        else:
            print(f"    WARNING: {ch2_reg_path} not found")
    
    # Create coordinate transformation info
    print("\n5. Creating coordinate transformation info...")
    
    transform_info = {
        'source_fov_id': fov_id,
        'ch1_label': ch1_label,
        'ch2_label': ch2_label,
        'coordinate_offset': {
            'left': int(min_left),
            'top': int(min_top)
        },
        'original_ch1_crop': {
            'top': int(ch1_crop['final_top']),
            'left': int(ch1_crop['final_left']),
            'bottom': int(ch1_crop['final_bottom']),
            'right': int(ch1_crop['final_right']),
            'height': int(ch1_crop['height']),
            'width': int(ch1_crop['width'])
        },
        'original_ch2_crop': {
            'top': int(ch2_crop['final_top']),
            'left': int(ch2_crop['final_left']),
            'bottom': int(ch2_crop['final_bottom']),
            'right': int(ch2_crop['final_right']),
            'height': int(ch2_crop['height']),
            'width': int(ch2_crop['width'])
        },
        'adjusted_ch1_crop': {
            'top': int(ch1_crop['final_top'] - min_top),
            'left': int(ch1_crop['final_left'] - min_left),
            'bottom': int(ch1_crop['final_bottom'] - min_top),
            'right': int(ch1_crop['final_right'] - min_left),
            'height': int(ch1_crop['height']),
            'width': int(ch1_crop['width'])
        },
        'adjusted_ch2_crop': {
            'top': int(ch2_crop['final_top'] - min_top),
            'left': int(ch2_crop['final_left'] - min_left),
            'bottom': int(ch2_crop['final_bottom'] - min_top),
            'right': int(ch2_crop['final_right'] - min_left),
            'height': int(ch2_crop['height']),
            'width': int(ch2_crop['width'])
        },
        'test_image_shape': {
            'height': int(new_height),
            'width': int(new_width)
        },
        'created': datetime.now().isoformat()
    }
    
    transform_file = output_path / f'FoV{fov_id}_NE{ch1_label}-{ch2_label}_transform.json'
    with open(transform_file, 'w') as f:
        json.dump(transform_info, f, indent=2)
    
    print(f"  Saved: {transform_file.name}")
    created_files.append(transform_file)
    
    # Create README
    print("\n6. Creating README...")
    
    readme_content = f"""
TEST DATASET - Single NE Pair
=============================

Created: {datetime.now().isoformat()}
Source: FoV {fov_id}, Ch1 NE {ch1_label} ↔ Ch2 NE {ch2_label}

EXTRACTED IMAGES:
{chr(10).join(f'  - {f.name}' for f in created_files if f.suffix == '.tif')}

COORDINATE TRANSFORMATION:
  Original image coordinates shifted by: (-{min_left}, -{min_top})
  
  Original Ch1 NE {ch1_label} crop: 
    top={ch1_crop['final_top']}, left={ch1_crop['final_left']}
    bottom={ch1_crop['final_bottom']}, right={ch1_crop['final_right']}
  
  Adjusted Ch1 NE {ch1_label} crop (in test images):
    top={ch1_crop['final_top'] - min_top}, left={ch1_crop['final_left'] - min_left}
    bottom={ch1_crop['final_bottom'] - min_top}, right={ch1_crop['final_right'] - min_left}
  
  Original Ch2 NE {ch2_label} crop:
    top={ch2_crop['final_top']}, left={ch2_crop['final_left']}
    bottom={ch2_crop['final_bottom']}, right={ch2_crop['final_right']}
  
  Adjusted Ch2 NE {ch2_label} crop (in test images):
    top={ch2_crop['final_top'] - min_top}, left={ch2_crop['final_left'] - min_left}
    bottom={ch2_crop['final_bottom'] - min_top}, right={ch2_crop['final_right'] - min_left}

USAGE:
  These images can be used for:
  - Visual inspection of spline fits
  - Debugging refinement issues
  - Quick testing of distance calculations
  - Creating figures for presentations
  
  Load transform.json to get coordinate mappings between original and test images.

NOTE:
  This is a VISUAL/TESTING dataset only. 
  To run the full pipeline, use the original images with the pipeline metadata.
"""
    
    readme_file = output_path / 'README.txt'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"  Saved: {readme_file.name}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST DATASET CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nLocation: {output_path.absolute()}")
    print(f"\nFiles created: {len(created_files)}")
    for f in created_files:
        print(f"  - {f.name}")
    
    print(f"\nCoordinate offset: (-{min_left}, -{min_top})")
    print(f"Test image size: {new_height}×{new_width} pixels")
    
    return {
        'output_dir': str(output_path.absolute()),
        'fov_id': fov_id,
        'ch1_label': ch1_label,
        'ch2_label': ch2_label,
        'files': [str(f) for f in created_files],
        'coordinate_offset': {'left': min_left, 'top': min_top},
        'transform_file': str(transform_file)
    }


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

def quick_extract_from_pipeline(
    fov_id,
    ch1_label,
    ch2_label,
    pipeline_output_dir='local_yeast_output/dual_label',
    output_dir=None
):
    """
    Extract test dataset using existing pipeline output.
    
    Usage:
        quick_extract_from_pipeline('0083', '08', '09')
    
    Args:
        fov_id: FoV identifier
        ch1_label: Ch1 NE label
        ch2_label: Ch2 NE label
        pipeline_output_dir: Path to pipeline output with metadata
        output_dir: Optional custom output directory
    """
    
    pipeline_path = Path(pipeline_output_dir)
    
    # Load FoV dict
    print(f"Loading metadata from {pipeline_path}...")
    
    # Try to find FoV dict (adjust path based on your structure)
    # This assumes you have a saved FoV dict somewhere
    fov_dict_candidates = [
        pipeline_path / 'fov_metadata.pkl',
        pipeline_path / 'fov_dict.pkl',
        pipeline_path / '../fov_metadata.json',
    ]
    
    fov_dict = None
    for candidate in fov_dict_candidates:
        if candidate.exists():
            if candidate.suffix == '.pkl':
                with open(candidate, 'rb') as f:
                    fov_dict = pickle.load(f)
            elif candidate.suffix == '.json':
                with open(candidate, 'r') as f:
                    fov_dict = json.load(f)
            print(f"  Loaded FoV dict from: {candidate}")
            break
    
    if fov_dict is None:
        raise FileNotFoundError(
            "Could not find FoV metadata. Please provide path to FoV dict.\n"
            f"Searched: {[str(c) for c in fov_dict_candidates]}"
        )
    
    # Load crop boxes
    ch1_crop_path = pipeline_path / 'initial_fit' / 'ch1_crop_BMY9999_99_99_9999.json'
    ch2_crop_path = pipeline_path / 'initial_fit' / 'ch2_crop_BMY9999_99_99_9999.json'
    
    with open(ch1_crop_path, 'r') as f:
        ch1_crops = json.load(f)
    with open(ch2_crop_path, 'r') as f:
        ch2_crops = json.load(f)
    
    print("  Loaded crop boxes")
    
    # Set output directory
    if output_dir is None:
        output_dir = f'test_FoV{fov_id}_NE{ch1_label}-{ch2_label}'
    
    # Extract
    return extract_test_ne_from_metadata(
        fov_id, ch1_label, ch2_label,
        fov_dict, ch1_crops, ch2_crops,
        output_dir=output_dir
    )


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    print("TEST DATASET EXTRACTOR - From Pipeline Metadata")
    print()
    print("Usage:")
    print("  from extract_test_dataset import quick_extract_from_pipeline")
    print("  result = quick_extract_from_pipeline('0083', '08', '09')")
    print()
    print("Or with full control:")
    print("  from extract_test_dataset import extract_test_ne_from_metadata")
    print("  result = extract_test_ne_from_metadata(")
    print("      fov_id='0083',")
    print("      ch1_label='08',")
    print("      ch2_label='09',")
    print("      fov_dict=your_fov_dict,")
    print("      ch1_crops=your_ch1_crops,")
    print("      ch2_crops=your_ch2_crops,")
    print("      output_dir='my_test_dataset'")
    print("  )")
