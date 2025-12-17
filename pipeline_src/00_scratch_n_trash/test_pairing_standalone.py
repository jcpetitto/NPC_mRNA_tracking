"""
Test the new mask-based pairing function without re-running pipeline.
Uses saved detection results to test pairing.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from utils.ne_mask_pairing import pair_nes_by_masks


def test_pairing_from_checkpoint(
    experiment_name='BMY9999_99_99_9999',
    checkpoint_dir='local_yeast_output/dual_label/checkpoints',
    min_iou=0.7,
    max_centroid_dist=10
):
    """
    Test pairing using saved masks from checkpoint.
    """
    
    print(f"\n{'='*70}")
    print(f"TESTING MASK-BASED PAIRING")
    print(f"{'='*70}\n")
    
    # Load the checkpoint that has masks
    checkpoint_path = Path(checkpoint_dir) / experiment_name / 'state_after_init_ne.pkl'
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run the pipeline at least through initial detection first.")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        img_proc = pickle.load(f)
    
    # Extract masks
    ch1_masks = img_proc._get_ch1_masks()
    ch2_masks = img_proc._get_ch2_masks()
    
    if not ch1_masks or not ch2_masks:
        print("ERROR: No masks found in checkpoint!")
        print("The checkpoint may be from before masks were added.")
        print("You'll need to re-run initial detection with the updated code.")
        return None
    
    print(f"\nFound masks for {len(ch1_masks)} FoVs in Ch1")
    print(f"Found masks for {len(ch2_masks)} FoVs in Ch2")
    
    # Test pairing for each FoV
    all_pairs = {}
    
    for fov_id in ch1_masks.keys():
        if fov_id not in ch2_masks:
            print(f"\nSkipping FoV {fov_id}: No Ch2 masks")
            continue
        
        ch1_fov_masks = ch1_masks[fov_id]
        ch2_fov_masks = ch2_masks[fov_id]
        
        print(f"\n--- FoV {fov_id} ---")
        print(f"  Ch1 labels: {list(ch1_fov_masks.keys())}")
        print(f"  Ch2 labels: {list(ch2_fov_masks.keys())}")
        
        # Run pairing
        pairs = pair_nes_by_masks(
            ch1_masks_dict=ch1_fov_masks,
            ch2_masks_dict=ch2_fov_masks,
            min_iou=min_iou,
            max_centroid_distance_pixels=max_centroid_dist
        )
        
        print(f"  Pairs found: {pairs}")
        
        if len(pairs) == 0:
            print(f"  ⚠️  No valid pairs found!")
        else:
            print(f"  ✓ Found {len(pairs)} valid pairs")
        
        all_pairs[fov_id] = pairs
    
    # Summary
    total_ch1 = sum(len(ch1_masks[fov]) for fov in ch1_masks.keys())
    total_ch2 = sum(len(ch2_masks[fov]) for fov in ch2_masks.keys())
    total_pairs = sum(len(pairs) for pairs in all_pairs.values())
    
    print(f"\n{'='*70}")
    print(f"PAIRING SUMMARY")
    print(f"{'='*70}")
    print(f"Total Ch1 NEs:    {total_ch1}")
    print(f"Total Ch2 NEs:    {total_ch2}")
    print(f"Total Pairs:      {total_pairs}")
    print(f"Pairing Rate:     {total_pairs / min(total_ch1, total_ch2) * 100:.1f}%")
    print(f"{'='*70}\n")
    
    return all_pairs


def visualize_pairing_comparison(
    experiment_name='BMY9999_99_99_9999',
    checkpoint_dir='local_yeast_output/dual_label/checkpoints',
    output_dir='pairing_test_results'
):
    """
    Compare old vs new pairing side-by-side.
    """
    
    print("Creating pairing comparison visualization...")
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir) / experiment_name / 'state_after_filtering.pkl'
    with open(checkpoint_path, 'rb') as f:
        img_proc = pickle.load(f)
    
    # Get OLD pairing
    old_pairs = img_proc.get_ne_pairs_by_FoV()
    
    # Get NEW pairing by testing
    init_checkpoint_path = Path(checkpoint_dir) / experiment_name / 'state_after_init_ne.pkl'
    with open(init_checkpoint_path, 'rb') as f:
        img_proc_init = pickle.load(f)
    
    ch1_masks = img_proc_init._get_ch1_masks()
    ch2_masks = img_proc_init._get_ch2_masks()
    
    if not ch1_masks or not ch2_masks:
        print("ERROR: No masks in checkpoint. Need to re-run with updated code.")
        return
    
    new_pairs = {}
    for fov_id in ch1_masks.keys():
        if fov_id in ch2_masks:
            pairs = pair_nes_by_masks(
                ch1_masks_dict=ch1_masks[fov_id],
                ch2_masks_dict=ch2_masks[fov_id],
                min_iou=0.7,
                max_centroid_distance_pixels=10
            )
            new_pairs[fov_id] = pairs
    
    # Create comparison figure
    fig, axes = plt.subplots(len(ch1_masks), 2, figsize=(14, 7*len(ch1_masks)))
    if len(ch1_masks) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, fov_id in enumerate(sorted(ch1_masks.keys())):
        # OLD pairing
        ax = axes[idx, 0]
        old_fov_pairs = old_pairs.get(fov_id, {})
        
        ch1_labels = list(ch1_masks[fov_id].keys())
        ch2_labels = list(ch2_masks[fov_id].keys())
        
        # Draw labels
        for i, label in enumerate(ch1_labels):
            ax.text(0.2, 0.9 - i*0.1, f"Ch1:{label}", fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        for i, label in enumerate(ch2_labels):
            ax.text(0.7, 0.9 - i*0.1, f"Ch2:{label}", fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral'))
        
        # Draw connections
        for ch1_label, ch2_label in old_fov_pairs.items():
            ch1_idx = ch1_labels.index(ch1_label) if ch1_label in ch1_labels else -1
            ch2_idx = ch2_labels.index(ch2_label) if ch2_label in ch2_labels else -1
            if ch1_idx >= 0 and ch2_idx >= 0:
                ax.plot([0.35, 0.65], [0.9 - ch1_idx*0.1, 0.9 - ch2_idx*0.1],
                       'g-', linewidth=2, alpha=0.7)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'OLD Pairing - FoV {fov_id}\n{len(old_fov_pairs)} pairs', 
                    fontweight='bold')
        ax.axis('off')
        
        # NEW pairing
        ax = axes[idx, 1]
        new_fov_pairs = new_pairs.get(fov_id, {})
        
        for i, label in enumerate(ch1_labels):
            ax.text(0.2, 0.9 - i*0.1, f"Ch1:{label}", fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        for i, label in enumerate(ch2_labels):
            ax.text(0.7, 0.9 - i*0.1, f"Ch2:{label}", fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral'))
        
        # Draw connections
        for ch1_label, ch2_label in new_fov_pairs.items():
            ch1_idx = ch1_labels.index(ch1_label) if ch1_label in ch1_labels else -1
            ch2_idx = ch2_labels.index(ch2_label) if ch2_label in ch2_labels else -1
            if ch1_idx >= 0 and ch2_idx >= 0:
                ax.plot([0.35, 0.65], [0.9 - ch1_idx*0.1, 0.9 - ch2_idx*0.1],
                       'b-', linewidth=2, alpha=0.7)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'NEW Pairing - FoV {fov_id}\n{len(new_fov_pairs)} pairs',
                    fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Pairing Comparison: Old (Crop Boxes) vs New (Tight Masks)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'pairing_comparison_{experiment_name}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison saved: {output_path / f'pairing_comparison_{experiment_name}.png'}")
    
    # Print detailed comparison
    print(f"\n{'='*70}")
    print("DETAILED COMPARISON")
    print(f"{'='*70}")
    for fov_id in sorted(ch1_masks.keys()):
        old_fov_pairs = old_pairs.get(fov_id, {})
        new_fov_pairs = new_pairs.get(fov_id, {})
        
        print(f"\nFoV {fov_id}:")
        print(f"  Old pairs: {old_fov_pairs}")
        print(f"  New pairs: {new_fov_pairs}")
        
        # Find differences
        old_set = set(old_fov_pairs.items())
        new_set = set(new_fov_pairs.items())
        
        lost = old_set - new_set
        gained = new_set - old_set
        
        if lost:
            print(f"  ❌ Lost pairs: {dict(lost)}")
        if gained:
            print(f"  ✅ Gained pairs: {dict(gained)}")
        if not lost and not gained:
            print(f"  ➡️  No change")


if __name__ == "__main__":
    # Test 1: Run pairing with different thresholds
    print("\n" + "="*70)
    print("TEST 1: Standard thresholds (IOU=0.7, dist=10px)")
    print("="*70)
    pairs_standard = test_pairing_from_checkpoint(min_iou=0.7, max_centroid_dist=10)
    
    print("\n" + "="*70)
    print("TEST 2: Looser thresholds (IOU=0.5, dist=15px)")
    print("="*70)
    pairs_loose = test_pairing_from_checkpoint(min_iou=0.5, max_centroid_dist=15)
    
    print("\n" + "="*70)
    print("TEST 3: Stricter thresholds (IOU=0.8, dist=5px)")
    print("="*70)
    pairs_strict = test_pairing_from_checkpoint(min_iou=0.8, max_centroid_dist=5)
    
    # Test 2: Visual comparison
    try:
        visualize_pairing_comparison()
    except Exception as e:
        print(f"\nCouldn't create comparison visualization: {e}")