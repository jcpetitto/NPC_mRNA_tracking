"""
Automated summary report generator for dual-label NE analysis.
Produces publication-ready figures and statistics.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime

def generate_full_report(
    experiment_name='BMY9999_99_99_9999',
    output_base='local_yeast_output/dual_label',
    report_dir='reports'
):
    """
    Generates a complete analysis report with figures and statistics.
    """
    
    print(f"\n{'='*70}")
    print(f"GENERATING REPORT FOR: {experiment_name}")
    print(f"{'='*70}\n")
    
    # Create report directory
    report_path = Path(report_dir) / experiment_name
    report_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize report
    report = {
        'experiment': experiment_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sections': {}
    }
    
    # --- SECTION 1: Registration Statistics ---
    print("📊 Section 1: Registration Stability Analysis...")
    try:
        with open(f'{output_base}/registration/registration_summary_stats_{experiment_name}.json', 'r') as f:
            reg_stats = json.load(f)
        
        report['sections']['registration'] = {
            'fov_level': reg_stats.get('fov_level', {}),
            'label_level': reg_stats.get('ne_label_level', {}),
            'slice_level': reg_stats.get('slice_level', {})
        }
        
        # Create registration summary figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        levels = ['fov_level', 'ne_label_level', 'slice_level']
        titles = ['FoV Level', 'NE Label Level', 'Slice Level']
        
        for ax, level, title in zip(axes, levels, titles):
            if level in reg_stats and reg_stats[level]['n'] > 0:
                stats = reg_stats[level]
                
                metrics = ['sigma_reg', 'angle_std', 'scale_std']
                values = [stats.get(m, 0) for m in metrics]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                bars = ax.bar(range(len(metrics)), values, color=colors, alpha=0.7)
                ax.set_xticks(range(len(metrics)))
                ax.set_xticklabels(['σ_reg\n(pixels)', 'Angle SD\n(degrees)', 'Scale SD\n(units)'], 
                                  fontsize=9)
                ax.set_title(f'{title}\n(N={stats["n"]})', fontweight='bold')
                ax.set_ylabel('Variability', fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Registration Stability Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(report_path / 'registration_stability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Registration statistics compiled")
        
    except Exception as e:
        print(f"  ✗ Registration data not found: {e}")
        report['sections']['registration'] = None
    
    # --- SECTION 2: Distance Measurements ---
    print("\n📏 Section 2: Dual-Label Distance Analysis...")
    try:
        with open(f'{output_base}/distances/dual_dist_result_{experiment_name}.json', 'r') as f:
            distances_data = json.load(f)
        
        # Compile all distances
        all_distances = []
        pair_info = []
        
        for fov_id, pairs in distances_data.items():
            for pair_key, stats in pairs.items():
                mean_dist = stats['mean_distance']
                std_dist = stats['std_distance']
                
                # Filter biologically plausible distances
                if abs(mean_dist) < 500:  # Within 500 nm
                    all_distances.extend(stats['distances'])
                    pair_info.append({
                        'fov': fov_id,
                        'pair': pair_key,
                        'mean': mean_dist,
                        'std': std_dist,
                        'median': stats.get('median_distance', np.nan),
                        'n_points': len(stats['distances']),
                        'data_data_mean': stats.get('data_data_mean', np.nan)
                    })
        
        # Create summary statistics
        all_distances = np.array(all_distances)
        report['sections']['distances'] = {
            'n_valid_pairs': len(pair_info),
            'n_total_measurements': len(all_distances),
            'overall_mean_nm': float(np.mean(all_distances)),
            'overall_std_nm': float(np.std(all_distances)),
            'overall_median_nm': float(np.median(all_distances)),
            'min_nm': float(np.min(all_distances)),
            'max_nm': float(np.max(all_distances)),
            'pair_details': pair_info
        }
        
        # Create distance analysis figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Histogram of all distances
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(all_distances, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(all_distances), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {np.mean(all_distances):.1f} nm')
        ax1.axvline(np.median(all_distances), color='orange', linestyle='--', linewidth=2,
                   label=f'Median = {np.median(all_distances):.1f} nm')
        ax1.set_xlabel('Distance (nm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribution of All Distance Measurements (N={len(all_distances)})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Per-pair summary
        ax2 = fig.add_subplot(gs[1, 0])
        df = pd.DataFrame(pair_info)
        df_sorted = df.sort_values('mean')
        
        colors = ['green' if abs(m) < 200 else 'orange' for m in df_sorted['mean']]
        ax2.barh(range(len(df_sorted)), df_sorted['mean'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels([f"{row['fov']}/{row['pair']}" for _, row in df_sorted.iterrows()], 
                           fontsize=9)
        ax2.set_xlabel('Mean Distance (nm)', fontsize=11, fontweight='bold')
        ax2.set_title('Mean Distance by NE Pair', fontsize=12, fontweight='bold')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Mean vs Std scatter
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(df['mean'], df['std'], s=100, alpha=0.6, c=colors, edgecolors='black')
        ax3.set_xlabel('Mean Distance (nm)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Std Dev (nm)', fontsize=11, fontweight='bold')
        ax3.set_title('Distance Variability', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = f"""
        SUMMARY STATISTICS
        {'─'*60}
        Valid NE Pairs:           {len(pair_info)}
        Total Measurements:       {len(all_distances)}
        
        Overall Mean Distance:    {np.mean(all_distances):.2f} ± {np.std(all_distances):.2f} nm
        Median Distance:          {np.median(all_distances):.2f} nm
        Range:                    {np.min(all_distances):.2f} to {np.max(all_distances):.2f} nm
        
        Sign Convention:
          • Positive = Ch1 outside / Ch2 inside
          • Negative = Ch2 outside / Ch1 inside
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Dual-Label Distance Analysis: {experiment_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(report_path / 'distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Distance analysis complete: {len(pair_info)} valid pairs")
        
    except Exception as e:
        print(f"  ✗ Distance data not found: {e}")
        report['sections']['distances'] = None
    
    # --- SECTION 3: Sample Visualizations ---
    print("\n🎨 Section 3: Creating sample visualizations...")
    try:
        # Load splines for visualization
        with open(f'{output_base}/merged_splines/bridged_splines_ch1_{experiment_name}.pkl', 'rb') as f:
            bridged_ch1 = pickle.load(f)
        with open(f'{output_base}/merged_splines/bridged_splines_ch2_{experiment_name}.pkl', 'rb') as f:
            bridged_ch2 = pickle.load(f)
        
        # Pick first valid pair for visualization
        if pair_info:
            sample = pair_info[0]
            fov_id = sample['fov']
            ch1_label, ch2_label = sample['pair'].split('_vs_')
            
            # Create overlay plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Plot Ch1 spline
            if fov_id in bridged_ch1 and ch1_label in bridged_ch1[fov_id]:
                for seg in bridged_ch1[fov_id][ch1_label]['data_segments']:
                    u_vals = np.linspace(0, 1, 200)
                    pts = np.array([seg(u) for u in u_vals])
                    ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=3, alpha=0.8)
            
            # Plot Ch2 spline
            if fov_id in bridged_ch2 and ch2_label in bridged_ch2[fov_id]:
                for seg in bridged_ch2[fov_id][ch2_label]['data_segments']:
                    u_vals = np.linspace(0, 1, 200)
                    pts = np.array([seg(u) for u in u_vals])
                    ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=3, alpha=0.8)
            
            ax.set_aspect('equal')
            ax.set_title(f'Sample NE Pair: {fov_id}/{ch1_label} (Ch1) vs {ch2_label} (Ch2)\n'
                        f'Mean Distance: {sample["mean"]:.1f} nm', 
                        fontsize=14, fontweight='bold')
            ax.legend(['Channel 1', 'Channel 2'], fontsize=12)
            ax.grid(alpha=0.3)
            
            plt.savefig(report_path / 'sample_overlay.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Sample visualization created")
    
    except Exception as e:
        print(f"  ✗ Could not create sample visualization: {e}")
    
    # --- Save JSON Report ---
    with open(report_path / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # --- Create Text Summary ---
    summary_text = f"""
{'='*70}
ANALYSIS SUMMARY REPORT
Experiment: {experiment_name}
Generated: {report['timestamp']}
{'='*70}

REGISTRATION STABILITY
{'─'*70}
"""
    
    if report['sections'].get('registration'):
        reg = report['sections']['registration']['slice_level']
        summary_text += f"""
Slice-Level Precision (N={reg.get('n', 0)} measurements):
  • Combined σ_reg:     {reg.get('sigma_reg', 0):.4f} pixels
  • Angle Variability:  {reg.get('angle_std', 0):.4f}°
  • Scale Variability:  {reg.get('scale_std', 0):.6f}
"""
    else:
        summary_text += "  No registration data available\n"
    
    summary_text += f"""
DISTANCE MEASUREMENTS
{'─'*70}
"""
    
    if report['sections'].get('distances'):
        dist = report['sections']['distances']
        summary_text += f"""
Valid NE Pairs: {dist['n_valid_pairs']}
Total Measurements: {dist['n_total_measurements']}

Overall Statistics:
  • Mean Distance:   {dist['overall_mean_nm']:.2f} ± {dist['overall_std_nm']:.2f} nm
  • Median Distance: {dist['overall_median_nm']:.2f} nm
  • Range:           {dist['min_nm']:.2f} to {dist['max_nm']:.2f} nm

Per-Pair Summary:
"""
        for pair in dist['pair_details']:
            summary_text += f"  {pair['fov']}/{pair['pair']}: {pair['mean']:.1f} ± {pair['std']:.1f} nm (N={pair['n_points']})\n"
    else:
        summary_text += "  No distance data available\n"
    
    summary_text += f"""
{'='*70}
Report files saved to: {report_path}/
  • analysis_report.json
  • registration_stability.png
  • distance_analysis.png
  • sample_overlay.png
  • summary.txt
{'='*70}
"""
    
    with open(report_path / 'summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\n{summary_text}")
    print(f"✅ REPORT GENERATION COMPLETE!\n")
    
    return report


if __name__ == "__main__":
    generate_full_report()