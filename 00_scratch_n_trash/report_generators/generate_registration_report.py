"""
Registration quality report with filtered vs. unfiltered comparison.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def generate_registration_report(
    experiment_name='BMY9999_99_99_9999',
    output_base='local_yeast_output/dual_label',
    report_dir='reports_for_boss'
):
    """
    Generate comprehensive registration quality report.
    """
    
    print(f"\n{'='*70}")
    print(f"REGISTRATION QUALITY REPORT: {experiment_name}")
    print(f"{'='*70}\n")
    
    report_path = Path(report_dir) / experiment_name
    report_path.mkdir(parents=True, exist_ok=True)
    
    # Load registration stability report
    with open(f'{output_base}/registration/registration_stability_report_{experiment_name}.json', 'r') as f:
        stability_report = json.load(f)
    
    # Load summary stats
    with open(f'{output_base}/registration/registration_summary_stats_{experiment_name}.json', 'r') as f:
        summary_stats = json.load(f)
    
    # Parse the stability report - handle both dict and list formats
    all_data = []
    
    # Check if stability_report is a list (flat format) or dict (nested format)
    if isinstance(stability_report, list):
        # Flat format: [{fov: ..., ne_label: ..., slice: ..., ...}, ...]
        for entry in stability_report:
            if 'kept' in entry:
                all_data.append({
                    'fov': entry.get('fov', 'unknown'),
                    'ne_label': entry.get('ne_label', 'unknown'),
                    'slice': entry.get('slice', 0),
                    'angle_delta': entry.get('angle_delta', 0),
                    'scale_delta': entry.get('scale_delta', 0),
                    'rdif_y': entry.get('rdif_y', 0),
                    'rdif_x': entry.get('rdif_x', 0),
                    'kept': entry['kept']
                })
    else:
        # Nested dict format: {fov_id: {ne_label: {slice: {...}}}}
        for fov_id, fov_data in stability_report.items():
            if not isinstance(fov_data, dict):
                continue
            for ne_label, ne_data in fov_data.items():
                if not isinstance(ne_data, dict):
                    continue
                for slice_idx, slice_data in ne_data.items():
                    if isinstance(slice_data, dict) and 'kept' in slice_data:
                        all_data.append({
                            'fov': fov_id,
                            'ne_label': ne_label,
                            'slice': int(slice_idx) if str(slice_idx).isdigit() else 0,
                            'angle_delta': slice_data.get('angle_delta', 0),
                            'scale_delta': slice_data.get('scale_delta', 0),
                            'rdif_y': slice_data.get('rdif_y', 0),
                            'rdif_x': slice_data.get('rdif_x', 0),
                            'kept': slice_data['kept']
                        })
    
    if not all_data:
        print("ERROR: No valid registration data found in stability report!")
        return None
    
    df = pd.DataFrame(all_data)
    df_kept = df[df['kept'] == True]
    df_filtered = df[df['kept'] == False]
    
    print(f"Total slices: {len(df)}")
    print(f"Kept: {len(df_kept)}")
    print(f"Filtered: {len(df_filtered)}")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # --- ROW 1: Distribution Comparisons ---
    metrics = [
        ('angle_delta', 'Angle Δ (degrees)', (-1, 1)),
        ('scale_delta', 'Scale Δ (units)', (-0.02, 0.02)),
        ('rdif_y', 'Y-Shift Δ (pixels)', (-2, 2))
    ]
    
    for idx, (metric, label, xlim) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        
        if len(df_kept) > 0:
            ax.hist(df_kept[metric], bins=30, alpha=0.7, color='green', 
                   label=f'Kept (N={len(df_kept)})', edgecolor='black')
        if len(df_filtered) > 0:
            ax.hist(df_filtered[metric], bins=30, alpha=0.7, color='red',
                   label=f'Filtered (N={len(df_filtered)})', edgecolor='black')
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # --- ROW 2: Statistics Comparison Table ---
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    def calc_stats(data, metric):
        if len(data) == 0:
            return {'mean': 0, 'std': 0, 'median': 0, 'n': 0}
        return {
            'mean': data[metric].mean(),
            'std': data[metric].std(),
            'median': data[metric].median(),
            'n': len(data)
        }
    
    # Build comparison table
    table_data = [
        ['Metric', 'ALL Data', '', 'KEPT Data', '', 'FILTERED Data', ''],
        ['', 'Mean ± SD', 'Median', 'Mean ± SD', 'Median', 'Mean ± SD', 'Median']
    ]
    
    metric_names = {
        'angle_delta': 'Angle Δ (deg)',
        'scale_delta': 'Scale Δ',
        'rdif_y': 'Y-Shift Δ (px)',
        'rdif_x': 'X-Shift Δ (px)'
    }
    
    for metric, name in metric_names.items():
        all_stats = calc_stats(df, metric)
        kept_stats = calc_stats(df_kept, metric)
        filt_stats = calc_stats(df_filtered, metric)
        
        table_data.append([
            name,
            f"{all_stats['mean']:.4f} ± {all_stats['std']:.4f}",
            f"{all_stats['median']:.4f}",
            f"{kept_stats['mean']:.4f} ± {kept_stats['std']:.4f}",
            f"{kept_stats['median']:.4f}",
            f"{filt_stats['mean']:.4f} ± {filt_stats['std']:.4f}",
            f"{filt_stats['median']:.4f}"
        ])
    
    # Add sample counts
    table_data.append([
        'Sample Count',
        str(len(df)),
        '',
        str(len(df_kept)),
        '',
        str(len(df_filtered)),
        ''
    ])
    
    # Add sigma_reg
    slice_stats = summary_stats.get('slice_level', {})
    table_data.append([
        'σ_reg (combined)',
        f"{slice_stats.get('sigma_reg', 0):.4f} px",
        '',
        '—',
        '',
        '—',
        ''
    ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header rows
    for j in range(7):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white', size=11)
        table[(1, j)].set_facecolor('#81C784')
        table[(1, j)].set_text_props(weight='bold', size=9)
    
    # Highlight kept vs filtered columns
    for i in range(2, len(table_data)):
        for j in [3, 4]:  # Kept columns
            table[(i, j)].set_facecolor('#e8f5e9')
        for j in [5, 6]:  # Filtered columns
            table[(i, j)].set_facecolor('#ffebee')
    
    ax_table.set_title('Registration Stability Statistics: All vs. Kept vs. Filtered',
                      fontsize=13, fontweight='bold', pad=20)
    
    # --- ROW 3: Filtering Impact Visualization ---
    ax_filter = fig.add_subplot(gs[2, :2])
    
    # Calculate percentage filtered per FoV
    fov_summary = []
    for fov_id in df['fov'].unique():
        fov_data = df[df['fov'] == fov_id]
        n_total = len(fov_data)
        n_kept = len(fov_data[fov_data['kept'] == True])
        n_filtered = n_total - n_kept
        pct_filtered = (n_filtered / n_total * 100) if n_total > 0 else 0
        fov_summary.append({
            'fov': fov_id,
            'total': n_total,
            'kept': n_kept,
            'filtered': n_filtered,
            'pct_filtered': pct_filtered
        })
    
    fov_df = pd.DataFrame(fov_summary).sort_values('pct_filtered', ascending=False)
    
    x = np.arange(len(fov_df))
    ax_filter.bar(x, fov_df['kept'], label='Kept', color='green', alpha=0.7)
    ax_filter.bar(x, fov_df['filtered'], bottom=fov_df['kept'], 
                 label='Filtered', color='red', alpha=0.7)
    
    ax_filter.set_xlabel('FoV ID', fontsize=11, fontweight='bold')
    ax_filter.set_ylabel('Number of Slices', fontsize=11, fontweight='bold')
    ax_filter.set_title('Registration Filtering Summary by FoV', fontsize=12, fontweight='bold')
    ax_filter.set_xticks(x)
    ax_filter.set_xticklabels(fov_df['fov'], rotation=45, ha='right')
    ax_filter.legend(fontsize=10)
    ax_filter.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (idx, row) in enumerate(fov_df.iterrows()):
        if row['pct_filtered'] > 0:
            ax_filter.text(i, row['total'] + 0.5, f"{row['pct_filtered']:.1f}%",
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    
    # --- Summary Text Box ---
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis('off')
    
    total_slices = len(df)
    kept_slices = len(df_kept)
    filtered_slices = len(df_filtered)
    pct_filtered = (filtered_slices / total_slices * 100) if total_slices > 0 else 0
    
    summary_text = f"""
FILTERING SUMMARY
{'─'*30}

Total Slices:      {total_slices}
Kept:              {kept_slices} ({100-pct_filtered:.1f}%)
Filtered:          {filtered_slices} ({pct_filtered:.1f}%)

Registration Quality (Kept):
  σ_reg:           {slice_stats.get('sigma_reg', 0):.4f} px
  Angle SD:        {calc_stats(df_kept, 'angle_delta')['std']:.4f}°
  Scale SD:        {calc_stats(df_kept, 'scale_delta')['std']:.6f}
  Y-Shift SD:      {calc_stats(df_kept, 'rdif_y')['std']:.4f} px
  X-Shift SD:      {calc_stats(df_kept, 'rdif_x')['std']:.4f} px

Filtering Criterion:
  ±2σ cutoff applied
"""
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                           edgecolor='black', linewidth=2))
    
    plt.suptitle(f'Registration Quality Analysis: {experiment_name}\n'
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(report_path / 'registration_quality_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Registration report saved: {report_path / 'registration_quality_report.png'}")
    
    # Generate MARKDOWN summary
    md_content = f"""# Registration Quality Report

**Experiment:** {experiment_name}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary Statistics

| Metric | Total | Kept | Filtered |
|--------|-------|------|----------|
| **Slices** | {total_slices} | {kept_slices} ({100-pct_filtered:.1f}%) | {filtered_slices} ({pct_filtered:.1f}%) |

## Registration Quality (Kept Data Only)

| Parameter | Value |
|-----------|-------|
| **σ_reg (combined precision)** | {slice_stats.get('sigma_reg', 0):.4f} pixels |
| **Angle Variability (SD)** | {calc_stats(df_kept, 'angle_delta')['std']:.4f}° |
| **Scale Variability (SD)** | {calc_stats(df_kept, 'scale_delta')['std']:.6f} |
| **Y-Shift Variability (SD)** | {calc_stats(df_kept, 'rdif_y')['std']:.4f} pixels |
| **X-Shift Variability (SD)** | {calc_stats(df_kept, 'rdif_x')['std']:.4f} pixels |

## Detailed Statistics by Metric

### All Data vs. Kept vs. Filtered

"""
    
    for metric, name in metric_names.items():
        all_stats = calc_stats(df, metric)
        kept_stats = calc_stats(df_kept, metric)
        filt_stats = calc_stats(df_filtered, metric)
        
        md_content += f"""
#### {name}

| Dataset | Mean | Std Dev | Median | N |
|---------|------|---------|--------|---|
| **All** | {all_stats['mean']:.4f} | {all_stats['std']:.4f} | {all_stats['median']:.4f} | {all_stats['n']} |
| **Kept** | {kept_stats['mean']:.4f} | {kept_stats['std']:.4f} | {kept_stats['median']:.4f} | {kept_stats['n']} |
| **Filtered** | {filt_stats['mean']:.4f} | {filt_stats['std']:.4f} | {filt_stats['median']:.4f} | {filt_stats['n']} |

"""
    
    md_content += f"""
## Per-FoV Breakdown

| FoV | Total Slices | Kept | Filtered | % Filtered |
|-----|--------------|------|----------|------------|
"""
    
    for _, row in fov_df.iterrows():
        md_content += f"| {row['fov']} | {row['total']} | {row['kept']} | {row['filtered']} | {row['pct_filtered']:.1f}% |\n"
    
    md_content += f"""
---

## Methodology

- **Filtering Criterion:** ±2σ cutoff applied to registration parameters
- **σ_reg Calculation:** Combined precision metric from all registration parameters
- **Quality Assessment:** Lower σ_reg indicates better registration stability

"""
    
    # Save markdown
    with open(report_path / 'registration_report.md', 'w') as f:
        f.write(md_content)
    
    print(f"✓ Markdown report saved: {report_path / 'registration_report.md'}\n")
    
    return {
        'total_slices': total_slices,
        'kept_slices': kept_slices,
        'filtered_slices': filtered_slices,
        'sigma_reg': slice_stats.get('sigma_reg', 0)
    }


if __name__ == "__main__":
    generate_registration_report()