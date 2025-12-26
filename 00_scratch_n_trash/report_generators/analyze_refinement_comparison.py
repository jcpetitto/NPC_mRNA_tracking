"""
Analyze and compare different refinement quality control approaches.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_refinement_results(base_dir, pattern='*_summary_stats.json'):
    """Load all refinement summary statistics."""
    results = []
    for json_file in Path(base_dir).glob(pattern):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    return pd.DataFrame(results)

def compare_qc_methods(likelihood_dir, poisson_dir):
    """
    Compare Likelihood Ratio vs Poisson Outlier approaches.
    
    Args:
        likelihood_dir: Directory with likelihood ratio results
        poisson_dir: Directory with Poisson outlier results
    """
    
    # Load results
    lr_results = load_refinement_results(likelihood_dir)
    po_results = load_refinement_results(poisson_dir)
    
    lr_results['method'] = 'Likelihood Ratio'
    po_results['method'] = 'Poisson Outlier'
    
    combined = pd.concat([lr_results, po_results], ignore_index=True)
    
    # === COMPARISON REPORT ===
    report_path = Path('refinement_method_comparison.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REFINEMENT QC METHOD COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        for method in ['Likelihood Ratio', 'Poisson Outlier']:
            data = combined[combined['method'] == method]
            f.write(f"\n{method}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean success rate: {data['success_rate'].mean():.1%}\n")
            f.write(f"Std dev: {data['success_rate'].std():.1%}\n")
            f.write(f"Range: {data['success_rate'].min():.1%} - {data['success_rate'].max():.1%}\n")
            
            # By channel
            for ch in data['channel'].unique():
                ch_data = data[data['channel'] == ch]
                f.write(f"\n  Channel {ch}:\n")
                f.write(f"    Success rate: {ch_data['success_rate'].mean():.1%}\n")
                f.write(f"    N profiles: {ch_data['total_profiles'].sum()}\n")
                f.write(f"    N success: {ch_data['n_success'].sum()}\n")
    
    print(f"Comparison report saved: {report_path}")
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Success rate by method
    sns.boxplot(data=combined, x='method', y='success_rate', ax=axes[0, 0])
    axes[0, 0].set_title('Success Rate by QC Method')
    axes[0, 0].set_ylabel('Success Rate')
    
    # Success rate by channel and method
    sns.barplot(data=combined, x='channel', y='success_rate', hue='method', ax=axes[0, 1])
    axes[0, 1].set_title('Success Rate by Channel')
    axes[0, 1].set_ylabel('Success Rate')
    
    # Total profiles by method
    method_totals = combined.groupby('method').agg({
        'total_profiles': 'sum',
        'n_success': 'sum'
    })
    method_totals['failure'] = method_totals['total_profiles'] - method_totals['n_success']
    
    method_totals[['n_success', 'failure']].plot(kind='bar', stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title('Total Outcomes by Method')
    axes[1, 0].set_ylabel('Number of Profiles')
    axes[1, 0].legend(['Success', 'Failure'])
    
    # Gain vs success rate
    for method in ['Likelihood Ratio', 'Poisson Outlier']:
        data = combined[combined['method'] == method]
        axes[1, 1].scatter(data['camera_gain'], data['success_rate'], 
                          label=method, alpha=0.6, s=100)
    axes[1, 1].set_xlabel('Camera Gain (e-/ADU)')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Gain vs Success Rate')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('refinement_method_comparison.png', dpi=300)
    print(f"Comparison plot saved: refinement_method_comparison.png")
    
    return combined

if __name__ == "__main__":
    # Example usage
    comparison = compare_qc_methods(
        likelihood_dir='local_yeast_output/dual_label/refined_fit',
        poisson_dir='local_yeast_output/dual_label/refined_fit_poisson'  # You'd need to save old results here
    )