"""
Comprehensive logging for dual-label distance calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

class DistanceCalculationLogger:
    """
    Logs detailed statistics about dual-label distance calculations,
    including impact of different filtering approaches.
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.distance_log = []
        self.segment_quality_log = []
        
    def log_distance_calculation(self, fov_id, ne_ch1, ne_ch2, 
                                  ch1_segment, ch2_segment,
                                  distance_result):
        """
        Log a single distance calculation between paired segments.
        
        Args:
            fov_id: Field of view identifier
            ne_ch1: Ch1 NE label
            ne_ch2: Ch2 NE label  
            ch1_segment: Segment info from ch1
            ch2_segment: Segment info from ch2
            distance_result: Dict with distance calculation results
        """
        
        entry = {
            'timestamp': self.timestamp,
            'fov_id': fov_id,
            'ne_ch1': ne_ch1,
            'ne_ch2': ne_ch2,
            'ch1_segment': ch1_segment,
            'ch2_segment': ch2_segment,
            
            # Distance metrics
            'mean_distance': distance_result.get('mean_distance'),
            'median_distance': distance_result.get('median_distance'),
            'std_distance': distance_result.get('std_distance'),
            'min_distance': distance_result.get('min_distance'),
            'max_distance': distance_result.get('max_distance'),
            'n_points_sampled': distance_result.get('n_points'),
            
            # Quality metrics
            'ch1_n_refined_points': distance_result.get('ch1_n_refined'),
            'ch2_n_refined_points': distance_result.get('ch2_n_refined'),
            'ch1_success_rate': distance_result.get('ch1_success_rate'),
            'ch2_success_rate': distance_result.get('ch2_success_rate'),
        }
        
        self.distance_log.append(entry)
        
    def log_segment_quality(self, fov_id, channel, ne_label, segment_label,
                           n_total, n_success, n_curvature_fail, 
                           n_likelihood_fail, n_optimization_fail):
        """
        Log refinement quality metrics for a segment.
        """
        
        entry = {
            'timestamp': self.timestamp,
            'fov_id': fov_id,
            'channel': channel,
            'ne_label': ne_label,
            'segment_label': segment_label,
            'n_total_profiles': n_total,
            'n_success': n_success,
            'n_curvature_fail': n_curvature_fail,
            'n_likelihood_fail': n_likelihood_fail,
            'n_optimization_fail': n_optimization_fail,
            'success_rate': n_success / n_total if n_total > 0 else 0,
            'curvature_fail_rate': n_curvature_fail / n_total if n_total > 0 else 0,
        }
        
        self.segment_quality_log.append(entry)
        
    def save_reports(self, fov_id):
        """
        Save comprehensive distance calculation reports.
        
        Outputs:
            1. Text summary
            2. CSV with per-pair details
            3. JSON with aggregate statistics
        """
        
        if not self.distance_log:
            print("No distance calculations to log.")
            return
            
        df_dist = pd.DataFrame(self.distance_log)
        df_qual = pd.DataFrame(self.segment_quality_log)
        
        # === AGGREGATE STATISTICS ===
        stats = {
            'timestamp': self.timestamp,
            'fov_id': fov_id,
            'n_pairs_calculated': len(df_dist),
            'overall_mean_distance': df_dist['mean_distance'].mean(),
            'overall_std_distance': df_dist['mean_distance'].std(),
            'distance_range': [df_dist['mean_distance'].min(), 
                              df_dist['mean_distance'].max()],
        }
        
        # Channel-specific quality
        if not df_qual.empty:
            for ch in df_qual['channel'].unique():
                ch_data = df_qual[df_qual['channel'] == ch]
                stats[f'ch{ch}_overall_success_rate'] = ch_data['success_rate'].mean()
                stats[f'ch{ch}_curvature_fail_rate'] = ch_data['curvature_fail_rate'].mean()
        
        # === TEXT REPORT ===
        report_path = self.output_dir / f"{self.timestamp}_{fov_id}_distance_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"DUAL-LABEL DISTANCE CALCULATION REPORT - {self.timestamp}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"FoV ID: {fov_id}\n")
            f.write(f"Number of paired segments: {stats['n_pairs_calculated']}\n")
            f.write(f"\n")
            
            f.write(f"DISTANCE STATISTICS\n")
            f.write(f"-" * 40 + "\n")
            f.write(f"Overall mean distance: {stats['overall_mean_distance']:.2f} ± {stats['overall_std_distance']:.2f} nm\n")
            f.write(f"Distance range: {stats['distance_range'][0]:.2f} - {stats['distance_range'][1]:.2f} nm\n")
            f.write(f"\n")
            
            if not df_qual.empty:
                f.write(f"REFINEMENT QUALITY IMPACT\n")
                f.write(f"-" * 40 + "\n")
                for ch in sorted(df_qual['channel'].unique()):
                    f.write(f"\nChannel {ch}:\n")
                    f.write(f"  Overall success rate: {stats[f'ch{ch}_overall_success_rate']:.1%}\n")
                    f.write(f"  Curvature failure rate: {stats[f'ch{ch}_curvature_fail_rate']:.1%}\n")
                    
                    ch_data = df_qual[df_qual['channel'] == ch]
                    f.write(f"  Total profiles: {ch_data['n_total_profiles'].sum()}\n")
                    f.write(f"  Successful fits: {ch_data['n_success'].sum()}\n")
                    f.write(f"  Failed curvature: {ch_data['n_curvature_fail'].sum()}\n")
                    f.write(f"  Failed likelihood: {ch_data['n_likelihood_fail'].sum()}\n")
                    f.write(f"  Failed optimization: {ch_data['n_optimization_fail'].sum()}\n")
            
            else:
                f.write(f"\n")
                f.write(f"PER-PAIR DETAILS\n")
                f.write(f"-" * 40 + "\n")
                for idx, row in df_dist.iterrows():
                    f.write(f"\nPair {idx+1}: NE {row['ne_ch1']} (Ch1) ↔ NE {row['ne_ch2']} (Ch2)\n")
                    f.write(f"  Mean distance: {row['mean_distance']:.2f} ± {row['std_distance']:.2f} nm\n")
                    f.write(f"  Range: {row['min_distance']:.2f} - {row['max_distance']:.2f} nm\n")
                    f.write(f"  Points sampled: {row['n_points_sampled']}\n")
                    f.write(f"  Ch1 refined: {row['ch1_n_refined_points']} points ({row['ch1_success_rate']:.1%})\n")
                    f.write(f"  Ch2 refined: {row['ch2_n_refined_points']} points ({row['ch2_success_rate']:.1%})\n")
        
        # === SAVE CSVs ===
        dist_csv = self.output_dir / f"{self.timestamp}_{fov_id}_distance_details.csv"
        df_dist.to_csv(dist_csv, index=False)
        
        qual_csv = self.output_dir / f"{self.timestamp}_{fov_id}_quality_details.csv"
        df_qual.to_csv(qual_csv, index=False)
        
        # === SAVE JSON ===
        json_path = self.output_dir / f"{self.timestamp}_{fov_id}_distance_stats.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n=== Distance calculation reports saved ===")
        print(f"  Report: {report_path.name}")
        print(f"  Distance CSV: {dist_csv.name}")
        print(f"  Quality CSV: {qual_csv.name}")
        print(f"  JSON: {json_path.name}")
        
        return stats


def analyze_filter_impact(with_filters_dir, without_filters_dir):
    """
    Compare distance calculations with vs without curvature/likelihood filters.
    
    Shows how stringent quality control affects final biological measurements.
    """
    
    # Load results
    with_df = pd.concat([
        pd.read_csv(f) for f in Path(with_filters_dir).glob('*_distance_details.csv')
    ])
    without_df = pd.concat([
        pd.read_csv(f) for f in Path(without_filters_dir).glob('*_distance_details.csv')
    ])
    
    with_df['filtering'] = 'With Filters'
    without_df['filtering'] = 'Without Filters'
    
    combined = pd.concat([with_df, without_df], ignore_index=True)
    
    # === COMPARISON REPORT ===
    report_path = Path('filter_impact_on_distances.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUALITY FILTER IMPACT ON DISTANCE MEASUREMENTS\n")
        f.write("="*80 + "\n\n")
        
        for filtering in ['With Filters', 'Without Filters']:
            data = combined[combined['filtering'] == filtering]
            f.write(f"\n{filtering}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of measurements: {len(data)}\n")
            f.write(f"Mean distance: {data['mean_distance'].mean():.2f} ± {data['mean_distance'].std():.2f} nm\n")
            f.write(f"Median distance: {data['mean_distance'].median():.2f} nm\n")
            f.write(f"Range: {data['mean_distance'].min():.2f} - {data['mean_distance'].max():.2f} nm\n")
            
            # Measurement precision
            f.write(f"\nMeasurement precision:\n")
            f.write(f"  Mean std within measurements: {data['std_distance'].mean():.2f} nm\n")
            f.write(f"  Std across measurements: {data['mean_distance'].std():.2f} nm\n")
        
        # Statistical comparison
        with_data = combined[combined['filtering'] == 'With Filters']['mean_distance']
        without_data = combined[combined['filtering'] == 'Without Filters']['mean_distance']
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(with_data, without_data)
        
        f.write(f"\n\nSTATISTICAL COMPARISON\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"t-statistic: {t_stat:.3f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}\n")
    
    print(f"Filter impact report saved: {report_path}")
    
    # === VISUALIZATION ===
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distance distributions
    sns.violinplot(data=combined, x='filtering', y='mean_distance', ax=axes[0, 0])
    axes[0, 0].set_title('Distance Distribution by Filtering Approach')
    axes[0, 0].set_ylabel('Mean Distance (nm)')
    
    # Precision comparison
    sns.boxplot(data=combined, x='filtering', y='std_distance', ax=axes[0, 1])
    axes[0, 1].set_title('Measurement Precision')
    axes[0, 1].set_ylabel('Std Dev within Measurement (nm)')
    
    # Success rate impact
    sns.scatterplot(data=combined, x='ch1_success_rate', y='mean_distance', 
                    hue='filtering', ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title('Ch1 Success Rate vs Distance')
    axes[1, 0].set_xlabel('Ch1 Success Rate')
    axes[1, 0].set_ylabel('Mean Distance (nm)')
    
    # N points impact
    sns.scatterplot(data=combined, x='n_points_sampled', y='std_distance',
                    hue='filtering', ax=axes[1, 1], alpha=0.6)
    axes[1, 1].set_title('Sampling Density vs Precision')
    axes[1, 1].set_xlabel('Number of Points Sampled')
    axes[1, 1].set_ylabel('Distance Std Dev (nm)')
    
    plt.tight_layout()
    plt.savefig('filter_impact_analysis.png', dpi=300)
    print(f"Visualization saved: filter_impact_analysis.png")
    
    return combined