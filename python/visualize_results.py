#!/usr/bin/env python3
"""
Visualization script for tiered storage evaluation results
Creates comprehensive plots showcasing threshold adjustments and accuracy-latency tradeoffs
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_threshold_data():
    """Load data for all thresholds"""
    thresholds = [0.6, 0.75, 0.9]
    data = {}
    
    for th in thresholds:
        baseline_file = f"data/query_log_baseline_cloud_th{th}.csv"
        tiered_file = f"data/query_log_tiered_th{th}.csv"
        
        if not os.path.exists(baseline_file) or not os.path.exists(tiered_file):
            print(f"Warning: Missing data files for threshold {th}")
            continue
        
        # Load baseline latencies
        baseline_latencies = []
        with open(baseline_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                baseline_latencies.append(float(row['latency_ms']))
        
        # Load tiered latencies
        tiered_latencies = []
        with open(tiered_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tiered_latencies.append(float(row['latency_ms']))
        
        # Calculate statistics
        data[th] = {
            'baseline': {
                'latencies': baseline_latencies,
                'mean': np.mean(baseline_latencies),
                'median': np.median(baseline_latencies),
                'p95': np.percentile(baseline_latencies, 95),
                'p99': np.percentile(baseline_latencies, 99),
            },
            'tiered': {
                'latencies': tiered_latencies,
                'mean': np.mean(tiered_latencies),
                'median': np.median(tiered_latencies),
                'p95': np.percentile(tiered_latencies, 95),
                'p99': np.percentile(tiered_latencies, 99),
            }
        }
    
    return data


def load_summary_metrics():
    """Load additional metrics from evaluation summary"""
    summary_path = "data/evaluation_summary.txt"
    if not os.path.exists(summary_path):
        return None
    
    metrics = {}
    current_threshold = None
    
    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith('Threshold:'):
                current_threshold = float(line.split(':')[1].strip())
                metrics[current_threshold] = {}
            elif current_threshold:
                if 'Average results per query:' in line:
                    metrics[current_threshold]['avg_results'] = float(line.split(':')[1].strip())
                elif 'Average similarity score:' in line:
                    metrics[current_threshold]['avg_similarity'] = float(line.split(':')[1].strip())
                elif 'Performance retention:' in line:
                    metrics[current_threshold]['retention'] = float(line.split(':')[1].strip().rstrip('%'))
                elif 'Tier distribution:' in line:
                    # Parse T1=100.0%, T2=0.0%, T3=0.0%
                    parts = line.split(':')[1].strip()
                    for part in parts.split(','):
                        if 'T1=' in part:
                            metrics[current_threshold]['t1_pct'] = float(part.split('=')[1].rstrip('%'))
                        elif 'T2=' in part:
                            metrics[current_threshold]['t2_pct'] = float(part.split('=')[1].rstrip('%'))
                        elif 'T3=' in part:
                            metrics[current_threshold]['t3_pct'] = float(part.split('=')[1].rstrip('%'))
    
    return metrics


def plot_latency_comparison(data, output_dir):
    """Plot 1: Latency comparison across thresholds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Latency Comparison: Baseline vs Tiered System Across Thresholds', 
                 fontsize=16, fontweight='bold')
    
    thresholds = sorted(data.keys())
    baseline_means = [data[th]['baseline']['mean'] for th in thresholds]
    tiered_means = [data[th]['tiered']['mean'] for th in thresholds]
    baseline_p95s = [data[th]['baseline']['p95'] for th in thresholds]
    tiered_p95s = [data[th]['tiered']['p95'] for th in thresholds]
    
    # Plot 1: Mean latency comparison
    ax1 = axes[0, 0]
    x = np.arange(len(thresholds))
    width = 0.35
    ax1.bar(x - width/2, baseline_means, width, label='Baseline (Cloud)', 
            color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, tiered_means, width, label='Tiered System', 
            color='#A23B72', alpha=0.8)
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title('Mean Latency by Threshold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{th:.2f}' for th in thresholds])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bl, tr) in enumerate(zip(baseline_means, tiered_means)):
        ax1.text(i - width/2, bl + 2, f'{bl:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, tr + 2, f'{tr:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: P95 latency comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, baseline_p95s, width, label='Baseline (Cloud)', 
            color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, tiered_p95s, width, label='Tiered System', 
            color='#A23B72', alpha=0.8)
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('P95 Latency (ms)')
    ax2.set_title('P95 Latency by Threshold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{th:.2f}' for th in thresholds])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bl, tr) in enumerate(zip(baseline_p95s, tiered_p95s)):
        ax2.text(i - width/2, bl + 2, f'{bl:.1f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, tr + 2, f'{tr:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Latency trend lines
    ax3 = axes[1, 0]
    ax3.plot(thresholds, baseline_means, 'o-', label='Baseline Mean', 
             color='#2E86AB', linewidth=2, markersize=8)
    ax3.plot(thresholds, tiered_means, 's-', label='Tiered Mean', 
             color='#A23B72', linewidth=2, markersize=8)
    ax3.plot(thresholds, baseline_p95s, 'o--', label='Baseline P95', 
             color='#2E86AB', linewidth=2, markersize=8, alpha=0.6)
    ax3.plot(thresholds, tiered_p95s, 's--', label='Tiered P95', 
             color='#A23B72', linewidth=2, markersize=8, alpha=0.6)
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Latency Trends Across Thresholds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement percentage
    ax4 = axes[1, 1]
    improvements = []
    for th in thresholds:
        baseline_mean = data[th]['baseline']['mean']
        tiered_mean = data[th]['tiered']['mean']
        if baseline_mean > 0:
            improvement = ((baseline_mean - tiered_mean) / baseline_mean) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    colors = ['#06A77D' if imp > 0 else '#D00000' for imp in improvements]
    bars = ax4.bar(x, improvements, width=0.6, color=colors, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Similarity Threshold')
    ax4.set_ylabel('Performance Improvement (%)')
    ax4.set_title('Tiered System Performance vs Baseline')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{th:.2f}' for th in thresholds])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, imp in enumerate(improvements):
        ax4.text(i, imp + (3 if imp > 0 else -5), f'{imp:.1f}%', 
                ha='center', va='bottom' if imp > 0 else 'top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/latency_comparison.png")
    plt.close()


def plot_accuracy_latency_tradeoff(data, metrics, output_dir):
    """Plot 2: Accuracy-Latency Tradeoff"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Accuracy-Latency Tradeoff Analysis', fontsize=16, fontweight='bold')
    
    thresholds = sorted(data.keys())
    
    # Extract metrics
    avg_results = [metrics.get(th, {}).get('avg_results', 0) for th in thresholds]
    avg_similarity = [metrics.get(th, {}).get('avg_similarity', 0) for th in thresholds]
    tiered_means = [data[th]['tiered']['mean'] for th in thresholds]
    baseline_means = [data[th]['baseline']['mean'] for th in thresholds]
    
    # Plot 1: Threshold vs Average Results
    ax1 = axes[0, 0]
    ax1.plot(thresholds, avg_results, 'o-', color='#F18F01', linewidth=2.5, markersize=10)
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Average Results per Query')
    ax1.set_title('Query Result Count vs Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.5, top=max(avg_results) * 1.1 if max(avg_results) > 0 else 5.5)
    
    # Add value labels
    for th, res in zip(thresholds, avg_results):
        ax1.text(th, res + 0.2, f'{res:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Threshold vs Average Similarity
    ax2 = axes[0, 1]
    ax2.plot(thresholds, avg_similarity, 's-', color='#C73E1D', linewidth=2.5, markersize=10)
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Average Similarity Score')
    ax2.set_title('Result Quality vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-0.05, top=1.0)
    
    # Add value labels
    for th, sim in zip(thresholds, avg_similarity):
        if sim > 0:
            ax2.text(th, sim + 0.03, f'{sim:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Latency vs Average Results (Tradeoff curve)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(avg_results, tiered_means, s=150, c=thresholds, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Average Results per Query')
    ax3.set_ylabel('Tiered System Latency (ms)')
    ax3.set_title('Latency vs Result Count Tradeoff')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Threshold', rotation=270, labelpad=15)
    
    # Add threshold labels
    for th, res, lat in zip(thresholds, avg_results, tiered_means):
        ax3.annotate(f'Th={th}', (res, lat), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Plot 4: Latency vs Similarity (Quality-Latency tradeoff)
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(avg_similarity, tiered_means, s=150, c=thresholds, 
                          cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Average Similarity Score')
    ax4.set_ylabel('Tiered System Latency (ms)')
    ax4.set_title('Latency vs Result Quality Tradeoff')
    ax4.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('Threshold', rotation=270, labelpad=15)
    
    # Add threshold labels
    for th, sim, lat in zip(thresholds, avg_similarity, tiered_means):
        if sim > 0:
            ax4.annotate(f'Th={th}', (sim, lat), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_latency_tradeoff.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_latency_tradeoff.png")
    plt.close()


def plot_tier_access_distribution(metrics, output_dir):
    """Plot 3: Tier Access Distribution"""
    if not metrics:
        print("Warning: No metrics available for tier access distribution")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Tier Access Distribution Across Thresholds', fontsize=16, fontweight='bold')
    
    thresholds = sorted(metrics.keys())
    
    # Extract tier percentages
    t1_pcts = [metrics[th].get('t1_pct', 0) for th in thresholds]
    t2_pcts = [metrics[th].get('t2_pct', 0) for th in thresholds]
    t3_pcts = [metrics[th].get('t3_pct', 0) for th in thresholds]
    
    # Plot 1: Stacked bar chart
    ax1 = axes[0]
    x = np.arange(len(thresholds))
    width = 0.6
    
    bars1 = ax1.bar(x, t1_pcts, width, label='Tier 1 (RAM)', color='#06A77D', alpha=0.8)
    bars2 = ax1.bar(x, t2_pcts, width, bottom=t1_pcts, label='Tier 2 (Local Disk)', 
                    color='#F18F01', alpha=0.8)
    bars3 = ax1.bar(x, t3_pcts, width, bottom=np.array(t1_pcts) + np.array(t2_pcts), 
                    label='Tier 3 (Remote Disk)', color='#2E86AB', alpha=0.8)
    
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Access Percentage (%)')
    ax1.set_title('Tier Access Distribution (Stacked)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{th:.2f}' for th in thresholds])
    # Move legend outside the plot area to avoid overlap
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (t1, t2, t3) in enumerate(zip(t1_pcts, t2_pcts, t3_pcts)):
        if t1 > 5:
            ax1.text(i, t1/2, f'{t1:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
        if t2 > 5:
            ax1.text(i, t1 + t2/2, f'{t2:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
        if t3 > 5:
            ax1.text(i, t1 + t2 + t3/2, f'{t3:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
    
    # Plot 2: Line chart showing tier access trends
    ax2 = axes[1]
    ax2.plot(thresholds, t1_pcts, 'o-', label='Tier 1 (RAM)', 
             color='#06A77D', linewidth=2.5, markersize=10)
    ax2.plot(thresholds, t2_pcts, 's-', label='Tier 2 (Local Disk)', 
             color='#F18F01', linewidth=2.5, markersize=10)
    ax2.plot(thresholds, t3_pcts, '^-', label='Tier 3 (Remote Disk)', 
             color='#2E86AB', linewidth=2.5, markersize=10)
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Access Percentage (%)')
    ax2.set_title('Tier Access Trends')
    # Move legend outside the plot area to avoid overlap
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tier_access_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/tier_access_distribution.png")
    plt.close()


def plot_comprehensive_dashboard(data, metrics, output_dir):
    """Plot 4: Comprehensive dashboard showing all key findings"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Tiered Storage System: Comprehensive Evaluation Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    thresholds = sorted(data.keys())
    
    # 1. Latency comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    baseline_means = [data[th]['baseline']['mean'] for th in thresholds]
    tiered_means = [data[th]['tiered']['mean'] for th in thresholds]
    x = np.arange(len(thresholds))
    width = 0.35
    ax1.bar(x - width/2, baseline_means, width, label='Baseline (Cloud)', 
            color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, tiered_means, width, label='Tiered System', 
            color='#A23B72', alpha=0.8)
    ax1.set_xlabel('Similarity Threshold', fontweight='bold')
    ax1.set_ylabel('Mean Latency (ms)', fontweight='bold')
    ax1.set_title('Mean Latency Comparison: Baseline vs Tiered System', 
                  fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{th:.2f}' for th in thresholds])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Performance retention (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    retentions = [metrics.get(th, {}).get('retention', 0) for th in thresholds]
    colors = ['#06A77D' if r > 100 else '#D00000' for r in retentions]
    bars = ax2.bar(x, retentions, width=0.6, color=colors, alpha=0.8)
    ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, label='Baseline (100%)')
    ax2.set_xlabel('Threshold', fontweight='bold')
    ax2.set_ylabel('Performance Retention (%)', fontweight='bold')
    ax2.set_title('Performance vs Baseline', fontweight='bold', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{th:.2f}' for th in thresholds])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Accuracy metrics (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    avg_results = [metrics.get(th, {}).get('avg_results', 0) for th in thresholds]
    ax3.plot(thresholds, avg_results, 'o-', color='#F18F01', linewidth=2.5, markersize=10)
    ax3.set_xlabel('Threshold', fontweight='bold')
    ax3.set_ylabel('Avg Results/Query', fontweight='bold')
    ax3.set_title('Result Count', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=-0.5)
    
    # 4. Similarity scores (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    avg_similarity = [metrics.get(th, {}).get('avg_similarity', 0) for th in thresholds]
    ax4.plot(thresholds, avg_similarity, 's-', color='#C73E1D', linewidth=2.5, markersize=10)
    ax4.set_xlabel('Threshold', fontweight='bold')
    ax4.set_ylabel('Avg Similarity', fontweight='bold')
    ax4.set_title('Result Quality', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=-0.05, top=1.0)
    
    # 5. Tradeoff curve (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    scatter = ax5.scatter(avg_results, tiered_means, s=200, c=thresholds, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    ax5.set_xlabel('Results/Query', fontweight='bold')
    ax5.set_ylabel('Latency (ms)', fontweight='bold')
    ax5.set_title('Accuracy-Latency Tradeoff', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    for th, res, lat in zip(thresholds, avg_results, tiered_means):
        ax5.annotate(f'{th}', (res, lat), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 6. Tier distribution (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    t1_pcts = [metrics.get(th, {}).get('t1_pct', 0) for th in thresholds]
    t2_pcts = [metrics.get(th, {}).get('t2_pct', 0) for th in thresholds]
    t3_pcts = [metrics.get(th, {}).get('t3_pct', 0) for th in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.6
    ax6.bar(x, t1_pcts, width, label='Tier 1 (RAM)', color='#06A77D', alpha=0.8)
    ax6.bar(x, t2_pcts, width, bottom=t1_pcts, label='Tier 2 (Local Disk)', 
            color='#F18F01', alpha=0.8)
    ax6.bar(x, t3_pcts, width, bottom=np.array(t1_pcts) + np.array(t2_pcts), 
            label='Tier 3 (Remote Disk)', color='#2E86AB', alpha=0.8)
    ax6.set_xlabel('Similarity Threshold', fontweight='bold')
    ax6.set_ylabel('Access Percentage (%)', fontweight='bold')
    ax6.set_title('Tier Access Distribution Across Thresholds', 
                  fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'{th:.2f}' for th in thresholds])
    ax6.legend(loc='upper right', ncol=3)
    ax6.set_ylim(0, 105)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comprehensive_dashboard.png")
    plt.close()


def main():
    """Main function to generate all visualizations"""
    print("=" * 70)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = "data/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading evaluation data...")
    data = load_threshold_data()
    if not data:
        print("Error: No data found. Please run evaluate.py first.")
        return
    
    metrics = load_summary_metrics()
    if not metrics:
        print("Warning: Could not load summary metrics. Some plots may be incomplete.")
    
    print(f"Loaded data for {len(data)} thresholds: {sorted(data.keys())}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    try:
        plot_latency_comparison(data, output_dir)
        plot_accuracy_latency_tradeoff(data, metrics, output_dir)
        plot_tier_access_distribution(metrics, output_dir)
        plot_comprehensive_dashboard(data, metrics, output_dir)
        
        print()
        print("=" * 70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("=" * 70)
        print(f"All visualizations saved to: {output_dir}/")
        print()
        print("Generated files:")
        print("  1. latency_comparison.png - Latency metrics across thresholds")
        print("  2. accuracy_latency_tradeoff.png - Accuracy-latency tradeoff analysis")
        print("  3. tier_access_distribution.png - Tier access patterns")
        print("  4. comprehensive_dashboard.png - Complete evaluation overview")
        print()
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

