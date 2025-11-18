#!/usr/bin/env python3
"""
Compare baseline vs tiered system results
"""

import csv
import statistics
import os


def load_summary():
    """Load evaluation summary"""
    summary_path = "data/evaluation_summary.txt"
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found: {summary_path}")
        print("Please run evaluate.py first.")
        return None
    
    summary = {}
    with open(summary_path, 'r') as f:
        for line in f:
            if 'Local disk:' in line and 'Baseline' in line:
                summary['baseline_disk'] = float(line.split(':')[1].strip().split()[0])
            elif 'Tier 2 (Local disk):' in line:
                summary['tiered_tier2'] = float(line.split(':')[1].strip().split()[0])
            elif 'Tier 1 (RAM):' in line:
                summary['tiered_ram'] = float(line.split(':')[1].strip().split()[0])
            elif 'Tier 3 (Remote disk):' in line:
                summary['tiered_tier3'] = float(line.split(':')[1].strip().split()[0])
            elif 'Mean latency:' in line and 'Baseline' in line:
                summary['baseline_mean'] = float(line.split(':')[1].strip().split()[0])
            elif 'Mean latency:' in line and 'Tiered' in line:
                summary['tiered_mean'] = float(line.split(':')[1].strip().split()[0])
            elif 'Local disk saved:' in line:
                # Extract number before % sign
                pct_str = line.split(':')[1].strip().split()[0]
                summary['disk_saved_pct'] = float(pct_str.rstrip('%'))
            elif 'Performance retention:' in line:
                # Extract number before % sign
                pct_str = line.split(':')[1].strip().split()[0]
                summary['perf_retention'] = float(pct_str.rstrip('%'))
    
    return summary


def analyze_log(log_file):
    """Analyze query log"""
    if not os.path.exists(log_file):
        return None
    
    latencies = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            latencies.append(float(row['latency_ms']))
    
    if not latencies:
        return None
    
    sorted_lat = sorted(latencies)
    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p95': sorted_lat[int(len(sorted_lat) * 0.95)],
        'p99': sorted_lat[int(len(sorted_lat) * 0.99)],
    }


def main():
    print("=" * 70)
    print("BASELINE vs TIERED SYSTEM COMPARISON")
    print("=" * 70)
    print()
    
    # Load summary
    summary = load_summary()
    if not summary:
        return
    
    # Load detailed stats from logs
    baseline_log = "data/query_log_baseline.csv"
    tiered_log = "data/query_log_tiered.csv"
    
    baseline_stats = analyze_log(baseline_log)
    tiered_stats = analyze_log(tiered_log)
    
    if not baseline_stats or not tiered_stats:
        print("Error: Could not load query logs. Please run evaluate.py first.")
        return
    
    # Storage comparison
    print("STORAGE COMPARISON")
    print("-" * 70)
    print(f"Baseline (all Tier 2):")
    print(f"  Local disk: {summary.get('baseline_disk', 0):.4f} GB")
    print()
    print(f"Tiered system:")
    print(f"  Tier 1 (RAM):        {summary.get('tiered_ram', 0):.4f} GB")
    print(f"  Tier 2 (Local disk):  {summary.get('tiered_tier2', 0):.4f} GB")
    print(f"  Tier 3 (Remote disk): {summary.get('tiered_tier3', 0):.4f} GB")
    print()
    print(f"Results:")
    print(f"  Local disk saved:    {summary.get('disk_saved_pct', 0):.2f}%")
    print(f"  RAM overhead:        {summary.get('tiered_ram', 0):.4f} GB")
    print()
    
    # Performance comparison
    print("PERFORMANCE COMPARISON")
    print("-" * 70)
    print(f"Baseline:")
    print(f"  Mean latency:  {baseline_stats['mean']:.2f} ms")
    print(f"  Median:        {baseline_stats['median']:.2f} ms")
    print(f"  P95:           {baseline_stats['p95']:.2f} ms")
    print(f"  P99:           {baseline_stats['p99']:.2f} ms")
    print()
    print(f"Tiered:")
    print(f"  Mean latency:  {tiered_stats['mean']:.2f} ms")
    print(f"  Median:        {tiered_stats['median']:.2f} ms")
    print(f"  P95:           {tiered_stats['p95']:.2f} ms")
    print(f"  P99:           {tiered_stats['p99']:.2f} ms")
    print()
    
    # Calculate differences
    mean_diff = tiered_stats['mean'] - baseline_stats['mean']
    mean_diff_pct = (mean_diff / baseline_stats['mean'] * 100) if baseline_stats['mean'] > 0 else 0
    
    p95_diff = tiered_stats['p95'] - baseline_stats['p95']
    p95_diff_pct = (p95_diff / baseline_stats['p95'] * 100) if baseline_stats['p95'] > 0 else 0
    
    print(f"Performance Difference:")
    print(f"  Mean latency:  {mean_diff:+.2f} ms ({mean_diff_pct:+.1f}%)")
    print(f"  P95 latency:   {p95_diff:+.2f} ms ({p95_diff_pct:+.1f}%)")
    print()
    
    # Efficiency summary
    print("EFFICIENCY SUMMARY")
    print("-" * 70)
    print(f"With {summary.get('disk_saved_pct', 0):.1f}% local disk reduction:")
    if mean_diff < 0:
        print(f"  Tiered system is {abs(mean_diff_pct):.1f}% FASTER")
        print(f"  Performance improvement: {abs(mean_diff_pct):.1f}%")
    else:
        print(f"  Tiered system maintains {summary.get('perf_retention', 0):.1f}% of baseline performance")
        print(f"  Performance overhead: {mean_diff_pct:.1f}%")
    print()
    print(f"Storage breakdown:")
    print(f"  - {summary.get('disk_saved_pct', 0):.1f}% less local disk usage")
    print(f"  - {summary.get('tiered_ram', 0):.4f} GB RAM for hot data")
    print(f"  - {summary.get('tiered_tier3', 0):.4f} GB remote storage for cold data")
    print()


if __name__ == "__main__":
    main()

