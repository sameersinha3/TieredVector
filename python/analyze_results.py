#!/usr/bin/env python3
"""
Analyze query log results and display latency statistics
"""

import argparse
import csv
import statistics
import os


def analyze_log(log_file, system_name):
    """Analyze query log and return statistics"""
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return None
    
    latencies = []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            latencies.append(float(row['latency_ms']))
    
    if not latencies:
        print(f"No data found in {log_file}")
        return None
    
    sorted_lat = sorted(latencies)
    
    stats = {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'p95': sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 0 else 0,
        'p99': sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 0 else 0,
    }
    
    return stats


def print_statistics(stats, system_name):
    """Print formatted statistics"""
    print("=" * 70)
    print(f"{system_name.upper()} SYSTEM STATISTICS")
    print("=" * 70)
    print(f"Total queries: {stats['count']}")
    print()
    print("Latency Statistics (ms):")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  P95:    {stats['p95']:.2f}")
    print(f"  P99:    {stats['p99']:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze query log results')
    parser.add_argument('--log', type=str, required=True, help='Path to query log CSV file')
    parser.add_argument('--system', type=str, required=True, choices=['baseline', 'tiered'],
                       help='System name (baseline or tiered)')
    args = parser.parse_args()
    
    stats = analyze_log(args.log, args.system)
    if stats:
        print_statistics(stats, args.system)


if __name__ == "__main__":
    main()

