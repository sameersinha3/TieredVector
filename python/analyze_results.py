"""
Analyze evaluation results: Memory-Performance Tradeoff

Reads query logs and calculates:
- Memory ratios (% RAM used vs baseline)
- Performance retention (% of baseline performance maintained)
- Efficiency metrics
"""
import csv
import os
import sys
import statistics
import argparse
import numpy as np


def calculate_latency_stats(latencies):
    """Calculate latency statistics."""
    if not latencies:
        return {}
    
    sorted_lat = sorted(latencies)
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 0 else 0,
        "p99": sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 0 else 0,
    }


def analyze_query_log(log_path, system_type):
    """Analyze query log and return statistics."""
    if not os.path.exists(log_path):
        print(f"Error: Query log not found at {log_path}")
        return None
    
    latencies = []
    latencies_t1 = []
    latencies_t2 = []
    latencies_t3 = []
    tier_patterns = {}
    total_queries = 0
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows from different system type if specified
            if row.get("system_type", "") != system_type and "system_type" in row:
                continue
            
            total_queries += 1
            
            # Overall latency
            if row.get("latency_ms"):
                try:
                    latencies.append(float(row["latency_ms"]))
                except ValueError:
                    pass
            
            # Per-tier latencies
            for tier, col in [("t1", "latency_t1_ms"), ("t2", "latency_t2_ms"), ("t3", "latency_t3_ms")]:
                if row.get(col):
                    try:
                        val = float(row[col])
                        if val > 0:
                            if tier == "t1":
                                latencies_t1.append(val)
                            elif tier == "t2":
                                latencies_t2.append(val)
                            elif tier == "t3":
                                latencies_t3.append(val)
                    except ValueError:
                        pass
            
            # Tier access patterns
            tier_hits = row.get("tier_hits", "")
            if tier_hits not in tier_patterns:
                tier_patterns[tier_hits] = []
            if latencies:
                tier_patterns[tier_hits].append(latencies[-1])
    
    stats = {
        "total_queries": total_queries,
        "overall": calculate_latency_stats(latencies),
        "tier1": calculate_latency_stats(latencies_t1),
        "tier2": calculate_latency_stats(latencies_t2),
        "tier3": calculate_latency_stats(latencies_t3),
        "tier_patterns": tier_patterns
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--log", type=str, required=True, help="Path to query log CSV file")
    parser.add_argument("--system", type=str, choices=["baseline", "tiered"], default="tiered",
                       help="System type")
    
    args = parser.parse_args()
    
    stats = analyze_query_log(args.log, args.system)
    if stats is None:
        return
    
    print("=" * 70)
    print(f"ANALYSIS: {args.system.upper()} SYSTEM")
    print("=" * 70)
    print(f"Total queries: {stats['total_queries']}")
    print()
    
    if stats['overall']:
        print("Overall Latency (ms):")
        print(f"  Mean:   {stats['overall']['mean']:.2f}")
        print(f"  Median: {stats['overall']['median']:.2f}")
        print(f"  Min:    {stats['overall']['min']:.2f}")
        print(f"  Max:    {stats['overall']['max']:.2f}")
        print(f"  P95:    {stats['overall']['p95']:.2f}")
        print(f"  P99:    {stats['overall']['p99']:.2f}")
        print()
    
    if stats['tier1']:
        print("Tier 1 Latency (ms):")
        print(f"  Mean:   {stats['tier1']['mean']:.2f}")
        print(f"  Median: {stats['tier1']['median']:.2f}")
        print(f"  P95:    {stats['tier1']['p95']:.2f}")
        print()
    
    if stats['tier2']:
        print("Tier 2 Latency (ms):")
        print(f"  Mean:   {stats['tier2']['mean']:.2f}")
        print(f"  Median: {stats['tier2']['median']:.2f}")
        print(f"  P95:    {stats['tier2']['p95']:.2f}")
        print()
    
    if stats['tier3']:
        print("Tier 3 Latency (ms):")
        print(f"  Mean:   {stats['tier3']['mean']:.2f}")
        print(f"  Median: {stats['tier3']['median']:.2f}")
        print(f"  P95:    {stats['tier3']['p95']:.2f}")
        print()
    
    if stats['tier_patterns']:
        print("Latency by Tier Access Pattern:")
        for pattern, pattern_lats in sorted(stats['tier_patterns'].items()):
            pattern_name = f"Tier {pattern}" if pattern else "No tiers"
            pattern_stats = calculate_latency_stats(pattern_lats)
            print(f"  {pattern_name:20} ({len(pattern_lats):3} queries): "
                  f"mean={pattern_stats['mean']:.2f}ms, "
                  f"median={pattern_stats['median']:.2f}ms, "
                  f"p95={pattern_stats['p95']:.2f}ms")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

