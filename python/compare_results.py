"""
Compare baseline vs tiered system results: Memory-Performance Tradeoff

Shows:
- Memory reduction (% RAM saved)
- Performance retention (% of baseline performance maintained)
- Efficiency summary
"""
import csv
import os
import sys
import argparse
import numpy as np


def read_evaluation_summary(summary_path):
    """Read evaluation summary file."""
    if not os.path.exists(summary_path):
        return None
    
    results = {}
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            
            if "Baseline System" in line:
                current_section = "baseline"
            elif "Tiered System" in line:
                current_section = "tiered"
            elif "Results:" in line:
                current_section = "results"
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if current_section:
                    if current_section not in results:
                        results[current_section] = {}
                    results[current_section][key] = value
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs tiered results")
    parser.add_argument("--summary", type=str, default="data/evaluation_summary.txt",
                       help="Path to evaluation summary file")
    
    args = parser.parse_args()
    
    summary = read_evaluation_summary(args.summary)
    
    if summary is None:
        print(f"Error: Evaluation summary not found at {args.summary}")
        print("Please run evaluate.py first to generate the summary.")
        return
    
    print("=" * 70)
    print("MEMORY-PERFORMANCE TRADEOFF COMPARISON")
    print("=" * 70)
    print()
    
    # Extract values
    baseline = summary.get("baseline", {})
    tiered = summary.get("tiered", {})
    results = summary.get("results", {})
    
    # Memory comparison
    print("Memory Usage:")
    baseline_ram = baseline.get("RAM Usage", "N/A")
    tiered_ram = tiered.get("RAM Usage", "N/A")
    ram_reduction = results.get("RAM Reduction", "N/A")
    
    print(f"  Baseline: {baseline_ram} GB (100% of data)")
    print(f"  Tiered:   {tiered_ram}")
    print(f"  Reduction: {ram_reduction}")
    print()
    
    # Performance comparison
    print("Performance (Latency):")
    print(f"  Baseline Mean:  {baseline.get('Mean Latency', 'N/A')} ms")
    print(f"  Tiered Mean:    {tiered.get('Mean Latency', 'N/A')} ms")
    print(f"  Baseline P95:   {baseline.get('P95 Latency', 'N/A')} ms")
    print(f"  Tiered P95:     {tiered.get('P95 Latency', 'N/A')} ms")
    print()
    
    # Performance retention
    print("Performance Retention:")
    perf_mean = results.get("Performance Retention (Mean)", "N/A")
    perf_p95 = results.get("Performance Retention (P95)", "N/A")
    print(f"  Mean latency: {perf_mean}% of baseline")
    print(f"  P95 latency:  {perf_p95}% of baseline")
    print()
    
    # Efficiency summary
    print("=" * 70)
    print("EFFICIENCY SUMMARY")
    print("=" * 70)
    
    try:
        ram_pct = float(ram_reduction.replace("%", ""))
        perf_pct = float(perf_mean.replace("%", ""))
        
        print(f"With {100 - ram_pct:.1f}% RAM usage, the tiered system retains")
        print(f"{perf_pct:.1f}% of baseline performance.")
        print()
        print(f"This represents:")
        print(f"  ✓ {ram_pct:.1f}% reduction in memory footprint")
        print(f"  ✓ {perf_pct:.1f}% performance retention")
        print()
        
        if perf_pct >= 80:
            print("Result: Excellent efficiency - significant memory savings")
            print("        while maintaining high performance.")
        elif perf_pct >= 60:
            print("Result: Good efficiency - substantial memory savings")
            print("        with acceptable performance.")
        else:
            print("Result: Tradeoff favors memory savings over performance.")
    except (ValueError, TypeError):
        print("Unable to calculate efficiency metrics from summary.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

