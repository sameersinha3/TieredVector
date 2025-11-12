"""
Compare costs between baseline and tiered systems.
Shows cost savings while maintaining same performance.
"""
import subprocess
import sys
import os


def run_calc_cost(log_path, system_type):
    """Run calc_cost.py and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, "python/calc_cost.py", "--log", log_path, "--system", system_type],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def extract_cost(output):
    """Extract total cost from calc_cost output."""
    for line in output.split('\n'):
        if 'TOTAL COST:' in line:
            try:
                cost_str = line.split('$')[-1].strip()
                return float(cost_str)
            except:
                pass
    return None


def extract_latency_stats(output):
    """Extract latency statistics from output."""
    stats = {}
    lines = output.split('\n')
    
    for i, line in enumerate(lines):
        if 'Overall Latency (ms):' in line:
            # Next few lines contain stats
            for j in range(i+1, min(i+7, len(lines))):
                if 'Mean:' in lines[j]:
                    stats['mean'] = float(lines[j].split(':')[1].strip())
                elif 'P95:' in lines[j]:
                    stats['p95'] = float(lines[j].split(':')[1].strip())
                elif 'P99:' in lines[j]:
                    stats['p99'] = float(lines[j].split(':')[1].strip())
            break
    
    return stats


def main():
    baseline_log = "data/query_log_baseline.csv"
    tiered_log = "data/query_log_tiered.csv"
    
    if not os.path.exists(baseline_log):
        print(f"Error: Baseline log not found at {baseline_log}")
        print("Please run evaluate.py first.")
        sys.exit(1)
    
    if not os.path.exists(tiered_log):
        print(f"Error: Tiered log not found at {tiered_log}")
        print("Please run evaluate.py first.")
        sys.exit(1)
    
    print("=" * 70)
    print("COST COMPARISON: Baseline vs Tiered System")
    print("=" * 70)
    print()
    
    # Calculate baseline costs
    print("Calculating baseline costs...")
    baseline_out, baseline_err, baseline_code = run_calc_cost(baseline_log, "baseline")
    if baseline_code != 0:
        print(f"Error calculating baseline costs: {baseline_err}")
        sys.exit(1)
    
    # Calculate tiered costs
    print("Calculating tiered costs...")
    tiered_out, tiered_err, tiered_code = run_calc_cost(tiered_log, "tiered")
    if tiered_code != 0:
        print(f"Error calculating tiered costs: {tiered_err}")
        sys.exit(1)
    
    # Extract costs
    baseline_cost = extract_cost(baseline_out)
    tiered_cost = extract_cost(tiered_out)
    
    if baseline_cost is None or tiered_cost is None:
        print("Error: Could not extract costs from output")
        print("\nBaseline output:")
        print(baseline_out)
        print("\nTiered output:")
        print(tiered_out)
        sys.exit(1)
    
    # Extract latency stats
    baseline_stats = extract_latency_stats(baseline_out)
    tiered_stats = extract_latency_stats(tiered_out)
    
    # Calculate savings
    cost_savings = baseline_cost - tiered_cost
    cost_savings_pct = (cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    # Display comparison
    print("\n" + "=" * 70)
    print("COST COMPARISON RESULTS")
    print("=" * 70)
    print()
    
    print(f"Baseline System Cost:  ${baseline_cost:.6f}")
    print(f"Tiered System Cost:     ${tiered_cost:.6f}")
    print(f"Cost Savings:           ${cost_savings:.6f} ({cost_savings_pct:.2f}% reduction)")
    print()
    
    if baseline_stats and tiered_stats:
        print("=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        print()
        
        print("Mean Latency:")
        print(f"  Baseline: {baseline_stats.get('mean', 0):.2f} ms")
        print(f"  Tiered:   {tiered_stats.get('mean', 0):.2f} ms")
        print(f"  Difference: {tiered_stats.get('mean', 0) - baseline_stats.get('mean', 0):+.2f} ms")
        print()
        
        print("P95 Latency:")
        print(f"  Baseline: {baseline_stats.get('p95', 0):.2f} ms")
        print(f"  Tiered:   {tiered_stats.get('p95', 0):.2f} ms")
        print(f"  Difference: {tiered_stats.get('p95', 0) - baseline_stats.get('p95', 0):+.2f} ms")
        print()
        
        print("P99 Latency:")
        print(f"  Baseline: {baseline_stats.get('p99', 0):.2f} ms")
        print(f"  Tiered:   {tiered_stats.get('p99', 0):.2f} ms")
        print(f"  Difference: {tiered_stats.get('p99', 0) - baseline_stats.get('p99', 0):+.2f} ms")
        print()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    
    if cost_savings > 0:
        print(f"✓ Tiered system achieves {cost_savings_pct:.2f}% cost reduction")
        if baseline_stats and tiered_stats:
            mean_diff = tiered_stats.get('mean', 0) - baseline_stats.get('mean', 0)
            p95_diff = tiered_stats.get('p95', 0) - baseline_stats.get('p95', 0)
            
            if abs(mean_diff) < baseline_stats.get('mean', 1) * 0.1 and abs(p95_diff) < baseline_stats.get('p95', 1) * 0.1:
                print("✓ Performance is maintained within 10% of baseline")
                print("✓ Evaluation demonstrates: Same Performance → Cost Reductions")
            else:
                print(f"⚠ Performance differs: Mean {mean_diff:+.2f}ms, P95 {p95_diff:+.2f}ms")
    else:
        print("⚠ Tiered system did not achieve cost savings")
        print("  This may indicate:")
        print("  - Not enough queries hitting cheaper tiers")
        print("  - Storage sizes need adjustment")
        print("  - Temperature thresholds need tuning")
    
    print()
    print("=" * 70)
    
    # Show detailed outputs
    print("\n" + "=" * 70)
    print("DETAILED BASELINE COST ANALYSIS")
    print("=" * 70)
    print(baseline_out)
    
    print("\n" + "=" * 70)
    print("DETAILED TIERED COST ANALYSIS")
    print("=" * 70)
    print(tiered_out)


if __name__ == "__main__":
    main()

