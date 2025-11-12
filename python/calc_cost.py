"""
Cost calculator for tiered vector storage system.
Adapted to work with ChromaDB (Tier 2) instead of LMDB.
"""
import csv
import os
import sys
import statistics
import argparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pickle
except ImportError:
    pickle = None

try:
    import chromadb
except ImportError:
    chromadb = None

COST_CFG = "config/cost.yaml"


def load_prices():
    if yaml is None:
        raise RuntimeError("Please install pyyaml to use cost calculator.")
    with open(COST_CFG, "r") as f:
        return yaml.safe_load(f)


def calculate_storage_sizes(system_type="tiered"):
    """
    Calculate actual storage sizes for each tier from data.
    Returns (ram_gb, ssd_gb, blob_gb)
    """
    ram_bytes = 0
    ssd_bytes = 0
    blob_bytes = 0
    
    # Tier 1: Calculate from Redis (estimate based on query log or use fallback)
    # For baseline system, we'll estimate from Redis keys
    # For tiered system, we can estimate from tier1_hot.pkl if it exists
    if os.path.exists("data/tier1_hot.pkl") and pickle:
        try:
            with open("data/tier1_hot.pkl", "rb") as f:
                hot_docs = pickle.load(f)
            for rec in hot_docs:
                # Estimate: id + text + embedding (768 dims * 4 bytes)
                ram_bytes += len(rec.get("id", "").encode()) + len(rec.get("text", "").encode()) + (768 * 4)
        except Exception as e:
            print(f"Warning: Could not calculate Tier 1 size from tier1_hot.pkl: {e}", file=sys.stderr)
    
    # Tier 2: Calculate from ChromaDB directory
    ssd_bytes = 0
    tier2_path = "./tier2_chroma_db"
    if os.path.exists(tier2_path) and chromadb:
        try:
            # ChromaDB stores data in a directory structure
            # Estimate based on directory size
            import subprocess
            if sys.platform != "win32":
                result = subprocess.run(
                    ["du", "-sb", tier2_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    ssd_bytes = int(result.stdout.split()[0])
                else:
                    # Fallback: walk directory
                    for root, dirs, files in os.walk(tier2_path):
                        for f in files:
                            ssd_bytes += os.path.getsize(os.path.join(root, f))
            else:
                # Windows fallback
                for root, dirs, files in os.walk(tier2_path):
                    for f in files:
                        ssd_bytes += os.path.getsize(os.path.join(root, f))
        except Exception as e:
            print(f"Warning: Could not calculate Tier 2 size: {e}", file=sys.stderr)
    
    # Tier 3: Calculate from remote ChromaDB (estimate based on query log)
    # For now, we'll use fallback values as we can't directly access remote DB size
    # In production, you'd query the remote DB for size
    
    # Convert to GB
    ram_gb = ram_bytes / (1024 ** 3)
    ssd_gb = ssd_bytes / (1024 ** 3)
    blob_gb = 0  # Will use fallback
    
    return ram_gb, ssd_gb, blob_gb


def main():
    parser = argparse.ArgumentParser(description="Calculate costs from query log")
    parser.add_argument("--log", type=str, default="data/query_log.csv",
                       help="Path to query log CSV file")
    parser.add_argument("--system", type=str, choices=["baseline", "tiered"],
                       default="tiered", help="System type")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Query log not found at {args.log}")
        return

    prices = load_prices()
    unit = prices["unit_prices"]
    exp = prices["experiment"]

    # Calculate actual storage sizes
    actual_ram_gb, actual_ssd_gb, actual_blob_gb = calculate_storage_sizes(args.system)
    
    # Use actual sizes if available, otherwise fall back to config
    ram_gb = actual_ram_gb if actual_ram_gb > 0 else exp.get("ram_gb", 1.0)
    ssd_gb = actual_ssd_gb if actual_ssd_gb > 0 else exp.get("ssd_gb", 2.0)
    blob_gb = actual_blob_gb if actual_blob_gb > 0 else exp.get("blob_gb", 5.0)
    
    # For baseline system, all data is in RAM
    if args.system == "baseline":
        # Estimate RAM size from query log or use a larger fallback
        # In practice, you'd measure Redis memory usage
        if actual_ram_gb == 0:
            # Rough estimate: assume 10k docs * ~3KB per doc = ~30MB = ~0.03GB
            # But use config fallback for now
            ram_gb = exp.get("ram_gb", 1.0)
        ssd_gb = 0.0
        blob_gb = 0.0

    total_q = 0
    total_b1 = 0
    total_b2 = 0
    total_b3 = 0
    total_ops_t1 = 0
    total_ops_t2 = 0
    total_ops_t3 = 0
    total_latency_ms = 0.0
    latencies = []
    latencies_t1 = []
    latencies_t2 = []
    latencies_t3 = []
    embed_times = []
    latency_by_tier_pattern = {}

    # Read query log
    with open(args.log, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            # Skip rows from different system type
            if row.get("system_type", "") != args.system and "system_type" in row:
                continue
                
            total_q += 1
            total_b1 += float(row.get("bytes_t1", 0) or 0)
            total_b2 += float(row.get("bytes_t2", 0) or 0)
            total_b3 += float(row.get("bytes_t3", 0) or 0)
            total_ops_t1 += int(row.get("ops_t1", 0) or 0)
            total_ops_t2 += int(row.get("ops_t2", 0) or 0)
            total_ops_t3 += int(row.get("ops_t3", 0) or 0)
            
            latency = float(row.get("latency_ms", 0) or 0)
            total_latency_ms += latency
            latencies.append(latency)
            
            # Per-tier latencies
            if "latency_t1_ms" in row and row["latency_t1_ms"]:
                try:
                    t1_lat = float(row["latency_t1_ms"])
                    if t1_lat > 0:
                        latencies_t1.append(t1_lat)
                except ValueError:
                    pass
            if "latency_t2_ms" in row and row["latency_t2_ms"]:
                try:
                    t2_lat = float(row["latency_t2_ms"])
                    if t2_lat > 0:
                        latencies_t2.append(t2_lat)
                except ValueError:
                    pass
            if "latency_t3_ms" in row and row["latency_t3_ms"]:
                try:
                    t3_lat = float(row["latency_t3_ms"])
                    if t3_lat > 0:
                        latencies_t3.append(t3_lat)
                except ValueError:
                    pass
            
            # Embedding time
            if "embed_time_ms" in row and row["embed_time_ms"]:
                try:
                    embed_times.append(float(row["embed_time_ms"]))
                except ValueError:
                    pass
            
            # Group by tier access pattern
            tier_hits = row.get("tier_hits", "")
            if tier_hits not in latency_by_tier_pattern:
                latency_by_tier_pattern[tier_hits] = []
            latency_by_tier_pattern[tier_hits].append(latency)

    # Convert bytes to GB
    gb_t1 = total_b1 / (1024 ** 3)
    gb_t2 = total_b2 / (1024 ** 3)
    gb_t3 = total_b3 / (1024 ** 3)

    # Calculate costs
    
    # 1. Storage costs (provisioned storage for the duration)
    month_frac = exp["hours"] / (24 * 30)
    c_ram_storage = ram_gb * unit["ram_gb_month"] * month_frac
    c_ssd_storage = ssd_gb * unit["ssd_gb_month"] * month_frac
    c_blob_storage = blob_gb * unit["blob_gb_month"] * month_frac
    storage_cost = c_ram_storage + c_ssd_storage + c_blob_storage
    
    # 2. Read operation costs
    # Tier 1: RAM reads are "free" (no operation cost, just storage)
    c_ram_ops = 0.0
    
    # Tier 2: SSD read operations (ChromaDB queries)
    c_ssd_ops = (total_ops_t2 / 1000.0) * unit.get("ssd_read_1k", 0.001)
    
    # Tier 3: GCS GET operations (ChromaDB remote queries)
    c_blob_ops = (total_ops_t3 / 1000.0) * unit["blob_get_1k"]
    
    # 3. Egress costs (data transfer out from GCS/remote)
    c_egress = gb_t3 * unit["egress_gb"]
    
    # 4. CPU cost for embedding generation
    if embed_times:
        avg_embedding_time_sec = (sum(embed_times) / len(embed_times)) / 1000.0
        cpu_hours = (total_q * avg_embedding_time_sec) / 3600.0
    else:
        # Fallback: estimate ~50ms per query
        avg_embedding_time_sec = 0.05
        cpu_hours = (total_q * avg_embedding_time_sec) / 3600.0
    c_cpu = cpu_hours * unit["cpu_core_hour"]
    
    # Total costs
    operation_cost = c_ram_ops + c_ssd_ops + c_blob_ops
    total_cost = storage_cost + operation_cost + c_egress + c_cpu
    
    # Latency statistics
    def calc_stats(lat_list):
        if not lat_list:
            return {}
        sorted_lat = sorted(lat_list)
        return {
            "mean": statistics.mean(lat_list),
            "median": statistics.median(lat_list),
            "min": min(lat_list),
            "max": max(lat_list),
            "p95": sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 0 else 0,
            "p99": sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 0 else 0,
        }
    
    overall_latency_stats = calc_stats(latencies)
    t1_latency_stats = calc_stats(latencies_t1)
    t2_latency_stats = calc_stats(latencies_t2)
    t3_latency_stats = calc_stats(latencies_t3)
    
    # Output results
    print("=" * 70)
    print(f"COST ANALYSIS - {args.system.upper()} SYSTEM")
    print("=" * 70)
    print(f"Queries processed: {total_q}")
    print()
    
    print("Storage Sizes (GB):")
    print(f"  Tier 1 (RAM):     {ram_gb:.4f} GB {'(calculated)' if actual_ram_gb > 0 else '(fallback)'}")
    print(f"  Tier 2 (SSD):     {ssd_gb:.4f} GB {'(calculated)' if actual_ssd_gb > 0 else '(fallback)'}")
    print(f"  Tier 3 (GCS):     {blob_gb:.4f} GB {'(calculated)' if actual_blob_gb > 0 else '(fallback)'}")
    print()
    
    print("Operations:")
    print(f"  Tier 1 reads:      {total_ops_t1:,}")
    print(f"  Tier 2 reads:      {total_ops_t2:,}")
    print(f"  Tier 3 reads:      {total_ops_t3:,}")
    print()
    
    print("Data Transferred (GB):")
    print(f"  Tier 1:           {gb_t1:.6f}")
    print(f"  Tier 2:           {gb_t2:.6f}")
    print(f"  Tier 3:           {gb_t3:.6f}")
    print()
    
    print("Cost Breakdown:")
    print(f"  Storage costs:")
    print(f"    Tier 1 (RAM):   ${c_ram_storage:.6f}")
    print(f"    Tier 2 (SSD):   ${c_ssd_storage:.6f}")
    print(f"    Tier 3 (GCS):   ${c_blob_storage:.6f}")
    print(f"    Total storage:  ${storage_cost:.6f}")
    print(f"  Operation costs:")
    print(f"    Tier 2 (SSD):   ${c_ssd_ops:.6f}")
    print(f"    Tier 3 (GCS):   ${c_blob_ops:.6f}")
    print(f"    Total ops:       ${operation_cost:.6f}")
    print(f"  Egress (GCS):      ${c_egress:.6f}")
    print(f"  CPU (embeddings):  ${c_cpu:.6f}")
    print()
    print(f"TOTAL COST:          ${total_cost:.6f}")
    print(f"Cost per 1k queries: ${total_cost / max(1, (total_q / 1000.0)):.6f}")
    print()
    
    print("=" * 70)
    print("LATENCY ANALYSIS")
    print("=" * 70)
    if overall_latency_stats:
        print("Overall Latency (ms):")
        print(f"  Mean:   {overall_latency_stats['mean']:.2f}")
        print(f"  Median: {overall_latency_stats['median']:.2f}")
        print(f"  Min:    {overall_latency_stats['min']:.2f}")
        print(f"  Max:    {overall_latency_stats['max']:.2f}")
        print(f"  P95:    {overall_latency_stats['p95']:.2f}")
        print(f"  P99:    {overall_latency_stats['p99']:.2f}")
        print()
    
    if t1_latency_stats:
        print("Tier 1 Latency (ms):")
        print(f"  Mean:   {t1_latency_stats['mean']:.2f}")
        print(f"  Median: {t1_latency_stats['median']:.2f}")
        print(f"  P95:    {t1_latency_stats['p95']:.2f}")
        print()
    
    if t2_latency_stats:
        print("Tier 2 Latency (ms):")
        print(f"  Mean:   {t2_latency_stats['mean']:.2f}")
        print(f"  Median: {t2_latency_stats['median']:.2f}")
        print(f"  P95:    {t2_latency_stats['p95']:.2f}")
        print()
    
    if t3_latency_stats:
        print("Tier 3 Latency (ms):")
        print(f"  Mean:   {t3_latency_stats['mean']:.2f}")
        print(f"  Median: {t3_latency_stats['median']:.2f}")
        print(f"  P95:    {t3_latency_stats['p95']:.2f}")
        print()
    
    if latency_by_tier_pattern:
        print("Latency by Tier Access Pattern:")
        for pattern, pattern_lats in sorted(latency_by_tier_pattern.items()):
            pattern_name = f"Tier {pattern}" if pattern else "No tiers"
            pattern_stats = calc_stats(pattern_lats)
            print(f"  {pattern_name:20} ({len(pattern_lats):3} queries): "
                  f"mean={pattern_stats['mean']:.2f}ms, "
                  f"median={pattern_stats['median']:.2f}ms, "
                  f"p95={pattern_stats['p95']:.2f}ms")
    
    print("=" * 70)
    
    return total_cost


if __name__ == "__main__":
    main()

