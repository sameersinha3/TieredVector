#!/usr/bin/env python3
"""
Evaluation Script: Baseline (Tier 2 only) vs Tiered System (3 tiers)

Baseline: All documents in Tier 2 (Local ChromaDB/disk)
Tiered: Temperature-based distribution across Tier 1 (RAM), Tier 2 (local disk), Tier 3 (remote disk)
"""

import argparse
import numpy as np
import os
import pickle
import redis
import chromadb
import time
import csv
import statistics
from dotenv import load_dotenv
from storage_manager import StorageManager

load_dotenv()

def setup_baseline_system(doc_embeddings, doc_ids, tier2_path='./tier2_baseline_db'):
    """Setup baseline: All documents in Tier 2 (local disk)"""
    print("=" * 70)
    print("SETTING UP BASELINE SYSTEM (All in Tier 2)")
    print("=" * 70)
    
    # Clear Redis (baseline doesn't use Tier 1)
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    keys = r.keys("doc*")
    if keys:
        r.delete(*keys)
        print(f"Cleared {len(keys)} documents from Redis")
    
    # Setup Tier 2 ChromaDB for baseline
    if os.path.exists(tier2_path):
        import shutil
        shutil.rmtree(tier2_path)
        print(f"Cleared existing baseline database at {tier2_path}")
    
    tier2_client = chromadb.PersistentClient(path=tier2_path)
    baseline_collection = tier2_client.get_or_create_collection(
        name="baseline_vectors",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Normalize embeddings
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings_normalized = np.divide(
        doc_embeddings,
        doc_norms,
        out=np.zeros_like(doc_embeddings),
        where=(doc_norms != 0)
    ).astype(np.float32)
    
    # Store all documents in Tier 2
    print(f"Storing {len(doc_ids)} documents in Tier 2 (baseline)...")
    batch_size = 500
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i+batch_size]
        batch_embeddings = [doc_embeddings_normalized[j].tolist() for j in range(i, min(i+batch_size, len(doc_ids)))]
        baseline_collection.upsert(ids=batch_ids, embeddings=batch_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Stored {min(i+batch_size, len(doc_ids))}/{len(doc_ids)} documents...")
    
    print(f"Baseline setup complete: {len(doc_ids)} documents in Tier 2")
    return baseline_collection, tier2_client


def setup_tiered_system():
    """Setup tiered system: Use existing simulate_temperature.py distribution"""
    print("=" * 70)
    print("SETTING UP TIERED SYSTEM (3 tiers)")
    print("=" * 70)
    
    # Check if tier_results.npz exists
    if not os.path.exists("tier_results.npz"):
        print("ERROR: tier_results.npz not found. Please run simulate_temperature.py first.")
        raise FileNotFoundError("tier_results.npz not found")
    
    # Load tier assignments
    tier_data = np.load("tier_results.npz")
    tier_assignment = tier_data['tier_assignment']
    
    print(f"Tier distribution loaded:")
    print(f"  Tier 1 (RAM): {np.sum(tier_assignment == 1)} documents")
    print(f"  Tier 2 (Local disk): {np.sum(tier_assignment == 2)} documents")
    print(f"  Tier 3 (Remote disk): {np.sum(tier_assignment == 3)} documents")
    
    # Initialize StorageManager (will use existing distribution)
    manager = StorageManager(
        tier3_host=os.getenv("VM_IP"),
        tier3_port=8000,
        tier3_collection="cold_vectors"
    )
    
    if not manager.initialize():
        raise RuntimeError("Failed to initialize tiered storage manager")
    
    return manager, tier_assignment


def calculate_baseline_storage(doc_embeddings):
    """Calculate baseline storage: All in Tier 2 (local disk)"""
    # Estimate disk storage for all documents in Tier 2
    # Each embedding: 768 dims * 4 bytes (float32) = 3072 bytes
    # Plus overhead: ~20% for ChromaDB metadata/index
    embedding_size = len(doc_embeddings) * 768 * 4  # bytes
    disk_size_gb = (embedding_size * 1.2) / (1024 ** 3)  # Add 20% overhead
    return disk_size_gb


def calculate_tiered_storage(tier_assignment, doc_embeddings):
    """Calculate tiered storage breakdown"""
    tier1_count = np.sum(tier_assignment == 1)
    tier2_count = np.sum(tier_assignment == 2)
    tier3_count = np.sum(tier_assignment == 3)
    
    # Tier 1 RAM usage (only Tier 1 documents)
    tier1_indices = np.where(tier_assignment == 1)[0]
    ram_bytes = 0
    for idx in tier1_indices:
        if idx < len(doc_embeddings):
            embedding = doc_embeddings[idx]
            ram_bytes += len(pickle.dumps(embedding.astype(np.float32)))
    ram_gb = ram_bytes / (1024 ** 3)
    
    # Tier 2 local disk usage
    tier2_size = tier2_count * 768 * 4 * 1.2  # bytes with overhead
    tier2_gb = tier2_size / (1024 ** 3)
    
    # Tier 3 remote disk usage
    tier3_size = tier3_count * 768 * 4 * 1.2  # bytes with overhead
    tier3_gb = tier3_size / (1024 ** 3)
    
    return ram_gb, tier2_gb, tier3_gb, tier1_count, tier2_count, tier3_count


def run_queries_baseline(baseline_collection, query_embeddings_normalized, k=5, threshold=0.75, num_queries=50):
    """Run queries against baseline system"""
    print("=" * 70)
    print(f"RUNNING QUERIES: BASELINE SYSTEM ({num_queries} queries)")
    print("=" * 70)
    
    latencies = []
    
    for i in range(num_queries):
        query_emb = query_embeddings_normalized[i]
        start_time = time.time()
        
        # Query Tier 2 only (ChromaDB)
        results = baseline_collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_queries} queries...")
    
    return latencies


def run_queries_tiered(manager, query_embeddings_normalized, k=5, threshold=0.75, num_queries=50):
    """Run queries against tiered system"""
    print("=" * 70)
    print(f"RUNNING QUERIES: TIERED SYSTEM ({num_queries} queries)")
    print("=" * 70)
    
    latencies = []
    
    for i in range(num_queries):
        query_emb = query_embeddings_normalized[i]
        start_time = time.time()
        
        # Query tiered system (expects numpy array)
        results = manager.retrieve_document(query_emb, k=k, threshold=threshold)
        
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_queries} queries...")
    
    return latencies


def save_query_log(latencies, filename, system_name):
    """Save query log to CSV"""
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'latency_ms', 'system'])
        for i, latency in enumerate(latencies):
            writer.writerow([i, latency, system_name])
    
    print(f"Query log saved to {filepath}")


def calculate_statistics(latencies):
    """Calculate latency statistics"""
    if not latencies:
        return {}
    
    sorted_lat = sorted(latencies)
    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'p95': sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 0 else 0,
        'p99': sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline vs tiered storage system')
    parser.add_argument('--queries', type=int, default=50, help='Number of queries to run (default: 50, limited by available queries)')
    parser.add_argument('--k', type=int, default=5, help='Number of results per query')
    parser.add_argument('--threshold', type=float, default=0.75, help='Similarity threshold')
    args = parser.parse_args()
    
    # Load data first to check available queries
    print("Loading data...")
    doc_embeddings = np.load("wiki_embeddings.npy")
    query_embeddings = np.load("query_embeddings.npy")
    doc_ids = [f"doc{i}" for i in range(len(doc_embeddings))]
    
    # Limit queries to available count
    available_queries = len(query_embeddings)
    num_queries = min(args.queries, available_queries)
    if args.queries > available_queries:
        print(f"Warning: Requested {args.queries} queries but only {available_queries} available. Using {num_queries} queries.")
    
    print("=" * 70)
    print("TIERED STORAGE EVALUATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Queries: {num_queries} (requested: {args.queries}, available: {available_queries})")
    print(f"  Results per query (k): {args.k}")
    print(f"  Similarity threshold: {args.threshold}")
    print()
    print(f"Loaded {len(doc_embeddings)} document embeddings")
    print(f"Loaded {len(query_embeddings)} query embeddings")
    print()
    
    # Normalize query embeddings
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings_normalized = np.divide(
        query_embeddings,
        query_norms,
        out=np.zeros_like(query_embeddings),
        where=(query_norms != 0)
    ).astype(np.float32)
    
    # Calculate storage
    baseline_disk_gb = calculate_baseline_storage(doc_embeddings)
    print(f"\nBaseline storage: {baseline_disk_gb:.4f} GB (all in Tier 2/local disk)")
    
    # Setup baseline system
    baseline_collection, baseline_client = setup_baseline_system(
        doc_embeddings, doc_ids, tier2_path='./tier2_baseline_db'
    )
    
    # Run baseline queries
    baseline_latencies = run_queries_baseline(
        baseline_collection, query_embeddings_normalized,
        k=args.k, threshold=args.threshold, num_queries=num_queries
    )
    
    baseline_stats = calculate_statistics(baseline_latencies)
    save_query_log(baseline_latencies, "query_log_baseline.csv", "baseline")
    
    print(f"\nBaseline Performance:")
    print(f"  Mean latency: {baseline_stats['mean']:.2f} ms")
    print(f"  Median latency: {baseline_stats['median']:.2f} ms")
    print(f"  P95 latency: {baseline_stats['p95']:.2f} ms")
    print(f"  P99 latency: {baseline_stats['p99']:.2f} ms")
    
    # Setup tiered system
    tiered_manager, tier_assignment = setup_tiered_system()
    
    # Calculate tiered storage
    tiered_ram_gb, tiered_tier2_gb, tiered_tier3_gb, t1_count, t2_count, t3_count = calculate_tiered_storage(
        tier_assignment, doc_embeddings
    )
    
    print(f"\nTiered storage breakdown:")
    print(f"  Tier 1 (RAM): {tiered_ram_gb:.4f} GB ({t1_count} documents)")
    print(f"  Tier 2 (Local disk): {tiered_tier2_gb:.4f} GB ({t2_count} documents)")
    print(f"  Tier 3 (Remote disk): {tiered_tier3_gb:.4f} GB ({t3_count} documents)")
    print(f"  Total local disk (Tier 2): {tiered_tier2_gb:.4f} GB")
    
    # Run tiered queries
    tiered_latencies = run_queries_tiered(
        tiered_manager, query_embeddings_normalized,
        k=args.k, threshold=args.threshold, num_queries=num_queries
    )
    
    tiered_stats = calculate_statistics(tiered_latencies)
    save_query_log(tiered_latencies, "query_log_tiered.csv", "tiered")
    
    print(f"\nTiered Performance:")
    print(f"  Mean latency: {tiered_stats['mean']:.2f} ms")
    print(f"  Median latency: {tiered_stats['median']:.2f} ms")
    print(f"  P95 latency: {tiered_stats['p95']:.2f} ms")
    print(f"  P99 latency: {tiered_stats['p99']:.2f} ms")
    
    # Calculate metrics
    local_disk_saved_gb = baseline_disk_gb - tiered_tier2_gb
    local_disk_saved_pct = (local_disk_saved_gb / baseline_disk_gb * 100) if baseline_disk_gb > 0 else 0
    
    # Performance retention: how much of baseline performance is retained
    # Lower latency = better, so retention = baseline_latency / tiered_latency * 100
    # > 100% means tiered is faster, < 100% means tiered is slower
    performance_retention = (baseline_stats['mean'] / tiered_stats['mean'] * 100) if tiered_stats['mean'] > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nStorage Comparison:")
    print(f"  Baseline (all Tier 2):     {baseline_disk_gb:.4f} GB local disk")
    print(f"  Tiered (Tier 2 only):       {tiered_tier2_gb:.4f} GB local disk")
    print(f"  Local disk saved:           {local_disk_saved_gb:.4f} GB ({local_disk_saved_pct:.2f}%)")
    print(f"  Tiered RAM overhead:        {tiered_ram_gb:.4f} GB ({t1_count} documents)")
    print(f"  Tiered remote storage:      {tiered_tier3_gb:.4f} GB ({t3_count} documents)")
    
    print(f"\nPerformance Comparison:")
    print(f"  Baseline mean latency:      {baseline_stats['mean']:.2f} ms")
    print(f"  Tiered mean latency:        {tiered_stats['mean']:.2f} ms")
    print(f"  Performance retention:      {performance_retention:.1f}%")
    if tiered_stats['mean'] < baseline_stats['mean']:
        improvement = ((baseline_stats['mean'] - tiered_stats['mean']) / baseline_stats['mean'] * 100)
        print(f"  Tiered is {improvement:.1f}% faster")
    else:
        degradation = ((tiered_stats['mean'] - baseline_stats['mean']) / baseline_stats['mean'] * 100)
        print(f"  Tiered is {degradation:.1f}% slower")
    
    print(f"\nEfficiency Summary:")
    print(f"  With {local_disk_saved_pct:.1f}% local disk reduction, tiered system")
    if tiered_stats['mean'] < baseline_stats['mean']:
        print(f"  achieves {100-performance_retention:.1f}% performance improvement")
    else:
        print(f"  maintains {performance_retention:.1f}% of baseline performance")
    print(f"  while using {tiered_ram_gb:.4f} GB RAM for hot data")
    print(f"  and {tiered_tier3_gb:.4f} GB remote storage for cold data")
    
    # Save summary
    os.makedirs("data", exist_ok=True)
    summary_path = "data/evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TIERED STORAGE EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Baseline System (All Tier 2):\n")
        f.write(f"  Local disk: {baseline_disk_gb:.4f} GB\n")
        f.write(f"  Mean latency: {baseline_stats['mean']:.2f} ms\n")
        f.write(f"  P95 latency: {baseline_stats['p95']:.2f} ms\n\n")
        f.write(f"Tiered System:\n")
        f.write(f"  Tier 1 (RAM): {tiered_ram_gb:.4f} GB ({t1_count} docs)\n")
        f.write(f"  Tier 2 (Local disk): {tiered_tier2_gb:.4f} GB ({t2_count} docs)\n")
        f.write(f"  Tier 3 (Remote disk): {tiered_tier3_gb:.4f} GB ({t3_count} docs)\n")
        f.write(f"  Mean latency: {tiered_stats['mean']:.2f} ms\n")
        f.write(f"  P95 latency: {tiered_stats['p95']:.2f} ms\n\n")
        f.write(f"Results:\n")
        f.write(f"  Local disk saved: {local_disk_saved_pct:.2f}%\n")
        f.write(f"  Performance retention: {performance_retention:.1f}%\n")
        f.write(f"  Baseline mean latency: {baseline_stats['mean']:.2f} ms\n")
        f.write(f"  Tiered mean latency: {tiered_stats['mean']:.2f} ms\n")
    
    print(f"\nSummary saved to {summary_path}")
    
    # Cleanup
    baseline_client.delete_collection("baseline_vectors")
    tiered_manager.close()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

