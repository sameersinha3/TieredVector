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
    print("SETTING UP BASELINE SYSTEM (All in Tier 2 - Local Disk)")
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


def setup_baseline_cloud_system(doc_embeddings, doc_ids):
    """Setup baseline: All documents in Tier 3 (remote CloudLab/cloud storage)"""
    print("=" * 70)
    print("SETTING UP BASELINE SYSTEM (All in Tier 3 - Cloud Storage)")
    print("=" * 70)
    
    # Clear Redis (baseline doesn't use Tier 1)
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    keys = r.keys("doc*")
    if keys:
        r.delete(*keys)
        print(f"Cleared {len(keys)} documents from Redis")
    
    # Clear local Tier 2 (baseline doesn't use local disk)
    tier2_path = './tier2_baseline_db'
    if os.path.exists(tier2_path):
        import shutil
        shutil.rmtree(tier2_path)
        print(f"Cleared local baseline database at {tier2_path}")
    
    # Connect to remote ChromaDB (Tier 3)
    vm_ip = os.getenv("VM_IP")
    if not vm_ip:
        raise ValueError("VM_IP not found in environment. Please set VM_IP in your .env file.")
    
    tier3_client = chromadb.HttpClient(host=vm_ip, port=8000)
    
    # Create or get baseline collection on remote
    baseline_collection = tier3_client.get_or_create_collection(
        name="baseline_cloud_vectors",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Clear existing documents in baseline collection
    try:
        existing_ids = baseline_collection.get()['ids']
        if existing_ids:
            baseline_collection.delete(ids=existing_ids)
            print(f"Cleared {len(existing_ids)} existing documents from remote baseline collection")
    except Exception as e:
        print(f"Note: Could not clear existing documents: {e}")
    
    # Normalize embeddings
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings_normalized = np.divide(
        doc_embeddings,
        doc_norms,
        out=np.zeros_like(doc_embeddings),
        where=(doc_norms != 0)
    ).astype(np.float32)
    
    # Store all documents in Tier 3 (remote)
    print(f"Storing {len(doc_ids)} documents in Tier 3 (remote cloud baseline)...")
    batch_size = 500
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i+batch_size]
        batch_embeddings = [doc_embeddings_normalized[j].tolist() for j in range(i, min(i+batch_size, len(doc_ids)))]
        baseline_collection.upsert(ids=batch_ids, embeddings=batch_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Stored {min(i+batch_size, len(doc_ids))}/{len(doc_ids)} documents...")
    
    print(f"Cloud baseline setup complete: {len(doc_ids)} documents in Tier 3 (remote)")
    return baseline_collection, tier3_client


def setup_tiered_system(doc_embeddings):
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
    
    # Initialize StorageManager
    manager = StorageManager(
        tier3_host=os.getenv("VM_IP"),
        tier3_port=8000,
        tier3_collection="cold_vectors"
    )
    
    if not manager.initialize():
        raise RuntimeError("Failed to initialize tiered storage manager")
    
    # Repopulate Tier 1 (Redis) - baseline system cleared it
    print("\nRepopulating Tier 1 (Redis)...")
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    keys = r.keys("doc*")
    if keys:
        r.delete(*keys)
        print(f"  Cleared {len(keys)} existing documents from Redis")
    
    # Normalize embeddings
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings_normalized = np.divide(
        doc_embeddings,
        doc_norms,
        out=np.zeros_like(doc_embeddings),
        where=(doc_norms != 0)
    ).astype(np.float32)
    
    # Populate Tier 1
    tier1_indices = np.where(tier_assignment == 1)[0]
    for idx in tier1_indices:
        r.set(f'doc{idx}', pickle.dumps(doc_embeddings_normalized[idx]))
    print(f"  Populated {len(tier1_indices)} documents in Tier 1 (Redis)")
    
    # Note: Tier 2 and Tier 3 should already be populated from simulate_temperature.py
    # We don't clear them to avoid re-population overhead
    
    return manager, tier_assignment


def calculate_baseline_storage(doc_embeddings, baseline_type='local'):
    """Calculate baseline storage"""
    # Estimate storage for all documents
    # Each embedding: 768 dims * 4 bytes (float32) = 3072 bytes
    # Plus overhead: ~20% for ChromaDB metadata/index
    embedding_size = len(doc_embeddings) * 768 * 4  # bytes
    storage_size_gb = (embedding_size * 1.2) / (1024 ** 3)  # Add 20% overhead
    
    if baseline_type == 'local':
        # Local disk storage (Tier 2)
        return storage_size_gb, 0.0  # (local_disk_gb, remote_disk_gb)
    else:  # cloud
        # Remote cloud storage (Tier 3)
        return 0.0, storage_size_gb  # (local_disk_gb, remote_disk_gb)


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


def run_queries_baseline(baseline_collection, query_embeddings_normalized, k=5, threshold=0.75, num_queries=50, baseline_type='local'):
    """Run queries against baseline system"""
    print("=" * 70)
    baseline_name = "LOCAL DISK" if baseline_type == 'local' else "CLOUD STORAGE"
    print(f"RUNNING QUERIES: BASELINE SYSTEM ({baseline_name}) - {num_queries} queries")
    print("=" * 70)
    
    latencies = []
    
    for i in range(num_queries):
        query_emb = query_embeddings_normalized[i]
        start_time = time.time()
        
        # Query baseline collection (Tier 2 for local, Tier 3 for cloud)
        results = baseline_collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_queries} queries...")
    
    return latencies


def run_queries_tiered(manager, query_embeddings_normalized, k=5, threshold=0.75, num_queries=50, query_texts=None):
    """Run queries against tiered system"""
    print("=" * 70)
    print(f"RUNNING QUERIES: TIERED SYSTEM ({num_queries} queries)")
    print("=" * 70)
    
    latencies = []
    tier_hits = {"T1": 0, "T2": 0, "T3": 0}  # Track tier access
    query_results = []  # Store first 5 queries with results
    
    for i in range(num_queries):
        query_emb = query_embeddings_normalized[i]
        start_time = time.time()
        
        # Query tiered system (expects numpy array)
        results = manager.retrieve_document(query_emb, k=k, threshold=threshold)
        
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        # Track tier hits
        if results:
            for result in results:
                source = result.get("source", "")
                if "T1" in source or "redis" in source.lower():
                    tier_hits["T1"] += 1
                elif "T2" in source or "local" in source.lower():
                    tier_hits["T2"] += 1
                elif "T3" in source or "remote" in source.lower():
                    tier_hits["T3"] += 1
        
        # Store first 5 queries with results
        if i < 5:
            query_text = query_texts[i] if query_texts and i < len(query_texts) else f"Query {i+1}"
            query_results.append({
                "query_id": i,
                "query_text": query_text,
                "results": results if results else [],
                "latency_ms": latency_ms
            })
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_queries} queries...")
    
    return latencies, tier_hits, query_results


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
    parser.add_argument('--threshold', type=float, default=0.75, help='Similarity threshold (used if --multi-threshold not set)')
    parser.add_argument('--baseline-type', type=str, default='cloud', choices=['local', 'cloud'],
                       help='Baseline type: "local" (all in Tier 2/local disk) or "cloud" (all in Tier 3/cloud storage)')
    parser.add_argument('--multi-threshold', action='store_true', 
                       help='Run evaluation with multiple thresholds (0.6, 0.75, 0.9)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations after evaluation (requires matplotlib)')
    parser.add_argument('--eval-queries', action='store_true',
                       help='Use separate evaluation query set (query_embeddings_eval.npy) instead of temperature simulation queries')
    args = parser.parse_args()
    
    # Determine thresholds to test
    if args.multi_threshold:
        thresholds = [0.6, 0.75, 0.9]
    else:
        thresholds = [args.threshold]
    
    # Load data first to check available queries
    print("Loading data...")
    doc_embeddings = np.load("wiki_embeddings.npy")
    
    # Choose which query set to use
    if args.eval_queries:
        eval_query_file = "query_embeddings_eval.npy"
        if not os.path.exists(eval_query_file):
            print(f"Error: Evaluation query file not found: {eval_query_file}")
            print("Please run: python load_eval_queries.py first")
            return
        query_embeddings = np.load(eval_query_file)
        print(f"Using evaluation query set: {len(query_embeddings)} queries")
    else:
        query_embeddings = np.load("query_embeddings.npy")
        print(f"Using temperature simulation query set: {len(query_embeddings)} queries")
    
    doc_ids = [f"doc{i}" for i in range(len(doc_embeddings))]
    
    # Load query texts if available
    query_texts = None
    try:
        # Try to load from cache file
        if os.path.exists("query_embeddings_cache.npz"):
            cache = np.load("query_embeddings_cache.npz", allow_pickle=True)
            query_texts = cache['queries'].tolist()
            print(f"Loaded {len(query_texts)} query texts from cache")
        else:
            # Try to load from evaluation queries JSON file
            if args.eval_queries and os.path.exists("query_embeddings_eval_queries.json"):
                import json
                with open("query_embeddings_eval_queries.json", 'r') as f:
                    query_texts = json.load(f)
                print(f"Loaded {len(query_texts)} query texts from evaluation queries file")
            else:
                # Load from dataset
                from datasets import load_dataset
                import itertools
                query_dataset = load_dataset("natural_questions", split="train", streaming=True)
                # Use appropriate offset for evaluation queries
                start_offset = 1000 if args.eval_queries else 0
                query_texts = [entry["question"]["text"] for entry in itertools.islice(query_dataset, start_offset, start_offset + len(query_embeddings))]
                print(f"Loaded {len(query_texts)} query texts from dataset")
    except Exception as e:
        print(f"Warning: Could not load query texts: {e}")
        query_texts = None
    
    # Load document metadata (title, text) for display
    doc_metadata = {}
    try:
        from datasets import load_dataset
        print("Loading document metadata (titles and text)...")
        doc_dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
        # Create mapping from doc index to metadata
        for idx in range(min(len(doc_embeddings), len(doc_dataset))):
            doc_metadata[f"doc{idx}"] = {
                "title": doc_dataset[idx]["title"],
                "text": doc_dataset[idx]["text"][:200] + "..." if len(doc_dataset[idx]["text"]) > 200 else doc_dataset[idx]["text"]
            }
        print(f"Loaded metadata for {len(doc_metadata)} documents")
    except Exception as e:
        print(f"Warning: Could not load document metadata: {e}")
        doc_metadata = {}
    
    # Limit queries to available count
    available_queries = len(query_embeddings)
    num_queries = min(args.queries, available_queries)
    if args.queries > available_queries:
        print(f"Warning: Requested {args.queries} queries but only {available_queries} available. Using {num_queries} queries.")
    
    print("=" * 70)
    print("TIERED STORAGE EVALUATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Baseline type: {args.baseline_type.upper()}")
    print(f"  Queries: {num_queries} (requested: {args.queries}, available: {available_queries})")
    print(f"  Results per query (k): {args.k}")
    if args.multi_threshold:
        print(f"  Similarity thresholds: {thresholds}")
    else:
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
    
    # Calculate storage (same for all thresholds)
    baseline_local_gb, baseline_remote_gb = calculate_baseline_storage(doc_embeddings, args.baseline_type)
    if args.baseline_type == 'local':
        print(f"\nBaseline storage: {baseline_local_gb:.4f} GB (all in Tier 2/local disk)")
    else:
        print(f"\nBaseline storage: {baseline_remote_gb:.4f} GB (all in Tier 3/cloud storage)")
    
    # Setup baseline system (once, reused for all thresholds)
    if args.baseline_type == 'local':
        baseline_collection, baseline_client = setup_baseline_system(
            doc_embeddings, doc_ids, tier2_path='./tier2_baseline_db'
        )
    else:  # cloud
        baseline_collection, baseline_client = setup_baseline_cloud_system(
            doc_embeddings, doc_ids
        )
    
    # Setup tiered system (once, reused for all thresholds)
    tiered_manager, tier_assignment = setup_tiered_system(doc_embeddings)
    
    # Calculate tiered storage (same for all thresholds)
    tiered_ram_gb, tiered_tier2_gb, tiered_tier3_gb, t1_count, t2_count, t3_count = calculate_tiered_storage(
        tier_assignment, doc_embeddings
    )
    
    print(f"\nTiered storage breakdown:")
    print(f"  Tier 1 (RAM): {tiered_ram_gb:.4f} GB ({t1_count} documents)")
    print(f"  Tier 2 (Local disk): {tiered_tier2_gb:.4f} GB ({t2_count} documents)")
    print(f"  Tier 3 (Remote disk): {tiered_tier3_gb:.4f} GB ({t3_count} documents)")
    print(f"  Total local disk (Tier 2): {tiered_tier2_gb:.4f} GB")
    
    # Storage comparison (same for all thresholds)
    if args.baseline_type == 'local':
        local_disk_saved_gb = baseline_local_gb - tiered_tier2_gb
        local_disk_saved_pct = (local_disk_saved_gb / baseline_local_gb * 100) if baseline_local_gb > 0 else 0
        remote_storage_added_gb = tiered_tier3_gb
    else:  # cloud
        remote_storage_saved_gb = baseline_remote_gb - tiered_tier3_gb
        remote_storage_saved_pct = (remote_storage_saved_gb / baseline_remote_gb * 100) if baseline_remote_gb > 0 else 0
        local_disk_added_gb = tiered_tier2_gb
    
    # Store results for all thresholds
    all_results = []
    
    # Run evaluation for each threshold
    for threshold in thresholds:
        print("\n" + "=" * 70)
        print(f"EVALUATING WITH THRESHOLD: {threshold}")
        print("=" * 70)
        
        # Run baseline queries
        baseline_latencies = run_queries_baseline(
            baseline_collection, query_embeddings_normalized,
            k=args.k, threshold=threshold, num_queries=num_queries,
            baseline_type=args.baseline_type
        )
        
        baseline_stats = calculate_statistics(baseline_latencies)
        baseline_log_name = f"query_log_baseline_{args.baseline_type}_th{threshold}.csv"
        save_query_log(baseline_latencies, baseline_log_name, f"baseline-{args.baseline_type}-th{threshold}")
        
        baseline_name = "LOCAL DISK" if args.baseline_type == 'local' else "CLOUD STORAGE"
        print(f"\nBaseline Performance ({baseline_name}, threshold={threshold}):")
        print(f"  Mean latency: {baseline_stats['mean']:.2f} ms")
        print(f"  Median latency: {baseline_stats['median']:.2f} ms")
        print(f"  P95 latency: {baseline_stats['p95']:.2f} ms")
        print(f"  P99 latency: {baseline_stats['p99']:.2f} ms")
        
        # Run tiered queries
        tiered_latencies, tier_hits, query_results = run_queries_tiered(
            tiered_manager, query_embeddings_normalized,
            k=args.k, threshold=threshold, num_queries=num_queries,
            query_texts=query_texts
        )
        
        # Enrich query results with document metadata
        for qr in query_results:
            for result in qr['results']:
                doc_id = result.get('id', '')
                if doc_id in doc_metadata:
                    result['title'] = doc_metadata[doc_id]['title']
                    result['text_snippet'] = doc_metadata[doc_id]['text']
        
        tiered_stats = calculate_statistics(tiered_latencies)
        save_query_log(tiered_latencies, f"query_log_tiered_th{threshold}.csv", f"tiered-th{threshold}")
        
        print(f"\nTiered Performance (threshold={threshold}):")
        print(f"  Mean latency: {tiered_stats['mean']:.2f} ms")
        print(f"  Median latency: {tiered_stats['median']:.2f} ms")
        print(f"  P95 latency: {tiered_stats['p95']:.2f} ms")
        print(f"  P99 latency: {tiered_stats['p99']:.2f} ms")
        
        # Calculate average results per query
        avg_results = sum(len(qr['results']) for qr in query_results) / len(query_results) if query_results else 0
        avg_similarity = 0
        total_results = 0
        for qr in query_results:
            for result in qr['results']:
                avg_similarity += result.get('score', 0)
                total_results += 1
        if total_results > 0:
            avg_similarity /= total_results
        
        # Performance retention
        performance_retention = (baseline_stats['mean'] / tiered_stats['mean'] * 100) if tiered_stats['mean'] > 0 else 0
        
        # Store results
        total_hits = sum(tier_hits.values())
        all_results.append({
            'threshold': threshold,
            'baseline_stats': baseline_stats,
            'tiered_stats': tiered_stats,
            'tier_hits': tier_hits,
            'total_hits': total_hits,
            'query_results': query_results,
            'avg_results': avg_results,
            'avg_similarity': avg_similarity,
            'performance_retention': performance_retention
        })
        
        # Print tier access statistics
        print(f"\nTier Access Statistics (threshold={threshold}):")
        if total_hits > 0:
            print(f"  Tier 1 (RAM) hits:        {tier_hits['T1']} ({tier_hits['T1']/total_hits*100:.1f}%)")
            print(f"  Tier 2 (Local disk) hits: {tier_hits['T2']} ({tier_hits['T2']/total_hits*100:.1f}%)")
            print(f"  Tier 3 (Remote disk) hits: {tier_hits['T3']} ({tier_hits['T3']/total_hits*100:.1f}%)")
            print(f"  Average results per query: {avg_results:.2f}")
            print(f"  Average similarity score:  {avg_similarity:.4f}")
        else:
            print(f"  No tier hits recorded")
    
    # Print comprehensive comparison
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION RESULTS - THRESHOLD COMPARISON")
    print("=" * 70)
    
    print(f"\nStorage Comparison (same for all thresholds):")
    if args.baseline_type == 'local':
        print(f"  Baseline (all Tier 2):        {baseline_local_gb:.4f} GB local disk")
        print(f"  Tiered (Tier 2 only):         {tiered_tier2_gb:.4f} GB local disk")
        print(f"  Local disk saved:             {local_disk_saved_gb:.4f} GB ({local_disk_saved_pct:.2f}%)")
        print(f"  Tiered RAM overhead:          {tiered_ram_gb:.4f} GB ({t1_count} documents)")
        print(f"  Tiered remote storage:        {tiered_tier3_gb:.4f} GB ({t3_count} documents)")
    else:  # cloud
        print(f"  Baseline (all Tier 3):        {baseline_remote_gb:.4f} GB cloud storage")
        print(f"  Tiered (Tier 3 only):         {tiered_tier3_gb:.4f} GB cloud storage")
        print(f"  Cloud storage saved:          {remote_storage_saved_gb:.4f} GB ({remote_storage_saved_pct:.2f}%)")
        print(f"  Tiered RAM overhead:          {tiered_ram_gb:.4f} GB ({t1_count} documents)")
        print(f"  Tiered local disk:            {tiered_tier2_gb:.4f} GB ({t2_count} documents)")
    
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON BY THRESHOLD")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Baseline Mean':<15} {'Tiered Mean':<15} {'Retention':<12} {'Avg Results':<12} {'Avg Similarity':<15} {'T1%':<8} {'T2%':<8} {'T3%':<8}")
    print("-" * 110)
    for result in all_results:
        th = result['threshold']
        bl_mean = result['baseline_stats']['mean']
        tr_mean = result['tiered_stats']['mean']
        ret = result['performance_retention']
        avg_res = result['avg_results']
        avg_sim = result['avg_similarity']
        total_h = result['total_hits']
        if total_h > 0:
            t1_pct = result['tier_hits']['T1'] / total_h * 100
            t2_pct = result['tier_hits']['T2'] / total_h * 100
            t3_pct = result['tier_hits']['T3'] / total_h * 100
        else:
            t1_pct = t2_pct = t3_pct = 0
        print(f"{th:<12.2f} {bl_mean:<15.2f} {tr_mean:<15.2f} {ret:<12.1f} {avg_res:<12.2f} {avg_sim:<15.4f} {t1_pct:<8.1f} {t2_pct:<8.1f} {t3_pct:<8.1f}")
    
    # Detailed breakdown for each threshold
    print(f"\n" + "=" * 70)
    print("DETAILED BREAKDOWN BY THRESHOLD")
    print("=" * 70)
    for result in all_results:
        th = result['threshold']
        print(f"\nThreshold: {th}")
        print(f"  Baseline: Mean={result['baseline_stats']['mean']:.2f}ms, P95={result['baseline_stats']['p95']:.2f}ms")
        print(f"  Tiered:   Mean={result['tiered_stats']['mean']:.2f}ms, P95={result['tiered_stats']['p95']:.2f}ms")
        print(f"  Performance retention: {result['performance_retention']:.1f}%")
        print(f"  Average results per query: {result['avg_results']:.2f}")
        print(f"  Average similarity score: {result['avg_similarity']:.4f}")
        if result['total_hits'] > 0:
            print(f"  Tier distribution: T1={result['tier_hits']['T1']/result['total_hits']*100:.1f}%, "
                  f"T2={result['tier_hits']['T2']/result['total_hits']*100:.1f}%, "
                  f"T3={result['tier_hits']['T3']/result['total_hits']*100:.1f}%")
    
    # Show sample queries from first threshold
    if all_results:
        print(f"\n" + "=" * 70)
        print("SAMPLE QUERIES AND RETRIEVED DOCUMENTS (First 5, Threshold={})".format(all_results[0]['threshold']))
        print("=" * 70)
        for qr in all_results[0]['query_results'][:5]:
            print(f"\nQuery {qr['query_id']+1}: {qr['query_text']}")
            print(f"  Latency: {qr['latency_ms']:.2f} ms")
            print(f"  Retrieved {len(qr['results'])} documents:")
            for idx, result in enumerate(qr['results'][:5], 1):
                doc_id = result.get('id', 'N/A')
                score = result.get('score', 0)
                source = result.get('source', 'N/A')
                title = result.get('title', 'N/A')
                text_snippet = result.get('text_snippet', '')
                print(f"    {idx}. {doc_id} - {title}")
                print(f"       Score: {score:.4f} | Source: {source}")
                if text_snippet:
                    print(f"       Text: {text_snippet}")
    
    # Save comprehensive summary
    os.makedirs("data", exist_ok=True)
    summary_path = "data/evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TIERED STORAGE EVALUATION SUMMARY - MULTI-THRESHOLD ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        baseline_label = "All Tier 2 (Local Disk)" if args.baseline_type == 'local' else "All Tier 3 (Cloud Storage)"
        f.write(f"Baseline System ({baseline_label}):\n")
        if args.baseline_type == 'local':
            f.write(f"  Local disk: {baseline_local_gb:.4f} GB\n")
        else:
            f.write(f"  Cloud storage: {baseline_remote_gb:.4f} GB\n")
        f.write(f"\nTiered System:\n")
        f.write(f"  Tier 1 (RAM): {tiered_ram_gb:.4f} GB ({t1_count} docs)\n")
        f.write(f"  Tier 2 (Local disk): {tiered_tier2_gb:.4f} GB ({t2_count} docs)\n")
        f.write(f"  Tier 3 (Remote disk): {tiered_tier3_gb:.4f} GB ({t3_count} docs)\n")
        if args.baseline_type == 'local':
            f.write(f"  Local disk saved: {local_disk_saved_pct:.2f}%\n")
        else:
            f.write(f"  Cloud storage saved: {remote_storage_saved_pct:.2f}%\n")
        f.write(f"\n" + "=" * 70 + "\n")
        f.write("PERFORMANCE BY THRESHOLD\n")
        f.write("=" * 70 + "\n\n")
        for result in all_results:
            th = result['threshold']
            f.write(f"Threshold: {th}\n")
            f.write(f"  Baseline: Mean={result['baseline_stats']['mean']:.2f}ms, P95={result['baseline_stats']['p95']:.2f}ms\n")
            f.write(f"  Tiered:   Mean={result['tiered_stats']['mean']:.2f}ms, P95={result['tiered_stats']['p95']:.2f}ms\n")
            f.write(f"  Performance retention: {result['performance_retention']:.1f}%\n")
            f.write(f"  Average results per query: {result['avg_results']:.2f}\n")
            f.write(f"  Average similarity score: {result['avg_similarity']:.4f}\n")
            if result['total_hits'] > 0:
                f.write(f"  Tier distribution: T1={result['tier_hits']['T1']/result['total_hits']*100:.1f}%, "
                       f"T2={result['tier_hits']['T2']/result['total_hits']*100:.1f}%, "
                       f"T3={result['tier_hits']['T3']/result['total_hits']*100:.1f}%\n")
            f.write("\n")
    
    print(f"\nSummary saved to {summary_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        try:
            from visualize_results import main as generate_visualizations
            generate_visualizations()
        except ImportError:
            print("Warning: matplotlib not available. Skipping visualization generation.")
            print("Install with: pip install matplotlib")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    # Cleanup
    if args.baseline_type == 'local':
        baseline_client.delete_collection("baseline_vectors")
    else:  # cloud
        baseline_client.delete_collection("baseline_cloud_vectors")
    tiered_manager.close()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

