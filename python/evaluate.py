"""
Evaluation script for Option 2: Same Performance → Cost Reductions

This script:
1. Sets up a baseline system (all data in Tier 1/Redis)
2. Sets up a tiered system (temperature-based tiering)
3. Runs queries against both systems
4. Ensures both meet performance targets
5. Calculates and compares costs
"""
import os
import sys
import numpy as np
import pickle
import redis
import chromadb
import time
import yaml
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from storage_manager import StorageManager
from query_logger import QueryLogger, QueryTimer

load_dotenv()

# Configuration
COST_CFG = "config/cost.yaml"
QUERY_LOG_BASELINE = "data/query_log_baseline.csv"
QUERY_LOG_TIERED = "data/query_log_tiered.csv"
EMBEDDING_DIM = 768  # Cohere multilingual-22-12 embedding dimension


def load_config():
    """Load cost and performance configuration."""
    with open(COST_CFG, 'r') as f:
        return yaml.safe_load(f)


def estimate_embedding_size(embedding: np.ndarray) -> float:
    """Estimate size of embedding in bytes."""
    # Embedding vector + overhead
    return embedding.nbytes + 100  # ~100 bytes overhead for metadata


def setup_baseline_system(doc_embeddings: np.ndarray, 
                         doc_ids: List[str],
                         redis_client: redis.Redis) -> Dict:
    """
    Setup baseline system: all documents in Tier 1 (Redis).
    Returns storage statistics.
    """
    print("=" * 70)
    print("SETTING UP BASELINE SYSTEM (All in Tier 1/Redis)")
    print("=" * 70)
    
    total_bytes = 0
    stored_count = 0
    
    for i, embedding in enumerate(doc_embeddings):
        doc_id = doc_ids[i] if i < len(doc_ids) else f"doc{i}"
        key = f"doc{i}"
        
        # Store in Redis
        value = pickle.dumps(embedding.astype(np.float32))
        redis_client.set(key, value)
        
        total_bytes += len(value)
        stored_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Stored {i + 1}/{len(doc_embeddings)} documents...")
    
    ram_gb = total_bytes / (1024 ** 3)
    print(f"\nBaseline system ready:")
    print(f"  Documents stored: {stored_count}")
    print(f"  Total size: {ram_gb:.4f} GB")
    
    return {
        "ram_gb": ram_gb,
        "ssd_gb": 0.0,
        "blob_gb": 0.0,
        "doc_count": stored_count
    }


def setup_tiered_system(tier_results_path: str = None,
                       redis_client: redis.Redis = None) -> StorageManager:
    """
    Setup tiered system using temperature-based tiering.
    Returns configured StorageManager.
    """
    print("=" * 70)
    print("SETTING UP TIERED SYSTEM (Temperature-based)")
    print("=" * 70)
    
    # Try to find tier_results.npz in common locations
    if tier_results_path is None:
        possible_paths = [
            "tier_results.npz",  # Current directory
            "python/tier_results.npz",  # From project root
            "../tier_results.npz",  # From python/ subdirectory
        ]
        for path in possible_paths:
            if os.path.exists(path):
                tier_results_path = path
                break
        
        if tier_results_path is None:
            raise FileNotFoundError(
                f"Tier results not found. Checked: {possible_paths}. "
                "Please run simulate_temperature.py first."
            )
    
    # Load tier assignments from simulate_temperature.py output
    if not os.path.exists(tier_results_path):
        raise FileNotFoundError(
            f"Tier results not found at {tier_results_path}. "
            "Please run simulate_temperature.py first."
        )
    
    tier_data = np.load(tier_results_path)
    tier_assignment = tier_data['tier_assignment']
    
    # Initialize storage manager
    storage_manager = StorageManager(
        redis_host='localhost',
        redis_port=6379,
        tier2_path='./tier2_chroma_db',
        tier3_host=os.getenv("VM_IP"),
        tier3_port=8000,
        tier3_collection='cold_vectors'
    )
    
    if not storage_manager.initialize():
        raise RuntimeError("Failed to initialize tiered storage manager")
    
    print(f"Tiered system initialized")
    print(f"  Tier 1 count: {np.sum(tier_assignment == 1)}")
    print(f"  Tier 2 count: {np.sum(tier_assignment == 2)}")
    print(f"  Tier 3 count: {np.sum(tier_assignment == 3)}")
    
    return storage_manager


def run_query_baseline(query_embedding: np.ndarray,
                      k: int,
                      threshold: float,
                      redis_client: redis.Redis,
                      timer: QueryTimer) -> Tuple[List[Dict], List[int], Dict]:
    """
    Run query against baseline system (all in Redis).
    Returns (results, tier_hits, metrics).
    """
    timer.start_tier(1)
    
    # Normalize query
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        return [], [], {"ops_t1": 0, "bytes_t1": 0, "latency_t1_ms": 0}
    
    query_norm = (query_embedding / norm).astype(np.float32)
    
    # Scan all keys in Redis
    keys = redis_client.keys("doc*")
    candidates = []
    ops_t1 = len(keys)
    bytes_t1 = 0
    
    for key in keys:
        emb_bytes = redis_client.get(key)
        if emb_bytes:
            emb_norm = pickle.loads(emb_bytes)
            score = np.dot(query_norm, emb_norm)
            
            if score >= threshold:
                doc_id = key.decode() if isinstance(key, bytes) else key
                candidates.append((score, doc_id))
            
            bytes_t1 += len(emb_bytes)
    
    candidates.sort(reverse=True, key=lambda x: x[0])
    results = [{"id": did, "score": s, "source": "redis (T1)"} 
               for s, did in candidates[:k]]
    
    latency_t1_ms = timer.end_tier(1)
    
    tier_hits = [1] if results else []
    
    metrics = {
        "ops_t1": ops_t1,
        "bytes_t1": bytes_t1,
        "latency_t1_ms": latency_t1_ms
    }
    
    return results, tier_hits, metrics


def run_query_tiered(query_embedding: np.ndarray,
                    k: int,
                    threshold: float,
                    storage_manager: StorageManager,
                    timer: QueryTimer) -> Tuple[List[Dict], List[int], Dict]:
    """
    Run query against tiered system.
    Returns (results, tier_hits, metrics).
    """
    tier_hits = []
    metrics = {
        "ops_t1": 0, "ops_t2": 0, "ops_t3": 0,
        "bytes_t1": 0, "bytes_t2": 0, "bytes_t3": 0,
        "latency_t1_ms": 0, "latency_t2_ms": 0, "latency_t3_ms": 0
    }
    
    # Normalize query
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        return [], [], metrics
    
    query_norm = (query_embedding / norm).astype(np.float32)
    query_list = query_norm.tolist()
    
    retrieved = []
    retrieved_ids = set()
    
    # Tier 1: Redis
    if storage_manager.redis_client:
        timer.start_tier(1)
        try:
            keys = storage_manager.redis_client.keys("doc*")
            candidates = []
            metrics["ops_t1"] = len(keys)
            
            for key in keys:
                emb_bytes = storage_manager.redis_client.get(key)
                if emb_bytes:
                    emb_norm = pickle.loads(emb_bytes)
                    score = np.dot(query_norm, emb_norm)
                    
                    if score >= threshold:
                        doc_id = key.decode() if isinstance(key, bytes) else key
                        candidates.append((score, doc_id))
                    
                    metrics["bytes_t1"] += len(emb_bytes)
            
            candidates.sort(reverse=True, key=lambda x: x[0])
            for s, did in candidates:
                if len(retrieved) < k and did not in retrieved_ids:
                    retrieved.append({"id": did, "score": s, "source": "redis (T1)"})
                    retrieved_ids.add(did)
            
            if retrieved:
                tier_hits.append(1)
            
            metrics["latency_t1_ms"] = timer.end_tier(1)
            
            if len(retrieved) >= k:
                return retrieved, tier_hits, metrics
        except Exception as e:
            print(f"[Tier 1 error] {e}")
            metrics["latency_t1_ms"] = timer.end_tier(1)
    
    # Tier 2: Local ChromaDB
    remaining = k - len(retrieved)
    if remaining > 0 and storage_manager.tier2_collection:
        timer.start_tier(2)
        try:
            results = storage_manager.tier2_collection.query(
                query_embeddings=[query_list],
                n_results=remaining
            )
            
            ids = results['ids'][0]
            distances = results['distances'][0]
            
            metrics["ops_t2"] = len(ids)
            
            for doc_id, dist in zip(ids, distances):
                if doc_id in retrieved_ids:
                    continue
                
                score = 1.0 - dist
                if score >= threshold:
                    if len(retrieved) < k:
                        retrieved.append({"id": doc_id, "score": score, "source": "chroma_local (T2)"})
                        retrieved_ids.add(doc_id)
                        # Estimate bytes: embedding size
                        metrics["bytes_t2"] += estimate_embedding_size(query_norm)
            
            if ids:
                tier_hits.append(2)
            
            metrics["latency_t2_ms"] = timer.end_tier(2)
            
            if len(retrieved) >= k:
                return retrieved, tier_hits, metrics
        except Exception as e:
            print(f"[Tier 2 error] {e}")
            metrics["latency_t2_ms"] = timer.end_tier(2)
    
    # Tier 3: Remote ChromaDB
    remaining = k - len(retrieved)
    if remaining > 0 and storage_manager.tier3_collection:
        timer.start_tier(3)
        try:
            results = storage_manager.tier3_collection.query(
                query_embeddings=[query_list],
                n_results=remaining
            )
            
            ids = results['ids'][0]
            distances = results['distances'][0]
            
            metrics["ops_t3"] = len(ids)
            
            for doc_id, dist in zip(ids, distances):
                if doc_id in retrieved_ids:
                    continue
                
                score = 1.0 - dist
                if score >= threshold:
                    if len(retrieved) < k:
                        retrieved.append({"id": doc_id, "score": score, "source": "chroma_remote (T3)"})
                        retrieved_ids.add(doc_id)
                        # Estimate bytes: embedding size + network overhead
                        metrics["bytes_t3"] += estimate_embedding_size(query_norm) * 1.1  # 10% network overhead
            
            if ids:
                tier_hits.append(3)
            
            metrics["latency_t3_ms"] = timer.end_tier(3)
        except Exception as e:
            print(f"[Tier 3 error] {e}")
            metrics["latency_t3_ms"] = timer.end_tier(3)
    
    return retrieved, tier_hits, metrics


def run_evaluation(query_embeddings: np.ndarray,
                  config: Dict,
                  num_queries: int = 100,
                  k: int = 5,
                  threshold: float = 0.75):
    """
    Run evaluation comparing baseline vs tiered system.
    """
    print("=" * 70)
    print("EVALUATION: Same Performance → Cost Reductions")
    print("=" * 70)
    print(f"Running {num_queries} queries with k={k}, threshold={threshold}")
    print()
    
    # Load document embeddings and IDs
    doc_embeddings_path = "wiki_embeddings.npy"
    if not os.path.exists(doc_embeddings_path):
        raise FileNotFoundError(f"Document embeddings not found at {doc_embeddings_path}")
    
    doc_embeddings = np.load(doc_embeddings_path)[:10000]  # Use first 10k
    doc_ids = [f"doc{i}" for i in range(len(doc_embeddings))]
    
    # Connect to Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    try:
        redis_client.ping()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Redis: {e}")
    
    # Setup baseline system
    baseline_stats = setup_baseline_system(doc_embeddings, doc_ids, redis_client)
    
    # Setup tiered system
    tiered_manager = setup_tiered_system()
    
    # Initialize loggers
    baseline_logger = QueryLogger(QUERY_LOG_BASELINE)
    tiered_logger = QueryLogger(QUERY_LOG_TIERED)
    
    # Run queries against baseline
    print("\n" + "=" * 70)
    print("RUNNING QUERIES: BASELINE SYSTEM")
    print("=" * 70)
    
    baseline_latencies = []
    for i in range(min(num_queries, len(query_embeddings))):
        query_emb = query_embeddings[i]
        
        with QueryTimer() as timer:
            results, tier_hits, metrics = run_query_baseline(
                query_emb, k, threshold, redis_client, timer
            )
        
        latency_ms = timer.get_total_ms()
        baseline_latencies.append(latency_ms)
        
        baseline_logger.log_query(
            query_id=i,
            latency_ms=latency_ms,
            tier_hits=tier_hits,
            ops_t1=metrics["ops_t1"],
            bytes_t1=metrics["bytes_t1"],
            latency_t1_ms=metrics["latency_t1_ms"],
            results_count=len(results),
            system_type="baseline"
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_queries} queries...")
    
    # Calculate baseline performance
    baseline_latencies.sort()
    baseline_p95 = baseline_latencies[int(len(baseline_latencies) * 0.95)]
    baseline_p99 = baseline_latencies[int(len(baseline_latencies) * 0.99)]
    baseline_mean = np.mean(baseline_latencies)
    
    print(f"\nBaseline Performance:")
    print(f"  Mean latency: {baseline_mean:.2f} ms")
    print(f"  P95 latency: {baseline_p95:.2f} ms")
    print(f"  P99 latency: {baseline_p99:.2f} ms")
    
    # Performance targets
    targets = config.get("performance_targets", {})
    p95_target = targets.get("p95_latency_ms", 200.0)
    p99_target = targets.get("p99_latency_ms", 500.0)
    mean_target = targets.get("mean_latency_ms", 150.0)
    
    print(f"\nPerformance Targets:")
    print(f"  Mean < {mean_target} ms: {'✓' if baseline_mean < mean_target else '✗'}")
    print(f"  P95 < {p95_target} ms: {'✓' if baseline_p95 < p95_target else '✗'}")
    print(f"  P99 < {p99_target} ms: {'✓' if baseline_p99 < p99_target else '✗'}")
    
    # Run queries against tiered system
    print("\n" + "=" * 70)
    print("RUNNING QUERIES: TIERED SYSTEM")
    print("=" * 70)
    
    tiered_latencies = []
    for i in range(min(num_queries, len(query_embeddings))):
        query_emb = query_embeddings[i]
        
        with QueryTimer() as timer:
            results, tier_hits, metrics = run_query_tiered(
                query_emb, k, threshold, tiered_manager, timer
            )
        
        latency_ms = timer.get_total_ms()
        tiered_latencies.append(latency_ms)
        
        tiered_logger.log_query(
            query_id=i,
            latency_ms=latency_ms,
            tier_hits=tier_hits,
            ops_t1=metrics["ops_t1"],
            ops_t2=metrics["ops_t2"],
            ops_t3=metrics["ops_t3"],
            bytes_t1=metrics["bytes_t1"],
            bytes_t2=metrics["bytes_t2"],
            bytes_t3=metrics["bytes_t3"],
            latency_t1_ms=metrics["latency_t1_ms"] if metrics["latency_t1_ms"] > 0 else None,
            latency_t2_ms=metrics["latency_t2_ms"] if metrics["latency_t2_ms"] > 0 else None,
            latency_t3_ms=metrics["latency_t3_ms"] if metrics["latency_t3_ms"] > 0 else None,
            results_count=len(results),
            system_type="tiered"
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_queries} queries...")
    
    # Calculate tiered performance
    tiered_latencies.sort()
    tiered_p95 = tiered_latencies[int(len(tiered_latencies) * 0.95)]
    tiered_p99 = tiered_latencies[int(len(tiered_latencies) * 0.99)]
    tiered_mean = np.mean(tiered_latencies)
    
    print(f"\nTiered Performance:")
    print(f"  Mean latency: {tiered_mean:.2f} ms")
    print(f"  P95 latency: {tiered_p95:.2f} ms")
    print(f"  P99 latency: {tiered_p99:.2f} ms")
    
    # Verify tiered system meets performance targets
    print(f"\nPerformance Targets (Tiered):")
    print(f"  Mean < {mean_target} ms: {'✓' if tiered_mean < mean_target else '✗'}")
    print(f"  P95 < {p95_target} ms: {'✓' if tiered_p95 < p95_target else '✗'}")
    print(f"  P99 < {p99_target} ms: {'✓' if tiered_p99 < p99_target else '✗'}")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Mean latency:  Baseline={baseline_mean:.2f}ms, Tiered={tiered_mean:.2f}ms, "
          f"Diff={tiered_mean - baseline_mean:.2f}ms ({((tiered_mean - baseline_mean) / baseline_mean * 100):+.1f}%)")
    print(f"P95 latency:   Baseline={baseline_p95:.2f}ms, Tiered={tiered_p95:.2f}ms, "
          f"Diff={tiered_p95 - baseline_p95:.2f}ms ({((tiered_p95 - baseline_p95) / baseline_p95 * 100):+.1f}%)")
    print(f"P99 latency:   Baseline={baseline_p99:.2f}ms, Tiered={tiered_p99:.2f}ms, "
          f"Diff={tiered_p99 - baseline_p99:.2f}ms ({((tiered_p99 - baseline_p99) / baseline_p99 * 100):+.1f}%)")
    
    print("\n" + "=" * 70)
    print("COST ANALYSIS")
    print("=" * 70)
    print("Run calc_cost.py to calculate and compare costs:")
    print(f"  Baseline: python python/calc_cost.py --log {QUERY_LOG_BASELINE} --system baseline")
    print(f"  Tiered:   python python/calc_cost.py --log {QUERY_LOG_TIERED} --system tiered")
    print("\nOr use compare_costs.py for side-by-side comparison.")
    
    # Cleanup
    tiered_manager.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate baseline vs tiered system")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--query-embeddings", type=str, default="query_embeddings.npy",
                       help="Path to query embeddings file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Load query embeddings
    if not os.path.exists(args.query_embeddings):
        print(f"Error: Query embeddings not found at {args.query_embeddings}")
        sys.exit(1)
    
    query_embeddings = np.load(args.query_embeddings)
    
    # Run evaluation
    run_evaluation(
        query_embeddings=query_embeddings,
        config=config,
        num_queries=args.queries,
        k=args.k,
        threshold=args.threshold
    )

