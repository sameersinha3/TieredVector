# Evaluation: Memory-Performance Tradeoff

Evaluates the efficiency of the tiered storage system by comparing local disk usage and performance against a baseline (all data in Tier 2/local disk).

## Quick Start

```bash
cd python

# Step 1: Ensure tiered system is set up
python simulate_temperature.py

# Step 2: Run evaluation
python evaluate.py --queries 50 --k 5 --threshold 0.75

# Step 3: Analyze results
python analyze_results.py --log data/query_log_baseline.csv --system baseline
python analyze_results.py --log data/query_log_tiered.csv --system tiered

# Step 4: Compare systems
python compare_results.py
```

## Evaluation Process

### Step 1: Setup Tiered System

```bash
python simulate_temperature.py
```

**What it does:**
- Calculates document temperatures based on query access patterns
- Distributes documents across 3 tiers:
  - Tier 1 (Redis/RAM): ~500 hot documents (5%)
  - Tier 2 (Local ChromaDB): ~2000 warm documents (20%)
  - Tier 3 (Remote ChromaDB): ~7500 cold documents (75%)
- Saves distribution to `tier_results.npz`

**Prerequisites:**
- Redis running locally
- ChromaDB server running on CloudLab VM (or local simulation)
- `wiki_embeddings.npy` and `query_embeddings.npy` files exist

### Step 2: Run Evaluation

```bash
python evaluate.py [--queries N] [--k K] [--threshold T]
```

**Parameters:**
- `--queries`: Number of queries to run (default: 50, limited by available queries in query_embeddings.npy)
- `--k`: Number of results per query (default: 5)
- `--threshold`: Similarity threshold (default: 0.75)

**What it does:**

1. **Setup Baseline System:**
   - Stores all documents in Tier 2 (Local ChromaDB)
   - Clears Tier 1 (Redis) - baseline doesn't use RAM
   - Creates separate baseline database at `./tier2_baseline_db`

2. **Setup Tiered System:**
   - Uses existing distribution from `simulate_temperature.py`
   - Initializes all 3 tiers (Redis, Local ChromaDB, Remote ChromaDB)

3. **Run Queries:**
   - Runs same queries against both systems
   - Measures latency for each query
   - Logs results to CSV files

4. **Calculate Metrics:**
   - Local disk storage comparison
   - RAM overhead (Tier 1)
   - Remote storage (Tier 3)
   - Performance comparison

**Output Files:**
- `data/query_log_baseline.csv` - Baseline system query metrics
- `data/query_log_tiered.csv` - Tiered system query metrics
- `data/evaluation_summary.txt` - Summary of results

### Step 3: Analyze Results

```bash
python analyze_results.py --log <log_file> --system <baseline|tiered>
```

**Output:** Detailed latency statistics (mean, median, p95, p99) per system.

**Example:**
```bash
python analyze_results.py --log data/query_log_baseline.csv --system baseline
python analyze_results.py --log data/query_log_tiered.csv --system tiered
```

### Step 4: Compare Systems

```bash
python compare_results.py
```

**Output:** Side-by-side comparison showing:
- Storage breakdown (local disk, RAM, remote)
- Performance comparison (latency metrics)
- Efficiency summary

## Metrics Explained

### Local Disk Savings

```
Local disk saved = (Baseline Tier 2 - Tiered Tier 2) / Baseline Tier 2 × 100%
```

**Example:** If baseline uses 0.0300 GB and tiered uses 0.0060 GB:
- Local disk saved = (0.0300 - 0.0060) / 0.0300 × 100% = 80%

### RAM Overhead

Tier 1 (Redis) stores hot documents in RAM. This is reported as overhead since baseline uses no RAM.

### Remote Storage

Tier 3 (Remote ChromaDB) stores cold documents on remote disk. This is reported separately as it represents distribution strategy.

### Performance Retention

```
Performance retention = (Baseline latency / Tiered latency) × 100%
```

- **> 100%**: Tiered is faster (better)
- **< 100%**: Tiered is slower (acceptable if within threshold)
- **100%**: Same performance

## Expected Results

The tiered system should demonstrate:

- ✓ **Significant local disk reduction** (typically 70-90% less local disk)
- ✓ **Performance tradeoff** (may be slower due to tier traversal overhead)
- ✓ **Efficient storage distribution** (hot data in RAM, warm in local disk, cold in remote)

## Configuration

### Evaluation Parameters

Edit `evaluate.py` or use command-line arguments:
- `--queries`: Number of test queries
- `--k`: Results per query
- `--threshold`: Similarity threshold

### Tier Distribution

Edit `simulate_temperature.py` to adjust:
- `tier1_threshold`: Percentile for Tier 1 (default: 95th)
- `tier2_threshold`: Percentile for Tier 2 (default: 75th)

## Output Files

- `data/query_log_baseline.csv` - Baseline system metrics
- `data/query_log_tiered.csv` - Tiered system metrics  
- `data/evaluation_summary.txt` - Summary of memory-performance tradeoff
- `tier2_baseline_db/` - Baseline database (created during evaluation)


