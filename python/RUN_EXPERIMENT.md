# Steps to Run the Experiment

## Prerequisites Check

1. **Required Data Files:**
   - ✅ `wiki_embeddings.npy` (document embeddings)
   - ✅ `query_embeddings.npy` (query embeddings)
   - ✅ `tier_results.npz` (optional - tier distribution from previous run)

2. **Services Running:**
   - Redis server (localhost:6379)
   - ChromaDB server on CloudLab VM (if using remote Tier 3)
   - Or local ChromaDB simulation (if VM_IP=127.0.0.1)

## Option 1: Quick Run (Use Existing Tier Distribution)

If you have `tier_results.npz` from a previous run:

```bash
cd python

# Run evaluation (uses existing tier_results.npz)
python evaluate.py --queries 50 --k 5 --threshold 0.75
```

**What happens:**
- Clears all tiers (Redis, Local ChromaDB, Remote ChromaDB)
- Loads tier distribution from `tier_results.npz`
- Populates tiers based on existing distribution
- Runs baseline and tiered queries
- Generates results

## Option 2: Recalculate Temperatures On-the-Fly

If you want to recalculate temperatures from scratch:

```bash
cd python

# Run evaluation with temperature recalculation
python evaluate.py --queries 50 --k 5 --threshold 0.75 --recalculate
```

**What happens:**
- Clears all tiers
- Recalculates temperatures from query embeddings
- Reallocates all documents using StorageManager
- Runs baseline and tiered queries
- Generates results

## Option 3: Full Setup (Regenerate Tier Distribution)

If you want to regenerate `tier_results.npz` first:

```bash
cd python

# Step 1: Regenerate tier distribution
python simulate_temperature.py

# Step 2: Run evaluation
python evaluate.py --queries 50 --k 5 --threshold 0.75
```

## After Running Evaluation

### View Results

1. **Summary File:**
   ```bash
   cat data/evaluation_summary.txt
   ```

2. **Detailed Analysis:**
   ```bash
   python analyze_results.py --log data/query_log_baseline.csv --system baseline
   python analyze_results.py --log data/query_log_tiered.csv --system tiered
   ```

3. **Comparison:**
   ```bash
   python compare_results.py
   ```

## Output Files

After running `evaluate.py`, you'll get:

- `data/query_log_baseline.csv` - Baseline system query metrics
- `data/query_log_tiered.csv` - Tiered system query metrics
- `data/evaluation_summary.txt` - Summary of results

## Troubleshooting

### Redis not running
```bash
# Start Redis (macOS)
brew services start redis

# Or run directly
redis-server
```

### ChromaDB connection issues
- Check `.env` file has correct `VM_IP`
- If using local simulation, set `VM_IP=127.0.0.1`
- Verify ChromaDB server is running on the VM

### Missing data files
- `wiki_embeddings.npy`: Run `python load_dataset.py`
- `query_embeddings.npy`: Run `python load_queries.py`

## Recommended Workflow

For a fresh experiment:

```bash
cd python

# 1. Ensure data files exist
ls -lh wiki_embeddings.npy query_embeddings.npy

# 2. Run evaluation (will clear and repopulate tiers)
python evaluate.py --queries 50 --k 5 --threshold 0.75 --recalculate

# 3. View results
python compare_results.py
cat data/evaluation_summary.txt
```

## Command-Line Options

```bash
python evaluate.py [OPTIONS]

Options:
  --queries N       Number of queries to run (default: 50)
  --k K             Number of results per query (default: 5)
  --threshold T     Similarity threshold (default: 0.75)
  --recalculate     Recalculate temperatures on the fly
```

## Notes

- The evaluation automatically clears all tiers before setup for a clean state
- Using `--recalculate` is slower but ensures fresh temperature calculations
- Without `--recalculate`, it uses existing `tier_results.npz` (faster)

