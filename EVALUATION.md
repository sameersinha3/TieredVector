# Evaluation: Memory-Performance Tradeoff

Evaluates the efficiency of the tiered system by comparing memory usage and performance retention against a baseline (all data in Tier 1/RAM).

## Quick Start

```bash
cd python

# Run evaluation
python evaluate.py --queries 100 --k 5 --threshold 0.75

# Analyze results
python analyze_results.py --log ../data/query_log_baseline.csv --system baseline
python analyze_results.py --log ../data/query_log_tiered.csv --system tiered

# Compare systems
python compare_results.py
```

## Evaluation Process

### Step 1: Run Evaluation

```bash
python evaluate.py [--queries N] [--k K] [--threshold T]
```

**What it does:**
- Sets up baseline system (all data in Tier 1/Redis - 100% RAM)
- Sets up tiered system (temperature-based distribution - X% RAM)
- Runs queries against both systems
- Calculates memory ratios and performance retention
- Logs metrics to `data/query_log_baseline.csv` and `data/query_log_tiered.csv`
- Saves summary to `data/evaluation_summary.txt`

**Output:** Memory-performance tradeoff analysis showing:
- % RAM used vs baseline
- % Performance retained vs baseline
- Efficiency summary

### Step 2: Analyze Results

```bash
python analyze_results.py --log <log_file> --system <baseline|tiered>
```

**Output:** Detailed latency statistics (mean, median, p95, p99) per system.

### Step 3: Compare Systems

```bash
python compare_results.py
```

**Output:** Side-by-side comparison showing:
- Memory reduction percentage
- Performance retention percentage
- Efficiency summary

## Metrics

### Memory Ratio
```
RAM fraction = (Tiered RAM / Baseline RAM) × 100%
RAM saved = 100% - RAM fraction
```

### Performance Retention
```
Performance retained = (Baseline latency / Tiered latency) × 100%
```
Higher percentage = better performance retention.

## Expected Results

The tiered system should demonstrate:
- ✓ Significant RAM reduction (typically 80-95% less RAM)
- ✓ High performance retention (typically 70-90% of baseline)
- ✓ Efficient memory-performance tradeoff

**Example:** "With 5% RAM usage, we retain 80% of baseline performance, representing a 95% reduction in memory footprint."

## Configuration

Edit `config/evaluation.yaml` to adjust:
- Performance targets (`performance_targets`)
- Evaluation parameters (`evaluation`)

## Output Files

- `data/query_log_baseline.csv` - Baseline system metrics
- `data/query_log_tiered.csv` - Tiered system metrics
- `data/evaluation_summary.txt` - Summary of memory-performance tradeoff
