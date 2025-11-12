# Evaluation: Same Performance → Cost Reductions

Evaluates cost savings of the tiered system while maintaining the same performance as a baseline (all data in Tier 1).

## Quick Start

```bash
cd python

# 1. Run evaluation
python evaluate.py --queries 100 --k 5 --threshold 0.75

# 2. Calculate costs
python calc_cost.py --log ../data/query_log_baseline.csv --system baseline
python calc_cost.py --log ../data/query_log_tiered.csv --system tiered

# 3. Compare results
python compare_costs.py
```

## Evaluation Process

### Step 1: Run Evaluation

```bash
python evaluate.py [--queries N] [--k K] [--threshold T]
```

**What it does:**
- Sets up baseline system (all data in Tier 1/Redis)
- Sets up tiered system (temperature-based distribution)
- Runs queries against both systems
- Logs metrics to `data/query_log_baseline.csv` and `data/query_log_tiered.csv`

**Output:** Query logs with latency, tier hits, operations, and bytes transferred.

### Step 2: Calculate Costs

```bash
python calc_cost.py --log <log_file> --system <baseline|tiered>
```

**Output:** Cost breakdown (storage, operations, egress, CPU) and latency statistics.

### Step 3: Compare Results

```bash
python compare_costs.py
```

**Output:** Side-by-side comparison showing:
- Cost savings percentage
- Performance comparison (mean, p95, p99 latency)
- Verification that performance targets are met

## Performance Targets

Both systems must meet (configurable in `config/cost.yaml`):
- Mean latency < 150ms
- P95 latency < 200ms  
- P99 latency < 500ms

## Expected Results

The tiered system should demonstrate:
- ✓ Same or similar performance
- ✓ Lower total cost 
- ✓ Cost savings from intelligent tier placement

## Configuration

Edit `config/cost.yaml` to adjust:
- Performance targets (`performance_targets`)
- Unit prices (`unit_prices`)
- Experiment duration (`experiment.hours`)

## Output Files

- `data/query_log_baseline.csv` - Baseline system metrics
- `data/query_log_tiered.csv` - Tiered system metrics
