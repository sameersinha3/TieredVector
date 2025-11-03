# Prediction-Based Dynamic Temperature Adjustment

This module implements predictive temperature modeling and dynamic tier migration for the TieredVector system.

## Overview

The prediction system monitors document access patterns and predicts future "temperature" changes to proactively migrate documents between storage tiers before they're needed, reducing retrieval latency while optimizing storage costs.

## Components

### 1. **TemperaturePredictor** (`temperature_predictor.py`)
- Tracks document access history and temperature evolution
- Extracts features for ML-based prediction:
  - Access frequency (1h, 6h, 24h windows)
  - Temporal patterns (hour of day, day of week)
  - Temperature trends
  - Query diversity
- Trains per-document Linear Regression models
- Predicts future temperatures and identifies migration candidates

### 2. **MigrationManager** (`migration_manager.py`)
- Executes tier migrations based on predictions
- Manages tier capacities and eviction policies
- Handles concurrent migrations with configurable limits
- Tracks migration statistics and performance metrics
- Integrates with existing storage infrastructure (Redis, LMDB, GCS)

### 3. **PredictionDemo** (`prediction_demo.py`)
- Demonstrates the complete prediction and migration workflow
- Generates synthetic access patterns with temporal variations
- Visualizes temperature evolution over time
- Shows migration decisions and results

## Key Features

### Adaptive Temperature Model
```python
# Temperature update formula (matches simulate_temperature.py)
new_temp = alpha * current_temp + access_weight
# alpha = 0.9 (retention factor)
```

### Feature Engineering
- **Temporal Features**: Cyclical encoding of time (sin/cos transforms)
- **Access Patterns**: Multi-window frequency analysis
- **Query Diversity**: Standard deviation of query embeddings
- **Temperature Trends**: Linear regression on recent temperatures

### Migration Decision Logic
```python
if predicted_tier != current_tier and temp_change >= threshold:
    schedule_migration()
```

### Eviction Policies
- **Lowest Temperature**: Evict coldest document when tier is full
- **LRU** (TODO): Least recently used
- **LFU** (TODO): Least frequently used

## Integration Points

### With Existing System
1. **Storage Manager**: Uses same tier storage backends
2. **Temperature Thresholds**: Matches percentiles from `simulate_temperature.py`
3. **Access Logging**: Hooks into query processing pipeline

### With Team Members' Work
1. **Sameer's Vertex AI**: Migration manager supports Vertex AI tier
2. **Yu's Evaluation**: Provides metrics for performance analysis

## Usage

### Basic Example
```python
from temperature_predictor import TemperaturePredictor
from migration_manager import MigrationManager

# Initialize predictor
predictor = TemperaturePredictor(
    history_window_hours=24,
    prediction_horizon_hours=6,
    migration_threshold=0.15
)

# Log access
predictor.log_access(doc_id=123, timestamp=datetime.now(), query_embedding=embedding)

# Train model
predictor.train_predictor(doc_id=123)

# Get migration candidates
candidates = predictor.get_migration_candidates(current_time=datetime.now())

# Execute migrations
manager = MigrationManager()
results = await manager.migrate_documents(candidates)
```

### Running the Demo
```bash
python prediction_demo.py
```

## Configuration Parameters

### TemperaturePredictor
- `history_window_hours`: How far back to look for training (default: 24)
- `prediction_horizon_hours`: How far ahead to predict (default: 6)
- `migration_threshold`: Minimum temperature change to trigger migration (default: 0.15)

### MigrationManager
- `max_concurrent_migrations`: Parallel migration limit (default: 10)
- `migration_batch_size`: Documents per batch (default: 100)
- `eviction_policy`: How to free space in full tiers (default: 'lru')

## Performance Considerations

### Prediction Overhead
- Feature extraction: O(1) per document
- Model training: O(n) where n = history length
- Prediction: O(1) per document

### Migration Costs
- Tier 1→2: ~1ms (memory to disk)
- Tier 2→3: ~10ms (disk to cloud)
- Tier 3→2: ~50ms (cloud to disk)
- Tier 2→1: ~1ms (disk to memory)

## Future Enhancements

1. **Advanced Models**
   - LSTM/GRU for temporal sequences
   - Attention mechanisms for query patterns
   - Ensemble methods for robustness

2. **Online Learning**
   - Incremental model updates
   - Adaptive threshold tuning
   - Concept drift detection

3. **Cost Optimization**
   - Multi-objective optimization (latency vs cost)
   - Budget-aware migration scheduling
   - Predictive pre-fetching

4. **Distributed Coordination**
   - Cross-node migration coordination
   - Global temperature synchronization
   - Distributed prediction consensus

## Metrics for Evaluation

### Prediction Accuracy
- Temperature prediction MAE/RMSE
- Tier prediction accuracy
- False positive/negative migration rates

### System Performance
- Average retrieval latency reduction
- Storage cost savings
- Migration overhead impact

### Operational Metrics
- Migration success rate
- Tier capacity utilization
- Temperature distribution stability

## Testing

```bash
# Unit tests (TODO)
python -m pytest tests/test_temperature_predictor.py
python -m pytest tests/test_migration_manager.py

# Integration tests (TODO)
python -m pytest tests/test_integration.py
```

## Notes

- Current implementation uses Linear Regression for simplicity
- Production deployment should include Redis/LMDB/GCS setup
- Consider batch predictions for efficiency at scale
- Monitor prediction drift and retrain periodically
