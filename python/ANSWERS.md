# Answers to Questions

## 1. How does simulate_temperature work?

**Process:**
1. Loads all document embeddings and query embeddings
2. Normalizes all embeddings (for cosine similarity)
3. Uses `NearestNeighbors` to find top-100 similar documents for each query
4. Calculates "temperature" for each document:
   - For each query, finds its 100 nearest documents
   - For each document, sums up similarity scores from all queries that reference it
   - Higher temperature = more frequently/strongly accessed
5. Assigns tiers based on temperature percentiles:
   - Tier 1 (95th percentile): Hottest documents → Redis/RAM
   - Tier 2 (75th percentile): Warm documents → Local ChromaDB
   - Tier 3 (below 75th): Cold documents → Remote ChromaDB
6. Stores documents in respective tiers

**Key insight:** Documents that appear in many query results or have high similarity scores get higher temperature → stored in faster tiers.

## 2. Current Query Count

**500 queries** in `query_embeddings.npy` (shape: 500, 768)

## 3. First 50 Sample Queries

See output above - includes questions like:
- "when is the last episode of season 8 of the walking dead"
- "in greek mythology who was the goddess of spring growth"
- "benefits of colonial life for single celled organisms"
- etc.

## 4. Does evaluate.py clear Redis/ChromaDB for tiered system?

**NO** - `setup_tiered_system()` does NOT clear existing data:
- It only initializes `StorageManager` which connects to existing Redis/ChromaDB
- It assumes `simulate_temperature.py` already populated the tiers
- This could cause issues if there's leftover data from previous runs

**Recommendation:** Add clearing logic to ensure clean state.

## 5. Dynamic Temperature Updates

**Current state:** `StorageManager` has promotion/demotion methods but no dynamic temperature recalculation.

**Implementation needed:**
- Method to update temperature thresholds
- Method to recalculate temperatures based on query access patterns
- Method to reallocate documents based on new temperatures

