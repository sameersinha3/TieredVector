# TieredVector: 3-Tier Vector Storage System

A high-performance RAG system that automatically manages document vectors across three storage tiers based on access frequency (temperature).

## Architecture

- **Tier 1 (Hot)**: Redis - Fastest access for high-temperature vectors
- **Tier 2 (Warm)**: RocksDB - Medium speed for medium-temperature vectors  
- **Tier 3 (Cold)**: GCS - Slowest but cheapest for low-temperature vectors

## Components

- **C++ Backend**: High-performance storage management with HTTP API
- **Python Data Loader**: Loads datasets and calculates document temperatures
- **Python ML Service**: Handles embedding generation and similarity search

## Prerequisites

### System Dependencies
```bash
# Install RocksDB
brew install rocksdb

# Install Redis C client
brew install hiredis

# Install Redis server
brew install redis
```

### Python Dependencies
```bash
# Install Python packages
pip install numpy sentence-transformers scikit-learn datasets requests
```

## Quick Start

### 1. Start Redis
```bash
brew services start redis
```

### 2. Build C++ System
```bash
cd cpp
./build.sh
```

### 3. Start C++ Storage Server
```bash
cd cpp/build
./tiered_vector
```

The server will start on port 8082 and display:
```
Starting 3-Tier Vector Storage System with HTTP API...
Initializing 3-tier storage system...
Redis connected
RocksDB opened
GCS client not available (Tier 3 disabled)

Server is running. Press Ctrl+C to stop.
Endpoints:
  POST /store - Store a document
  GET /status - Get server status
```

### 4. Load Your Data
```bash
cd python
python data_loader.py
```

This will:
- Load your `wiki_embeddings.npy` file
- Calculate document temperatures using Natural Questions dataset
- Send documents to C++ storage with temperature-based tier assignment

## API Endpoints

### Store Document
```bash
curl -X POST http://localhost:8082/store \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": 0,
    "embedding": [0.1, 0.2, ...],
    "temperature": 0.85
  }'
```

### Get Status
```bash
curl -X GET http://localhost:8082/status
```

## Configuration

### Temperature Thresholds
Documents are assigned to tiers based on temperature:
- **Tier 1**: temperature ≥ 95th percentile (default ~0.8)
- **Tier 2**: 75th percentile ≤ temperature < 95th percentile (default ~0.5-0.8)
- **Tier 3**: temperature < 75th percentile (default <0.5)