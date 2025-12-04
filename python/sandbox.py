import cohere
import numpy as np
import os
import hashlib
import json

from dotenv import load_dotenv
from storage_manager import StorageManager

load_dotenv()

manager = StorageManager(
    tier3_host=os.getenv("VM_IP"),
    tier3_port=8000,
    tier3_collection="cold_vectors" 
)

manager.initialize()

query = "cloud storage systems"

# Cache directory for query embeddings
CACHE_DIR = "query_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Create hash of query text for cache filename
query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
cache_file = os.path.join(CACHE_DIR, f"{query_hash}.npy")
cache_meta_file = os.path.join(CACHE_DIR, f"{query_hash}.json")

# Check if cached embedding exists
query_embedding = None
if os.path.exists(cache_file):
    try:
        query_embedding = np.load(cache_file)
        # Load metadata to show what query this cache is for
        if os.path.exists(cache_meta_file):
            with open(cache_meta_file, 'r') as f:
                meta = json.load(f)
                print(f"Using cached embedding for query: '{meta.get('query', 'unknown')}'")
        else:
            print(f"Using cached embedding (hash: {query_hash[:8]}...)")
    except Exception as e:
        print(f"Error loading cache: {e}, generating new embedding")

# Generate embedding if not cached
if query_embedding is None:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if not cohere_api_key:
        raise ValueError(
            "COHERE_API_KEY not found in environment. "
            "Please set COHERE_API_KEY in your .env file."
        )
    
    try:
        co = cohere.Client(cohere_api_key)
query_embedding = np.array(co.embed(
    model="multilingual-22-12",
    texts=[query]
).embeddings[0])
        print("Generated embedding using Cohere API")
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding using Cohere API: {e}")
    
    # Save to cache
    try:
        np.save(cache_file, query_embedding)
        # Save metadata
        with open(cache_meta_file, 'w') as f:
            json.dump({
                'query': query,
                'model': 'cohere-multilingual-22-12',
                'dimension': len(query_embedding)
            }, f)
        print(f"Cached embedding saved to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

results = manager.retrieve_document(query_embedding, k=5, threshold=0.8)
print(results)

manager.close()