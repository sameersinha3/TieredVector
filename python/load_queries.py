import cohere
from dotenv import load_dotenv
import itertools
import numpy as np
import os
import time
import hashlib

from datasets import load_dataset

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(API_KEY)

# Configuration
NUM_QUERIES = 50000
BATCH_SIZE = 8613  # Recommended batch size (90% of max 9570)
RATE_LIMIT_PER_MIN = 100  # Trial API: 100/min, Production: 2000/min
DELAY_BETWEEN_BATCHES = 60.0 / (RATE_LIMIT_PER_MIN / BATCH_SIZE)  # Calculate delay to stay under limit

CACHE_FILE = "query_embeddings_cache.npz"
OLD_CACHE_FILE = "query_embeddings.npy"


def hash_query(query_text):
    """Generate hash for query text"""
    return hashlib.md5(query_text.encode('utf-8')).hexdigest()


def load_existing_cache():
    """Load existing embeddings cache (new or old format)"""
    existing_queries = []
    existing_embeddings = []
    query_hash_map = {}  # hash -> index
    
    # Try new format first
    if os.path.exists(CACHE_FILE):
        print(f"Loading existing cache from {CACHE_FILE}...")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        existing_queries = cache['queries'].tolist()
        existing_embeddings = cache['embeddings']
        
        # Build hash map
        for idx, query in enumerate(existing_queries):
            query_hash_map[hash_query(query)] = idx
        
        print(f"  Found {len(existing_queries)} existing queries in cache")
        return existing_queries, existing_embeddings, query_hash_map
    
    # Try old format and migrate
    elif os.path.exists(OLD_CACHE_FILE):
        print(f"Found old format cache ({OLD_CACHE_FILE}). Migrating to new format...")
        old_embeddings = np.load(OLD_CACHE_FILE)
        old_count = len(old_embeddings)
        
        print(f"  Loading {old_count} queries from dataset to match embeddings...")
        query_dataset = load_dataset("natural_questions", split="train", streaming=True)
        old_queries = [entry["question"]["text"] for entry in itertools.islice(query_dataset, old_count)]
        
        # Build hash map
        for idx, query in enumerate(old_queries):
            query_hash_map[hash_query(query)] = idx
        
        # Save in new format
        np.savez(CACHE_FILE, queries=np.array(old_queries, dtype=object), embeddings=old_embeddings)
        print(f"  Migrated {old_count} queries to new format")
        
        return old_queries, old_embeddings, query_hash_map
    
    return [], np.array([]), {}


def find_new_queries(all_queries, query_hash_map):
    """Find queries that need to be embedded"""
    new_queries = []
    new_indices = []
    
    for idx, query in enumerate(all_queries):
        query_hash = hash_query(query)
        if query_hash not in query_hash_map:
            new_queries.append(query)
            new_indices.append(idx)
    
    return new_queries, new_indices


def main():
    print("=" * 70)
    print("LOADING QUERIES WITH INCREMENTAL EMBEDDING")
    print("=" * 70)
    print()
    
    # Load existing cache
    existing_queries, existing_embeddings, query_hash_map = load_existing_cache()
    
    # Load all queries from dataset
    print(f"Loading {NUM_QUERIES} queries from Natural Questions dataset...")
    query_dataset = load_dataset("natural_questions", split="train", streaming=True)
    all_queries = [entry["question"]["text"] for entry in itertools.islice(query_dataset, NUM_QUERIES)]
    print(f"Loaded {len(all_queries)} queries")
    print()
    
    # Find new queries that need embedding
    new_queries, new_indices = find_new_queries(all_queries, query_hash_map)
    
    if len(new_queries) == 0:
        print("All queries already embedded!")
        print(f"Total queries in cache: {len(existing_queries)}")
        print(f"Requested queries: {NUM_QUERIES}")
        
        # Truncate if we need fewer queries
        if len(existing_queries) > NUM_QUERIES:
            print(f"\nTruncating to {NUM_QUERIES} queries...")
            final_queries = all_queries[:NUM_QUERIES]
            final_embeddings = existing_embeddings[:NUM_QUERIES]
            
            np.savez(CACHE_FILE, queries=np.array(final_queries, dtype=object), embeddings=final_embeddings)
            np.save(OLD_CACHE_FILE, final_embeddings)  # Also save old format for compatibility
            
            print(f"Saved {NUM_QUERIES} queries to cache")
        else:
            # Save old format for compatibility
            np.save(OLD_CACHE_FILE, existing_embeddings)
            print(f"Saved old format for compatibility")
        return
    
    print(f"Found {len(new_queries)} new queries to embed")
    print(f"Already cached: {len(existing_queries)} queries")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {(len(new_queries) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"Delay between batches: {DELAY_BETWEEN_BATCHES:.2f} seconds")
    print()
    
    # Embed new queries with rate limiting
    new_embeddings_list = []
    total_batches = (len(new_queries) + BATCH_SIZE - 1) // BATCH_SIZE
    
    i = 0
    while i < len(new_queries):
        batch_num = (i // BATCH_SIZE) + 1
        batch_queries = new_queries[i:i + BATCH_SIZE]
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_queries)} queries)...", end=" ")
        
        try:
            response = co.embed(
                model="multilingual-22-12",
                texts=batch_queries
            )
            batch_embeddings = np.array(response.embeddings)
            new_embeddings_list.append(batch_embeddings)
            print(f"✓ Got {len(batch_embeddings)} embeddings")
            
            # Rate limiting: wait between batches (except for the last one)
            if i + BATCH_SIZE < len(new_queries):
                time.sleep(DELAY_BETWEEN_BATCHES)
            
            i += BATCH_SIZE  # Move to next batch only on success
                
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                print(f"✗ Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                # Retry this batch (don't increment i)
            else:
                print(f"✗ Error: {e}")
                raise
    
    # Combine new embeddings
    if new_embeddings_list:
        new_embeddings = np.vstack(new_embeddings_list)
        print(f"\nGenerated {len(new_embeddings)} new embeddings")
    else:
        new_embeddings = np.array([])
    
    # Merge with existing embeddings
    if len(existing_embeddings) > 0:
        # Combine: existing + new
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        all_query_texts = existing_queries + new_queries
    else:
        all_embeddings = new_embeddings
        all_query_texts = new_queries
    
    # Truncate to NUM_QUERIES if needed
    if len(all_query_texts) > NUM_QUERIES:
        print(f"\nTruncating to {NUM_QUERIES} queries...")
        all_query_texts = all_query_texts[:NUM_QUERIES]
        all_embeddings = all_embeddings[:NUM_QUERIES]
    
    # Save in new format
    np.savez(CACHE_FILE, queries=np.array(all_query_texts, dtype=object), embeddings=all_embeddings)
    
    # Also save old format for backward compatibility
    np.save(OLD_CACHE_FILE, all_embeddings)
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total queries in cache: {len(all_query_texts)}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    print(f"New embeddings generated: {len(new_embeddings)}")
    print(f"Existing embeddings reused: {len(existing_embeddings)}")
    print(f"Saved to {CACHE_FILE} (new format)")
    print(f"Saved to {OLD_CACHE_FILE} (old format for compatibility)")
    print()


if __name__ == "__main__":
    main()
