import cohere
from dotenv import load_dotenv
import itertools
import numpy as np
import os
import time

from datasets import load_dataset

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(API_KEY)

# Configuration
NUM_QUERIES = 50000
BATCH_SIZE = 8613  # Recommended batch size (90% of max 9570)
RATE_LIMIT_PER_MIN = 100  # Trial API: 100/min, Production: 2000/min
DELAY_BETWEEN_BATCHES = 60.0 / (RATE_LIMIT_PER_MIN / BATCH_SIZE)

def load_existing_embeddings():
    """Load existing query embeddings if they exist"""
    if os.path.exists("query_embeddings.npy"):
        existing = np.load("query_embeddings.npy")
        print(f"Found existing query_embeddings.npy with {len(existing)} queries")
        return existing
    return None

def main():
    print("=" * 70)
    print("LOADING QUERIES WITH INCREMENTAL EMBEDDING")
    print("=" * 70)
    print()
    
    # Load existing embeddings
    existing_embeddings = load_existing_embeddings()
    existing_count = len(existing_embeddings) if existing_embeddings is not None else 0
    
    # Load queries from dataset
    print(f"Loading {NUM_QUERIES} queries from Natural Questions dataset...")
    query_dataset = load_dataset("natural_questions", split="train", streaming=True)
    all_queries = [entry["question"]["text"] for entry in itertools.islice(query_dataset, NUM_QUERIES)]
    print(f"Loaded {len(all_queries)} queries")
    print()
    
    # Determine which queries need embedding
    if existing_count >= NUM_QUERIES:
        # We have enough or more queries
        print(f"Existing embeddings ({existing_count}) >= requested ({NUM_QUERIES})")
        if existing_count > NUM_QUERIES:
            print(f"Truncating to {NUM_QUERIES} queries...")
            query_embeddings = existing_embeddings[:NUM_QUERIES]
        else:
            query_embeddings = existing_embeddings
        print(f"Using existing embeddings, no API calls needed!")
    else:
        # We need more queries
        print(f"Existing embeddings: {existing_count}")
        print(f"Requested: {NUM_QUERIES}")
        print(f"Need to embed: {NUM_QUERIES - existing_count} new queries")
        print()
        
        # Get new queries that need embedding
        new_queries = all_queries[existing_count:NUM_QUERIES]
        print(f"Embedding {len(new_queries)} new queries...")
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
        
        # Combine with existing embeddings
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            query_embeddings = np.vstack([existing_embeddings, new_embeddings])
            print(f"Combined with {len(existing_embeddings)} existing embeddings")
        else:
            query_embeddings = new_embeddings
    
    # Save final embeddings
    np.save("query_embeddings.npy", query_embeddings)
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total queries in file: {len(query_embeddings)}")
    print(f"Embedding dimension: {query_embeddings.shape[1]}")
    if existing_count < NUM_QUERIES:
        print(f"Existing embeddings reused: {existing_count}")
        print(f"New embeddings generated: {NUM_QUERIES - existing_count}")
    else:
        print(f"All embeddings reused from existing file")
    print(f"Saved to query_embeddings.npy")
    print()

if __name__ == "__main__":
    main()

