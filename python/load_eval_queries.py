#!/usr/bin/env python3
"""
Load 100 new evaluation queries from Natural Questions dataset
These queries are separate from the ones used for temperature simulation
"""

import cohere
from dotenv import load_dotenv
import itertools
import numpy as np
import os
import time
import json

from datasets import load_dataset

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment. Please set COHERE_API_KEY in your .env file.")

co = cohere.Client(API_KEY)

# Configuration
NUM_EVAL_QUERIES = 100
START_OFFSET = 1000  # Start from query 1000 to ensure we get different queries than temperature simulation
BATCH_SIZE = 1  # Process 1 query at a time to minimize rate limit issues
DELAY_BETWEEN_BATCHES = 6  # 6 seconds between queries = ~10 queries per minute (safe for trial keys)

def main():
    print("=" * 70)
    print("LOADING EVALUATION QUERIES (Separate from Temperature Simulation)")
    print("=" * 70)
    print()
    
    # Check if evaluation queries already exist
    eval_embeddings_file = "query_embeddings_eval.npy"
    eval_queries_file = "query_embeddings_eval_queries.json"
    
    if os.path.exists(eval_embeddings_file):
        existing_embeddings = np.load(eval_embeddings_file)
        if len(existing_embeddings) >= NUM_EVAL_QUERIES:
            print(f"Found existing evaluation queries: {len(existing_embeddings)} queries")
            if os.path.exists(eval_queries_file):
                with open(eval_queries_file, 'r') as f:
                    queries = json.load(f)
                print(f"Sample queries:")
                for i, q in enumerate(queries[:3]):
                    print(f"  {i+1}. {q}")
            print(f"\nUsing existing evaluation queries from {eval_embeddings_file}")
            return
    
    # Load queries from dataset starting from offset
    print(f"Loading {NUM_EVAL_QUERIES} queries from Natural Questions dataset...")
    print(f"Starting from offset {START_OFFSET} to get different queries...")
    query_dataset = load_dataset("natural_questions", split="train", streaming=True)
    
    # Skip first START_OFFSET queries, then take NUM_EVAL_QUERIES
    all_queries_iter = itertools.islice(query_dataset, START_OFFSET, START_OFFSET + NUM_EVAL_QUERIES)
    all_queries = [entry["question"]["text"] for entry in all_queries_iter]
    
    print(f"Loaded {len(all_queries)} queries")
    print(f"\nSample queries:")
    for i, q in enumerate(all_queries[:5]):
        print(f"  {i+1}. {q}")
    print()
    
    # Embed queries with careful rate limiting
    print(f"Embedding {len(all_queries)} queries using Cohere API...")
    print(f"Batch size: {BATCH_SIZE} query per batch")
    print(f"Delay between queries: {DELAY_BETWEEN_BATCHES} seconds")
    print(f"Total batches: {len(all_queries)}")
    print(f"\nNote: Trial API keys have strict per-minute rate limits (~10 calls/minute).")
    print(f"This will take approximately {len(all_queries) * DELAY_BETWEEN_BATCHES / 60:.1f} minutes.")
    print(f"Starting in 5 seconds... (to let any previous rate limits clear)")
    time.sleep(5)
    print()
    
    query_embeddings_list = []
    
    # Process in batches with careful rate limiting
    i = 0
    retry_count = 0
    max_retries = 5
    
    while i < len(all_queries):
        batch_queries = all_queries[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(all_queries) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"[{batch_num}/{total_batches}] Processing batch ({len(batch_queries)} queries)...", end=" ", flush=True)
        
        try:
            response = co.embed(
                model="multilingual-22-12",
                texts=batch_queries
            )
            batch_embeddings = np.array(response.embeddings)
            query_embeddings_list.append(batch_embeddings)
            print(f"✓ Got {len(batch_embeddings)} embeddings")
            retry_count = 0  # Reset retry count on success
            
            # Rate limiting: wait between batches
            if i + BATCH_SIZE < len(all_queries):
                print(f"  Waiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
                time.sleep(DELAY_BETWEEN_BATCHES)
            
            i += BATCH_SIZE  # Move to next batch only on success
                
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"\n✗ Too many rate limit errors. Stopping.")
                    print(f"Progress: {len(query_embeddings_list) * BATCH_SIZE} / {len(all_queries)} queries embedded")
                    if len(query_embeddings_list) > 0:
                        print(f"\nSaving partial results...")
                        partial_embeddings = np.vstack(query_embeddings_list)
                        np.save(eval_embeddings_file + ".partial", partial_embeddings)
                        print(f"Saved {len(partial_embeddings)} embeddings to {eval_embeddings_file}.partial")
                    raise RuntimeError("Rate limit exceeded. Please try again later or use a production API key.")
                
                # Wait longer for rate limit - trial keys often have 1-minute windows
                wait_time = 70  # Wait 70 seconds to ensure rate limit window clears
                print(f"✗ Rate limit hit (attempt {retry_count}/{max_retries}). Waiting {wait_time} seconds...")
                for remaining in range(wait_time, 0, -10):
                    print(f"  {remaining} seconds remaining...", end='\r')
                    time.sleep(min(10, remaining))
                print(f"  Retrying...")
                # Retry this batch (don't increment i)
            else:
                print(f"✗ Error: {e}")
                raise
    
    # Combine all embeddings
    if query_embeddings_list:
        query_embeddings = np.vstack(query_embeddings_list)
    else:
        raise RuntimeError("No embeddings generated")
    
    # Save embeddings
    np.save(eval_embeddings_file, query_embeddings)
    
    # Save query texts for reference
    with open(eval_queries_file, 'w') as f:
        json.dump(all_queries, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total evaluation queries: {len(query_embeddings)}")
    print(f"Embedding dimension: {query_embeddings.shape[1]}")
    print(f"Saved to: {eval_embeddings_file}")
    print(f"Query texts saved to: {eval_queries_file}")
    print()
    print("These queries are separate from the ones used for temperature simulation.")
    print("They will be used for evaluation to test the tiered system with unseen queries.")
    print()

if __name__ == "__main__":
    main()

