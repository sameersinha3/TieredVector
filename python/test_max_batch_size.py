#!/usr/bin/env python3
"""
Test script to find the maximum batch size for Cohere embedding API.
Uses binary search approach: start with large batch, halve on error.
"""

import cohere
from dotenv import load_dotenv
import itertools
import numpy as np
import os
import time
import sys

from datasets import load_dataset

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    print("Error: COHERE_API_KEY not found in .env file")
    sys.exit(1)

co = cohere.Client(API_KEY)

# Configuration
TEST_QUERIES = 50000  # Total queries to test with
MIN_BATCH_SIZE = 1    # Minimum batch size to test
MAX_BATCH_SIZE = TEST_QUERIES  # Start with full dataset

def test_batch_size(queries, batch_size):
    """Test if a specific batch size works"""
    test_queries = queries[:batch_size]
    
    print(f"  Testing batch size: {batch_size} queries...", end=" ")
    
    try:
        start_time = time.time()
        response = co.embed(
            model="multilingual-22-12",
            texts=test_queries
        )
        elapsed = time.time() - start_time
        
        if len(response.embeddings) == batch_size:
            print(f"✓ SUCCESS ({elapsed:.2f}s)")
            return True, elapsed
        else:
            print(f"✗ FAILED (got {len(response.embeddings)} embeddings, expected {batch_size})")
            return False, elapsed
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            print(f"✗ RATE LIMIT (waiting 60s...)")
            time.sleep(60)
            # Retry once
            try:
                retry_start = time.time()
                response = co.embed(
                    model="multilingual-22-12",
                    texts=test_queries
                )
                elapsed = time.time() - retry_start
                if len(response.embeddings) == batch_size:
                    print(f"  Retry: ✓ SUCCESS ({elapsed:.2f}s)")
                    return True, elapsed
                else:
                    print(f"  Retry: ✗ FAILED")
                    return False, elapsed
            except:
                return False, 0
        elif "payload" in error_msg or "too large" in error_msg or "413" in error_msg:
            print(f"✗ PAYLOAD TOO LARGE")
            return False, 0
        elif "400" in error_msg or "bad request" in error_msg:
            print(f"✗ BAD REQUEST: {str(e)[:100]}")
            return False, 0
        else:
            print(f"✗ ERROR: {str(e)[:100]}")
            return False, 0


def find_max_batch_size(queries, min_size, max_size):
    """Binary search to find maximum working batch size"""
    print("=" * 70)
    print("FINDING MAXIMUM BATCH SIZE")
    print("=" * 70)
    print(f"Testing with {len(queries)} total queries")
    print(f"Search range: {min_size} - {max_size}")
    print()
    
    # Binary search
    low = min_size
    high = max_size
    best_size = 0
    best_time = 0
    
    test_history = []
    
    while low <= high:
        mid = (low + high) // 2
        print(f"[Binary Search] Testing: {mid} (range: {low}-{high})")
        
        success, elapsed = test_batch_size(queries, mid)
        test_history.append((mid, success, elapsed))
        
        if success:
            best_size = mid
            best_time = elapsed
            # Try larger batch
            low = mid + 1
            print(f"  → Success! Trying larger batch...")
        else:
            # Try smaller batch
            high = mid - 1
            print(f"  → Failed. Trying smaller batch...")
        
        print()
        
        # Safety: don't test too many times
        if len(test_history) > 20:
            print("  Stopping after 20 tests to avoid excessive API calls")
            break
    
    return best_size, best_time, test_history


def main():
    print("Loading queries from Natural Questions dataset...")
    query_dataset = load_dataset("natural_questions", split="train", streaming=True)
    queries = [entry["question"]["text"] for entry in itertools.islice(query_dataset, TEST_QUERIES)]
    
    print(f"Loaded {len(queries)} queries")
    print(f"First query: {queries[0][:100]}...")
    print()
    
    # Find max batch size
    max_size, elapsed, history = find_max_batch_size(queries, MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, len(queries)))
    
    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Maximum working batch size: {max_size} queries")
    if max_size > 0:
        print(f"Time for max batch: {elapsed:.2f} seconds")
        print(f"Throughput: {max_size/elapsed:.1f} queries/second")
    print()
    
    print("Test History:")
    print("-" * 70)
    print(f"{'Batch Size':<15} {'Result':<10} {'Time (s)':<12}")
    print("-" * 70)
    for size, success, time_taken in history:
        result = "✓ PASS" if success else "✗ FAIL"
        time_str = f"{time_taken:.2f}" if time_taken > 0 else "N/A"
        print(f"{size:<15} {result:<10} {time_str:<12}")
    
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if max_size > 0:
        # Use 90% of max for safety margin
        recommended = int(max_size * 0.9)
        print(f"Recommended batch size: {recommended} queries (90% of max {max_size})")
        print(f"This provides a safety margin for payload size variations.")
    else:
        print("No working batch size found. Check API key and network connection.")
    
    print()


if __name__ == "__main__":
    main()

