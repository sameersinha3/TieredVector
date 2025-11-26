#!/usr/bin/env python3
"""
Check storage on remote CloudLab ChromaDB
"""

import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
VM_IP = os.getenv("VM_IP")

if not VM_IP:
    print("Error: VM_IP not found in .env file")
    exit(1)

try:
    print(f"Connecting to ChromaDB at {VM_IP}:8000...")
    client = chromadb.HttpClient(host=VM_IP, port=8000)
    
    # Get collection
    collection = client.get_collection("cold_vectors")
    
    # Get all documents
    result = collection.get()
    
    print(f"\nTier 3 (Remote ChromaDB) Storage:")
    print(f"  Collection: cold_vectors")
    print(f"  Total documents: {len(result['ids'])}")
    print(f"  Host: {VM_IP}:8000")
    
    # Estimate storage size (rough)
    # Each embedding: 768 dims * 4 bytes (float32) = 3072 bytes
    # Plus overhead
    estimated_bytes = len(result['ids']) * 768 * 4 * 1.2  # 20% overhead
    estimated_gb = estimated_bytes / (1024 ** 3)
    print(f"  Estimated size: {estimated_gb:.4f} GB")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure ChromaDB server is running on the VM")


