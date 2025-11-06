import lmdb
import numpy as np
import json
import os
import pickle
import redis
import chromadb # CHANGED: Added chromadb

from dotenv import load_dotenv
# REMOVED: from google.cloud import firestore
# REMOVED: from google.oauth2 import service_account
from sklearn.neighbors import NearestNeighbors


'''
Initial temperatures based on a query dataset (these will be moved around later)
'''
load_dotenv()
doc_embeddings = np.load("wiki_embeddings.npy")[:10000] # Only use 10k documents for now 
query_embeddings = np.load("query_embeddings.npy")

k = 100
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(doc_embeddings)

print("Found Nearest Neighbors")

temperature = np.zeros(len(doc_embeddings))

distances, indices = nbrs.kneighbors(query_embeddings)
for query_dists, query_topk in zip(distances, indices):
    for rank, (dist, idx) in enumerate(zip(query_dists, query_topk)):
        temperature[idx] += 1/dist


print("Assigned Temperatures")

tier1_threshold = np.percentile(temperature, 95)
tier2_threshold = np.percentile(temperature, 75)

tier_assignment = np.zeros(len(doc_embeddings), dtype=int)
tier_assignment[temperature >= tier1_threshold] = 1
tier_assignment[(temperature < tier1_threshold) & (temperature >= tier2_threshold)] = 2
tier_assignment[temperature < tier2_threshold] = 3


num_tier1 = np.sum(tier_assignment == 1)
num_tier2 = np.sum(tier_assignment == 2)
num_tier3 = np.sum(tier_assignment == 3)

print(f"Tier 1 count: {num_tier1}")
print(f"Tier 2 count: {num_tier2}")
print(f"Tier 3 count: {num_tier3}")

np.savez(
    "tier_results.npz",
    temperature=temperature,
    tier_assignment=tier_assignment,
    tier1_threshold=tier1_threshold,
    tier2_threshold=tier2_threshold
)

# Connect to local Redis
r = redis.Redis(host='localhost', port=6379)

# Store Tier 1 vectors
tier1_indices = np.where(tier_assignment == 1)[0]
for idx in tier1_indices:
    r.set(f'doc{idx}', pickle.dumps(doc_embeddings[idx]))

print("Redis Complete")

# Store Tier 2 vectors in LMDB (key-value store)
env = lmdb.open('tier2_lmdb', map_size=1_000_000_000)  
tier2_indices = np.where(tier_assignment == 2)[0]
with env.begin(write=True) as txn:
    for idx in tier2_indices:
        key = f'doc{idx}'.encode('utf-8')
        value = pickle.dumps(doc_embeddings[idx])
        txn.put(key, value)
env.close()

print("LMDB Complete")

# --- START: CHROMA DB TIER 3 ---

tier3_indices = np.where(tier_assignment == 3)[0]  # array of indices
embeddings = np.array(doc_embeddings)  # shape (N, d)


CHROMA_HOST_IP = os.getenv("VM_IP")

print(f"Connecting to ChromaDB at {CHROMA_HOST_IP}...")
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST_IP, port=8000)
    collection = chroma_client.get_or_create_collection(name="cold_vectors")
    print("ChromaDB connected.")
except Exception as e:
    print(f"Failed to connect to ChromaDB: {e}")
    print("Please ensure the ChromaDB server is running on your VM and port 8000 is open.")
    exit()


# Prepare data for ChromaDB batch upload
ids_to_add = [f"doc{i}" for i in tier3_indices]
embeddings_to_add = [embeddings[i].astype(np.float32).tolist() for i in tier3_indices]

batch_size = 500  # ChromaDB can handle large batches, 500 is very safe
total_docs = 0

for j in range(0, len(ids_to_add), batch_size):
    batch_ids = ids_to_add[j:j+batch_size]
    batch_embeddings = embeddings_to_add[j:j+batch_size]
    
    print(f"Committing batch of {len(batch_ids)} documents to ChromaDB...")
    
    # Use 'upsert' - it's robust (creates or updates)
    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings
    )
    total_docs += len(batch_ids)

print(f"ChromaDB Complete. Stored {total_docs} vectors.")

# --- END: CHROMA DB TIER 3 ---


print(f"Stored {len(tier1_indices)} vectors in Redis (Tier 1)")
print(f"Stored {len(tier2_indices)} vectors in LMDB (Tier 2)")
print(f"Stored {len(tier3_indices)} vectors in Cloud (ChromaDB) (Tier 3)")