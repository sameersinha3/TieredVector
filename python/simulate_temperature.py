import numpy as np
import json
import os
import pickle
import redis
import chromadb 

from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors


load_dotenv()
doc_embeddings = np.load("wiki_embeddings.npy")[:10000] # Only use 10k documents for now 
query_embeddings = np.load("query_embeddings.npy")

print("Normalizing all doc embeddings...")
doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
doc_embeddings_normalized = np.divide(
    doc_embeddings, 
    doc_norms, 
    out=np.zeros_like(doc_embeddings), 
    where=(doc_norms != 0)
).astype(np.float32)

print("Normalizing all query embeddings...")
query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
query_embeddings_normalized = np.divide(
    query_embeddings, 
    query_norms, 
    out=np.zeros_like(query_embeddings), 
    where=(query_norms != 0)
).astype(np.float32)

print("All vectors normalized.")


k = 100
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(doc_embeddings_normalized)
print("Found Nearest Neighbors")

temperature = np.zeros(len(doc_embeddings_normalized))

# Use normalized queries to find neighbors
distances, indices = nbrs.kneighbors(query_embeddings_normalized)
for query_dists, query_topk in zip(distances, indices):

    for rank, (sim, idx) in enumerate(zip(query_dists, query_topk)):
        if sim > 0: # Only count positive similarity
            temperature[idx] += sim # Add similarity score


print("Assigned Temperatures")

# --- TIER ASSIGNMENT (Unchanged) ---
tier1_threshold = np.percentile(temperature, 95)
tier2_threshold = np.percentile(temperature, 75)

tier_assignment = np.zeros(len(doc_embeddings_normalized), dtype=int)
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

print("Storing Tier 1 in Redis...")
r = redis.Redis(host='localhost', port=6379)
tier1_indices = np.where(tier_assignment == 1)[0]

for idx in tier1_indices:
    r.set(f'doc{idx}', pickle.dumps(doc_embeddings_normalized[idx]))

print(f"Redis (Tier 1) stored {len(tier1_indices)} NORMALIZED vectors.")

# --- END: REDIS (Tier 1) ---


# 3. --- (CHANGED) TIER 2 (LMDB) REPLACED WITH LOCAL CHROMA DB ---
print("Storing Tier 2 in Local ChromaDB...")
try:
    tier2_client = chromadb.PersistentClient(path="./tier2_chroma_db")
    tier2_collection = tier2_client.get_or_create_collection(
        name="warm_vectors",
        metadata={"hnsw:space": "cosine"} # Use cosine
    )
    print("Tier 2 (Local ChromaDB) initialized at ./tier2_chroma_db")
except Exception as e:
    print(f"Failed to initialize Tier 2 ChromaDB: {e}")
    exit() # Can't continue if T2 fails

tier2_indices = np.where(tier_assignment == 2)[0]

ids_to_add_t2 = [f"doc{i}" for i in tier2_indices]
embeddings_to_add_t2 = [doc_embeddings_normalized[i].tolist() for i in tier2_indices]

batch_size = 500 # Re-use batch size
total_docs_t2 = 0

for j in range(0, len(ids_to_add_t2), batch_size):
    batch_ids = ids_to_add_t2[j:j+batch_size]
    batch_embeddings = embeddings_to_add_t2[j:j+batch_size]
    
    print(f"Committing batch of {len(batch_ids)} documents to Tier 2 ChromaDB...")
    tier2_collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings
    )
    total_docs_t2 += len(batch_ids)

print(f"ChromaDB Tier 2 Complete. Stored {total_docs_t2} vectors.")

# --- END: CHROMA DB TIER 2 ---


# --- START: CHROMA DB TIER 3 (REMOTE) ---
tier3_indices = np.where(tier_assignment == 3)[0]

CHROMA_HOST_IP = os.getenv("VM_IP")

print(f"Connecting to ChromaDB at {CHROMA_HOST_IP}...")
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST_IP, port=8000)
    # (CHANGED) Set collection to use cosine distance
    collection = chroma_client.get_or_create_collection(
        name="cold_vectors",
        metadata={"hnsw:space": "cosine"} # Use cosine
    )
    print("ChromaDB connected.")
except Exception as e:
    print(f"Failed to connect to ChromaDB: {e}")
    print("Please ensure the ChromaDB server is running on your VM and port 8000 is open.")
    exit()


ids_to_add_t3 = [f"doc{i}" for i in tier3_indices]
embeddings_to_add_t3 = [doc_embeddings_normalized[i].tolist() for i in tier3_indices]

batch_size = 500
total_docs_t3 = 0

for j in range(0, len(ids_to_add_t3), batch_size):
    batch_ids = ids_to_add_t3[j:j+batch_size]
    batch_embeddings = embeddings_to_add_t3[j:j+batch_size]
    
    print(f"Committing batch of {len(batch_ids)} documents to Tier 3 ChromaDB...")
    
    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings
    )
    total_docs_t3 += len(batch_ids)

print(f"ChromaDB Tier 3 Complete. Stored {total_docs_t3} vectors.")

print(f"Stored {len(tier1_indices)} vectors in Redis (Tier 1)")
print(f"Stored {len(tier2_indices)} vectors in Local ChromaDB (Tier 2)")
print(f"Stored {len(tier3_indices)} vectors in Remote ChromaDB (Tier 3)")