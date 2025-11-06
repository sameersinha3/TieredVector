import lmdb
import numpy as np
import json
import os
import pickle
import redis

from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
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

tier3_indices = np.where(tier_assignment == 3)[0]  # array of indices
embeddings = np.array(doc_embeddings)  # shape (N, d)
project = "VectorTier"
location = os.getenv("REGION")
bucket = os.getenv("BUCKET_NAME")
blob = os.getenv("BLOB")

# Initialize GCS Storage
credentials = service_account.Credentials.from_service_account_file(
    os.getenv("SA_KEY")
)
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket(bucket)
blob = bucket.blob(blob)

count = 0
with blob.open("w") as f:
    for i in tier3_indices:
        datapoint = {
            "datapoint_id": f"doc{i}",  # unique ID for retrieval
            "feature_vector": embeddings[i].astype(np.float32).tolist(),
        }
        
        # Write each datapoint as a new line in the file
        f.write(json.dumps(datapoint) + "\n")
        count += 1

print(f"Stored {len(tier1_indices)} vectors in Redis (Tier 1)")
print(f"Stored {len(tier2_indices)} vectors in LMDB (Tier 2)")
print(f"Stored {len(tier3_indices)} vectors in GCS (Tier 3)")