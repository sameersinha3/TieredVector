import numpy as np
import redis
import pickle
import plyvel

from google.cloud import storage
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

'''
Initial temperatures based on a query dataset (these will be moved around later)
'''

doc_embeddings = np.load("wiki_embeddings.npy")
query_dataset = load_dataset("natural_questions", split="train")
queries = [entry['question'] for entry in dataset]

model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = model.encode(queries)

k = 3
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(doc_embeddings)

temperature = np.zeros(len(doc_embeddings))
alpha = 0.9

_, indices = nbrs.kneighbors(query_embeddings)
for query_topk in indices:
    for idx in query_topk:
        temperature[idx] = alpha * temperature[idx] + 1

# If we change the values here make sure to modify config.go instead
tier1_threshold = np.percentile(temperature, 95)
tier2_threshold = np.percentile(temperature, 75)

tier_assignment = np.zeros(len(doc_embeddings), dtype=int)
tier_assignment[temperature >= tier1_threshold] = 1
tier_assignment[(temperature < tier1_threshold) & (temperature >= tier2_threshold)] = 2
tier_assignment[temperature < tier2_threshold] = 3

# Connect to local Redis
r = redis.Redis(host='localhost', port=6379)

# Store Tier 1 vectors
tier1_indices = np.where(tier_assignment == 1)[0]
for idx in tier1_indices:
    r.set(f'vector:{idx}', pickle.dumps(doc_embeddings[idx]))

# Store Tier 2 vectors
db = plyvel.DB('./rocksdb_data/', create_if_missing=True)
tier2_indices = np.where(tier_assignment == 2)[0]
for idx in tier2_indices:
    db.put(f'vector_{idx}'.encode(), pickle.dumps(doc_embeddings[idx]))

# Store Tier 3 vectors up to 1000 - want to remain within free tier
client = storage.Client()
bucket = client.get_bucket('your-bucket-name') # Need to configure GCS storage
tier3_indices = np.where(tier_assignment == 3)[0]
for idx in tier3_indices[:1000]:
    blob = bucket.blob(f'vector_{idx}')
    blob.upload_from_string(pickle.dumps(doc_embeddings[idx]))