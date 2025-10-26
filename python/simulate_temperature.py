import cohere
import numpy as np
import redis
import pickle
import lmdb
import os

from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv


'''
Initial temperatures based on a query dataset (these will be moved around later)
'''
print("START")
load_dotenv()  # load .env file
API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(API_KEY)

queries = [
    "What is the capital of France?",
    "Who wrote Harry Potter?",
    "When was the Declaration of Independence signed?",
    "What is the tallest mountain in the world?",
    "How many continents are there?"
]
doc_embeddings = np.load("wiki_embeddings.npy")
'''
query_dataset = load_dataset("natural_questions", split="train[:10]")
queries = [entry['question'] for entry in query_dataset]
print(f"Number of examples in the dataset: {len(query_dataset)}")
'''
response = co.embed(
    model="multilingual-22-12",
    texts=queries
)

query_embeddings = np.array(response.embeddings)

print("Finished Encoding Queries")

k = 3
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(doc_embeddings)

print("Found Nearest Neighbors")

temperature = np.zeros(len(doc_embeddings))
alpha = 0.9

_, indices = nbrs.kneighbors(query_embeddings)
for query_topk in indices:
    for idx in query_topk:
        temperature[idx] = alpha * temperature[idx] + 1

print("Assigned Temperatures")

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

print("Redis Complete")

# Store Tier 2 vectors in LMDB (key-value store)
env = lmdb.open('tier2_lmdb', map_size=1_000_000_000)  
tier2_indices = np.where(tier_assignment == 2)[0]
with env.begin(write=True) as txn:
    for idx in tier2_indices:
        key = f'vector_{idx}'.encode('utf-8')
        value = pickle.dumps(doc_embeddings[idx])
        txn.put(key, value)
env.close()

print("LMDB Complete")
# Store Tier 3 vectors up to 1000 - want to remain within free tier
# Commented out GCS for now 
# client = storage.Client()
# bucket = client.get_bucket('your-bucket-name') 
# tier3_indices = np.where(tier_assignment == 3)[0]
# for idx in tier3_indices[:1000]:
#     blob = bucket.blob(f'vector_{idx}')
#     blob.upload_from_string(pickle.dumps(doc_embeddings[idx]))

print(f"Stored {len(tier1_indices)} vectors in Redis (Tier 1)")
print(f"Stored {len(tier2_indices)} vectors in DBM (Tier 2)")
print(f"Tier 1 threshold: {tier1_threshold:.3f}")
print(f"Tier 2 threshold: {tier2_threshold:.3f}")