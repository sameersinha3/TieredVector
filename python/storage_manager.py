import redis
import pickle
import numpy as np
import lmdb
import json
from google.cloud import storage
from google.oauth2 import service_account


'''
STORAGE MANAGER CLASS MODIFIED TO USE GCS CONFIGURATION

Example Usage in sandbox.py (temporary file)
'''

class StorageManager:
    def __init__(self, redis_host='localhost', redis_port=6379, lmdb_path='./tier2_lmdb',
                 gcs_project='VectorTier', gcs_bucket=None, gcs_blob_name=None, sa_key_path=None):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.lmdb_path = lmdb_path

        self.redis_client = None
        self.lmdb_env = None

        # GCS setup
        self.gcs_project = gcs_project
        self.gcs_bucket_name = gcs_bucket
        self.gcs_blob_name = gcs_blob_name
        self.sa_key_path = sa_key_path
        self.gcs_client = None
        self.gcs_bucket = None
        self.gcs_blob = None

        self.tier1_threshold = 0.9
        self.tier2_threshold = 0.7


    def initialize(self) -> bool:
        """Initialize Redis, LMDB, and GCS connections."""
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=False)
            self.redis_client.ping()
            print("Redis connected")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            return False
            
        try:
            self.lmdb_env = lmdb.open(self.lmdb_path, map_size=1_000_000_000)
            print("LMDB opened")
        except Exception as e:
            print(f"Failed to open LMDB: {e}")
            self.lmdb_env = None
            
        try:
            creds = service_account.Credentials.from_service_account_file(self.sa_key_path)
            self.gcs_client = storage.Client(project=self.gcs_project, credentials=creds)
            self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            self.gcs_blob = self.gcs_bucket.blob(self.gcs_blob_name)
            print("GCS initialized")
        except Exception as e:
            print(f"Failed to initialize GCS: {e}")
            self.gcs_client = None

        return True


    def determine_tier(self, temperature: float) -> int:
        if temperature >= self.tier1_threshold:
            return 1  # Redis
        elif temperature >= self.tier2_threshold:
            return 2  # LMDB
        else:
            return 3  # GCS


    def store_document(self, doc_id: int, embedding: np.ndarray, temperature: float) -> bool:
        tier = self.determine_tier(temperature)
        
        try:
            if tier == 1:
                return self._store_in_redis(doc_id, embedding)
            elif tier == 2:
                return self._store_in_lmdb(doc_id, embedding)
            else:
                return self._store_in_gcs(doc_id, embedding)
        except Exception as e:
            print(f"Failed to store document {doc_id}: {e}")
            return False


    def _store_in_redis(self, doc_id: int, embedding: np.ndarray) -> bool:
        try:
            key = f"doc{doc_id}"
            value = pickle.dumps(embedding)
            self.redis_client.set(key, value)
            return True
        except Exception as e:
            print(f"Redis store error for doc {doc_id}: {e}")
            return False


    def _store_in_lmdb(self, doc_id: int, embedding: np.ndarray) -> bool:
        if not self.lmdb_env:
            print(f"LMDB not available, skipping doc {doc_id}")
            return False
        try:
            with self.lmdb_env.begin(write=True) as txn:
                key = f"doc{doc_id}".encode()
                value = pickle.dumps(embedding)
                txn.put(key, value)
            return True
        except Exception as e:
            print(f"LMDB store error for doc {doc_id}: {e}")
            return False


    def _store_in_gcs(self, doc_id: int, embedding: np.ndarray) -> bool:
        if not self.gcs_blob:
            print("GCS not initialized")
            return False
        try:
            datapoint = {
                "datapoint_id": f"doc{doc_id}",
                "feature_vector": embedding.astype(np.float32).tolist(),
            }

            # Download existing data to append to it
            data = ""
            if self.gcs_blob.exists():
                data = self.gcs_blob.download_as_text()

            new_data = data + json.dumps(datapoint) + "\n"
            self.gcs_blob.upload_from_string(new_data)
            return True
        except Exception as e:
            print(f"GCS store error for doc {doc_id}: {e}")
            return False
    

    def _promote_from_gcs_to_lmdb(self, doc_id, remove=True):
        # If we want to remove the vector from GCS, we need to reupload the blob. We can toggle this with remove=False
        # If remove=False, we keep the vector in GCS and replicate it in LMDB.
        if remove:
            newlines = []
        data = self.gcs_blob.download_as_text().splitlines()
        for line in data:
            entry = json.loads(line)
            id, vec = entry["datapoint_id"][3:], entry["feature_vector"]
            if remove and int(id) != doc_id:
                newlines.append(line)
            if int(id) == doc_id:
                retrieved_id, retrieved_vec = id, vec
                break
        
        if remove:
            new_blob_text = "\n".join(newlines)
            self.gcs_blob.upload_from_string(new_blob_text, content_type="text/plain")
        
        self._store_in_lmdb(retrieved_id, retrieved_vec)
    
    def _promote_from_lmdb_to_redis(self, doc_id):
        with self.lmdb_env.begin() as txn:
            value = txn.get(f"doc{doc_id}".encode())  # LMDB keys must be bytes
            if not value:
                print(f"[Promote] doc_id {doc_id} not found in LMDB.")
                return

            # Deserialize the embedding
            embedding = pickle.loads(value)
            trimmed_id = doc_id[3:]

            self._store_in_redis(trimmed_id, embedding)
        
    def _demote_from_redis_to_lmdb(self, doc_id):
        trimmed_id = doc_id[3:] 

        value = self.redis_client.get(trimmed_id)
        embedding = pickle.loads(value)
        self._store_in_lmdb(trimmed_id, embedding)

        self.redis_client.delete(trimmed_id)

    def _demote_from_lmdb_to_gcs(self, doc_id):
        with self.lmdb_env.begin() as txn:
            value = txn.get(f"doc{doc_id}".encode()) 
            if not value:
                print(f"[Promote] doc_id {doc_id} not found in LMDB.")
                return

            # Deserialize the embedding
            embedding = pickle.loads(value)
        
        data = self.gcs_blob.download_as_text().splitlines()
        datapoint = {
                "datapoint_id": doc_id,
                "feature_vector": embedding.astype(np.float32).tolist(),
            }
        new_data = data + json.dumps(datapoint) + "\n"
        self.gcs_blob.upload_from_string(new_data)


    def retrieve_document(self, query_embedding: np.ndarray, k: int = 3, threshold: float = 0.75):
        query_embedding = np.array(query_embedding, dtype=np.float32)
        retrieved = []

        # --- Tier 1: Redis ---
        if self.redis_client:
            try:
                keys = self.redis_client.keys("vector:*")
                candidates = []
                for key in keys:
                    emb = pickle.loads(self.redis_client.get(key))
                    score = np.dot(query_embedding, emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                    )
                    if score >= threshold:
                        candidates.append((score, key.decode().split(":")[1]))
                candidates.sort(reverse=True, key=lambda x: x[0])
                docs = [{"id": did, "score": s, "source": "redis"} for s, did in candidates[:k]]
                retrieved.extend(docs)
                if len(retrieved) >= k:
                    return retrieved
            except Exception as e:
                print(f"[Redis simple search] error: {e}")


        # --- Tier 2: LMDB ---
        remaining = k - len(retrieved)
        if remaining > 0 and self.lmdb_env:
            try:
                with self.lmdb_env.begin() as txn:
                    cursor = txn.cursor()
                    best = []
                    for key, value in cursor:
                        emb = pickle.loads(value)
                        score = np.dot(query_embedding, emb) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                        )
                        if score >= threshold:
                            best.append((score, key.decode()))
                    best.sort(reverse=True, key=lambda x: x[0])
                    docs = [{"id": k, "score": s, "source": "lmdb"} for s, k in best[:remaining]]
                    retrieved.extend(docs)
                    if len(retrieved) >= k:
                        return retrieved
            except Exception as e:
                print(f"[LMDB] search error: {e}")

        # --- Tier 3: GCS ---
        remaining = k - len(retrieved)
        if remaining > 0 and self.gcs_blob:
            try:
                data = self.gcs_blob.download_as_text().splitlines()
                candidates = []
                for line in data:
                    entry = json.loads(line)
                    emb = np.array(entry["feature_vector"], dtype=np.float32)
                    score = np.dot(query_embedding, emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                    )
                    if score >= threshold:
                        candidates.append((score, entry["datapoint_id"]))
                candidates.sort(reverse=True, key=lambda x: x[0])
                docs = [{"id": did, "score": s, "source": "gcs"} for s, did in candidates[:remaining]]
                retrieved.extend(docs)
            except Exception as e:
                print(f"[GCS] retrieval error: {e}")

        return retrieved if retrieved else None


    def close(self):
        if self.redis_client:
            self.redis_client.close()
        if self.lmdb_env:
            self.lmdb_env.close()
        print("Storage connections closed")
