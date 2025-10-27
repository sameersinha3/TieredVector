import redis
import pickle
import numpy as np
from typing import List, Optional
import lmdb
from google.cloud import storage


class StorageManager:

    def __init__(self, redis_host='localhost', redis_port=6379, lmdb_path='./tier2_lmdb',
                 gcs_bucket_name="vectier", project="VectorTier"):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.lmdb_path = lmdb_path
        self.gcs_bucket_name = gcs_bucket_name
        self.project = project

        self.redis_client = None
        self.lmdb = None
        self.gcs_client = None
        self.gcs_bucket_handle = None

        self.tier1_threshold=0.9
        self.tier2_threshold=0.7

    def initialize(self) -> bool:
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
            if self.gcs_bucket_name:
                self.gcs_client = storage.Client(project=self.project)
                self.gcs_bucket_handle = self.gcs_client.get_bucket(self.gcs_bucket_name)
                print(f"GCS bucket '{self.gcs_bucket_name}' initialized")
        except Exception as e:
            print(f"Failed to initialize GCS: {e}")
            self.gcs_client = None
            self.gcs_bucket_handle = None
            
        return True
        
    def determine_tier(self, temperature: float) -> int:
        """Determine which tier a document should be stored in based on temperature."""
        if temperature >= self.tier1_threshold:
            return 1  # Redis
        elif temperature >= self.tier2_threshold:
            return 2
        else:
            return 3
            
    def store_document(self, doc_id: int, embedding: np.ndarray, temperature: float) -> bool:
        tier = self.determine_tier(temperature)
        
        try:
            if tier == 1:
                success = self._store_in_redis(doc_id, embedding)
            elif tier == 2:
                success = self._store_in_lmdb(doc_id, embedding)
            else:  # tier == 3
                success = self._store_in_gcs(doc_id, embedding)
                    
            return success
            
        except Exception as e:
            print(f"Failed to store document {doc_id}: {e}")
            return False
        
    def _store_in_redis(self, doc_id: int, embedding: np.ndarray) -> bool:
        try:
            key = f"vector:{doc_id}"
            value = pickle.dumps(embedding)
            self.redis_client.set(key, value)
            return True
        except Exception as e:
            print(f"Redis store error for doc {doc_id}: {e}")
            return False
            
    def _store_in_lmdb(self, doc_id: int, embedding: np.ndarray) -> bool:
        if self.lmdb_env is None:
            print(f"LMDB not available, skipping doc {doc_id}")
            return False
            
        try:
            with self.lmdb_env.begin(write=True) as txn:
                key = f"vector_{doc_id}".encode()
                value = pickle.dumps(embedding)
                txn.put(key, value)
            return True
        except Exception as e:
            print(f"LMDB store error for doc {doc_id}: {e}")
            return False
            
    def _store_in_gcs(self, doc_id: int, embedding: np.ndarray) -> bool:
        try:
            blob_name = f"vector_{doc_id}"
            blob = self.gcs_bucket_handle.blob(blob_name)
            value = pickle.dumps(embedding)
            blob.upload_from_string(value)
            return True
        except Exception as e:
            print(f"GCS store error for doc {doc_id}: {e}")
            return False
            
    def retrieve_document(self, query_embedding: np.ndarray, k: int = 3, threshold: float = 0.75):
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # --- Tier 1: Redis ---
        if self.redis_client:
            try:
                query_vector = query_embedding.tobytes()
                query = f"*=>[KNN {k} @embedding $vector AS score]"
                params = {"vector": query_vector}

                res = self.redis_client.ft("doc_index").search(
                    query, query_params=params, sort_by="score", dialect=2
                )
                if len(res.docs) > 0:
                    docs = []
                    for doc in res.docs:
                        sim = 1 - float(doc.score)
                        if sim >= threshold:
                            docs.append({
                                "id": doc.id,
                                "score": sim,
                                "text": getattr(doc, "text", None),
                                "source": "redis"
                            })
                    print("[Tier 1: Redis] Found relevant docs")
                    return docs
            except Exception as e:
                print(f"[Redis] search error: {e}")

        # --- Tier 2: LMDB ---
        if hasattr(self, "lmdb_env") and self.lmdb_env:
            try:
                with self.lmdb_env.begin() as txn:
                    cursor = txn.cursor()
                    best = []
                    for key, value in cursor:
                        data = pickle.loads(value)
                        emb = np.array(data["embedding"], dtype=np.float32)
                        score = np.dot(query_embedding, emb) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                        )
                        if score >= threshold:
                            best.append((score, data))
                    if best:
                        best.sort(reverse=True, key=lambda x: x[0])
                        docs = [
                            {
                                "id": d["id"],
                                "score": float(s),
                                "text": d["text"],
                                "source": "lmdb"
                            }
                            for s, d in best[:k]
                        ]
                        print("[Tier 2: LMDB] Found relevant docs")
                        return docs
            except Exception as e:
                print(f"[LMDB] search error: {e}")

        # --- Tier 3: Vertex AI ---
        if hasattr(self, "vertex_client") and self.vertex_client:
            try:
                index = self.vertex_client.get_index(name=self.vertex_index_name)
                res = index.find_neighbors(
                    queries=[query_embedding.tolist()],
                    num_neighbors=k
                )
                if res and len(res[0].neighbors) > 0:
                    docs = [
                        {
                            "id": match.datapoint.datapoint_id,
                            "score": 1 - match.distance,  # convert distance to similarity
                            "source": "vertex"
                        }
                        for match in res[0].neighbors
                    ]
                    print("[Tier 3: Vertex AI] Found relevant docs")
                    return docs
            except Exception as e:
                print(f"[Vertex] search error: {e}")

        # Nothing found
        print("No relevant docs found in any tier.")
        return None
    

    def close(self):
        if self.redis_client:
            self.redis_client.close()
        if hasattr(self, "lmdb_env") and self.lmdb_env:
            self.lmdb_env.close()
        print("Storage connections closed")

