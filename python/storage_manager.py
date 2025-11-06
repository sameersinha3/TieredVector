import redis
import pickle
import numpy as np
import lmdb
import json
import os

# CHANGED: Import chromadb
import chromadb

'''
STORAGE MANAGER CLASS MODIFIED TO USE CHROMA DB (VM) CONFIGURATION
'''

class StorageManager:
    def __init__(self, redis_host='localhost', redis_port=6379, lmdb_path='./tier2_lmdb',
                 # NEW: ChromaDB client settings
                 chroma_host=os.getenv("VM_IP")
                 chroma_port=8000,
                 chroma_collection='cold_vectors'):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.lmdb_path = lmdb_path

        self.redis_client = None
        self.lmdb_env = None

        # NEW: ChromaDB setup
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.chroma_collection_name = chroma_collection
        self.chroma_client = None
        self.chroma_collection = None # This will be our collection reference

        self.tier1_threshold = 0.9
        self.tier2_threshold = 0.7


    def initialize(self) -> bool:
        """Initialize Redis, LMDB, and ChromaDB connections."""
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
            
        # CHANGED: Initialize ChromaDB client instead of Firestore/GCS
        try:
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host, 
                port=self.chroma_port
            )
            # This ensures the collection exists
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.chroma_collection_name
            )
            print(f"ChromaDB connected (Host: {self.chroma_host}, Collection: {self.chroma_collection_name})")
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None

        return True


    def determine_tier(self, temperature: float) -> int:
        if temperature >= self.tier1_threshold:
            return 1  # Redis
        elif temperature >= self.tier2_threshold:
            return 2  # LMDB
        else:
            return 3  # CHANGED: ChromaDB


    def store_document(self, doc_id: int, embedding: np.ndarray, temperature: float) -> bool:
        tier = self.determine_tier(temperature)
        
        try:
            if tier == 1:
                return self._store_in_redis(doc_id, embedding)
            elif tier == 2:
                return self._store_in_lmdb(doc_id, embedding)
            else:
                # CHANGED: Store in ChromaDB
                return self._store_in_chroma(doc_id, embedding)
        except Exception as e:
            print(f"Failed to store document {doc_id}: {e}")
            return False

    # ... _store_in_redis and _store_in_lmdb are unchanged ...
    def _store_in_redis(self, doc_id: int, embedding: np.ndarray) -> bool:
        # (This code is unchanged)
        try:
            key = f"doc{doc_id}"
            value = pickle.dumps(embedding)
            self.redis_client.set(key, value)
            return True
        except Exception as e:
            print(f"Redis store error for doc {doc_id}: {e}")
            return False


    def _store_in_lmdb(self, doc_id: int, embedding: np.ndarray) -> bool:
        # (This code is unchanged)
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
            
    
    # CHANGED: Replaced _store_in_firestore with _store_in_chroma
    def _store_in_chroma(self, doc_id: int, embedding: np.ndarray) -> bool:
        """Stores a single document in ChromaDB."""
        if not self.chroma_collection:
            print("ChromaDB not initialized")
            return False
        try:
            doc_key = f"doc{doc_id}"
            
            # ChromaDB stores the vector itself
            # We use upsert to create or overwrite
            self.chroma_collection.upsert(
                ids=[doc_key],
                embeddings=[embedding.astype(np.float32).tolist()]
                # You could also add metadatas=[{"source": "wiki"}]
            )
            return True
        except Exception as e:
            print(f"ChromaDB store error for doc {doc_id}: {e}")
            return False
    

    # CHANGED: Replaced _promote_from_firestore_to_lmdb
    def _promote_from_chroma_to_lmdb(self, doc_id, remove=True):
        """Fetches a doc from ChromaDB and puts it in LMDB."""
        if not self.chroma_collection:
            print("ChromaDB not initialized")
            return
        
        try:
            doc_key = f"doc{doc_id}"
            # Get the embedding by its ID
            result = self.chroma_collection.get(ids=[doc_key], include=["embeddings"])
            
            if not result['embeddings']:
                print(f"[Promote] doc_id {doc_key} not found in ChromaDB.")
                return

            embedding = np.array(result['embeddings'][0], dtype=np.float32)
            self._store_in_lmdb(doc_id, embedding)

            if remove:
                self.chroma_collection.delete(ids=[doc_key])
                
        except Exception as e:
            print(f"[Promote Chroma->LMDB] Error: {e}")

    # ... _promote_from_lmdb_to_redis is unchanged ...
    
    # CHANGED: Replaced _demote_from_lmdb_to_gcs/firestore
    def _demote_from_lmdb_to_chroma(self, doc_id):
        """Demotes from LMDB to ChromaDB"""
        key_bytes = f"doc{doc_id}".encode()
        
        with self.lmdb_env.begin(write=True) as txn:
            value = txn.get(key_bytes) 
            if not value:
                print(f"[Demote] doc_id {doc_id} not found in LMDB.")
                return

            embedding = pickle.loads(value)
            trimmed_id = key_bytes.decode()[3:] # "123"
        
            # Store in ChromaDB
            self._store_in_chroma(trimmed_id, embedding)
            
            # Delete from LMDB
            txn.delete(key_bytes)


    def retrieve_document(self, query_embedding: np.ndarray, k: int = 3, threshold: float = 0.75):
        query_embedding = np.array(query_embedding, dtype=np.float32)
        retrieved = []
        retrieved_ids = set()

        # ... Tier 1 (Redis) and Tier 2 (LMDB) are unchanged ...
        # --- Tier 1: Redis ---
        if self.redis_client:
            try:
                keys = self.redis_client.keys("doc*") 
                candidates = []
                for key in keys:
                    emb = pickle.loads(self.redis_client.get(key))
                    score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    if score >= threshold:
                        doc_id = key.decode()
                        candidates.append((score, doc_id))
                
                candidates.sort(reverse=True, key=lambda x: x[0])
                for s, did in candidates:
                    if len(retrieved) < k and did not in retrieved_ids:
                        retrieved.append({"id": did, "score": s, "source": "redis"})
                        retrieved_ids.add(did)
                if len(retrieved) >= k: return retrieved
            except Exception as e: print(f"[Redis simple search] error: {e}")

        # --- Tier 2: LMDB ---
        remaining = k - len(retrieved)
        if remaining > 0 and self.lmdb_env:
            try:
                with self.lmdb_env.begin() as txn:
                    cursor = txn.cursor()
                    best = []
                    for key_bytes, value in cursor:
                        key = key_bytes.decode()
                        if key in retrieved_ids: continue
                        emb = pickle.loads(value)
                        score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                        if score >= threshold:
                            best.append((score, key))
                    
                    best.sort(reverse=True, key=lambda x: x[0])
                    for s, did in best:
                         if len(retrieved) < k and did not in retrieved_ids:
                            retrieved.append({"id": did, "score": s, "source": "lmdb"})
                            retrieved_ids.add(did)
                    if len(retrieved) >= k: return retrieved
            except Exception as e: print(f"[LMDB] search error: {e}")


        # --- Tier 3: ChromaDB ---
        # CHANGED: Search from ChromaDB instead of Firestore scan
        remaining = k - len(retrieved)
        if remaining > 0 and self.chroma_collection:
            try:
                # This is the magic!
                # ChromaDB does the search for you on the server.
                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=remaining
                    # You can also add a 'where' filter
                )
                
                ids = results['ids'][0]
                distances = results['distances'][0]
                
                for doc_id, dist in zip(ids, distances):
                    if doc_id in retrieved_ids:
                        continue
                        
                    # Note: Chroma gives distance (smaller is better). 
                    # Your code used similarity (higher is better).
                    # We'll just use distance for now.
                    # 1 - dist is cosine similarity if using 'cosine' distance
                    score = 1 - dist # Assumes cosine distance

                    if score >= threshold:
                        if len(retrieved) < k:
                            retrieved.append({"id": doc_id, "score": score, "source": "chroma"})
                            retrieved_ids.add(doc_id)
                        
            except Exception as e:
                print(f"[ChromaDB] retrieval error: {e}")

        return retrieved if retrieved else None


    def close(self):
        if self.redis_client:
            self.redis_client.close()
        if self.lmdb_env:
            self.lmdb_env.close()
        # NEW: Chroma client doesn't have a .close()
        print("Storage connections closed")