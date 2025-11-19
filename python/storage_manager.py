import redis
import pickle
import numpy as np
import chromadb
import os
from access_tracker import AccessTracker
from document_scorer import DocumentScorer
from tier_orchestrator import TierOrchestrator
import scoring_config as config

class StorageManager:
    def __init__(self, redis_host='localhost', redis_port=6379, 
                 tier2_path='./tier2_chroma_db',           # For Local Tier 2
                 tier3_host=os.getenv("VM_IP"),          # For Remote Tier 3
                 tier3_port=8000,
                 tier3_collection='cold_vectors',
                 enable_scoring=config.ENABLE_SCORING):
        
        # --- Tier 1 (Redis) ---
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None

        # --- Tier 2 (Local ChromaDB) ---
        self.tier2_path = tier2_path
        self.tier2_client = None
        self.tier2_collection = None

        # --- Tier 3 (Remote ChromaDB) ---
        self.tier3_host = tier3_host
        self.tier3_port = tier3_port
        self.tier3_collection_name = tier3_collection
        self.tier3_client = None
        self.tier3_collection = None

        # --- Temperature Thresholds ---
        self.tier1_threshold = 0.9
        self.tier2_threshold = 0.7

        self.enable_scoring = enable_scoring
        self.tracker = None
        self.scorer = None
        self.orchestrator = None
        self.accesses = {1: 0, 2: 0, 3: 0}


    def initialize(self) -> bool:
        """Initialize Redis, Local ChromaDB, and Remote ChromaDB connections."""
        
        # --- Tier 1 (Redis) ---
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=False)
            self.redis_client.ping()
            print("Tier 1 (Redis) connected")
        except Exception as e:
            print(f"Failed to connect to Tier 1 Redis: {e}")
            return False
            
        # --- Tier 2 (Local ChromaDB) ---
        try:
            self.tier2_client = chromadb.PersistentClient(path=self.tier2_path)
            self.tier2_collection = self.tier2_client.get_or_create_collection(
                name="warm_vectors",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Tier 2 (Local ChromaDB) initialized at {self.tier2_path}")
        except Exception as e:
            print(f"Failed to initialize Tier 2 ChromaDB: {e}")
            return False # Fail if local DB can't start
            
        # --- Tier 3 (Remote ChromaDB) ---
        try:
            self.tier3_client = chromadb.HttpClient(
                host=self.tier3_host, 
                port=self.tier3_port
            )
            self.tier3_collection = self.tier3_client.get_or_create_collection(
                name=self.tier3_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Tier 3 (Remote ChromaDB) connected (Host: {self.tier3_host})")
        except Exception as e:
            print(f"Failed to initialize Tier 3 ChromaDB: {e}")
            self.tier3_client = None # Allow app to run without remote DB

        if self.enable_scoring:
            try:
                self._initialize_scoring()
                if self.scorer:
                    assignments = self._scan_tier_assignments()
                    self.scorer.bulk_update_tiers(assignments)
            except Exception as e:
                print(f"Error initializing scoring: {e}")

        return True


    def determine_tier(self, temperature: float) -> int:
        if temperature >= self.tier1_threshold:
            return 1  # Redis
        elif temperature >= self.tier2_threshold:
            return 2  # CHANGED: Tier 2 (Local ChromaDB)
        else:
            return 3  # Tier 3 (Remote ChromaDB)


    def store_document(self, doc_id: int, embedding: np.ndarray, temperature: float) -> bool:
        tier = self.determine_tier(temperature)
        
        try:
            if tier == 1:
                return self._store_in_redis(doc_id, embedding)
            elif tier == 2:
                # CHANGED: Store in local ChromaDB
                return self._store_in_tier2_chroma(doc_id, embedding)
            else:
                # CHANGED: Store in remote ChromaDB
                return self._store_in_tier3_chroma(doc_id, embedding)
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

        # --- NEW: Replaces _store_in_lmdb ---
    def _store_in_tier2_chroma(self, doc_id: int, embedding: np.ndarray) -> bool:
        """Stores a single document in Tier 2 (Local ChromaDB)."""
        if not self.tier2_collection:
            print("Tier 2 ChromaDB not initialized")
            return False
        try:
            doc_key = f"doc{doc_id}"
            self.tier2_collection.upsert(
                ids=[doc_key],
                embeddings=[embedding.astype(np.float32).tolist()]
            )
            return True
        except Exception as e:
            print(f"Tier 2 ChromaDB store error for doc {doc_id}: {e}")
            return False

                # --- RENAMED: Was _store_in_chroma ---
    def _store_in_tier3_chroma(self, doc_id: int, embedding: np.ndarray) -> bool:
        """Stores a single document in Tier 3 (Remote ChromaDB)."""
        if not self.tier3_collection:
            print("Tier 3 ChromaDB not initialized")
            return False
        try:
            doc_key = f"doc{doc_id}"
            self.tier3_collection.upsert(
                ids=[doc_key],
                embeddings=[embedding.astype(np.float32).tolist()]
            )
            return True
        except Exception as e:
            print(f"Tier 3 ChromaDB store error for doc {doc_id}: {e}")
            return False
    
    
    def _promote_from_tier3_to_tier2(self, doc_id, remove=True):
        """Fetches a doc from Tier 3 (Remote) and puts it in Tier 2 (Local)."""
        if not self.tier3_collection or not self.tier2_collection:
            print("DBs not initialized for promotion")
            return
        
        try:
            doc_key = f"doc{doc_id}"
            result = self.tier3_collection.get(ids=[doc_key], include=["embeddings"])
            
            if not result['ids']:
                print(f"[Promote] doc_id {doc_key} not found in Tier 3.")
                return

            # Add to Tier 2
            self.tier2_collection.upsert(
                ids=result['ids'],
                embeddings=result['embeddings']
            )

            if remove:
                self.tier3_collection.delete(ids=[doc_key])
                
        except Exception as e:
            print(f"[Promote T3->T2] Error: {e}")

    
    def _demote_from_tier2_to_tier3(self, doc_id, remove=True):
        """Fetches a doc from Tier 2 (Local) and puts it in Tier 3 (Remote)."""
        if not self.tier3_collection or not self.tier2_collection:
            print("DBs not initialized for demotion")
            return

        try:
            doc_key = f"doc{doc_id}"
            result = self.tier2_collection.get(ids=[doc_key], include=["embeddings"])

            if not result['ids']:
                print(f"[Demote] doc_id {doc_key} not found in Tier 2.")
                return

            # Add to Tier 3
            self.tier3_collection.upsert(
                ids=result['ids'],
                embeddings=result['embeddings']
            )

            if remove:
                self.tier2_collection.delete(ids=[doc_key])

        except Exception as e:
            print(f"[Demote T2->T3] Error: {e}")


    def _promote_from_tier2_to_redis(self, doc_id, remove=True):
        """Fetches a doc from Tier 2 (Local) and puts it in Tier 1 (Redis)."""
        if not self.tier2_collection or not self.redis_client:
            print("DBs not initialized for promotion")
            return

        try:
            doc_key = f"doc{doc_id}"
            result = self.tier2_collection.get(ids=[doc_key], include=["embeddings"])

            if not result['ids']:
                print(f"[Promote] doc_id {doc_key} not found in Tier 2.")
                return
            
            # Convert list embedding back to numpy array for Redis
            embedding = np.array(result['embeddings'][0], dtype=np.float32)
            trimmed_id = doc_key[3:] # 'doc123' -> '123'
            
            # Store in Redis
            self._store_in_redis(trimmed_id, embedding)

            if remove:
                self.tier2_collection.delete(ids=[doc_key])
                
        except Exception as e:
            print(f"[Promote T2->Redis] Error: {e}")


    def _demote_from_redis_to_tier2(self, doc_id, remove=True):
        """Fetches a doc from Tier 1 (Redis) and puts it in Tier 2 (Local)."""
        if not self.tier2_collection or not self.redis_client:
            print("DBs not initialized for demotion")
            return
        
        try:
            doc_key = f"doc{doc_id}"
            value = self.redis_client.get(doc_key)
            if not value:
                print(f"[Demote] doc_id {doc_key} not found in Redis.")
                return

            # Get embedding from Redis
            embedding = pickle.loads(value)

            # Add to Tier 2
            self.tier2_collection.upsert(
                ids=[doc_key],
                embeddings=[embedding.astype(np.float32).tolist()]
            )

            if remove:
                self.redis_client.delete(doc_key)
        
        except Exception as e:
            print(f"[Demote Redis->T2] Error: {e}")


    def retrieve_document(self, query_embedding: np.ndarray, k: int = 3, threshold: float = 0.75):
        
        # 1. --- NORMALIZE QUERY ONCE ---
        # (This assumes all stored vectors are also normalized)
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            print("Warning: Zero-vector query.")
            return []
            
        query_norm = (query_embedding / norm).astype(np.float32)
        query_list = query_norm.tolist() # For ChromaDB
        
        retrieved = []
        retrieved_ids = set()

        # --- Tier 1: Redis (Manual Scan) ---
        if self.redis_client:
            try:
                keys = self.redis_client.keys("doc*") 
                candidates = []
                for key in keys:
                    # Load the stored NORMALIZED vector
                    emb_norm = pickle.loads(self.redis_client.get(key))
                    
                    # Dot product of two normalized vectors IS the cosine similarity
                    score = np.dot(query_norm, emb_norm)
                    
                    if score >= threshold:
                        doc_id = key.decode()
                        candidates.append((score, doc_id))
                
                candidates.sort(reverse=True, key=lambda x: x[0])
                for s, did in candidates:
                    if len(retrieved) < k and did not in retrieved_ids:
                        retrieved.append({"id": did, "score": s, "source": "redis (T1)"})
                        retrieved_ids.add(did)
                        self.accesses[1] += 1

                if len(retrieved) >= k:
                    return retrieved
            except Exception as e:
                print(f"[Redis simple search] error: {e}")


        # --- Tier 2: Local ChromaDB (ANN Search) ---
        remaining = k - len(retrieved)
        if remaining > 0 and self.tier2_collection:
            try:
                # Query using the normalized query list
                results = self.tier2_collection.query(
                    query_embeddings=[query_list],
                    n_results=remaining
                )
                
                ids = results['ids'][0]
                distances = results['distances'][0] # This is now Cosine Distance
                
                for doc_id, dist in zip(ids, distances):
                    if doc_id in retrieved_ids:
                        continue
                    
                    # Score = 1.0 - Cosine Distance
                    score = 1.0 - dist 

                    if score >= threshold:
                        if len(retrieved) < k:
                            retrieved.append({"id": doc_id, "score": score, "source": "chroma_local (T2)"})
                            retrieved_ids.add(doc_id)
                            self.accesses[2] += 1
            
            except Exception as e:
                print(f"[ChromaDB T2] retrieval error: {e}")
        
        if len(retrieved) >= k:
            return retrieved


        # --- Tier 3: Remote ChromaDB (ANN Search) ---
        remaining = k - len(retrieved) # Re-check remaining k
        if remaining > 0 and self.tier3_collection:
            try:
                # Query using the normalized query list
                results = self.tier3_collection.query(
                    query_embeddings=[query_list],
                    n_results=remaining
                )
                
                ids = results['ids'][0]
                distances = results['distances'][0] # This is now Cosine Distance
                
                for doc_id, dist in zip(ids, distances):
                    if doc_id in retrieved_ids:
                        continue
                    
                    score = 1.0 - dist

                    if score >= threshold:
                        if len(retrieved) < k:
                            retrieved.append({"id": doc_id, "score": score, "source": "chroma_remote (T3)"})
                            retrieved_ids.add(doc_id)
                            self.accesses[3] += 1
                        
            except Exception as e:
                print(f"[ChromaDB T3] retrieval error: {e}")

        if self.enable_scoring and self.tracker and retrieved:
            try:
                for doc in retrieved:
                    doc_id = doc['id']
                    if isinstance(doc_id, bytes):
                        doc_id = doc_id.decode()
                    self.tracker.record_access(doc_id)
                    if self.scorer:
                        src = doc.get('source', '')
                        if 'T1' in src:
                            tier = 1
                        elif 'T2' in src:
                            tier = 2
                        else:
                            tier = 3
                        self.scorer.update_document_tier(doc_id, tier)
            except Exception as e:
                print(f"Error tracking access: {e}")

        return retrieved if retrieved else None


    def close(self):
        if self.redis_client:
            self.redis_client.close()
        print("Storage connections closed")
    
    def summary(self):
        print("Access Patterns")
        print(f"Tier 1 accesses: {self.accesses[1]}")
        print(f"Tier 2 accesses: {self.accesses[2]}")
        print(f"Tier 3 accesses: {self.accesses[3]}")


    def _initialize_scoring(self):
        try:
            self.tracker = AccessTracker()
            self.scorer = DocumentScorer(self.tracker)
            self.orchestrator = TierOrchestrator(self, self.tracker, self.scorer)
        except Exception as e:
            print(f"Failed to initialize scoring system: {e}")
            self.enable_scoring = False
            self.tracker = None
            self.scorer = None
            self.orchestrator = None


    def _scan_tier_assignments(self) -> dict:
        assignments = {}
        try:
            if self.redis_client:
                keys = self.redis_client.keys("doc*")
                for key in keys:
                    doc_id = key.decode()
                    assignments[doc_id] = 1
            if self.tier2_collection:
                result = self.tier2_collection.get()
                for doc_id in result['ids']:
                    assignments[doc_id] = 2
            if self.tier3_collection:
                result = self.tier3_collection.get()
                for doc_id in result['ids']:
                    assignments[doc_id] = 3
        except Exception as e:
            print(f"Error scanning tier assignments: {e}")
        return assignments


    def get_scoring_stats(self) -> dict:
        stats = {'scoring_enabled': self.enable_scoring}
        if not self.enable_scoring:
            return stats
        if self.tracker:
            top_docs = self.tracker.get_top_accessed(10)
            stats['top_accessed'] = [
                {
                    'doc_id': doc_id,
                    'score': score,
                    'stats': self.tracker.get_access_stats(doc_id)
                }
                for doc_id, score in top_docs
            ]
        if self.scorer:
            stats['tier_distribution'] = self.scorer.get_tier_distribution()
            stats['migration_stats'] = self.scorer.get_migration_stats(24)
        if self.orchestrator:
            stats['orchestrator_status'] = self.orchestrator.get_status()
        return stats


    def force_migration(self, doc_id: str, target_tier: int) -> bool:
        if self.orchestrator:
            return self.orchestrator.force_migration(doc_id, target_tier)
        return False