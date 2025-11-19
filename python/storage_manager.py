import redis
import pickle
import numpy as np
import chromadb
import os

class StorageManager:
    def __init__(self, redis_host='localhost', redis_port=6379, 
                 tier2_path='./tier2_chroma_db',           # For Local Tier 2
                 tier3_host=os.getenv("VM_IP"),          # For Remote Tier 3
                 tier3_port=8000,
                 tier3_collection='cold_vectors'):
        
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
        # (This code is unchanged)
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
    
    
    # --- ALL PROMOTION/DEMOTION FUNCTIONS RE-WRITTEN ---

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
                    
                    # *** THIS IS THE CHANGE ***
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

                if len(retrieved) >= k:
                    return retrieved
            except Exception as e:
                print(f"[Redis simple search] error: {e}")


        # --- Tier 2: Local ChromaDB (ANN Search) ---
        remaining = k - len(retrieved)
        if remaining > 0 and self.tier2_collection:
            try:
                # *** THIS IS THE CHANGE ***
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
                    
                    # *** THIS IS THE CHANGE ***
                    # Score = 1.0 - Cosine Distance
                    score = 1.0 - dist 

                    if score >= threshold:
                        if len(retrieved) < k:
                            retrieved.append({"id": doc_id, "score": score, "source": "chroma_local (T2)"})
                            retrieved_ids.add(doc_id)
            
            except Exception as e:
                print(f"[ChromaDB T2] retrieval error: {e}")
        
        if len(retrieved) >= k:
            return retrieved


        # --- Tier 3: Remote ChromaDB (ANN Search) ---
        remaining = k - len(retrieved) # Re-check remaining k
        if remaining > 0 and self.tier3_collection:
            try:
                # *** THIS IS THE CHANGE ***
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
                        
            except Exception as e:
                print(f"[ChromaDB T3] retrieval error: {e}")

        return retrieved if retrieved else None


    def update_temperature_thresholds(self, tier1_threshold: float, tier2_threshold: float):
        """Update temperature thresholds for tier assignment"""
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        print(f"Updated thresholds: Tier1 >= {tier1_threshold}, Tier2 >= {tier2_threshold}")
    
    def get_document_location(self, doc_id: int) -> tuple:
        """Get current tier location of a document. Returns (tier, found)"""
        doc_key = f"doc{doc_id}"
        
        # Check Tier 1 (Redis)
        if self.redis_client:
            if self.redis_client.exists(doc_key):
                return (1, True)
        
        # Check Tier 2 (Local ChromaDB)
        if self.tier2_collection:
            try:
                result = self.tier2_collection.get(ids=[doc_key])
                if result['ids']:
                    return (2, True)
            except:
                pass
        
        # Check Tier 3 (Remote ChromaDB)
        if self.tier3_collection:
            try:
                result = self.tier3_collection.get(ids=[doc_key])
                if result['ids']:
                    return (3, True)
            except:
                pass
        
        return (None, False)
    
    def reallocate_document(self, doc_id: int, new_temperature: float, embedding: np.ndarray = None) -> bool:
        """Reallocate a document to a new tier based on updated temperature"""
        current_tier, found = self.get_document_location(doc_id)
        
        if not found:
            print(f"Document {doc_id} not found in any tier. Storing in appropriate tier.")
            if embedding is not None:
                return self.store_document(doc_id, embedding, new_temperature)
            else:
                print(f"Error: Cannot store document {doc_id} without embedding")
                return False
        
        new_tier = self.determine_tier(new_temperature)
        
        # If already in correct tier, no action needed
        if current_tier == new_tier:
            return True
        
        # Need to move document
        # First, get the embedding if not provided
        if embedding is None:
            embedding = self._get_document_embedding(doc_id, current_tier)
            if embedding is None:
                print(f"Error: Could not retrieve embedding for document {doc_id}")
                return False
        
        # Remove from current tier and add to new tier
        success = False
        
        if new_tier == 1:
            # Moving to Tier 1 (Redis)
            if current_tier == 2:
                self._promote_from_tier2_to_redis(doc_id, remove=True)
            elif current_tier == 3:
                # T3 -> T2 -> T1 (two steps)
                self._promote_from_tier3_to_tier2(doc_id, remove=True)
                self._promote_from_tier2_to_redis(doc_id, remove=True)
            success = True
        elif new_tier == 2:
            # Moving to Tier 2 (Local ChromaDB)
            if current_tier == 1:
                self._demote_from_redis_to_tier2(doc_id, remove=True)
            elif current_tier == 3:
                self._promote_from_tier3_to_tier2(doc_id, remove=True)
            success = True
        elif new_tier == 3:
            # Moving to Tier 3 (Remote ChromaDB)
            if current_tier == 1:
                # T1 -> T2 -> T3 (two steps)
                self._demote_from_redis_to_tier2(doc_id, remove=True)
                self._demote_from_tier2_to_tier3(doc_id, remove=True)
            elif current_tier == 2:
                self._demote_from_tier2_to_tier3(doc_id, remove=True)
            success = True
        
        return success
    
    def _get_document_embedding(self, doc_id: int, tier: int) -> np.ndarray:
        """Retrieve embedding from specified tier"""
        doc_key = f"doc{doc_id}"
        
        try:
            if tier == 1:
                if self.redis_client:
                    value = self.redis_client.get(doc_key)
                    if value:
                        return pickle.loads(value)
            elif tier == 2:
                if self.tier2_collection:
                    result = self.tier2_collection.get(ids=[doc_key], include=["embeddings"])
                    if result['ids']:
                        return np.array(result['embeddings'][0], dtype=np.float32)
            elif tier == 3:
                if self.tier3_collection:
                    result = self.tier3_collection.get(ids=[doc_key], include=["embeddings"])
                    if result['ids']:
                        return np.array(result['embeddings'][0], dtype=np.float32)
        except Exception as e:
            print(f"Error retrieving embedding for doc {doc_id} from tier {tier}: {e}")
        
        return None
    
    def recalculate_and_reallocate(self, temperatures: np.ndarray, doc_embeddings: np.ndarray, 
                                   tier1_percentile: float = 95, tier2_percentile: float = 75):
        """Recalculate tier assignments based on new temperatures and reallocate documents"""
        print("Recalculating tier assignments...")
        
        # Calculate new thresholds
        tier1_threshold = np.percentile(temperatures, tier1_percentile)
        tier2_threshold = np.percentile(temperatures, tier2_percentile)
        
        # Update thresholds
        self.update_temperature_thresholds(tier1_threshold, tier2_threshold)
        
        # Calculate new tier assignments
        new_tier_assignment = np.zeros(len(temperatures), dtype=int)
        new_tier_assignment[temperatures >= tier1_threshold] = 1
        new_tier_assignment[(temperatures < tier1_threshold) & (temperatures >= tier2_threshold)] = 2
        new_tier_assignment[temperatures < tier2_threshold] = 3
        
        print(f"New distribution: Tier1={np.sum(new_tier_assignment==1)}, "
              f"Tier2={np.sum(new_tier_assignment==2)}, Tier3={np.sum(new_tier_assignment==3)}")
        
        # Reallocate documents
        print("Reallocating documents...")
        reallocated = 0
        for doc_id in range(len(temperatures)):
            new_tier = new_tier_assignment[doc_id]
            temperature = temperatures[doc_id]
            embedding = doc_embeddings[doc_id] if doc_id < len(doc_embeddings) else None
            
            if self.reallocate_document(doc_id, temperature, embedding):
                reallocated += 1
            
            if (doc_id + 1) % 1000 == 0:
                print(f"  Reallocated {doc_id + 1}/{len(temperatures)} documents...")
        
        print(f"Reallocation complete: {reallocated}/{len(temperatures)} documents")
        return new_tier_assignment
    
    def clear_all_tiers(self):
        """Clear all documents from all tiers (for clean setup)"""
        print("Clearing all tiers...")
        
        # Clear Tier 1 (Redis)
        if self.redis_client:
            keys = self.redis_client.keys("doc*")
            if keys:
                self.redis_client.delete(*keys)
                print(f"  Cleared {len(keys)} documents from Redis (Tier 1)")
        
        # Clear Tier 2 (Local ChromaDB)
        if self.tier2_collection:
            try:
                all_ids = self.tier2_collection.get()['ids']
                if all_ids:
                    self.tier2_collection.delete(ids=all_ids)
                    print(f"  Cleared {len(all_ids)} documents from Local ChromaDB (Tier 2)")
            except Exception as e:
                print(f"  Error clearing Tier 2: {e}")
        
        # Clear Tier 3 (Remote ChromaDB)
        if self.tier3_collection:
            try:
                all_ids = self.tier3_collection.get()['ids']
                if all_ids:
                    self.tier3_collection.delete(ids=all_ids)
                    print(f"  Cleared {len(all_ids)} documents from Remote ChromaDB (Tier 3)")
            except Exception as e:
                print(f"  Error clearing Tier 3: {e}")
        
        print("All tiers cleared")

    def close(self):
        if self.redis_client:
            self.redis_client.close()
        print("Storage connections closed")