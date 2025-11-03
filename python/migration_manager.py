import numpy as np
from datetime import datetime
import asyncio
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import redis
import lmdb
import pickle
from google.cloud import storage
from storage_manager import StorageManager

class MigrationManager:
    """
    Manages document migration between tiers based on temperature predictions
    Integrates with existing storage infrastructure
    """
    
    def __init__(self, 
                 storage_manager: Optional[StorageManager] = None,
                 max_concurrent_migrations: int = 10,
                 migration_batch_size: int = 100,
                 eviction_policy: str = 'lru'):
        """
        Initialize migration manager
        
        Args:
            storage_manager: Instance of StorageManager for tier operations
            max_concurrent_migrations: Maximum concurrent migration operations
            migration_batch_size: Number of documents to migrate in one batch
            eviction_policy: Policy for evicting documents when tier is full ('lru', 'lfu', 'lowest_temp')
        """
        self.storage_manager = storage_manager or StorageManager()
        self.max_concurrent = max_concurrent_migrations
        self.batch_size = migration_batch_size
        self.eviction_policy = eviction_policy
        
        # Migration statistics
        self.migration_stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'tier_promotions': 0,
            'tier_demotions': 0,
            'migration_latencies': []
        }
        
        # Tier capacity management
        self.tier_capacities = {
            1: 10000,   # Redis capacity
            2: 100000,  # LMDB capacity
            3: float('inf')  # GCS unlimited
        }
        
        self.tier_usage = {
            1: 0,
            2: 0,
            3: 0
        }
        
        # Track document locations
        self.document_locations = {}  # doc_id -> tier
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def migrate_documents(self, migration_candidates: List[Dict]) -> Dict:
        """
        Execute batch migration of documents between tiers
        
        Args:
            migration_candidates: List of migration candidates from TemperaturePredictor
            
        Returns:
            Migration results summary
        """
        # Sort by priority (confidence * temperature change)
        sorted_candidates = sorted(
            migration_candidates,
            key=lambda x: x['confidence'] * abs(x['predicted_temp'] - x['current_temp']),
            reverse=True
        )
        
        # Process in batches
        results = {
            'migrated': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            for i in range(0, len(sorted_candidates), self.batch_size):
                batch = sorted_candidates[i:i + self.batch_size]
                
                # Check tier capacities and prepare migrations
                prepared_batch = []
                for candidate in batch:
                    if self._can_migrate(candidate):
                        prepared_batch.append(candidate)
                    else:
                        results['skipped'] += 1
                        
                # Execute migrations
                futures = [
                    executor.submit(self._migrate_single_document, candidate)
                    for candidate in prepared_batch
                ]
                
                for future in futures:
                    try:
                        success, details = future.result()
                        if success:
                            results['migrated'] += 1
                        else:
                            results['failed'] += 1
                        results['details'].append(details)
                    except Exception as e:
                        results['failed'] += 1
                        self.logger.error(f"Migration error: {e}")
                        
        # Update statistics
        self._update_stats(results)
        
        return results
        
    def _can_migrate(self, candidate: Dict) -> bool:
        """
        Check if migration is feasible considering tier capacities
        """
        target_tier = candidate['target_tier']
        current_tier = candidate['current_tier']
        
        # Check if target tier has capacity
        if target_tier < 3:  # Tier 3 (GCS) has unlimited capacity
            if self.tier_usage[target_tier] >= self.tier_capacities[target_tier]:
                # Need to evict from target tier
                if not self._evict_from_tier(target_tier):
                    return False
                    
        return True
        
    def _evict_from_tier(self, tier: int) -> bool:
        """
        Evict lowest priority document from a tier to make space
        """
        if self.eviction_policy == 'lowest_temp':
            # Find document with lowest temperature in this tier
            candidates = []
            for doc_id, doc_tier in self.document_locations.items():
                if doc_tier == tier:
                    # Get temperature from storage metadata
                    temp = self._get_document_temperature(doc_id)
                    candidates.append((doc_id, temp))
                    
            if not candidates:
                return False
                
            # Evict lowest temperature document
            candidates.sort(key=lambda x: x[1])
            evict_doc_id = candidates[0][0]
            
            # Move to next tier down
            next_tier = min(tier + 1, 3)
            self._migrate_single_document({
                'doc_id': evict_doc_id,
                'current_tier': tier,
                'target_tier': next_tier,
                'current_temp': candidates[0][1],
                'predicted_temp': candidates[0][1],
                'confidence': 1.0
            })
            
            return True
            
        # TODO: Implement other eviction policies (LRU, LFU)
        return False
        
    def _migrate_single_document(self, migration: Dict) -> Tuple[bool, Dict]:
        """
        Migrate a single document between tiers
        """
        start_time = datetime.now()
        doc_id = migration['doc_id']
        source_tier = migration['current_tier']
        target_tier = migration['target_tier']
        
        try:
            # Retrieve document from source tier
            doc_data = self._retrieve_from_tier(doc_id, source_tier)
            if doc_data is None:
                return False, {
                    'doc_id': doc_id,
                    'error': 'Document not found in source tier'
                }
                
            # Store in target tier
            success = self._store_in_tier(doc_id, doc_data, target_tier)
            if not success:
                return False, {
                    'doc_id': doc_id,
                    'error': 'Failed to store in target tier'
                }
                
            # Delete from source tier
            self._delete_from_tier(doc_id, source_tier)
            
            # Update tracking
            self.document_locations[doc_id] = target_tier
            self.tier_usage[source_tier] -= 1
            self.tier_usage[target_tier] += 1
            
            # Record migration
            latency = (datetime.now() - start_time).total_seconds()
            self.migration_stats['migration_latencies'].append(latency)
            
            if target_tier < source_tier:
                self.migration_stats['tier_promotions'] += 1
            else:
                self.migration_stats['tier_demotions'] += 1
                
            self.logger.info(f"Migrated doc {doc_id}: Tier {source_tier} -> {target_tier} "
                           f"(temp: {migration['current_temp']:.3f} -> {migration['predicted_temp']:.3f})")
            
            return True, {
                'doc_id': doc_id,
                'source_tier': source_tier,
                'target_tier': target_tier,
                'latency': latency
            }
            
        except Exception as e:
            self.logger.error(f"Migration failed for doc {doc_id}: {e}")
            return False, {
                'doc_id': doc_id,
                'error': str(e)
            }
            
    def _retrieve_from_tier(self, doc_id: int, tier: int) -> Optional[np.ndarray]:
        """
        Retrieve document embedding from specified tier
        """
        try:
            if tier == 1:  # Redis
                key = f"vector:{doc_id}"
                data = self.storage_manager.redis_client.get(key)
                if data:
                    return pickle.loads(data)
                    
            elif tier == 2:  # LMDB
                with self.storage_manager.lmdb_env.begin() as txn:
                    key = f"vector_{doc_id}".encode()
                    data = txn.get(key)
                    if data:
                        return pickle.loads(data)
                        
            elif tier == 3:  # GCS
                # TODO: Implement GCS retrieval
                # This would use storage_manager's GCS client
                pass
                
        except Exception as e:
            self.logger.error(f"Error retrieving doc {doc_id} from tier {tier}: {e}")
            
        return None
        
    def _store_in_tier(self, doc_id: int, embedding: np.ndarray, tier: int) -> bool:
        """
        Store document embedding in specified tier
        """
        try:
            if tier == 1:  # Redis
                return self.storage_manager._store_in_redis(doc_id, embedding)
                
            elif tier == 2:  # LMDB
                return self.storage_manager._store_in_lmdb(doc_id, embedding)
                
            elif tier == 3:  # GCS
                return self.storage_manager._store_in_gcs(doc_id, embedding)
                
        except Exception as e:
            self.logger.error(f"Error storing doc {doc_id} in tier {tier}: {e}")
            
        return False
        
    def _delete_from_tier(self, doc_id: int, tier: int) -> bool:
        """
        Delete document from specified tier
        """
        try:
            if tier == 1:  # Redis
                key = f"vector:{doc_id}"
                self.storage_manager.redis_client.delete(key)
                
            elif tier == 2:  # LMDB
                with self.storage_manager.lmdb_env.begin(write=True) as txn:
                    key = f"vector_{doc_id}".encode()
                    txn.delete(key)
                    
            elif tier == 3:  # GCS
                # TODO: Implement GCS deletion
                pass
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting doc {doc_id} from tier {tier}: {e}")
            return False
            
    def _get_document_temperature(self, doc_id: int) -> float:
        """
        Get current temperature for a document
        """
        # This would integrate with TemperaturePredictor
        # For now, return a placeholder
        return 0.5
        
    def _update_stats(self, results: Dict):
        """
        Update migration statistics
        """
        self.migration_stats['total_migrations'] += results['migrated'] + results['failed']
        self.migration_stats['successful_migrations'] += results['migrated']
        self.migration_stats['failed_migrations'] += results['failed']
        
    def get_stats(self) -> Dict:
        """
        Get migration statistics
        """
        stats = self.migration_stats.copy()
        
        # Calculate average latency
        if stats['migration_latencies']:
            stats['avg_latency'] = np.mean(stats['migration_latencies'])
            stats['p95_latency'] = np.percentile(stats['migration_latencies'], 95)
        else:
            stats['avg_latency'] = 0
            stats['p95_latency'] = 0
            
        # Add tier usage
        stats['tier_usage'] = self.tier_usage.copy()
        stats['tier_capacities'] = self.tier_capacities.copy()
        
        return stats
        
    def get_tier_distribution(self) -> Dict:
        """
        Get current distribution of documents across tiers
        """
        distribution = {1: 0, 2: 0, 3: 0}
        for tier in self.document_locations.values():
            distribution[tier] += 1
        return distribution
