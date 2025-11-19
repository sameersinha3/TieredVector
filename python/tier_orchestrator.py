import time
from document_scorer import DocumentScorer
from access_tracker import AccessTracker

class TierOrchestrator:
    def __init__(self, storage_manager, access_tracker, document_scorer):
        self.storage = storage_manager
        self.tracker = access_tracker
        self.scorer = document_scorer
        self.migration_count = 0
    
    def check_and_migrate(self):
        candidates = self.scorer.get_migration_candidates()
        
        migrated = 0
        for doc_id, action in candidates:
            if self._perform_migration(doc_id, action):
                migrated += 1
                self.migration_count += 1
        
        return {'migrated': migrated, 'total': self.migration_count}
    
    def _perform_migration(self, doc_id, action):
        current_tier = self.scorer.current_tiers.get(doc_id)
        if not current_tier:
            return False
        
        try:
            if action == 'promote':
                if current_tier == 3:
                    self.storage._promote_from_tier3_to_tier2(doc_id)
                    target_tier = 2
                elif current_tier == 2:
                    self.storage._promote_from_tier2_to_redis(doc_id)
                    target_tier = 1
                else:
                    return False
                    
            elif action == 'demote':
                if current_tier == 1:
                    self.storage._demote_from_redis_to_tier2(doc_id)
                    target_tier = 2
                elif current_tier == 2:
                    self.storage._demote_from_tier2_to_tier3(doc_id)
                    target_tier = 3
                else:
                    return False
            else:
                return False
            
            self.scorer.record_migration(doc_id, current_tier, target_tier)
            return True
            
        except Exception:
            return False
    
    def force_migration(self, doc_id, target_tier):
        current_tier = self.scorer.current_tiers.get(doc_id)
        if not current_tier or current_tier == target_tier:
            return current_tier == target_tier
        
        action = 'promote' if current_tier > target_tier else 'demote'
        
        while current_tier != target_tier:
            if not self._perform_migration(doc_id, action):
                return False
            current_tier = self.scorer.current_tiers.get(doc_id)
        
        return True
    
    def get_status(self):
        return {
            'total_migrations': self.migration_count,
            'tier_distribution': self.scorer.get_tier_distribution(),
            'top_documents': self.tracker.get_top_accessed(5)
        }
