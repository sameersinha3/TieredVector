from collections import defaultdict
from access_tracker import AccessTracker

class DocumentScorer:
    def __init__(self, access_tracker):
        self.tracker = access_tracker
        self.tier1_threshold = 0.7
        self.tier2_threshold = 0.3
        self.current_tiers = {}
        self.recent_migrations = defaultdict(list)
        
    def update_document_tier(self, doc_id, current_tier):
        self.current_tiers[doc_id] = current_tier
        
    def bulk_update_tiers(self, tier_assignments):
        self.current_tiers.update(tier_assignments)
    
    def get_score(self, doc_id):
        return self.tracker.get_access_score(doc_id)
    
    def score_to_tier(self, score):
        if score >= self.tier1_threshold:
            return 1
        elif score >= self.tier2_threshold:
            return 2
        else:
            return 3
    
    def get_migration_candidates(self):
        candidates = []
        
        for doc_id in self.current_tiers:
            score = self.get_score(doc_id)
            current_tier = self.current_tiers[doc_id]
            target_tier = self.score_to_tier(score)
            
            if current_tier != target_tier:
                if current_tier > target_tier:
                    candidates.append((doc_id, 'promote'))
                else:
                    candidates.append((doc_id, 'demote'))
        
        return candidates[:20]  # limit to 20 migrations
    
    def record_migration(self, doc_id, from_tier, to_tier):
        import time
        timestamp = time.time()
        self.recent_migrations[doc_id].append((timestamp, from_tier, to_tier))
        self.current_tiers[doc_id] = to_tier
    
    def get_tier_distribution(self):
        distribution = defaultdict(int)
        for tier in self.current_tiers.values():
            distribution[tier] += 1
        return dict(distribution)
    
    def get_migration_stats(self, hours=24):
        import time
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        total = 0
        for migrations in self.recent_migrations.values():
            for timestamp, from_tier, to_tier in migrations:
                if timestamp > cutoff_time:
                    total += 1
        
        return {'total_migrations': total}
