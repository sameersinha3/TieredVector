import time
from collections import defaultdict

class AccessTracker:
    def __init__(self):
        self.access_counts = defaultdict(int)
        self.last_access = {}
        
    def record_access(self, doc_id):
        self.access_counts[doc_id] += 1
        self.last_access[doc_id] = time.time()
    
    def get_access_score(self, doc_id):
        if doc_id not in self.access_counts:
            return 0.0
            
        count = self.access_counts[doc_id]
        last_time = self.last_access.get(doc_id, 0)
        current_time = time.time()
        
        # Simple scoring: frequency + recency
        freq_score = min(1.0, count / 100.0)  # normalize to 0-1
        
        if last_time > 0:
            hours_ago = (current_time - last_time) / 3600
            recency_score = max(0.0, 1.0 - hours_ago / 24.0)  # decay over 24 hours
        else:
            recency_score = 0.0
            
        return (freq_score + recency_score) / 2.0
    
    def get_access_stats(self, doc_id):
        return {
            'total_accesses': self.access_counts.get(doc_id, 0),
            'last_access': self.last_access.get(doc_id),
            'access_score': self.get_access_score(doc_id)
        }
    
    def get_top_accessed(self, n=10):
        scored_docs = []
        for doc_id in self.access_counts:
            score = self.get_access_score(doc_id)
            scored_docs.append((doc_id, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:n]
