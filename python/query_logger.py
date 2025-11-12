"""
Query logger for tracking query metrics during evaluation.
Logs latency, tier hits, bytes transferred, and operations for cost calculation.
"""
import csv
import os
import time
from typing import Dict, List, Optional
from datetime import datetime


class QueryLogger:
    """Logs query execution metrics to CSV for cost and performance analysis."""
    
    def __init__(self, log_path: str = "data/query_log.csv"):
        self.log_path = log_path
        self.fieldnames = [
            "timestamp",
            "query_id",
            "latency_ms",
            "latency_t1_ms",
            "latency_t2_ms", 
            "latency_t3_ms",
            "embed_time_ms",
            "tier_hits",  # e.g., "1", "1,2", "1,2,3"
            "ops_t1",
            "ops_t2",
            "ops_t3",
            "bytes_t1",
            "bytes_t2",
            "bytes_t3",
            "results_count",
            "system_type"  # "baseline" or "tiered"
        ]
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Create log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log_query(self, 
                  query_id: int,
                  latency_ms: float,
                  tier_hits: List[int],
                  ops_t1: int = 0,
                  ops_t2: int = 0,
                  ops_t3: int = 0,
                  bytes_t1: float = 0.0,
                  bytes_t2: float = 0.0,
                  bytes_t3: float = 0.0,
                  latency_t1_ms: Optional[float] = None,
                  latency_t2_ms: Optional[float] = None,
                  latency_t3_ms: Optional[float] = None,
                  embed_time_ms: Optional[float] = None,
                  results_count: int = 0,
                  system_type: str = "tiered"):
        """Log a single query execution."""
        tier_hits_str = ",".join(map(str, sorted(tier_hits))) if tier_hits else ""
        
        row = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "latency_ms": f"{latency_ms:.3f}",
            "latency_t1_ms": f"{latency_t1_ms:.3f}" if latency_t1_ms is not None else "",
            "latency_t2_ms": f"{latency_t2_ms:.3f}" if latency_t2_ms is not None else "",
            "latency_t3_ms": f"{latency_t3_ms:.3f}" if latency_t3_ms is not None else "",
            "embed_time_ms": f"{embed_time_ms:.3f}" if embed_time_ms is not None else "",
            "tier_hits": tier_hits_str,
            "ops_t1": ops_t1,
            "ops_t2": ops_t2,
            "ops_t3": ops_t3,
            "bytes_t1": f"{bytes_t1:.0f}",
            "bytes_t2": f"{bytes_t2:.0f}",
            "bytes_t3": f"{bytes_t3:.0f}",
            "results_count": results_count,
            "system_type": system_type
        }
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class QueryTimer:
    """Context manager for timing query execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.tier_timers = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
    
    def start_tier(self, tier: int):
        """Start timing a specific tier access."""
        self.tier_timers[tier] = time.time()
    
    def end_tier(self, tier: int) -> float:
        """End timing a specific tier access and return duration in ms."""
        if tier in self.tier_timers:
            duration = (time.time() - self.tier_timers[tier]) * 1000
            del self.tier_timers[tier]
            return duration
        return 0.0
    
    def get_total_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        elif self.start_time:
            return (time.time() - self.start_time) * 1000
        return 0.0

