import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

class TemperaturePredictor:
    """
    Basic temperature predictor for document tier management
    """
    
    def __init__(self, 
                 history_window_hours=24,
                 prediction_horizon_hours=6,
                 migration_threshold=0.15):
        
        # Prediction parameters
        self.history_window = timedelta(hours=history_window_hours)
        self.prediction_horizon = timedelta(hours=prediction_horizon_hours)
        self.migration_threshold = migration_threshold
        
        # Access history tracking
        self.access_history = defaultdict(lambda: deque(maxlen=1000))
        self.temperature_history = defaultdict(list)
        
        # Current temperatures
        self.current_temperatures = {}
        
        # Tier thresholds (matching simulate_temperature.py)
        self.tier1_threshold = 0.95  # percentile
        self.tier2_threshold = 0.75  # percentile
        
        # Simple prediction weights
        self.decay_factor = 0.9  # matches simulate_temperature.py alpha
        
    def log_access(self, doc_id: int, timestamp: datetime, query_embedding: np.ndarray):
        """
        Log a document access for temperature tracking
        
        Args:
            doc_id: Document ID
            timestamp: Access timestamp
            query_embedding: Query embedding that accessed this document
        """
        self.access_history[doc_id].append({
            'timestamp': timestamp,
            'query_embedding': query_embedding
        })
        
        # Update current temperature using exponential decay
        current_temp = self.current_temperatures.get(doc_id, 0.0)
        new_temp = 0.9 * current_temp + 1.0  # Same alpha as simulate_temperature.py
        self.current_temperatures[doc_id] = new_temp
        
        # Log temperature history
        self.temperature_history[doc_id].append({
            'timestamp': timestamp,
            'temperature': new_temp
        })
        
    def calculate_access_frequency(self, doc_id: int, current_time: datetime) -> float:
        """
        Calculate recent access frequency for a document
        """
        accesses = self.access_history[doc_id]
        window_start = current_time - self.history_window
        
        recent_accesses = sum(1 for a in accesses if a['timestamp'] >= window_start)
        return recent_accesses / self.history_window.total_seconds() * 3600  # accesses per hour
    
    def estimate_future_temperature(self, doc_id: int, current_time: datetime) -> float:
        """
        Simple temperature estimation based on recent access patterns
        """
        current_temp = self.current_temperatures.get(doc_id, 0.0)
        
        # Get access frequency
        access_freq = self.calculate_access_frequency(doc_id, current_time)
        
        # Simple prediction: if high access frequency, temperature will increase
        # if low access frequency, temperature will decay
        if access_freq > 1.0:  # More than 1 access per hour
            predicted_temp = current_temp * self.decay_factor + access_freq * 0.1
        else:
            predicted_temp = current_temp * self.decay_factor
            
        return predicted_temp
        
    def predict_temperature(self, doc_id: int, future_time: datetime) -> float:
        """
        Predict temperature at a future time
        
        Args:
            doc_id: Document ID
            future_time: Time to predict temperature for
            
        Returns:
            Predicted temperature
        """
        return self.estimate_future_temperature(doc_id, datetime.now())
        
    def get_migration_candidates(self, current_time: datetime) -> List[Dict]:
        """
        Identify documents that should be migrated between tiers
        
        Returns:
            List of migration candidates with source/target tiers
        """
        candidates = []
        future_time = current_time + self.prediction_horizon
        
        # Calculate current tier thresholds based on all temperatures
        all_temps = list(self.current_temperatures.values())
        if len(all_temps) < 10:
            return candidates
            
        tier1_temp_threshold = np.percentile(all_temps, self.tier1_threshold * 100)
        tier2_temp_threshold = np.percentile(all_temps, self.tier2_threshold * 100)
        
        for doc_id, current_temp in self.current_temperatures.items():
            # Predict future temperature
            predicted_temp = self.predict_temperature(doc_id, future_time)
            
            # Determine current and predicted tiers
            current_tier = self._get_tier(current_temp, tier1_temp_threshold, tier2_temp_threshold)
            predicted_tier = self._get_tier(predicted_temp, tier1_temp_threshold, tier2_temp_threshold)
            
            # Check if migration is needed
            temp_change = abs(predicted_temp - current_temp)
            if (predicted_tier != current_tier and 
                temp_change >= self.migration_threshold):
                
                candidates.append({
                    'doc_id': doc_id,
                    'current_tier': current_tier,
                    'target_tier': predicted_tier,
                    'current_temp': current_temp,
                    'predicted_temp': predicted_temp,
                    'confidence': min(1.0, temp_change / self.migration_threshold)
                })
                
        return candidates
    
    def _get_tier(self, temperature: float, tier1_threshold: float, tier2_threshold: float) -> int:
        """Determine tier based on temperature"""
        if temperature >= tier1_threshold:
            return 1
        elif temperature >= tier2_threshold:
            return 2
        else:
            return 3
            
    def should_migrate(self, doc_id: int, current_tier: int, predicted_tier: int, 
                      temp_change: float) -> bool:
        """
        Check if a document should be migrated
        """
        # Only migrate if tier changes and temperature change is significant
        return (current_tier != predicted_tier and 
                abs(temp_change) >= self.migration_threshold)
