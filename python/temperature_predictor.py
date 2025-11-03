import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pickle
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import redis
import lmdb
import json

class TemperaturePredictor:
    """
    Predicts document temperature changes and manages dynamic tier migration
    Based on access patterns, temporal patterns, and query similarity
    """
    
    def __init__(self, 
                 redis_host='localhost',
                 redis_port=6379,
                 lmdb_path='./tier2_lmdb',
                 history_window_hours=24,
                 prediction_horizon_hours=6,
                 migration_threshold=0.15):
        """
        Initialize temperature predictor
        
        Args:
            redis_host: Redis host for Tier 1 storage
            redis_port: Redis port
            lmdb_path: Path to LMDB for Tier 2 storage
            history_window_hours: Hours of history to consider for prediction
            prediction_horizon_hours: Hours ahead to predict
            migration_threshold: Temperature change threshold to trigger migration
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.lmdb_env = lmdb.open(lmdb_path, map_size=1_000_000_000)
        
        # Prediction parameters
        self.history_window = timedelta(hours=history_window_hours)
        self.prediction_horizon = timedelta(hours=prediction_horizon_hours)
        self.migration_threshold = migration_threshold
        
        # Access history tracking
        self.access_history = defaultdict(lambda: deque(maxlen=1000))
        self.temperature_history = defaultdict(list)
        
        # Prediction models per document
        self.models = {}
        self.scalers = {}
        
        # Current temperatures
        self.current_temperatures = {}
        
        # Tier thresholds (matching simulate_temperature.py)
        self.tier1_threshold = 0.95  # percentile
        self.tier2_threshold = 0.75  # percentile
        
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
        
    def extract_features(self, doc_id: int, current_time: datetime) -> np.ndarray:
        """
        Extract features for temperature prediction
        
        Features:
        - Access frequency in different time windows (1h, 6h, 24h)
        - Time since last access
        - Hour of day (cyclical encoding)
        - Day of week (cyclical encoding)
        - Recent temperature trend
        - Query diversity (std of recent query embeddings)
        """
        features = []
        
        # Access frequency features
        accesses = self.access_history[doc_id]
        time_windows = [1, 6, 24]  # hours
        
        for window_hours in time_windows:
            window_start = current_time - timedelta(hours=window_hours)
            window_accesses = sum(1 for a in accesses 
                                  if a['timestamp'] >= window_start)
            features.append(window_accesses / window_hours)  # Normalize by window size
            
        # Time since last access
        if accesses:
            time_since_last = (current_time - accesses[-1]['timestamp']).total_seconds() / 3600
        else:
            time_since_last = 24.0  # Default to 24 hours
        features.append(time_since_last)
        
        # Temporal features (cyclical encoding)
        hour = current_time.hour
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24)
        ])
        
        day_of_week = current_time.weekday()
        features.extend([
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ])
        
        # Temperature trend
        temp_history = self.temperature_history[doc_id]
        if len(temp_history) >= 2:
            recent_temps = [t['temperature'] for t in temp_history[-10:]]
            temp_trend = np.polyfit(range(len(recent_temps)), recent_temps, 1)[0]
        else:
            temp_trend = 0.0
        features.append(temp_trend)
        
        # Query diversity (standard deviation of recent query embeddings)
        if len(accesses) >= 2:
            recent_queries = [a['query_embedding'] for a in list(accesses)[-10:]]
            query_diversity = np.std(recent_queries)
        else:
            query_diversity = 0.0
        features.append(query_diversity)
        
        return np.array(features)
    
    def train_predictor(self, doc_id: int):
        """
        Train or update temperature predictor for a specific document
        """
        temp_history = self.temperature_history[doc_id]
        
        if len(temp_history) < 10:  # Need minimum history
            return
            
        # Prepare training data
        X = []
        y = []
        
        for i in range(len(temp_history) - 1):
            features = self.extract_features(doc_id, temp_history[i]['timestamp'])
            target_temp = temp_history[i + 1]['temperature']
            X.append(features)
            y.append(target_temp)
            
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        self.models[doc_id] = model
        self.scalers[doc_id] = scaler
        
    def predict_temperature(self, doc_id: int, future_time: datetime) -> float:
        """
        Predict temperature at a future time
        
        Args:
            doc_id: Document ID
            future_time: Time to predict temperature for
            
        Returns:
            Predicted temperature
        """
        if doc_id not in self.models:
            # Return current temperature if no model
            return self.current_temperatures.get(doc_id, 0.0)
            
        features = self.extract_features(doc_id, future_time)
        features_scaled = self.scalers[doc_id].transform(features.reshape(1, -1))
        predicted_temp = self.models[doc_id].predict(features_scaled)[0]
        
        # Ensure temperature is non-negative
        return max(0.0, predicted_temp)
        
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
            
    def execute_migration(self, migration: Dict) -> bool:
        """
        Execute a tier migration for a document
        
        Args:
            migration: Migration details from get_migration_candidates
            
        Returns:
            Success status
        """
        # Implementation would connect to storage_manager.py
        # This is a placeholder for the actual migration logic
        doc_id = migration['doc_id']
        source_tier = migration['current_tier']
        target_tier = migration['target_tier']
        
        print(f"Migrating doc {doc_id} from Tier {source_tier} to Tier {target_tier}")
        print(f"Temperature change: {migration['current_temp']:.3f} -> {migration['predicted_temp']:.3f}")
        
        # TODO: Implement actual migration logic with storage_manager
        return True
        
    def save_state(self, filepath: str):
        """Save predictor state to disk"""
        state = {
            'access_history': dict(self.access_history),
            'temperature_history': dict(self.temperature_history),
            'current_temperatures': self.current_temperatures,
            'models': self.models,
            'scalers': self.scalers
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state(self, filepath: str):
        """Load predictor state from disk"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.access_history = defaultdict(lambda: deque(maxlen=1000), state['access_history'])
        self.temperature_history = defaultdict(list, state['temperature_history'])
        self.current_temperatures = state['current_temperatures']
        self.models = state['models']
        self.scalers = state['scalers']
