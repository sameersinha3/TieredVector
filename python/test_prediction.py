"""
Quick test to verify temperature prediction logic
"""

import numpy as np
from datetime import datetime, timedelta
from temperature_predictor import TemperaturePredictor
from migration_manager import MigrationManager

# Initialize predictor
predictor = TemperaturePredictor()

# Simulate some accesses
current_time = datetime.now()
test_embedding = np.random.rand(384)  # Dummy embedding

# Log some accesses for doc 1
for i in range(5):
    access_time = current_time - timedelta(hours=i)
    predictor.log_access(doc_id=1, timestamp=access_time, query_embedding=test_embedding)
    
# Log fewer accesses for doc 2
predictor.log_access(doc_id=2, timestamp=current_time, query_embedding=test_embedding)

# Check temperatures
print(f"Doc 1 temperature: {predictor.current_temperatures.get(1, 0):.3f}")
print(f"Doc 2 temperature: {predictor.current_temperatures.get(2, 0):.3f}")

# Get migration candidates
candidates = predictor.get_migration_candidates(current_time)
print(f"\nFound {len(candidates)} migration candidates")

# Test migration manager
manager = MigrationManager()
if candidates:
    results = manager.migrate_documents(candidates)
    print(f"Migration results: {results}")
