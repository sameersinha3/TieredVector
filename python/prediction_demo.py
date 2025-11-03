"""
Demo script for prediction-based dynamic temperature adjustment
Shows how temperature prediction and tier migration work together
"""

import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from temperature_predictor import TemperaturePredictor
from migration_manager import MigrationManager
from storage_manager import StorageManager

class PredictionDemo:
    def __init__(self):
        # Initialize components
        self.predictor = TemperaturePredictor(
            history_window_hours=24,
            prediction_horizon_hours=6,
            migration_threshold=0.15
        )
        
        self.storage_manager = StorageManager()
        self.migration_manager = MigrationManager(
            storage_manager=self.storage_manager,
            max_concurrent_migrations=10,
            eviction_policy='lowest_temp'
        )
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Demo parameters
        self.num_documents = 1000
        self.num_queries = 5000
        self.simulation_hours = 48
        
    def generate_synthetic_data(self):
        """Generate synthetic document embeddings and queries"""
        print("Generating synthetic data...")
        
        # Generate document embeddings
        doc_texts = [f"Document about topic {i % 20}" for i in range(self.num_documents)]
        self.doc_embeddings = self.model.encode(doc_texts)
        
        # Generate query patterns with temporal variations
        self.query_patterns = {
            'morning': ['news', 'weather', 'traffic', 'stocks'],
            'afternoon': ['research', 'analysis', 'reports', 'data'],
            'evening': ['entertainment', 'movies', 'games', 'social'],
            'night': ['tutorials', 'documentation', 'reference', 'guides']
        }
        
    def simulate_access_patterns(self):
        """Simulate realistic access patterns over time"""
        print("\nSimulating access patterns...")
        
        current_time = datetime.now() - timedelta(hours=self.simulation_hours)
        time_step = timedelta(minutes=30)
        
        access_log = []
        
        while current_time < datetime.now():
            # Determine time period
            hour = current_time.hour
            if 6 <= hour < 12:
                period = 'morning'
            elif 12 <= hour < 17:
                period = 'afternoon'
            elif 17 <= hour < 22:
                period = 'evening'
            else:
                period = 'night'
                
            # Generate queries based on time period
            num_queries = np.random.poisson(10)  # Average 10 queries per 30 min
            
            for _ in range(num_queries):
                # Select query topic based on time period
                topics = self.query_patterns[period]
                query_text = f"Query about {np.random.choice(topics)}"
                query_embedding = self.model.encode([query_text])[0]
                
                # Find relevant documents
                similarities = np.dot(self.doc_embeddings, query_embedding)
                top_k_indices = np.argsort(similarities)[-5:][::-1]
                
                # Log accesses
                for doc_id in top_k_indices:
                    self.predictor.log_access(doc_id, current_time, query_embedding)
                    access_log.append({
                        'timestamp': current_time,
                        'doc_id': doc_id,
                        'period': period
                    })
                    
            current_time += time_step
            
        print(f"Generated {len(access_log)} document accesses")
        return access_log
        
    def train_predictors(self):
        """Train temperature predictors for documents with sufficient history"""
        print("\nTraining temperature predictors...")
        
        trained_count = 0
        for doc_id in self.predictor.temperature_history.keys():
            if len(self.predictor.temperature_history[doc_id]) >= 10:
                self.predictor.train_predictor(doc_id)
                trained_count += 1
                
        print(f"Trained predictors for {trained_count} documents")
        
    def demonstrate_predictions(self):
        """Demonstrate temperature predictions and migration candidates"""
        print("\nDemonstrating temperature predictions...")
        
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=6)
        
        # Get top 10 hottest documents
        hot_docs = sorted(
            self.predictor.current_temperatures.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\nTop 10 Hottest Documents:")
        print("Doc ID | Current Temp | Predicted Temp | Change")
        print("-" * 50)
        
        for doc_id, current_temp in hot_docs:
            predicted_temp = self.predictor.predict_temperature(doc_id, future_time)
            change = predicted_temp - current_temp
            print(f"{doc_id:6d} | {current_temp:11.3f} | {predicted_temp:13.3f} | {change:+6.3f}")
            
        # Get migration candidates
        candidates = self.predictor.get_migration_candidates(current_time)
        
        print(f"\nFound {len(candidates)} migration candidates")
        
        if candidates:
            print("\nTop Migration Candidates:")
            print("Doc ID | Current Tier | Target Tier | Confidence")
            print("-" * 50)
            
            for candidate in candidates[:10]:
                print(f"{candidate['doc_id']:6d} | {candidate['current_tier']:11d} | "
                      f"{candidate['target_tier']:10d} | {candidate['confidence']:10.3f}")
                
        return candidates
        
    async def execute_migrations(self, candidates):
        """Execute tier migrations"""
        print("\nExecuting migrations...")
        
        results = await self.migration_manager.migrate_documents(candidates)
        
        print(f"\nMigration Results:")
        print(f"- Migrated: {results['migrated']}")
        print(f"- Failed: {results['failed']}")
        print(f"- Skipped: {results['skipped']}")
        
        # Show migration statistics
        stats = self.migration_manager.get_stats()
        print(f"\nMigration Statistics:")
        print(f"- Total migrations: {stats['total_migrations']}")
        print(f"- Success rate: {stats['successful_migrations'] / max(1, stats['total_migrations']) * 100:.1f}%")
        print(f"- Tier promotions: {stats['tier_promotions']}")
        print(f"- Tier demotions: {stats['tier_demotions']}")
        
        if stats['avg_latency'] > 0:
            print(f"- Average latency: {stats['avg_latency']:.3f}s")
            print(f"- P95 latency: {stats['p95_latency']:.3f}s")
            
    def visualize_temperature_evolution(self):
        """Visualize temperature evolution for top documents"""
        print("\nGenerating temperature evolution visualization...")
        
        plt.figure(figsize=(12, 6))
        
        # Select top 5 documents by temperature
        top_docs = sorted(
            self.predictor.current_temperatures.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for doc_id, _ in top_docs:
            history = self.predictor.temperature_history[doc_id]
            if len(history) > 0:
                timestamps = [h['timestamp'] for h in history]
                temperatures = [h['temperature'] for h in history]
                plt.plot(timestamps, temperatures, label=f'Doc {doc_id}')
                
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Document Temperature Evolution')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('temperature_evolution.png')
        print("Saved visualization to temperature_evolution.png")
        
    def run_demo(self):
        """Run the complete demonstration"""
        print("=" * 60)
        print("Prediction-Based Dynamic Temperature Adjustment Demo")
        print("=" * 60)
        
        # Generate data
        self.generate_synthetic_data()
        
        # Simulate access patterns
        access_log = self.simulate_access_patterns()
        
        # Train predictors
        self.train_predictors()
        
        # Demonstrate predictions
        candidates = self.demonstrate_predictions()
        
        # Execute migrations (async)
        if candidates:
            asyncio.run(self.execute_migrations(candidates))
            
        # Visualize results
        self.visualize_temperature_evolution()
        
        # Show tier distribution
        distribution = self.migration_manager.get_tier_distribution()
        print(f"\nFinal Tier Distribution:")
        for tier, count in distribution.items():
            print(f"- Tier {tier}: {count} documents")
            
        print("\nDemo completed!")
        

if __name__ == "__main__":
    # Note: This demo uses synthetic data and in-memory storage
    # In production, it would connect to actual Redis/LMDB/GCS
    
    demo = PredictionDemo()
    demo.run_demo()
