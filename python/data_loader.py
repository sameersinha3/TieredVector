import numpy as np
import requests
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset

class DataLoader:
    def __init__(self, cpp_service_url="http://localhost:8082"):
        self.cpp_service_url = cpp_service_url
        self.model = None
        self.doc_embeddings = None
        self.similarity_index = None
        
    def load_model(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """Load the sentence transformer model"""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully")
        
    def load_embeddings(self, embeddings_file="wiki_embeddings.npy"):
        """Load pre-computed embeddings"""
        print(f"Loading embeddings from {embeddings_file}")
        self.doc_embeddings = np.load(embeddings_file)
        print(f"Embeddings loaded: {self.doc_embeddings.shape}")
        
        # Build similarity index
        print("Building similarity index...")
        self.similarity_index = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.similarity_index.fit(self.doc_embeddings)
        print("Similarity index built")
        
    def calculate_temperatures(self, query_dataset_name="natural_questions", split="train", max_queries=1000):
        """Calculate temperatures based on query dataset"""
        print(f"Loading query dataset: {query_dataset_name}")
        query_dataset = load_dataset(query_dataset_name, split=split)
        
        # Limit queries for faster processing
        queries = [entry['question'] for entry in query_dataset[:max_queries]]
        print(f"Processing {len(queries)} queries...")
        
        # Generate query embeddings
        query_embeddings = self.model.encode(queries)
        
        # Calculate temperatures
        temperature = np.zeros(len(self.doc_embeddings))
        alpha = 0.9
        
        print("Calculating document temperatures...")
        _, indices = self.similarity_index.kneighbors(query_embeddings)
        for i, query_topk in enumerate(indices):
            if i % 100 == 0:
                print(f"Processed {i}/{len(indices)} queries")
            for idx in query_topk:
                temperature[idx] = alpha * temperature[idx] + 1
                
        print("Temperature calculation complete")
        return temperature
        
    def determine_tiers(self, temperature):
        """Determine tier assignments based on temperature"""
        tier1_threshold = np.percentile(temperature, 95)
        tier2_threshold = np.percentile(temperature, 75)
        
        tier_assignment = np.zeros(len(temperature), dtype=int)
        tier_assignment[temperature >= tier1_threshold] = 1
        tier_assignment[(temperature < tier1_threshold) & (temperature >= tier2_threshold)] = 2
        tier_assignment[temperature < tier2_threshold] = 3
        
        print(f"Tier 1 threshold: {tier1_threshold:.3f}")
        print(f"Tier 2 threshold: {tier2_threshold:.3f}")
        print(f"Tier 1 documents: {np.sum(tier_assignment == 1)}")
        print(f"Tier 2 documents: {np.sum(tier_assignment == 2)}")
        print(f"Tier 3 documents: {np.sum(tier_assignment == 3)}")
        
        return tier_assignment, tier1_threshold, tier2_threshold
        
    def send_to_cpp(self, doc_id, embedding, temperature):
        """Send document to C++ storage system"""
        try:
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist()
            
            payload = {
                "doc_id": int(doc_id),
                "embedding": embedding_list,
                "temperature": float(temperature)
            }
            
            response = requests.post(
                f"{self.cpp_service_url}/store",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to store doc {doc_id}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending doc {doc_id} to C++: {e}")
            return False
            
    def load_data_to_cpp(self, max_docs=1000):
        """Load data into C++ storage system"""
        if self.doc_embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
            
        # Calculate temperatures
        temperature = self.calculate_temperatures()
        
        # Determine tiers
        tier_assignment, tier1_thresh, tier2_thresh = self.determine_tiers(temperature)
        
        # Send documents to C++
        print(f"\nLoading {min(max_docs, len(self.doc_embeddings))} documents to C++ storage...")
        
        success_count = 0
        for i in range(min(max_docs, len(self.doc_embeddings))):
            if self.send_to_cpp(i, self.doc_embeddings[i], temperature[i]):
                success_count += 1
                
            if i % 100 == 0:
                print(f"Loaded {i}/{min(max_docs, len(self.doc_embeddings))} documents")
                
        print(f"\nSuccessfully loaded {success_count} documents to C++ storage")
        return success_count

def main():
    # Initialize data loader
    loader = DataLoader()
    
    # Load model and embeddings
    loader.load_model()
    loader.load_embeddings()
    
    # Load data to C++ (limit to 1000 docs for testing)
    loader.load_data_to_cpp(max_docs=1000)

if __name__ == "__main__":
    main()
