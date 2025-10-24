#!/usr/bin/env python3
"""
Lightweight Python ML service for embedding generation and similarity search.
This service handles the ML/AI aspects while the Go service handles tier management.
"""

import numpy as np
import json
import logging
from typing import List, Dict, Optional
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLService:
    """ML service for embedding generation and similarity search."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.similarity_index = None
        self.doc_count = 0
        
        # Load model and embeddings
        self._load_model()
        self._load_embeddings()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_embeddings(self):
        """Load document embeddings."""
        try:
            embeddings_path = "wiki_embeddings.npy"
            if os.path.exists(embeddings_path):
                logger.info(f"Loading embeddings from {embeddings_path}")
                self.doc_embeddings = np.load(embeddings_path)
                self.doc_count = len(self.doc_embeddings)
                
                # Build similarity index
                self.similarity_index = NearestNeighbors(
                    n_neighbors=10, 
                    metric='cosine'
                )
                self.similarity_index.fit(self.doc_embeddings)
                logger.info(f"Loaded {self.doc_count} document embeddings")
            else:
                logger.warning(f"Embeddings file {embeddings_path} not found")
                # Create dummy embeddings for testing
                self.doc_count = 10000
                self.doc_embeddings = np.random.rand(self.doc_count, 384)
                self.similarity_index = NearestNeighbors(
                    n_neighbors=10, 
                    metric='cosine'
                )
                self.similarity_index.fit(self.doc_embeddings)
                logger.info("Created dummy embeddings for testing")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def encode_query(self, query_text: str) -> List[float]:
        """Encode a query text into an embedding."""
        try:
            embedding = self.model.encode([query_text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise
    
    def similarity_search(self, query_text: str, k: int = 5) -> Dict:
        """Perform similarity search for a query."""
        try:
            # Encode query
            query_embedding = self.model.encode([query_text])
            
            # Perform similarity search
            distances, indices = self.similarity_index.kneighbors(
                query_embedding, n_neighbors=k
            )
            
            # Convert to similarity scores (1 - distance)
            similarities = (1 - distances[0]).tolist()
            retrieved_docs = indices[0].tolist()
            
            return {
                "query_text": query_text,
                "retrieved_docs": retrieved_docs,
                "similarities": similarities,
                "query_embedding": query_embedding[0].tolist()
            }
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def get_similar_documents(self, doc_idx: int, n_similar: int = 5) -> List[int]:
        """Get similar documents for a given document index."""
        try:
            if doc_idx >= self.doc_count:
                return []
            
            distances, indices = self.similarity_index.kneighbors(
                [self.doc_embeddings[doc_idx]], n_neighbors=n_similar + 1
            )
            
            # Return similar docs (excluding the document itself)
            return indices[0][1:].tolist()
        except Exception as e:
            logger.error(f"Failed to get similar documents: {e}")
            return []
    
    def get_embedding(self, doc_idx: int) -> Optional[List[float]]:
        """Get embedding for a specific document."""
        try:
            if 0 <= doc_idx < self.doc_count:
                return self.doc_embeddings[doc_idx].tolist()
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding for doc {doc_idx}: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get service statistics."""
        return {
            "model_name": self.model_name,
            "doc_count": self.doc_count,
            "embedding_dimension": self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0,
            "similarity_index_built": self.similarity_index is not None
        }

# Initialize Flask app and ML service
app = Flask(__name__)
ml_service = None

def init_ml_service():
    """Initialize the ML service."""
    global ml_service
    try:
        model_name = os.getenv('MODEL_NAME', 'all-mpnet-base-v2')
        ml_service = MLService(model_name)
        logger.info("ML service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML service: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ml_service",
        "stats": ml_service.get_stats() if ml_service else None
    })

@app.route('/encode', methods=['POST'])
def encode_query():
    """Encode a query text into an embedding."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field"}), 400
        
        query_text = data['query']
        embedding = ml_service.encode_query(query_text)
        
        return jsonify({
            "query": query_text,
            "embedding": embedding,
            "dimension": len(embedding)
        })
    except Exception as e:
        logger.error(f"Error in encode endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def similarity_search():
    """Perform similarity search."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field"}), 400
        
        query_text = data['query']
        k = data.get('k', 5)
        
        result = ml_service.similarity_search(query_text, k)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/similar', methods=['POST'])
def get_similar_documents():
    """Get similar documents for a given document."""
    try:
        data = request.get_json()
        if not data or 'doc_idx' not in data:
            return jsonify({"error": "Missing 'doc_idx' field"}), 400
        
        doc_idx = data['doc_idx']
        n_similar = data.get('n_similar', 5)
        
        similar_docs = ml_service.get_similar_documents(doc_idx, n_similar)
        
        return jsonify({
            "doc_idx": doc_idx,
            "similar_docs": similar_docs,
            "count": len(similar_docs)
        })
    except Exception as e:
        logger.error(f"Error in similar endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/embedding/<int:doc_idx>', methods=['GET'])
def get_embedding(doc_idx):
    """Get embedding for a specific document."""
    try:
        embedding = ml_service.get_embedding(doc_idx)
        if embedding is None:
            return jsonify({"error": f"Document {doc_idx} not found"}), 404
        
        return jsonify({
            "doc_idx": doc_idx,
            "embedding": embedding,
            "dimension": len(embedding)
        })
    except Exception as e:
        logger.error(f"Error in embedding endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get service statistics."""
    try:
        return jsonify(ml_service.get_stats())
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_encode', methods=['POST'])
def batch_encode():
    """Encode multiple queries in batch."""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({"error": "Missing 'queries' field"}), 400
        
        queries = data['queries']
        if not isinstance(queries, list):
            return jsonify({"error": "'queries' must be a list"}), 400
        
        embeddings = []
        for query in queries:
            embedding = ml_service.encode_query(query)
            embeddings.append(embedding)
        
        return jsonify({
            "queries": queries,
            "embeddings": embeddings,
            "count": len(embeddings)
        })
    except Exception as e:
        logger.error(f"Error in batch_encode endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_search', methods=['POST'])
def batch_search():
    """Perform similarity search for multiple queries."""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({"error": "Missing 'queries' field"}), 400
        
        queries = data['queries']
        k = data.get('k', 5)
        
        if not isinstance(queries, list):
            return jsonify({"error": "'queries' must be a list"}), 400
        
        results = []
        for query in queries:
            result = ml_service.similarity_search(query, k)
            results.append(result)
        
        return jsonify({
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        logger.error(f"Error in batch_search endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize ML service
    init_ml_service()
    
    # Get configuration from environment
    port = int(os.getenv('PORT', 8081))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting ML service on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Start Flask app
    app.run(host=host, port=port, debug=debug)
