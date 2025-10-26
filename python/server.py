from flask import Flask, request, jsonify
import numpy as np
import logging
from storage_manager import StorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
storage_manager = None

@app.route('/store', methods=['POST'])
def store_document():
    """Store a document with embedding and temperature."""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'doc_id' not in data or 'embedding' not in data or 'temperature' not in data:
            return jsonify({'error': 'Missing required fields: doc_id, embedding, temperature'}), 400
            
        doc_id = int(data['doc_id'])
        embedding = np.array(data['embedding'], dtype=np.float32)
        temperature = float(data['temperature'])
        
        success = storage_manager.store_document(doc_id, embedding, temperature)
        
        if success:
            tier = storage_manager.determine_tier(temperature)
            return jsonify({
                'status': 'success',
                'doc_id': doc_id,
                'tier': tier,
                'temperature': temperature
            })
        else:
            return jsonify({'error': 'Failed to store document'}), 500
            
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrieve', methods=['POST'])
def retrieve_document():
    """Retrieve relevant document by query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
            
        query = data['query']        
        embedding = storage_manager.retrieve_document(query)
        
        if embedding is not None:
            return jsonify({
                'status': 'success',
                'embedding': embedding.tolist(),
                'dimension': len(embedding)
            })
        else:
            return jsonify({'error': 'Document not found'}), 404
            
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get server and storage status."""
    try:
        stats = storage_manager.get_tier_distribution()
        return jsonify({
            'status': 'running',
            'storage_stats': stats,
            'tier_thresholds': {
                'tier1_redis': storage_manager.tier1_threshold,
                'tier2_leveldb': storage_manager.tier2_threshold
            }
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

def create_app(redis_host='localhost', redis_port=6379, leveldb_path='./leveldb_data', gcs_bucket=None):
    """Create and configure the Flask app."""
    global storage_manager
    
    # Initialize storage manager
    storage_manager = StorageManager(
        redis_host=redis_host,
        redis_port=redis_port, 
        leveldb_path=leveldb_path,
        gcs_bucket=gcs_bucket
    )
    
    if not storage_manager.initialize():
        raise RuntimeError("Failed to initialize storage manager")
        
    return app

if __name__ == '__main__':
    # Create app with default settings
    app = create_app()
    
    print("Starting Python 3-Tier Vector Storage Server...")
    print("Endpoints:")
    print("  POST /store - Store a document")
    print("  POST /retrieve - Retrieve a document") 
    print("  GET /status - Get storage statistics")
    print("  GET /health - Health check")
    
    # Run server
    app.run(host='0.0.0.0', port=8082, debug=True)

