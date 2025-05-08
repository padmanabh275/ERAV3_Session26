"""
Backend server for the Web Page Indexer Chrome extension.
Handles FAISS indexing and searching operations.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from typing import Dict, List, Tuple
import logging
import sys

# Configure logging to show more details
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS more explicitly
CORS(app, resources={
    r"/*": {
        "origins": ["chrome-extension://*", "http://localhost:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize FAISS index and model
MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    logger.info(f"Model loaded successfully. Embedding dimension: {dimension}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Store page data
page_data: Dict[int, Dict] = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        return jsonify({
            'status': 'ok',
            'index_size': index.ntotal,
            'model_loaded': model is not None,
            'dimension': dimension
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all indexed documents."""
    try:
        documents = []
        for doc_id, data in page_data.items():
            documents.append({
                'id': doc_id,
                'title': data['title'],
                'url': data['url'],
                'content_preview': data['content'][:200] + '...' if len(data['content']) > 200 else data['content']
            })
        return jsonify({
            'total_documents': len(documents),
            'documents': documents
        })
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_connection():
    """Test connection endpoint."""
    return jsonify({'message': 'Server is running!'})

def load_index():
    """Load existing index and data if available."""
    try:
        if os.path.exists('faiss_index/index.faiss'):
            global index
            index = faiss.read_index('faiss_index/index.faiss')
            
            if os.path.exists('faiss_index/metadata.json'):
                with open('faiss_index/metadata.json', 'r') as f:
                    metadata = json.load(f)
                    global page_data
                    page_data = {int(k): v for k, v in metadata['page_data'].items()}
                    
            logger.info(f"Loaded index with {index.ntotal} documents")
    except Exception as e:
        logger.error(f"Error loading index: {e}")

def save_index():
    """Save index and metadata to disk."""
    try:
        os.makedirs('faiss_index', exist_ok=True)
        faiss.write_index(index, 'faiss_index/index.faiss')
        
        metadata = {
            'dimension': dimension,
            'total_documents': index.ntotal,
            'page_data': page_data
        }
        
        with open('faiss_index/metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        logger.info("Saved index and metadata")
    except Exception as e:
        logger.error(f"Error saving index: {e}")

@app.route('/index', methods=['POST', 'OPTIONS'])
def index_page():
    """Index a new page."""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.json
        if not data or 'content' not in data:
            logger.error("Missing content in request")
            return jsonify({'error': 'Missing content in request'}), 400
            
        content = data['content']
        logger.info(f"Indexing page: {data.get('url', 'unknown URL')}")
        
        # Generate embedding
        embedding = model.encode([content])[0]
        
        # Add to index
        index.add(np.array([embedding], dtype=np.float32))
        doc_id = index.ntotal - 1
        
        # Store metadata
        page_data[doc_id] = {
            'title': data.get('title', ''),
            'url': data.get('url', ''),
            'content': content
        }
        
        # Save index
        save_index()
        
        logger.info(f"Successfully indexed document {doc_id}")
        return jsonify({'id': doc_id, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error indexing page: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST', 'OPTIONS'])
def search():
    """Search for similar pages."""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.json
        if not data or 'query' not in data:
            logger.error("Missing query in request")
            return jsonify({'error': 'Missing query in request'}), 400
            
        query = data['query']
        logger.info(f"Processing search query: {query}")
        
        if index.ntotal == 0:
            logger.info("No documents indexed yet")
            return jsonify({'results': [], 'message': 'No documents indexed yet'})
        
        # Generate query embedding
        query_embedding = model.encode([query])[0]
        
        # Search in index
        k = min(5, index.ntotal)  # Number of results to return
        distances, indices = index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                page = page_data[idx]
                results.append({
                    'title': page['title'],
                    'url': page['url'],
                    'snippet': generate_snippet(page['content'], query),
                    'score': float(1 - distances[0][i])  # Convert distance to similarity score
                })
        
        logger.info(f"Found {len(results)} results for query: {query}")
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return jsonify({'error': str(e)}), 500

def generate_snippet(content: str, query: str, window_size: int = 10) -> str:
    """Generate a text snippet containing the query."""
    words = content.split()
    query_words = query.lower().split()
    
    # Find the best matching window
    best_window = ''
    best_score = 0
    
    for i in range(len(words) - window_size):
        window = ' '.join(words[i:i + window_size])
        score = sum(1 for qw in query_words if qw in window.lower())
        
        if score > best_score:
            best_score = score
            best_window = window
    
    return best_window + '...' if best_window else content[:100] + '...'

if __name__ == '__main__':
    try:
        load_index()
        logger.info("Starting server on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1) 