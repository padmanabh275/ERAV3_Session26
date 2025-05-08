"""
Generate FAISS index for document embeddings using either Nomic Atlas or sentence-transformers.

This module provides functionality to create and save FAISS indices for document embeddings
using either Nomic Atlas or sentence-transformers for embedding generation. It supports both
DataFrame and list inputs, and includes fallback mechanisms for testing purposes.
"""

import faiss
import numpy as np
from nomic import atlas
import json
import os
import pandas as pd
import time
import logging
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import subprocess
import sys
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IndexMetadata:
    """Class to hold index metadata information."""
    dimension: int
    total_documents: int
    document_texts: List[str]
    titles: Optional[List[str]] = None

def check_nomic_auth() -> bool:
    """
    Check if user is authenticated with Nomic.
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        # Try to get the current user to check authentication
        atlas.get_current_user()
        return True
    except Exception as e:
        logger.error("Not authenticated with Nomic. Please run 'nomic login' first.")
        return False

def authenticate_nomic() -> bool:
    """
    Attempt to authenticate with Nomic.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        logger.info("Attempting to authenticate with Nomic...")
        subprocess.run(["nomic", "login"], check=True)
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to authenticate with Nomic. Please run 'nomic login' manually.")
        return False
    except FileNotFoundError:
        logger.error("Nomic CLI not found. Please install it first: pip install nomic")
        return False

def prepare_documents_for_atlas(documents: Union[pd.DataFrame, List[str]]) -> List[Dict[str, str]]:
    """
    Prepare documents for Atlas mapping.
    
    Args:
        documents: Either a DataFrame with 'text' and optional 'title' columns,
                  or a list of document texts.
    
    Returns:
        List of dictionaries containing document text and optional titles.
    """
    if isinstance(documents, pd.DataFrame):
        data = []
        for _, row in documents.iterrows():
            doc = {'text': str(row['text'])}
            if 'title' in row:
                doc['title'] = str(row['title'])
            data.append(doc)
    else:
        data = [{'text': str(doc)} for doc in documents]
    return data

def create_faiss_index(
    documents: Union[pd.DataFrame, List[str]],
    dimension: int = 384
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Create a FAISS index from a list of documents using Nomic Atlas for embeddings.
    
    Args:
        documents: Input documents as DataFrame or list of strings
        dimension: Dimension of the embedding vectors (default: 384)
    
    Returns:
        Tuple of (FAISS index, embeddings array)
    
    Raises:
        Exception: If embedding generation fails or no embeddings are produced
    """
    # Check authentication first
    if not check_nomic_auth():
        if not authenticate_nomic():
            raise Exception("Authentication with Nomic failed. Please run 'nomic login' manually.")

    try:
        # Initialize FAISS index
        index = faiss.IndexFlatL2(dimension)
        
        logger.info("Creating Atlas map...")
        identifier = f"Web-Page-Indexer-{int(time.time())}"
        
        # Prepare data for Atlas
        data = prepare_documents_for_atlas(documents)
        
        # Validate data format
        if not all(isinstance(doc.get('text'), str) for doc in data):
            raise ValueError("All documents must have a 'text' field containing a string")
        
        logger.info(f"Creating Atlas map with {len(data)} documents...")
        try:
            dataset = atlas.map_data(
                data=data,
                indexed_field='text',
                identifier=identifier,
                description="Web page embeddings for semantic search"
            )
        except Exception as e:
            if "Schema not found" in str(e):
                logger.error("Schema error with Nomic Atlas. Please ensure you have the latest version:")
                logger.error("pip install --upgrade nomic")
                raise Exception("Nomic Atlas schema error. Please update the package and try again.")
            raise
        
        map = dataset.maps[0]
        
        logger.info("Retrieving embeddings...")
        embeddings = map.embeddings.latent
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(-1, dimension)
        
        if len(embeddings_array) == 0:
            raise Exception("No embeddings were generated")
        
        logger.info(f"Generated {len(embeddings_array)} embeddings")
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        return index, embeddings_array
        
    except Exception as e:
        logger.error(f"Error creating Atlas map: {str(e)}")
        if "Schema not found" in str(e):
            logger.error("Please ensure you have the latest version of nomic installed:")
            logger.error("pip install --upgrade nomic")
        raise

def save_index(index: faiss.Index, metadata: IndexMetadata, output_dir: str) -> None:
    """
    Save FAISS index and metadata to files.
    
    Args:
        index: FAISS index to save
        metadata: IndexMetadata object containing index information
        output_dir: Directory to save the index and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, 'index.faiss'))
    
    # Convert metadata to serializable format
    serializable_metadata = {
        'dimension': metadata.dimension,
        'total_documents': metadata.total_documents,
        'document_texts': metadata.document_texts,
        'titles': metadata.titles or []
    }
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(serializable_metadata, f)

def load_example_documents() -> pd.DataFrame:
    """
    Load example documents for testing purposes.
    
    Returns:
        DataFrame containing example documents
    """
    return pd.DataFrame({
        'text': [
            "This is a sample document for testing.",
            "Another document with different content.",
            "A third document to demonstrate the functionality.",
            "Machine learning is transforming the way we process data.",
            "Natural language processing helps computers understand human language.",
            "Deep learning models can recognize patterns in complex data.",
            "Artificial intelligence is revolutionizing various industries.",
            "Data science combines statistics and programming.",
            "Neural networks mimic the human brain's structure.",
            "Computer vision enables machines to interpret visual data.",
            "Big data analytics helps make informed decisions.",
            "Cloud computing provides scalable computing resources.",
            "Cybersecurity protects digital systems from threats.",
            "Blockchain technology ensures secure transactions.",
            "Internet of Things connects physical devices to the internet.",
            "Quantum computing promises revolutionary computational power.",
            "Robotics combines hardware and software for automation.",
            "Augmented reality enhances real-world experiences.",
            "Virtual reality creates immersive digital environments.",
            "Edge computing processes data closer to its source."
        ]
    })

def create_embeddings_with_transformer(
    documents: Union[pd.DataFrame, List[str]],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> np.ndarray:
    """
    Create embeddings using sentence-transformers.
    
    Args:
        documents: Input documents as DataFrame or list of strings
        model_name: Name of the sentence-transformer model to use
        batch_size: Batch size for processing documents
    
    Returns:
        numpy array of embeddings
    """
    logger.info(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extract texts from DataFrame if needed
    if isinstance(documents, pd.DataFrame):
        texts = documents['text'].tolist()
    else:
        texts = documents
    
    logger.info(f"Generating embeddings for {len(texts)} documents...")
    embeddings = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def create_faiss_index_with_transformer(
    documents: Union[pd.DataFrame, List[str]],
    model_name: str = 'all-MiniLM-L6-v2',
    dimension: int = 384,
    batch_size: int = 32
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Create a FAISS index using sentence-transformers for embeddings.
    
    Args:
        documents: Input documents as DataFrame or list of strings
        model_name: Name of the sentence-transformer model to use
        dimension: Dimension of the embedding vectors
        batch_size: Batch size for processing documents
    
    Returns:
        Tuple of (FAISS index, embeddings array)
    """
    try:
        # Generate embeddings
        embeddings_array = create_embeddings_with_transformer(
            documents,
            model_name=model_name,
            batch_size=batch_size
        )
        
        # Initialize and populate FAISS index
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        logger.info(f"Created FAISS index with {len(embeddings_array)} embeddings")
        return index, embeddings_array
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def main() -> None:
    """Main function to demonstrate index creation and saving."""
    try:
        logger.info("Loading news articles dataset...")
        news_articles = pd.read_csv(
            'https://raw.githubusercontent.com/nomic-ai/maps/main/data/ag_news_25k.csv',
            nrows=100
        )
        
        # Try Nomic Atlas first
        try:
            logger.info("Attempting to create index with Nomic Atlas...")
            index, embeddings = create_faiss_index(news_articles)
        except Exception as e:
            logger.warning(f"Nomic Atlas failed: {str(e)}")
            logger.info("Falling back to sentence-transformers...")
            index, embeddings = create_faiss_index_with_transformer(news_articles)
        
        metadata = IndexMetadata(
            dimension=embeddings.shape[1],
            total_documents=len(news_articles),
            document_texts=news_articles.text.values.tolist(),
            titles=news_articles.title.values.tolist() if 'title' in news_articles.columns else None
        )
        
        logger.info("Saving index and metadata...")
        save_index(index, metadata, 'faiss_index')
        
        logger.info("Index and metadata saved successfully!")
        
    except Exception as e:
        logger.error(f"Error with news articles: {str(e)}")
        logger.info("Falling back to example documents...")
        
        try:
            documents = load_example_documents()
            
            # Try Nomic Atlas first
            try:
                logger.info("Attempting to create index with Nomic Atlas...")
                index, embeddings = create_faiss_index(documents)
            except Exception as e:
                logger.warning(f"Nomic Atlas failed: {str(e)}")
                logger.info("Falling back to sentence-transformers...")
                index, embeddings = create_faiss_index_with_transformer(documents)
            
            metadata = IndexMetadata(
                dimension=embeddings.shape[1],
                total_documents=len(documents),
                document_texts=documents.text.values.tolist()
            )
            
            logger.info("Saving index and metadata...")
            save_index(index, metadata, 'faiss_index')
            
            logger.info("Index and metadata saved successfully with example documents!")
            
        except Exception as e:
            logger.error(f"Error with example documents: {str(e)}")
            logger.error("Failed to create index.")
            exit(1)

if __name__ == "__main__":
    main() 