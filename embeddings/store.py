"""
Vector store for embeddings using FAISS.

Manages storage and retrieval of document embeddings
using FAISS for efficient similarity search.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    """
    Stores and manages document embeddings using FAISS.
    
    Provides local storage for embeddings with efficient
    similarity search capabilities.
    """
    
    def __init__(
        self,
        store_path: Union[str, Path],
        dimension: Optional[int] = None,
        similarity_metric: str = "cosine"
    ) -> None:
        """
        Initialize the vector store.
        
        Args:
            store_path: Path to store embeddings and index
            dimension: Embedding dimension (auto-detected if not provided)
            similarity_metric: Similarity metric ('cosine' or 'l2')
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        
        self.store_path: Path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension: Optional[int] = dimension
        self.similarity_metric: str = similarity_metric
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        
        # Metadata storage: maps index position to document metadata
        self.metadata: List[Dict[str, Any]] = []
        
        # Text storage: maps document_id to original text for snippet extraction
        self.texts: Dict[str, str] = {}
        
        # Document ID to index mapping
        self.doc_id_to_idx: Dict[str, int] = {}
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Create a FAISS index based on similarity metric.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            FAISS index
        """
        if self.similarity_metric == "cosine":
            # For cosine similarity, use InnerProduct index with normalized vectors
            # We'll normalize vectors before adding to index
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine
        else:  # L2
            index = faiss.IndexFlatL2(dimension)
        
        return index
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        texts: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add embeddings to the store.
        
        Args:
            embeddings: Array of embeddings (n_vectors, dimension)
            metadata: List of metadata dictionaries for each embedding
            texts: Optional dictionary mapping document_id to text for snippet extraction
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Determine dimension from first embedding if not set
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = self._create_index(self.dimension)
        
        # Normalize embeddings for cosine similarity
        if self.similarity_metric == "cosine":
            # Normalize to unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Add to FAISS index
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        
        # Store metadata and texts
        start_idx = len(self.metadata)
        for i, meta in enumerate(metadata):
            idx = start_idx + i
            self.metadata.append(meta)
            
            # Store document ID mapping
            doc_id = meta.get("document_id")
            if doc_id:
                self.doc_id_to_idx[doc_id] = idx
            
            # Store text if provided
            if texts and doc_id:
                self.texts[doc_id] = texts.get(doc_id, "")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector (1, dimension) or (dimension,)
            top_k: Number of results to return
            
        Returns:
            List of tuples (metadata, similarity_score)
            For cosine similarity, scores are in [0, 1] range
            For L2, scores are distances (lower is better)
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query for cosine similarity
        if self.similarity_metric == "cosine":
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            norm[norm == 0] = 1
            query_embedding = query_embedding / norm
        
        # Convert to float32
        query_embedding = query_embedding.astype('float32')
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Process results
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                
                # Convert distance to similarity score
                if self.similarity_metric == "cosine":
                    # Inner product is already similarity (0-1 range)
                    score = float(dist)
                else:  # L2
                    # Convert distance to similarity (inverse, normalized)
                    # For display, we'll use 1 / (1 + distance) to get [0, 1] range
                    score = float(1 / (1 + dist))
                
                results.append((metadata, score))
        
        return results
    
    def get_text(self, document_id: str) -> Optional[str]:
        """
        Get original text for a document ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Original text or None if not found
        """
        return self.texts.get(document_id)
    
    def save(self) -> None:
        """Save the vector store to disk."""
        if self.index is None:
            return
        
        # Save FAISS index
        index_path = self.store_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.store_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # Save texts (using pickle for efficiency with large texts)
        texts_path = self.store_path / "texts.pkl"
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        
        # Save document ID mapping
        doc_id_map_path = self.store_path / "doc_id_map.json"
        with open(doc_id_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_id_to_idx, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        config = {
            "dimension": self.dimension,
            "similarity_metric": self.similarity_metric,
            "num_vectors": self.index.ntotal
        }
        config_path = self.store_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def load(self) -> None:
        """Load the vector store from disk."""
        index_path = self.store_path / "index.faiss"
        if not index_path.exists():
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = self.store_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        # Load texts
        texts_path = self.store_path / "texts.pkl"
        if texts_path.exists():
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)
        
        # Load document ID mapping
        doc_id_map_path = self.store_path / "doc_id_map.json"
        if doc_id_map_path.exists():
            with open(doc_id_map_path, 'r', encoding='utf-8') as f:
                self.doc_id_to_idx = json.load(f)
        
        # Load configuration
        config_path = self.store_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.dimension = config.get("dimension")
                self.similarity_metric = config.get("similarity_metric", "cosine")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "num_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "similarity_metric": self.similarity_metric,
            "num_texts": len(self.texts)
        }
