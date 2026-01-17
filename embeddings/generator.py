"""
Vector embedding generation.

Generates embeddings for documents using offline SentenceTransformers models.
Includes local disk-based caching for deterministic and repeatable results.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def generate_document_id(file_path: Union[str, Path], text: Optional[str] = None) -> str:
    """
    Generate a deterministic document ID.
    
    Uses file path hash for deterministic IDs. If text is provided,
    uses content hash for content-based IDs.
    
    Args:
        file_path: Path to the document file
        text: Optional document text for content-based hashing
        
    Returns:
        Deterministic document ID as hex string
    """
    file_path_str = str(Path(file_path).resolve())
    
    if text is not None:
        # Content-based ID (more robust for same content, different paths)
        content = f"{file_path_str}:{text[:1000]}"  # Use first 1000 chars for hash
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    else:
        # Path-based ID
        return hashlib.sha256(file_path_str.encode('utf-8')).hexdigest()[:16]


class EmbeddingGenerator:
    """
    Generates vector embeddings for text using SentenceTransformers.
    
    Features:
    - Uses local SentenceTransformers models (fully offline)
    - Disk-based caching for deterministic and repeatable results
    - Batch processing support
    - Automatic cache management
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformers model
            device: Device to run model on ('cpu' or 'cuda')
            cache_dir: Directory for caching embeddings (default: ./data/embeddings_cache)
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name: str = model_name
        self.device: str = device
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension: Optional[int] = None
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path("./data/embeddings_cache")
        else:
            cache_dir = Path(cache_dir)
        
        self.cache_dir: Path = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> None:
        """
        Load the SentenceTransformers model.
        
        Raises:
            RuntimeError: If model cannot be loaded
        """
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Get embedding dimension by encoding a dummy text
            dummy_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dimension = len(dummy_embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")
    
    def _get_cache_path(self, document_id: str) -> Path:
        """
        Get cache file path for a document ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{document_id}.npz"
    
    def _load_from_cache(self, document_id: str) -> Optional[np.ndarray]:
        """
        Load embedding from cache if it exists.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Cached embedding or None if not found
        """
        cache_path = self._get_cache_path(document_id)
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                embedding = data['embedding']
                # Verify dimension matches current model
                if self.embedding_dimension and len(embedding) == self.embedding_dimension:
                    return embedding
            except Exception:
                # If cache is corrupted, ignore it
                pass
        return None
    
    def _save_to_cache(self, document_id: str, embedding: np.ndarray) -> None:
        """
        Save embedding to cache.
        
        Args:
            document_id: Document identifier
            embedding: Embedding vector to cache
        """
        cache_path = self._get_cache_path(document_id)
        try:
            np.savez_compressed(cache_path, embedding=embedding)
        except Exception as e:
            # Log warning but don't fail
            print(f"Warning: Failed to cache embedding for {document_id}: {e}")
    
    def generate(
        self,
        text: str,
        document_id: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            document_id: Optional document ID (auto-generated if not provided)
            use_cache: Whether to use cache (default: True)
            
        Returns:
            Dictionary with format:
            {
                "document_id": str,
                "embedding": np.ndarray
            }
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            self.load_model()
        
        # Generate document ID if not provided
        if document_id is None:
            # Use text hash for content-based ID
            document_id = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        # Check cache first
        if use_cache:
            cached_embedding = self._load_from_cache(document_id)
            if cached_embedding is not None:
                return {
                    "document_id": document_id,
                    "embedding": cached_embedding
                }
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Cache the embedding
        if use_cache:
            self._save_to_cache(document_id, embedding)
        
        return {
            "document_id": document_id,
            "embedding": embedding
        }
    
    def generate_batch(
        self,
        texts: List[str],
        document_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            document_ids: Optional list of document IDs (auto-generated if not provided)
            use_cache: Whether to use cache (default: True)
            
        Returns:
            List of dictionaries, each with format:
            {
                "document_id": str,
                "embedding": np.ndarray
            }
        """
        if self.model is None:
            self.load_model()
        
        # Generate document IDs if not provided
        if document_ids is None:
            document_ids = [
                hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
                for text in texts
            ]
        
        results: List[Dict[str, Any]] = []
        texts_to_encode: List[str] = []
        indices_to_encode: List[int] = []
        
        # Check cache for each text
        for idx, (text, doc_id) in enumerate(zip(texts, document_ids)):
            if use_cache:
                cached_embedding = self._load_from_cache(doc_id)
                if cached_embedding is not None:
                    results.append({
                        "document_id": doc_id,
                        "embedding": cached_embedding
                    })
                    continue
            
            # Need to encode this text
            texts_to_encode.append(text)
            indices_to_encode.append(idx)
        
        # Generate embeddings for uncached texts
        if texts_to_encode:
            embeddings = self.model.encode(
                texts_to_encode,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Process results and cache
            for i, embedding in enumerate(embeddings):
                original_idx = indices_to_encode[i]
                doc_id = document_ids[original_idx]
                
                # Cache the embedding
                if use_cache:
                    self._save_to_cache(doc_id, embedding)
                
                results.append({
                    "document_id": doc_id,
                    "embedding": embedding
                })
        
        # Sort results to match original order
        result_dict = {r["document_id"]: r for r in results}
        return [result_dict[doc_id] for doc_id in document_ids]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.embedding_dimension is None:
            if self.model is None:
                self.load_model()
            # Dimension should be set after loading
            if self.embedding_dimension is None:
                raise RuntimeError("Could not determine embedding dimension")
        
        return self.embedding_dimension
    
    def clear_cache(self, document_id: Optional[str] = None) -> None:
        """
        Clear embedding cache.
        
        Args:
            document_id: Optional specific document ID to clear.
                        If None, clears all cached embeddings.
        """
        if document_id is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.npz"):
                cache_file.unlink()
        else:
            # Clear specific document cache
            cache_path = self._get_cache_path(document_id)
            if cache_path.exists():
                cache_path.unlink()
