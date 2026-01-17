"""
Document embeddings module.

This module handles generation of vector embeddings
for documents using local embedding models with disk-based caching.
"""

from typing import List

from .generator import EmbeddingGenerator, generate_document_id
from .store import VectorStore

__all__: List[str] = ["EmbeddingGenerator", "generate_document_id", "VectorStore"]

