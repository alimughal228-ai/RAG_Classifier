"""
Document retrieval module.

This module handles retrieval of relevant documents
based on queries using local vector search with FAISS.
"""

from typing import List

from .retriever import DocumentRetriever

__all__: List[str] = ["DocumentRetriever"]

