"""
Document retrieval functionality.

Retrieves relevant documents based on queries using
vector similarity search with FAISS and snippet extraction.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embeddings.generator import EmbeddingGenerator
from embeddings.store import VectorStore


def extract_snippet(text: str, query: str, max_length: int = 200) -> str:
    """
    Extract a relevant snippet from text based on query.
    
    Args:
        text: Full document text
        query: Search query
        max_length: Maximum snippet length in characters
        
    Returns:
        Extracted snippet with query context
    """
    if not text or not query:
        # Return beginning of text if no query
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # Find sentences containing query terms
    query_terms = query.lower().split()
    sentences = re.split(r'[.!?]\s+', text)
    
    # Score sentences by query term matches
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)
        if score > 0:
            scored_sentences.append((score, sentence))
    
    # Sort by score and get best sentence
    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentence = scored_sentences[0][1].strip()
        
        # If sentence is too long, truncate
        if len(best_sentence) > max_length:
            # Try to find a good truncation point
            words = best_sentence.split()
            snippet = ""
            for word in words:
                if len(snippet + word) > max_length - 3:
                    break
                snippet += word + " "
            return snippet.strip() + "..."
        
        return best_sentence
    
    # Fallback: return beginning of text
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


class DocumentRetriever:
    """
    Retrieves relevant documents based on queries.
    
    Uses FAISS vector similarity search to find relevant
    documents and extracts snippets for context.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ) -> None:
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance with indexed documents
            embedding_generator: Embedding generator for query embeddings
        """
        self.vector_store: VectorStore = vector_store
        self.embedding_generator: EmbeddingGenerator = embedding_generator
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with format:
            [
                {
                    "document_id": str,
                    "score": float,
                    "snippet": str
                }
            ]
        """
        # Generate query embedding
        query_embedding_result = self.embedding_generator.generate(
            text=query,
            use_cache=False  # Don't cache query embeddings
        )
        query_embedding = query_embedding_result["embedding"]
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Format results with snippets
        results: List[Dict[str, Any]] = []
        for metadata, score in search_results:
            document_id = metadata.get("document_id", "")
            
            # Get original text for snippet extraction
            text = self.vector_store.get_text(document_id)
            if text is None:
                # Fallback to metadata text if available
                text = metadata.get("text", "")
            
            # Extract snippet
            snippet = extract_snippet(text, query)
            
            result = {
                "document_id": document_id,
                "score": score,
                "snippet": snippet
            }
            
            # Add additional metadata if available
            if "file_path" in metadata:
                result["file_path"] = metadata["file_path"]
            if "file_name" in metadata:
                result["file_name"] = metadata["file_name"]
            
            results.append(result)
        
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents with similarity scores (raw format).
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of tuples (document_metadata, similarity_score)
        """
        # Generate query embedding
        query_embedding_result = self.embedding_generator.generate(
            text=query,
            use_cache=False
        )
        query_embedding = query_embedding_result["embedding"]
        
        # Search and return raw results
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
