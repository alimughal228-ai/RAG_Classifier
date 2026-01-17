"""
Document ingestion pipeline.

Orchestrates the complete ingestion workflow:
loading, preprocessing, and preparing documents.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


class IngestionPipeline:
    """
    Complete pipeline for ingesting documents.
    
    Coordinates loading, preprocessing, and initial
    processing of documents.
    """
    
    def __init__(
        self,
        loader: Optional[Any] = None,
        preprocessor: Optional[Any] = None
    ) -> None:
        """
        Initialize the ingestion pipeline.
        
        Args:
            loader: Document loader instance
            preprocessor: Document preprocessor instance
        """
        self.loader: Optional[Any] = loader
        self.preprocessor: Optional[Any] = preprocessor
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document through the pipeline.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with processed document data
        """
        pass
    
    def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of document paths
            
        Returns:
            List of processed documents
        """
        pass

