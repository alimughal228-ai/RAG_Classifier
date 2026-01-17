"""
Central configuration module.

Manages all configuration settings for the document AI system.
Supports configuration via environment variables and config files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_formats: List[str] = None
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".docx", ".txt", ".md"]


@dataclass
class ClassificationConfig:
    """Configuration for document classification."""
    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_type: str = "local"
    confidence_threshold: float = 0.5
    fallback_threshold: float = 0.3


@dataclass
class ExtractionConfig:
    """Configuration for information extraction."""
    
    model_path: Optional[str] = None
    model_type: str = "local"
    extract_entities: bool = True
    extract_key_values: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    embedding_dimension: Optional[int] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    
    store_path: str = "./data/vector_store"
    index_type: str = "local"
    similarity_metric: str = "cosine"


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    
    top_k: int = 10
    similarity_threshold: float = 0.0
    rerank: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Paths
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    output_dir: Path = Path("./output")
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Module configurations
    ingestion: IngestionConfig = None
    classification: ClassificationConfig = None
    extraction: ExtractionConfig = None
    embedding: EmbeddingConfig = None
    vector_store: VectorStoreConfig = None
    retrieval: RetrievalConfig = None
    
    def __post_init__(self) -> None:
        """Initialize default configurations."""
        if self.ingestion is None:
            self.ingestion = IngestionConfig()
        if self.classification is None:
            self.classification = ClassificationConfig()
        if self.extraction is None:
            self.extraction = ExtractionConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AppConfig instance with loaded settings
    """
    # TODO: Implement configuration loading from file/env
    return AppConfig()


def get_config() -> AppConfig:
    """
    Get the current application configuration.
    
    Returns:
        AppConfig instance
    """
    return load_config()

