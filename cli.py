"""
Command-line interface for the document AI system.

Provides CLI commands for all major operations:
ingestion, classification, extraction, embedding, and retrieval.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from classification.classifier import DocumentClassifier
from config import AppConfig, get_config


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Local, offline document AI system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to documents to ingest"
    )
    ingest_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for processed documents"
    )
    
    # Classification command
    classify_parser = subparsers.add_parser("classify", help="Classify documents")
    classify_parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to documents or directories (e.g., Dataset_files) to classify"
    )
    classify_parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for classification results (optional)"
    )
    
    # Extraction command
    extract_parser = subparsers.add_parser("extract", help="Extract information")
    extract_parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to documents to extract from"
    )
    extract_parser.add_argument(
        "--output",
        type=str,
        help="Output file for extraction results"
    )
    
    # Embedding command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to documents to embed"
    )
    embed_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for embeddings"
    )
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents for retrieval")
    index_parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to documents to index"
    )
    index_parser.add_argument(
        "--store-path",
        type=str,
        help="Path to vector store"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    query_parser.add_argument(
        "--store-path",
        type=str,
        help="Path to vector store"
    )
    
    # RAG pipeline command
    rag_parser = subparsers.add_parser("rag", help="Run RAG pipeline")
    rag_parser.add_argument(
        "--index",
        action="store_true",
        help="Index documents"
    )
    rag_parser.add_argument(
        "--query",
        type=str,
        help="Query to search"
    )
    rag_parser.add_argument(
        "files",
        nargs="*",
        type=str,
        help="Paths to documents (for indexing)"
    )
    
    # Process command - complete pipeline
    process_parser = subparsers.add_parser(
        "process",
        help="Process documents through complete pipeline (ingest, classify, extract, embed)"
    )
    process_parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing documents to process"
    )
    process_parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Output JSON file path (default: output.json)"
    )
    
    # Search command - semantic search
    search_parser = subparsers.add_parser(
        "search",
        help="Search processed documents using semantic search"
    )
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query"
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    search_parser.add_argument(
        "--store-path",
        type=str,
        help="Path to vector store (default: from config)"
    )
    
    # QA command - question answering (optional)
    qa_parser = subparsers.add_parser(
        "qa",
        help="Answer questions using Ollama (local) / GROQ (optional) / llama-cpp (optional)"
    )
    qa_parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer"
    )
    qa_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve for context (default: 5)"
    )
    qa_parser.add_argument(
        "--model-path",
        type=str,
        help="Path to GGUF model file (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf)"
    )
    qa_parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "ollama", "groq", "local"],
        default="auto",
        help="QA provider (default: auto)"
    )
    qa_parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model name (default: llama3.1:8b)"
    )
    qa_parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434)"
    )
    qa_parser.add_argument(
        "--store-path",
        type=str,
        help="Path to vector store (default: from config)"
    )
    qa_parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    
    return parser


def handle_ingest(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle document ingestion command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    # TODO: Implement ingestion logic
    print(f"Ingesting {len(args.files)} document(s)...")
    for file_path in args.files:
        print(f"  - {file_path}")


def handle_classify(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle document classification command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from ingestion.loader import get_loader
    
    # Initialize classifier
    classifier = DocumentClassifier(
        model_name=config.classification.model_name,
        confidence_threshold=config.classification.confidence_threshold,
        fallback_threshold=config.classification.fallback_threshold
    )
    
    print("Loading classification model...")
    classifier.load_model()
    print("Model loaded successfully.\n")
    
    # Collect all files to process
    files_to_process: List[Path] = []
    
    for file_arg in args.files:
        file_path = Path(file_arg)
        
        # If it's a directory, add all supported files from it
        if file_path.is_dir():
            supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
            for ext in supported_extensions:
                files_to_process.extend(file_path.glob(f"*{ext}"))
        elif file_path.is_file():
            files_to_process.append(file_path)
        else:
            print(f"Warning: File or directory not found: {file_path}")
    
    if not files_to_process:
        print("No files found to classify.")
        return
    
    print(f"Classifying {len(files_to_process)} document(s)...\n")
    
    results: List[Dict[str, Any]] = []
    
    for file_path in files_to_process:
        try:
            print(f"Processing: {file_path.name}...", end=" ")
            
            # Load document
            loader = get_loader(file_path)
            document = loader.load()
            text = document.get("text", "")
            
            if not text or not text.strip():
                result = {
                    "file": str(file_path),
                    "class": "Unclassifiable",
                    "confidence": 0.0,
                    "error": "No text extracted from document"
                }
                print(f"❌ No text extracted")
            else:
                # Classify
                classification = classifier.classify(text)
                result = {
                    "file": str(file_path),
                    "class": classification["class"],
                    "confidence": classification["confidence"]
                }
                print(f"✓ {classification['class']} (confidence: {classification['confidence']:.3f})")
            
            results.append(result)
            
        except Exception as e:
            error_result = {
                "file": str(file_path),
                "class": "Unclassifiable",
                "confidence": 0.0,
                "error": str(e)
            }
            results.append(error_result)
            print(f"❌ Error: {e}")
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    else:
        # Print summary
        print("\n" + "="*60)
        print("Classification Summary")
        print("="*60)
        for result in results:
            print(f"\nFile: {Path(result['file']).name}")
            print(f"  Class: {result['class']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
        
        # Print statistics
        print("\n" + "="*60)
        print("Statistics")
        print("="*60)
        class_counts: Dict[str, int] = {}
        for result in results:
            cls = result['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")


def handle_extract(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle information extraction command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from ingestion.loader import get_loader
    from extraction.extractor import InformationExtractor
    
    # Initialize classifier and extractor
    classifier = DocumentClassifier(
        model_name=config.classification.model_name,
        confidence_threshold=config.classification.confidence_threshold,
        fallback_threshold=config.classification.fallback_threshold
    )
    
    extractor = InformationExtractor()
    
    print("Loading classification model...")
    classifier.load_model()
    print("Model loaded successfully.\n")
    
    # Collect all files to process
    files_to_process: List[Path] = []
    
    for file_arg in args.files:
        file_path = Path(file_arg)
        
        # If it's a directory, add all supported files from it
        if file_path.is_dir():
            supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
            for ext in supported_extensions:
                files_to_process.extend(file_path.glob(f"*{ext}"))
        elif file_path.is_file():
            files_to_process.append(file_path)
        else:
            print(f"Warning: File or directory not found: {file_path}")
    
    if not files_to_process:
        print("No files found to extract from.")
        return
    
    print(f"Processing {len(files_to_process)} document(s)...\n")
    
    results: List[Dict[str, Any]] = []
    
    for file_path in files_to_process:
        try:
            print(f"Processing: {file_path.name}...")
            
            # Load document
            loader = get_loader(file_path)
            document = loader.load()
            text = document.get("text", "")
            
            if not text or not text.strip():
                result = {
                    "file": str(file_path),
                    "class": "Unclassifiable",
                    "extracted_data": {},
                    "error": "No text extracted from document"
                }
                print(f"  ❌ No text extracted\n")
                results.append(result)
                continue
            
            # Classify document
            classification = classifier.classify(text)
            doc_class = classification["class"]
            confidence = classification["confidence"]
            
            print(f"  Classified as: {doc_class} (confidence: {confidence:.3f})")
            
            # Extract data only if document is classified (not Other/Unclassifiable)
            if doc_class in ["Other", "Unclassifiable"]:
                extracted_data = {}
                print(f"  ⚠️  Skipping extraction for {doc_class} document type")
            else:
                extracted_data = extractor.extract(text, doc_class)
                print(f"  ✓ Extracted {len([v for v in extracted_data.values() if v is not None])} field(s)")
            
            result = {
                "file": str(file_path),
                "class": doc_class,
                "confidence": confidence,
                "extracted_data": extracted_data
            }
            
            results.append(result)
            print()
            
        except Exception as e:
            error_result = {
                "file": str(file_path),
                "class": "Unclassifiable",
                "extracted_data": {},
                "error": str(e)
            }
            results.append(error_result)
            print(f"  ❌ Error: {e}\n")
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    else:
        # Print summary
        print("="*60)
        print("Extraction Summary")
        print("="*60)
        for result in results:
            print(f"\nFile: {Path(result['file']).name}")
            print(f"  Class: {result['class']}")
            if 'confidence' in result:
                print(f"  Confidence: {result['confidence']:.3f}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                extracted = result.get('extracted_data', {})
                if extracted:
                    print("  Extracted Data:")
                    for key, value in extracted.items():
                        if value is not None:
                            print(f"    {key}: {value}")
                else:
                    print("  No data extracted (document type not supported)")
        
        # Print statistics
        print("\n" + "="*60)
        print("Statistics")
        print("="*60)
        class_counts: Dict[str, int] = {}
        extraction_counts: Dict[str, int] = {"success": 0, "skipped": 0, "error": 0}
        
        for result in results:
            cls = result['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
            if 'error' in result:
                extraction_counts["error"] += 1
            elif cls in ["Other", "Unclassifiable"]:
                extraction_counts["skipped"] += 1
            else:
                extracted = result.get('extracted_data', {})
                if any(v is not None for v in extracted.values()):
                    extraction_counts["success"] += 1
                else:
                    extraction_counts["skipped"] += 1
        
        print("\nClassification:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
        
        print("\nExtraction:")
        for status, count in extraction_counts.items():
            print(f"  {status.capitalize()}: {count}")


def handle_embed(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle embedding generation command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from ingestion.loader import get_loader
    from embeddings.generator import EmbeddingGenerator, generate_document_id
    
    # Initialize embedding generator
    cache_dir = config.data_dir / "embeddings_cache"
    generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=cache_dir
    )
    
    print("Loading embedding model...")
    generator.load_model()
    print(f"Model loaded: {config.embedding.model_name}")
    print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    print(f"Cache directory: {cache_dir}\n")
    
    # Collect all files to process
    files_to_process: List[Path] = []
    
    for file_arg in args.files:
        file_path = Path(file_arg)
        
        # If it's a directory, add all supported files from it
        if file_path.is_dir():
            supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
            for ext in supported_extensions:
                files_to_process.extend(file_path.glob(f"*{ext}"))
        elif file_path.is_file():
            files_to_process.append(file_path)
        else:
            print(f"Warning: File or directory not found: {file_path}")
    
    if not files_to_process:
        print("No files found to generate embeddings for.")
        return
    
    print(f"Processing {len(files_to_process)} document(s)...\n")
    
    results: List[Dict[str, Any]] = []
    
    for file_path in files_to_process:
        try:
            print(f"Processing: {file_path.name}...", end=" ")
            
            # Load document
            loader = get_loader(file_path)
            document = loader.load()
            text = document.get("text", "")
            
            if not text or not text.strip():
                print("❌ No text extracted")
                continue
            
            # Generate document ID
            doc_id = generate_document_id(file_path, text)
            
            # Generate embedding
            embedding_result = generator.generate(
                text=text,
                document_id=doc_id,
                use_cache=True
            )
            
            # Convert numpy array to list for JSON serialization in output
            embedding_array = embedding_result["embedding"]
            
            result = {
                "document_id": embedding_result["document_id"],
                "file_path": str(file_path),
                "file_name": file_path.name,
                "embedding_dimension": len(embedding_array),
                "embedding": embedding_array.tolist()  # Convert to list for JSON
            }
            
            results.append(result)
            print(f"✓ Generated (ID: {doc_id})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Note: Embeddings are also cached in: {cache_dir}")
    else:
        # Print summary
        print("\n" + "="*60)
        print("Embedding Generation Summary")
        print("="*60)
        for result in results:
            print(f"\nFile: {result['file_name']}")
            print(f"  Document ID: {result['document_id']}")
            print(f"  Embedding Dimension: {result['embedding_dimension']}")
            print(f"  Embedding (first 5 values): {result['embedding'][:5]}...")
        
        print(f"\n{'='*60}")
        print(f"Total embeddings generated: {len(results)}")
        print(f"Cache directory: {cache_dir}")
        print(f"\nTip: Use --output to save full embeddings to JSON file")


def handle_index(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle document indexing command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from ingestion.loader import get_loader
    from embeddings.generator import EmbeddingGenerator, generate_document_id
    from embeddings.store import VectorStore
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.data_dir / "embeddings_cache"
    )
    
    store_path = Path(args.store_path) if args.store_path else Path(config.vector_store.store_path)
    vector_store = VectorStore(
        store_path=store_path,
        similarity_metric=config.vector_store.similarity_metric
    )
    
    print("Loading embedding model...")
    embedding_generator.load_model()
    print(f"Model loaded: {config.embedding.model_name}\n")
    
    # Collect all files to process
    files_to_process: List[Path] = []
    
    for file_arg in args.files:
        file_path = Path(file_arg)
        
        # If it's a directory, add all supported files from it
        if file_path.is_dir():
            supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
            for ext in supported_extensions:
                files_to_process.extend(file_path.glob(f"*{ext}"))
        elif file_path.is_file():
            files_to_process.append(file_path)
        else:
            print(f"Warning: File or directory not found: {file_path}")
    
    if not files_to_process:
        print("No files found to index.")
        return
    
    print(f"Indexing {len(files_to_process)} document(s)...\n")
    
    embeddings_list: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []
    texts_dict: Dict[str, str] = {}
    
    for file_path in files_to_process:
        try:
            print(f"Processing: {file_path.name}...", end=" ")
            
            # Load document
            loader = get_loader(file_path)
            document = loader.load()
            text = document.get("text", "")
            
            if not text or not text.strip():
                print("❌ No text extracted")
                continue
            
            # Generate document ID
            doc_id = generate_document_id(file_path, text)
            
            # Generate embedding
            embedding_result = embedding_generator.generate(
                text=text,
                document_id=doc_id,
                use_cache=True
            )
            
            embedding = embedding_result["embedding"]
            embeddings_list.append(embedding)
            
            # Store metadata
            metadata = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "file_name": file_path.name
            }
            metadata_list.append(metadata)
            
            # Store text for snippet extraction
            texts_dict[doc_id] = text
            
            print(f"✓ Indexed (ID: {doc_id})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not embeddings_list:
        print("\nNo documents were successfully indexed.")
        return
    
    # Add all embeddings to vector store
    print(f"\nBuilding FAISS index with {len(embeddings_list)} documents...")
    embeddings_array = np.array(embeddings_list)
    vector_store.add_embeddings(
        embeddings=embeddings_array,
        metadata=metadata_list,
        texts=texts_dict
    )
    
    # Save vector store
    print("Saving index to disk...")
    vector_store.save()
    
    # Print statistics
    stats = vector_store.get_stats()
    print(f"\n{'='*60}")
    print("Indexing Complete")
    print(f"{'='*60}")
    print(f"Documents indexed: {stats['num_vectors']}")
    print(f"Embedding dimension: {stats['dimension']}")
    print(f"Similarity metric: {stats['similarity_metric']}")
    print(f"Index saved to: {store_path}")


def handle_query(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle document query command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from embeddings.generator import EmbeddingGenerator
    from embeddings.store import VectorStore
    from retrieval.retriever import DocumentRetriever
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.data_dir / "embeddings_cache"
    )
    
    store_path = Path(args.store_path) if args.store_path else Path(config.vector_store.store_path)
    
    # Load vector store
    print(f"Loading index from: {store_path}")
    vector_store = VectorStore(
        store_path=store_path,
        similarity_metric=config.vector_store.similarity_metric
    )
    vector_store.load()
    
    stats = vector_store.get_stats()
    if stats["num_vectors"] == 0:
        print("Error: No documents found in index. Please run 'index' command first.")
        return
    
    print(f"Index loaded: {stats['num_vectors']} documents")
    print(f"Embedding dimension: {stats['dimension']}")
    print(f"Similarity metric: {stats['similarity_metric']}\n")
    
    # Initialize retriever
    embedding_generator.load_model()
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
    
    # Perform query
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}\n")
    print("Searching...\n")
    
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k
    )
    
    if not results:
        print("No results found.")
        return
    
    # Output results
    print("="*60)
    print("Search Results")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Document ID: {result['document_id']}")
        print(f"    Similarity Score: {result['score']:.4f}")
        if 'file_name' in result:
            print(f"    File: {result['file_name']}")
        print(f"    Snippet: {result['snippet']}")
    
    print(f"\n{'='*60}")
    print(f"Found {len(results)} result(s)")


def handle_rag(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle RAG pipeline command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    # TODO: Implement RAG pipeline logic
    if args.index:
        print(f"Indexing {len(args.files)} document(s) for RAG...")
    if args.query:
        print(f"RAG Query: {args.query}")


def handle_process(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle document processing command (complete pipeline).
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from classification.classifier import DocumentClassifier
    from embeddings.generator import EmbeddingGenerator
    from extraction.extractor import InformationExtractor
    from pipelines.document_pipeline import DocumentPipeline
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Find all supported documents
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
    file_paths: List[Path] = []
    for ext in supported_extensions:
        file_paths.extend(input_dir.glob(f"*{ext}"))
    
    if not file_paths:
        print(f"No supported documents found in: {input_dir}")
        return
    
    print("="*60)
    print("Document Processing Pipeline")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Found {len(file_paths)} document(s)")
    print(f"Output file: {args.output}")
    print()
    
    # Initialize components
    print("Initializing pipeline components...")
    
    classifier = DocumentClassifier(
        model_name=config.classification.model_name,
        confidence_threshold=config.classification.confidence_threshold,
        fallback_threshold=config.classification.fallback_threshold
    )
    print("  - Loading classification model...", end=" ")
    classifier.load_model()
    print("✓")
    
    extractor = InformationExtractor()
    print("  - Information extractor: ✓")
    
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.data_dir / "embeddings_cache"
    )
    print("  - Loading embedding model...", end=" ")
    embedding_generator.load_model()
    print("✓")
    
    # Create pipeline
    pipeline = DocumentPipeline(
        classifier=classifier,
        extractor=extractor,
        embedding_generator=embedding_generator
    )
    
    print("\nPipeline initialized successfully.\n")
    
    # Process documents
    results = pipeline.process_batch(file_paths, verbose=True)
    
    # Save results (in assessment format by default)
    output_path = Path(args.output)
    pipeline.save_results(results, output_path, assessment_format=True)
    
    # Also save full format for reference
    full_output_path = output_path.parent / f"{output_path.stem}_full{output_path.suffix}"
    pipeline.save_results(results, full_output_path, assessment_format=False)
    
    # Also index documents for search
    print("\n" + "="*60)
    print("Indexing documents for search...")
    print("="*60)
    
    try:
        from embeddings.store import VectorStore
        
        store_path = config.data_dir / "vector_store"
        vector_store = VectorStore(
            store_path=store_path,
            similarity_metric=config.vector_store.similarity_metric
        )
        
        # Collect embeddings and metadata
        embeddings_list: List[np.ndarray] = []
        metadata_list: List[Dict[str, Any]] = []
        texts_dict: Dict[str, str] = {}
        
        for result in results:
            if result.get("status") == "success" and result.get("embedding"):
                doc_id = result["document_id"]
                embedding_vector = np.array(result["embedding"]["vector"])
                
                embeddings_list.append(embedding_vector)
                metadata_list.append({
                    "document_id": doc_id,
                    "file_path": result["file_path"],
                    "file_name": result["file_name"]
                })
                texts_dict[doc_id] = result.get("text", "")
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list)
            vector_store.add_embeddings(
                embeddings=embeddings_array,
                metadata=metadata_list,
                texts=texts_dict
            )
            vector_store.save()
            print(f"✓ Indexed {len(embeddings_list)} document(s) for search")
            print(f"  Index location: {store_path}")
        else:
            print("⚠️  No documents available for indexing")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not index documents: {e}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


def handle_search(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle semantic search command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    from embeddings.generator import EmbeddingGenerator
    from embeddings.store import VectorStore
    from retrieval.retriever import DocumentRetriever
    
    store_path = Path(args.store_path) if args.store_path else Path(config.vector_store.store_path)
    
    print("="*60)
    print("Semantic Search")
    print("="*60)
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}")
    print(f"Index: {store_path}")
    print()
    
    # Load vector store
    print("Loading search index...", end=" ")
    vector_store = VectorStore(
        store_path=store_path,
        similarity_metric=config.vector_store.similarity_metric
    )
    vector_store.load()
    
    stats = vector_store.get_stats()
    if stats["num_vectors"] == 0:
        print("❌")
        print("Error: No documents found in index.")
        print("Please run 'process' command first to index documents.")
        return
    
    print(f"✓ ({stats['num_vectors']} documents)")
    
    # Initialize retriever
    print("Initializing retriever...", end=" ")
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.data_dir / "embeddings_cache"
    )
    embedding_generator.load_model()
    
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
    print("✓")
    
    # Perform search
    print("\nSearching...\n")
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k
    )
    
    if not results:
        print("No results found.")
        return
    
    # Display results
    print("="*60)
    print("Search Results")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.4f}")
        print(f"    Document ID: {result['document_id']}")
        if 'file_name' in result:
            print(f"    File: {result['file_name']}")
        print(f"    Snippet: {result['snippet']}")
    
    print(f"\n{'='*60}")
    print(f"Found {len(results)} result(s)")
    print("="*60)


def handle_qa(args: argparse.Namespace, config: AppConfig) -> None:
    """
    Handle question answering command.
    
    Args:
        args: Parsed command arguments
        config: Application configuration
    """
    # Check for GROQ API (optional)
    try:
        from qa.groq_answerer import GroqQuestionAnswerer, is_groq_available, get_available_models
        groq_available = is_groq_available()
    except ImportError:
        groq_available = False
        GroqQuestionAnswerer = None
        get_available_models = lambda: []
    
    # Check for Ollama (local)
    try:
        from qa.ollama_answerer import OllamaAnswerer, is_ollama_available, get_ollama_models
        ollama_available = is_ollama_available(host=getattr(args, "ollama_host", "http://localhost:11434"))
    except ImportError:
        ollama_available = False
        OllamaAnswerer = None
        get_ollama_models = lambda *args, **kwargs: []

    # Check for local LLM (llama-cpp) as fallback
    try:
        from qa.answerer import QuestionAnswerer, is_qa_available
        local_qa_available = is_qa_available()
    except ImportError:
        local_qa_available = False
        QuestionAnswerer = None
    
    # If neither is available, show setup instructions
    if not groq_available and not ollama_available and not local_qa_available:
        print("="*60)
        print("Question Answering Not Available")
        print("="*60)
        print("Error: No QA method available.")
        print("\nTo enable question answering:")
        print("\nOption 1: Ollama (Local - Recommended)")
        print("  1. Install Ollama")
        print("  2. Pull a model: ollama pull llama3.1:8b")
        print("  3. Start server: ollama serve")
        print("\nOption 2: GROQ API (Optional)")
        print("  1. Install: pip install groq")
        print("  2. Add GROQ_API_KEY to .env file")
        print("\nOption 3: Local LLM (llama-cpp)")
        print("  1. Install: pip install llama-cpp-python")
        print("  2. Download a GGUF model (e.g., Mistral 7B)")
        print("  3. Specify model path with --model-path")
        return
    
    from embeddings.generator import EmbeddingGenerator
    from embeddings.store import VectorStore
    from retrieval.retriever import DocumentRetriever
    
    store_path = Path(args.store_path) if args.store_path else Path(config.vector_store.store_path)
    
    print("="*60)
    print("Question Answering")
    print("="*60)
    print(f"Question: {args.question}")
    print(f"Top-K: {args.top_k}")
    print(f"Index: {store_path}")
    print()
    
    # Load vector store
    print("Loading search index...", end=" ")
    vector_store = VectorStore(
        store_path=store_path,
        similarity_metric=config.vector_store.similarity_metric
    )
    vector_store.load()
    
    stats = vector_store.get_stats()
    if stats["num_vectors"] == 0:
        print("❌")
        print("Error: No documents found in index.")
        print("Please run 'process' command first to index documents.")
        return
    
    print(f"✓ ({stats['num_vectors']} documents)")
    
    # Initialize retriever
    print("Initializing retriever...", end=" ")
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.data_dir / "embeddings_cache"
    )
    embedding_generator.load_model()
    
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
    print("✓")
    
    # Retrieve relevant documents
    print(f"\nRetrieving top-{args.top_k} documents...", end=" ")
    retrieved_docs = retriever.retrieve(
        query=args.question,
        top_k=args.top_k
    )
    
    if not retrieved_docs:
        print("❌")
        print("No relevant documents found.")
        return
    
    print(f"✓ ({len(retrieved_docs)} documents)")
    
    # Get full text for retrieved documents (for better context)
    for doc in retrieved_docs:
        doc_id = doc.get("document_id")
        if doc_id:
            full_text = vector_store.get_text(doc_id)
            if full_text:
                doc["text"] = full_text
    
    # Initialize question answerer (provider selection)
    provider = (getattr(args, "provider", None) or "auto").lower()
    if provider not in ["auto", "ollama", "groq", "local"]:
        provider = "auto"

    if provider in ["auto", "ollama"] and ollama_available:
        print("Initializing Ollama...", end=" ")
        try:
            model = getattr(args, "ollama_model", None) or "llama3.1:8b"
            host = getattr(args, "ollama_host", None) or "http://localhost:11434"
            answerer = OllamaAnswerer(model=model, host=host)
            print(f"✓ (Model: {answerer.model})")
        except Exception as e:
            print("❌")
            print(f"Error initializing Ollama: {e}")
            return
    elif provider in ["auto", "groq"] and groq_available:
        print("Initializing GROQ API...", end=" ")
        try:
            answerer = GroqQuestionAnswerer()
            print(f"✓ (Model: {answerer.model})")
        except Exception as e:
            print("❌")
            print(f"Error initializing GROQ: {e}")
            print("\nCheck that GROQ_API_KEY is set in .env file")
            return
    elif provider in ["auto", "local"] and local_qa_available:
        print("Loading local LLM model...", end=" ")
        try:
            answerer = QuestionAnswerer(model_path=args.model_path, verbose=True)
            answerer.load_model()
            print("✓")
        except Exception as e:
            print("❌")
            print(f"Error loading LLM model: {e}")
            return
    else:
        print("Error: Requested QA provider is not available.")
        print("Try: --provider ollama  (recommended) or --provider groq")
        return
    
    # Generate answer
    print("\nGenerating answer...\n")
    result = answerer.answer(
        question=args.question,
        retrieved_docs=retrieved_docs,
        max_tokens=args.max_tokens
    )
    
    # Display results
    print("="*60)
    print("Answer")
    print("="*60)
    print(f"\n{result['answer']}\n")
    print("-"*60)
    print(f"Context: {result['context_used']} document(s) used")
    print(f"Model: {result['model']}")
    provider = result.get('provider', 'local')
    print(f"Provider: {provider.upper()}")
    if result.get('tokens_generated'):
        print(f"Tokens generated: {result['tokens_generated']}")
    if result.get('tokens_total'):
        print(f"Total tokens: {result['tokens_total']}")
    print("="*60)


def main() -> None:
    """
    Main entry point for the CLI.
    
    Parses arguments and routes to appropriate handlers.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = get_config()
    
    handlers = {
        "ingest": handle_ingest,
        "classify": handle_classify,
        "extract": handle_extract,
        "embed": handle_embed,
        "index": handle_index,
        "query": handle_query,
        "rag": handle_rag,
        "process": handle_process,
        "search": handle_search,
        "qa": handle_qa,
    }
    
    handler = handlers.get(args.command)
    if handler:
        handler(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

