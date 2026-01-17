"""
Complete document processing pipeline.

Orchestrates the full workflow:
1. Ingest documents
2. Classify documents
3. Extract structured data
4. Generate embeddings
5. Store results

Ensures layer isolation and clear data flow.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from classification.classifier import DocumentClassifier
from embeddings.generator import EmbeddingGenerator, generate_document_id
from extraction.extractor import InformationExtractor
from ingestion.loader import get_loader


class DocumentPipeline:
    """
    Complete document processing pipeline.
    
    Processes documents through all stages:
    ingestion -> classification -> extraction -> embedding generation
    """
    
    def __init__(
        self,
        classifier: Optional[DocumentClassifier] = None,
        extractor: Optional[InformationExtractor] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ) -> None:
        """
        Initialize the pipeline.
        
        Args:
            classifier: Document classifier instance
            extractor: Information extractor instance
            embedding_generator: Embedding generator instance
        """
        self.classifier: Optional[DocumentClassifier] = classifier
        self.extractor: Optional[InformationExtractor] = extractor
        self.embedding_generator: Optional[EmbeddingGenerator] = embedding_generator
    
    def process_document(
        self,
        file_path: Path,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with all processed data:
            {
                "document_id": str,
                "file_path": str,
                "file_name": str,
                "text": str,
                "classification": {
                    "class": str,
                    "confidence": float
                },
                "extracted_data": dict,
                "embedding": {
                    "dimension": int,
                    "vector": list
                },
                "status": str,
                "error": Optional[str]
            }
        """
        result: Dict[str, Any] = {
            "document_id": None,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "text": "",
            "classification": None,
            "extracted_data": {},
            "embedding": None,
            "status": "pending",
            "error": None
        }
        
        try:
            # Step 1: Ingest document
            if verbose:
                print(f"  [1/4] Ingesting: {file_path.name}...", end=" ")
            
            try:
                loader = get_loader(file_path)
                document = loader.load()
                text = document.get("text", "")
                
                if not text or not text.strip():
                    result["status"] = "failed"
                    result["error"] = "No text extracted from document"
                    if verbose:
                        print("❌ No text extracted")
                    return result
                
                result["text"] = text
                if verbose:
                    print("✓")
                    
            except Exception as e:
                result["status"] = "failed"
                result["error"] = f"Ingestion error: {str(e)}"
                if verbose:
                    print(f"❌ Error: {e}")
                return result
            
            # Step 2: Classify document
            if verbose:
                print(f"  [2/4] Classifying...", end=" ")
            
            try:
                if self.classifier is None:
                    raise RuntimeError("Classifier not initialized")
                
                classification = self.classifier.classify(text)
                result["classification"] = {
                    "class": classification["class"],
                    "confidence": classification["confidence"]
                }
                
                if verbose:
                    print(f"✓ {classification['class']} ({classification['confidence']:.3f})")
                    
            except Exception as e:
                result["status"] = "failed"
                result["error"] = f"Classification error: {str(e)}"
                if verbose:
                    print(f"❌ Error: {e}")
                return result
            
            # Step 3: Extract structured data
            if verbose:
                print(f"  [3/4] Extracting data...", end=" ")
            
            try:
                doc_class = result["classification"]["class"]
                
                if doc_class in ["Other", "Unclassifiable"]:
                    result["extracted_data"] = {}
                    if verbose:
                        print("⚠️  Skipped (document type not supported)")
                else:
                    if self.extractor is None:
                        raise RuntimeError("Extractor not initialized")
                    
                    extracted_data = self.extractor.extract(text, doc_class)
                    result["extracted_data"] = extracted_data
                    
                    extracted_count = len([v for v in extracted_data.values() if v is not None])
                    if verbose:
                        print(f"✓ {extracted_count} field(s)")
                        
            except Exception as e:
                result["status"] = "partial"
                result["error"] = f"Extraction error: {str(e)}"
                if verbose:
                    print(f"⚠️  Error: {e}")
                # Continue to embedding even if extraction fails
            
            # Step 4: Generate embedding
            if verbose:
                print(f"  [4/4] Generating embedding...", end=" ")
            
            try:
                if self.embedding_generator is None:
                    raise RuntimeError("Embedding generator not initialized")
                
                # Generate document ID
                doc_id = generate_document_id(file_path, text)
                result["document_id"] = doc_id
                
                # Generate embedding
                embedding_result = self.embedding_generator.generate(
                    text=text,
                    document_id=doc_id,
                    use_cache=True
                )
                
                embedding_vector = embedding_result["embedding"]
                result["embedding"] = {
                    "dimension": len(embedding_vector),
                    "vector": embedding_vector.tolist()
                }
                
                if verbose:
                    print(f"✓ (dim: {len(embedding_vector)})")
                    
            except Exception as e:
                result["status"] = "partial" if result["status"] == "pending" else result["status"]
                if result["error"]:
                    result["error"] += f"; Embedding error: {str(e)}"
                else:
                    result["error"] = f"Embedding error: {str(e)}"
                if verbose:
                    print(f"❌ Error: {e}")
                return result
            
            # Success
            result["status"] = "success"
            if verbose:
                print(f"  ✓ Document processed successfully")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = f"Unexpected error: {str(e)}"
            if verbose:
                print(f"  ❌ Unexpected error: {e}")
        
        return result
    
    def process_batch(
        self,
        file_paths: List[Path],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents through the pipeline.
        
        Args:
            file_paths: List of document file paths
            verbose: Whether to print progress messages
            
        Returns:
            List of processed document results
        """
        results: List[Dict[str, Any]] = []
        
        total = len(file_paths)
        if verbose:
            print(f"\nProcessing {total} document(s)...\n")
        
        for i, file_path in enumerate(file_paths, 1):
            if verbose:
                print(f"[{i}/{total}] {file_path.name}")
            
            result = self.process_document(file_path, verbose=verbose)
            results.append(result)
            
            if verbose:
                print()  # Blank line between documents
        
        return results
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        assessment_format: bool = False
    ) -> None:
        """
        Save processing results to output.json.
        
        Args:
            results: List of processed document results
            output_path: Path to output JSON file
            assessment_format: If True, save in assessment-required format (filename as key)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if assessment_format:
            # Save in assessment format (filename as key)
            from .format_converter import convert_to_assessment_format
            assessment_output = convert_to_assessment_format(results)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(assessment_output, f, indent=2, ensure_ascii=False)
        else:
            # Save full format (array of objects)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total documents processed: {len(results)}")
        
        # Print summary statistics
        status_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {}
        
        for result in results:
            status = result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            classification = result.get("classification")
            if classification:
                doc_class = classification.get("class", "Unknown")
                class_counts[doc_class] = class_counts.get(doc_class, 0) + 1
        
        print("\nStatus Summary:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status.capitalize()}: {count}")
        
        if class_counts:
            print("\nClassification Summary:")
            for doc_class, count in sorted(class_counts.items()):
                print(f"  {doc_class}: {count}")

