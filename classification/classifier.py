"""
Document classification functionality.

Classifies documents by type, category, or other attributes
using local, offline models.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# Document class prototypes - short reference texts for each category
CLASS_PROTOTYPES = {
    "Invoice": (
        "This is an invoice document containing billing information, "
        "itemized charges, payment terms, invoice number, date, "
        "vendor details, and total amount due."
    ),
    "Resume": (
        "This is a resume or curriculum vitae document containing "
        "personal information, work experience, education history, "
        "skills, qualifications, and professional summary."
    ),
    "Utility Bill": (
        "This is a utility bill document containing service charges, "
        "usage information, billing period, account number, "
        "utility provider details, and payment due date."
    ),
    "Other": (
        "This is a general document that does not fit into specific "
        "categories like invoice, resume, or utility bill."
    ),
}

# Keyword-based fallback heuristics
KEYWORD_HEURISTICS = {
    "Invoice": [
        "invoice", "bill", "invoice number", "billing", "payment terms",
        "itemized", "subtotal", "total due", "vendor", "purchase order",
        "invoice date", "due date", "amount", "tax", "shipping"
    ],
    "Resume": [
        "resume", "curriculum vitae", "cv", "work experience", "experience",
        "education", "skills", "qualifications", "objective", "summary",
        "employment", "degree", "university", "college", "certification",
        "references", "email", "phone", "contact", "professional",
        "years of experience", "work history", "career", "position"
    ],
    "Utility Bill": [
        "utility bill", "electric bill", "water bill", "gas bill",
        "utility service", "usage", "kwh", "therms", "gallons",
        "service period", "account number", "utility provider",
        "service address", "meter reading"
    ],
    "Other": []  # Catch-all category
}


class DocumentClassifier:
    """
    Classifies documents into categories.
    
    Uses offline SentenceTransformers model for primary classification
    with keyword-based heuristics as fallback.
    Categories: Invoice, Resume, Utility Bill, Other, Unclassifiable
    """
    
    # Classification categories
    CATEGORIES = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        confidence_threshold: float = 0.5,
        fallback_threshold: float = 0.3
    ) -> None:
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the SentenceTransformers model to use
            confidence_threshold: Minimum confidence for ML classification
            fallback_threshold: Minimum confidence for keyword fallback
        """
        self.model_name: str = model_name
        self.confidence_threshold: float = confidence_threshold
        self.fallback_threshold: float = fallback_threshold
        self.model: Optional[SentenceTransformer] = None
        self.class_embeddings: Optional[np.ndarray] = None
        self.class_names: List[str] = [cat for cat in CLASS_PROTOTYPES.keys()]
    
    def load_model(self) -> None:
        """
        Load the SentenceTransformers model and precompute class embeddings.
        
        Raises:
            Exception: If model cannot be loaded
        """
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Precompute embeddings for class prototypes
            prototype_texts = [
                CLASS_PROTOTYPES[class_name] 
                for class_name in self.class_names
            ]
            self.class_embeddings = self.model.encode(
                prototype_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load classification model: {e}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        
        Since embeddings are normalized, cosine similarity is just the dot product.
        
        Args:
            vec1: First normalized vector
            vec2: Second normalized vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # For normalized vectors, cosine similarity = dot product
        return float(np.dot(vec1, vec2))
    
    def _classify_with_embeddings(self, text: str) -> Tuple[str, float]:
        """
        Classify document using SentenceTransformers embeddings.
        
        Args:
            text: Document text to classify
            
        Returns:
            Tuple of (class_name, confidence_score)
        """
        if self.model is None or self.class_embeddings is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Generate embedding for input text
        text_embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute similarities with all class prototypes
        similarities = []
        for class_embedding in self.class_embeddings:
            similarity = self._cosine_similarity(text_embedding, class_embedding)
            similarities.append(similarity)
        
        # Find best match
        max_idx = int(np.argmax(similarities))
        best_class = self.class_names[max_idx]
        best_confidence = float(similarities[max_idx])
        
        return best_class, best_confidence
    
    def _classify_with_keywords(self, text: str) -> Tuple[str, float]:
        """
        Classify document using keyword-based heuristics (fallback).
        
        Args:
            text: Document text to classify
            
        Returns:
            Tuple of (class_name, confidence_score)
        """
        text_lower = text.lower()
        scores = {}
        
        for class_name, keywords in KEYWORD_HEURISTICS.items():
            if class_name == "Other":
                continue  # Skip Other category in keyword matching
            
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            # Use weighted scoring: more matches = higher confidence
            # For short documents, even 1-2 matches can be significant
            if matches > 0:
                # Base score from matches, with bonus for multiple matches
                score = min(matches / max(len(keywords) * 0.3, 3), 1.0)  # More lenient normalization
                # Boost score if multiple keywords found
                if matches >= 2:
                    score = min(score * 1.3, 0.7)  # Cap at 0.7 for keyword fallback
                elif matches >= 3:
                    score = min(score * 1.5, 0.8)  # Higher boost for 3+ matches
            else:
                score = 0.0
            scores[class_name] = score
        
        if not scores or max(scores.values()) == 0:
            return "Other", 0.3  # Low confidence for catch-all
        
        # Find best match
        best_class = max(scores, key=scores.get)
        best_confidence = scores[best_class]
        
        # Ensure minimum confidence for keyword matches
        if best_confidence > 0:
            best_confidence = max(best_confidence, 0.35)  # Minimum 0.35 for keyword matches
        
        return best_class, best_confidence
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a document.
        
        Args:
            text: Document text to classify
            
        Returns:
            Dictionary with classification results:
            {
                "class": str,  # One of: Invoice, Resume, Utility Bill, Other, Unclassifiable
                "confidence": float  # Confidence score between 0 and 1
            }
        """
        if not text or not text.strip():
            return {"class": "Unclassifiable", "confidence": 0.0}
        
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Primary: Try embedding-based classification
        try:
            ml_class, ml_confidence = self._classify_with_embeddings(text)
            
            # If ML confidence is above threshold, use it
            if ml_confidence >= self.confidence_threshold:
                return {"class": ml_class, "confidence": ml_confidence}
            
            # If ML confidence is below threshold but above fallback threshold,
            # use keyword fallback
            if ml_confidence >= self.fallback_threshold:
                keyword_class, keyword_confidence = self._classify_with_keywords(text)
                
                # Use keyword result if it's more confident
                if keyword_confidence > ml_confidence:
                    return {"class": keyword_class, "confidence": keyword_confidence}
                else:
                    # Use ML result but with lower confidence
                    return {"class": ml_class, "confidence": ml_confidence}
            
            # ML confidence is very low, try keyword fallback
            keyword_class, keyword_confidence = self._classify_with_keywords(text)
            
            if keyword_confidence >= self.fallback_threshold:
                return {"class": keyword_class, "confidence": keyword_confidence}
            
            # Both methods have low confidence - mark as unclassifiable
            return {"class": "Unclassifiable", "confidence": max(ml_confidence, keyword_confidence)}
            
        except Exception as e:
            # If embedding classification fails, use keyword fallback
            try:
                keyword_class, keyword_confidence = self._classify_with_keywords(text)
                if keyword_confidence >= self.fallback_threshold:
                    return {"class": keyword_class, "confidence": keyword_confidence}
            except Exception:
                pass
            
            # If everything fails, return unclassifiable
            return {"class": "Unclassifiable", "confidence": 0.0}
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of classification results, each with format:
            {"class": str, "confidence": float}
        """
        return [self.classify(text) for text in texts]

