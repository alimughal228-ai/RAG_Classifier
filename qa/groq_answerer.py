"""
GROQ API-based question answering.

Provides question answering using GROQ's fast inference API.
Uses best available models (Mixtral, Llama 3) for high-quality answers.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Load environment variables
if GROQ_AVAILABLE:
    load_dotenv()


# Currently supported GROQ models (as of 2025)
# Note: mixtral-8x7b-32768 has been decommissioned
GROQ_MODELS = [
    "llama-3.1-70b-versatile",  # Best overall - Llama 3.1 70B (128K context)
    "llama-3.1-8b-instant",      # Fast and efficient - Llama 3.1 8B
    "llama-3-70b-8192",          # Llama 3 70B (8K context)
    "llama-3-8b-8192",           # Llama 3 8B (8K context)
    "gemma-7b-it",               # Gemma 7B Instruct
]


class GroqQuestionAnswerer:
    """
    Question answering using GROQ API.
    
    Uses GROQ's fast inference API with best available models.
    Requires GROQ_API_KEY in environment or .env file.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the GROQ question answerer.
        
        Args:
            model: GROQ model name (defaults to best available)
            api_key: GROQ API key (defaults to GROQ_API_KEY env var)
            
        Raises:
            ImportError: If groq package is not installed
            ValueError: If API key is not found
        """
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq package is required. Install with: pip install groq"
            )
        
        # Get API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it in .env file or environment variable."
            )
        
        # Initialize GROQ client
        self.client = Groq(api_key=self.api_key)
        
        # Select best model
        self.model = model or GROQ_MODELS[0]  # Default to best model
    
    def _construct_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_context_length: int = 8000
    ) -> str:
        """
        Construct context window from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents with text/snippets
            max_context_length: Maximum context length in characters
            
        Returns:
            Constructed context string
        """
        context_parts: List[str] = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Get text - prefer full text, fallback to snippet
            text = doc.get("text") or doc.get("snippet", "")
            
            if not text:
                continue
            
            # Add document to context
            doc_header = f"[Document {i}]"
            doc_header_len = len(doc_header) + 2  # +2 for newlines
            
            # Check if adding this document would exceed limit
            if current_length + doc_header_len + len(text) > max_context_length:
                # Try to truncate the last document
                remaining = max_context_length - current_length - doc_header_len - 10
                if remaining > 100:  # Only add if meaningful space remains
                    truncated_text = text[:remaining] + "..."
                    doc_text = f"{doc_header}\n{truncated_text}\n\n"
                    context_parts.append(doc_text)
                break
            
            doc_text = f"{doc_header}\n{text}\n\n"
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts).strip()
    
    def _construct_messages(
        self,
        question: str,
        context: str
    ) -> List[Dict[str, str]]:
        """
        Construct messages for GROQ API (chat format).
        
        Args:
            question: User question
            context: Retrieved context from documents
            
        Returns:
            List of message dictionaries
        """
        system_message = """You are a helpful assistant that answers questions based on the provided context documents.

Your task:
- Answer questions clearly and concisely using ONLY the information from the context documents
- Extract specific facts, dates, amounts, names, and details from the documents
- If the question asks to find documents, list the specific documents that match
- If the answer cannot be found in the context, say "I could not find this information in the provided documents"
- Do not make up information or use knowledge outside the provided context
- Be specific and cite document numbers when relevant"""
        
        user_message = f"""Context Documents:
{context}

Question: {question}

Answer the question using ONLY the information from the context documents above. Be specific and cite which documents contain the relevant information."""
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate answer to question using retrieved documents via GROQ API.
        
        Args:
            question: User question
            retrieved_docs: List of retrieved documents (must have 'text' or 'snippet')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with answer and metadata:
            {
                "answer": str,
                "context_used": int,  # Number of documents used
                "model": str,
                "tokens_generated": int,
                "provider": "groq"
            }
        """
        # Construct context from retrieved documents
        context = self._construct_context(retrieved_docs)
        
        if not context:
            return {
                "answer": "I could not find any relevant context to answer your question.",
                "context_used": 0,
                "model": self.model,
                "tokens_generated": 0,
                "provider": "groq"
            }
        
        # Construct messages
        messages = self._construct_messages(question, context)
        
        # Generate answer via GROQ API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Get token usage
            usage = response.usage
            tokens_generated = usage.completion_tokens if usage else 0
            
            # Count documents used in context
            context_used = len([doc for doc in retrieved_docs if doc.get("text") or doc.get("snippet")])
            
            return {
                "answer": answer_text,
                "context_used": context_used,
                "model": self.model,
                "tokens_generated": tokens_generated,
                "tokens_prompt": usage.prompt_tokens if usage else 0,
                "tokens_total": usage.total_tokens if usage else 0,
                "provider": "groq"
            }
            
        except Exception as e:
            error_str = str(e)
            # Check if model is decommissioned
            if "decommissioned" in error_str.lower() or "model_decommissioned" in error_str:
                # Try fallback to a supported model
                fallback_models = [m for m in GROQ_MODELS if m != self.model]
                if fallback_models:
                    self.model = fallback_models[0]  # Use first available fallback
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stream=False
                        )
                        answer_text = response.choices[0].message.content.strip()
                        usage = response.usage
                        return {
                            "answer": answer_text,
                            "context_used": len([doc for doc in retrieved_docs if doc.get("text") or doc.get("snippet")]),
                            "model": self.model,
                            "tokens_generated": usage.completion_tokens if usage else 0,
                            "tokens_prompt": usage.prompt_tokens if usage else 0,
                            "tokens_total": usage.total_tokens if usage else 0,
                            "provider": "groq",
                            "note": f"Used fallback model (original model was decommissioned)"
                        }
                    except Exception as e2:
                        return {
                            "answer": f"Error: The selected model is not available. Please try a different model. Error: {str(e2)}",
                            "context_used": 0,
                            "model": self.model,
                            "tokens_generated": 0,
                            "provider": "groq",
                            "error": str(e2)
                        }
            
            return {
                "answer": f"Error generating answer: {error_str}",
                "context_used": 0,
                "model": self.model,
                "tokens_generated": 0,
                "provider": "groq",
                "error": error_str
            }
    
    def is_available(self) -> bool:
        """
        Check if GROQ question answering is available.
        
        Returns:
            True if API key is set and client is ready
        """
        return GROQ_AVAILABLE and self.api_key is not None


def is_groq_available() -> bool:
    """
    Check if GROQ API is available.
    
    Returns:
        True if groq package is installed and API key is set
    """
    if not GROQ_AVAILABLE:
        return False
    
    load_dotenv()
    return bool(os.getenv("GROQ_API_KEY"))


def get_available_models() -> List[str]:
    """
    Get list of available GROQ models.
    
    Returns:
        List of model names
    """
    return GROQ_MODELS.copy()

