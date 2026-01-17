"""
Local question answering using LLaMA.cpp backend.

Provides question answering capabilities using fully offline,
open-source LLMs (Mistral 7B, LLaMA, etc.) via GGUF format.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class QuestionAnswerer:
    """
    Question answering using local LLM via LLaMA.cpp.
    
    This is an optional module that requires llama-cpp-python
    and a GGUF model file to function.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize the question answerer.
        
        Args:
            model_path: Path to GGUF model file (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf)
            n_ctx: Context window size
            n_threads: Number of threads (None = auto)
            verbose: Whether to print model loading messages
            
        Raises:
            ImportError: If llama-cpp-python is not installed
            FileNotFoundError: If model file is not found
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for question answering. "
                "Install with: pip install llama-cpp-python"
            )
        
        if model_path is None:
            # Try to find model in default location
            default_paths = [
                Path("./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
                Path("./models/llama-2-7b-chat.Q4_K_M.gguf"),
                Path.home() / ".cache" / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            ]
            
            for path in default_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please download a GGUF model (e.g., Mistral 7B) and specify the path."
            )
        
        self.model_path: str = model_path
        self.n_ctx: int = n_ctx
        self.n_threads: int = n_threads or os.cpu_count() or 4
        self.verbose: bool = verbose
        
        self.model: Optional[Llama] = None
    
    def load_model(self) -> None:
        """
        Load the LLM model.
        
        Raises:
            RuntimeError: If model cannot be loaded
        """
        if self.model is not None:
            return
        
        try:
            if self.verbose:
                print(f"Loading model: {self.model_path}...")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose,
                n_gpu_layers=0  # CPU only
            )
            
            if self.verbose:
                print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model: {e}")
    
    def _construct_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_context_length: int = 3000
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
    
    def _construct_prompt(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Construct prompt for the LLM.
        
        Args:
            question: User question
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt
        """
        # Use instruction format for Mistral/LLaMA models
        prompt = f"""<s>[INST] You are a helpful assistant that answers questions based on the provided context documents.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based only on the information in the context documents. If the answer cannot be found in the context, say so. [/INST]"""
        
        return prompt
    
    def answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate answer to question using retrieved documents.
        
        Args:
            question: User question
            retrieved_docs: List of retrieved documents (must have 'text' or 'snippet')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with answer and metadata:
            {
                "answer": str,
                "context_used": int,  # Number of documents used
                "model": str,
                "tokens_generated": int
            }
        """
        if self.model is None:
            self.load_model()
        
        # Construct context from retrieved documents
        context = self._construct_context(retrieved_docs)
        
        if not context:
            return {
                "answer": "I could not find any relevant context to answer your question.",
                "context_used": 0,
                "model": self.model_path,
                "tokens_generated": 0
            }
        
        # Construct prompt
        prompt = self._construct_prompt(question, context)
        
        # Generate answer
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>", "[INST]", "[/INST]"],
                echo=False
            )
            
            answer_text = response["choices"][0]["text"].strip()
            tokens_generated = response.get("usage", {}).get("completion_tokens", 0)
            
            # Count documents used in context
            context_used = len([doc for doc in retrieved_docs if doc.get("text") or doc.get("snippet")])
            
            return {
                "answer": answer_text,
                "context_used": context_used,
                "model": Path(self.model_path).name,
                "tokens_generated": tokens_generated
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_used": 0,
                "model": self.model_path,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """
        Check if question answering is available.
        
        Returns:
            True if model is loaded and ready
        """
        return LLAMA_CPP_AVAILABLE and self.model is not None


def is_qa_available() -> bool:
    """
    Check if question answering module is available.
    
    Returns:
        True if llama-cpp-python is installed
    """
    return LLAMA_CPP_AVAILABLE

