"""
Ollama-based question answering (fully local).

Uses the locally running Ollama server (default: http://localhost:11434)
to generate answers from retrieved document context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def is_ollama_available(host: str = DEFAULT_OLLAMA_HOST, timeout_s: float = 1.5) -> bool:
    """
    Check whether Ollama is reachable.

    Args:
        host: Ollama host URL (e.g., http://localhost:11434)
        timeout_s: Request timeout in seconds

    Returns:
        True if Ollama responds, else False.
    """
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def get_ollama_models(host: str = DEFAULT_OLLAMA_HOST, timeout_s: float = 2.0) -> List[str]:
    """
    Return list of locally available Ollama model names.
    """
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout_s)
        if r.status_code != 200:
            return []
        data = r.json() or {}
        models = data.get("models") or []
        names: List[str] = []
        for m in models:
            name = (m or {}).get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names
    except Exception:
        return []


class OllamaAnswerer:
    """
    Question answering using Ollama (local).
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def _construct_context(self, retrieved_docs: List[Dict[str, Any]], max_chars: int = 8000) -> str:
        parts: List[str] = []
        used = 0
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get("text") or doc.get("snippet") or ""
            if not text:
                continue
            header = f"[Document {i}]"
            chunk = f"{header}\n{text}\n\n"
            if used + len(chunk) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    parts.append(chunk[:remaining] + "...")
                break
            parts.append(chunk)
            used += len(chunk)
        return "".join(parts).strip()

    def _prompt(self, question: str, context: str) -> str:
        return (
            "You are a helpful assistant. Answer ONLY from the context.\n"
            "If the question asks to find documents, list the document numbers that match.\n"
            'If not found, say: "I could not find this information in the provided documents".\n\n'
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Generate answer to question using Ollama.
        """
        context = self._construct_context(retrieved_docs)
        if not context:
            return {
                "answer": "I could not find any relevant context to answer your question.",
                "context_used": 0,
                "model": self.model,
                "tokens_generated": 0,
                "provider": "ollama",
            }

        payload = {
            "model": self.model,
            "prompt": self._prompt(question, context),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json() or {}
            answer_text = (data.get("response") or "").strip()
            return {
                "answer": answer_text or "I could not find this information in the provided documents.",
                "context_used": len([d for d in retrieved_docs if d.get("text") or d.get("snippet")]),
                "model": self.model,
                "tokens_generated": data.get("eval_count") or 0,
                "provider": "ollama",
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_used": 0,
                "model": self.model,
                "tokens_generated": 0,
                "provider": "ollama",
                "error": str(e),
            }


