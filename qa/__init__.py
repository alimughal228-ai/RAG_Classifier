"""
Question answering module (OPTIONAL).

This module provides question answering using:
- Ollama (fully local) - Recommended for easiest local setup
- GROQ API (optional)
- Local LLM via LLaMA.cpp (optional, fully offline)

Fully optional and isolated.
"""

from typing import List, Optional

try:
    from .answerer import QuestionAnswerer
    __all__: List[str] = ["QuestionAnswerer"]
except ImportError:
    QuestionAnswerer = None

try:
    from .ollama_answerer import OllamaAnswerer, is_ollama_available, get_ollama_models
    if "__all__" in globals():
        __all__.extend(["OllamaAnswerer", "is_ollama_available", "get_ollama_models"])
    else:
        __all__ = ["OllamaAnswerer", "is_ollama_available", "get_ollama_models"]
except ImportError:
    OllamaAnswerer = None
    is_ollama_available = lambda *args, **kwargs: False
    get_ollama_models = lambda *args, **kwargs: []

try:
    from .groq_answerer import GroqQuestionAnswerer, is_groq_available, get_available_models
    if "__all__" in globals():
        __all__.extend(["GroqQuestionAnswerer", "is_groq_available", "get_available_models"])
    else:
        __all__ = ["GroqQuestionAnswerer", "is_groq_available", "get_available_models"]
except ImportError:
    GroqQuestionAnswerer = None
    is_groq_available = lambda: False
    get_available_models = lambda: []
    # keep __all__ as-is

