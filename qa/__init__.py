"""
Question answering module (OPTIONAL).

This module provides question answering using:
- GROQ API (fast, cloud-based) - Recommended
- Local LLM via LLaMA.cpp (fully offline)

Fully optional and isolated.
"""

from typing import List, Optional

try:
    from .answerer import QuestionAnswerer
    __all__: List[str] = ["QuestionAnswerer"]
except ImportError:
    QuestionAnswerer = None

try:
    from .groq_answerer import GroqQuestionAnswerer, is_groq_available, get_available_models
    if QuestionAnswerer:
        __all__: List[str] = ["QuestionAnswerer", "GroqQuestionAnswerer", "is_groq_available", "get_available_models"]
    else:
        __all__: List[str] = ["GroqQuestionAnswerer", "is_groq_available", "get_available_models"]
except ImportError:
    GroqQuestionAnswerer = None
    is_groq_available = lambda: False
    get_available_models = lambda: []
    if QuestionAnswerer:
        __all__: List[str] = ["QuestionAnswerer"]
    else:
        __all__: List[str] = []

