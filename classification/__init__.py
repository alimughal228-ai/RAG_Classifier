"""
Document classification module.

This module handles document type classification and
categorization using offline models.
"""

from typing import List

from .classifier import DocumentClassifier

__all__: List[str] = ["DocumentClassifier"]

