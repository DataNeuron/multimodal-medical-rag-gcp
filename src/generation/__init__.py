# ============================================
# Generation Module
# ============================================
"""
RAG pipeline and LLM-based response generation.

Components:
    - rag_pipeline: Complete RAG workflow from query to response

Features:
    - Context-aware generation using retrieved documents
    - Medical domain-specific prompting
    - Citation and source tracking
    - Confidence scoring
"""

from .rag_pipeline import RAGPipeline, query_rag, generate_response

__all__ = [
    "RAGPipeline",
    "query_rag",
    "generate_response",
]
