# ============================================
# Retrieval Module
# ============================================
"""
Vector search and document retrieval for the RAG system.

Components:
    - vector_search: Vertex AI Vector Search integration

Features:
    - Approximate nearest neighbor search
    - Hybrid search (vector + metadata filtering)
    - Configurable similarity thresholds
"""

from .vector_search import VectorSearchClient, search_similar, create_index

__all__ = [
    "VectorSearchClient",
    "search_similar",
    "create_index",
]
