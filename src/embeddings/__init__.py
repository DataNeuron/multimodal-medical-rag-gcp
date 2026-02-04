# ============================================
# Embeddings Module
# ============================================
"""
Generate embeddings for text and images using Vertex AI.

Components:
    - text_embeddings: Generate text embeddings using Vertex AI text-embedding model
    - image_embeddings: Generate multimodal embeddings for medical images

Embedding models:
    - Text: text-embedding-004 (768 dimensions)
    - Multimodal: multimodalembedding@001 (1408 dimensions)
"""

from .text_embeddings import TextEmbedder, embed_text, embed_documents
from .image_embeddings import ImageEmbedder, embed_image, embed_images_batch

__all__ = [
    "TextEmbedder",
    "embed_text",
    "embed_documents",
    "ImageEmbedder",
    "embed_image",
    "embed_images_batch",
]
