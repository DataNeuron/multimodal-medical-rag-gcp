# ============================================
# Text Embeddings Module
# ============================================
"""
Generate text embeddings using Vertex AI's text-embedding model.

This module provides:
    - Single text embedding generation
    - Batch embedding for multiple documents
    - Chunking strategies for long documents

The default model (text-embedding-004) produces 768-dimensional vectors
optimized for semantic similarity search.
"""

import os
from typing import Optional

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

from src.utils.gcp_utils import initialize_vertex_ai


class TextEmbedder:
    """
    Text embedding generator using Vertex AI.

    Attributes:
        model_name: Name of the Vertex AI embedding model
        project_id: GCP project ID
        location: GCP region
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize the text embedder.

        Args:
            model_name: Embedding model name (default from env)
            project_id: GCP project ID (default from env)
            location: GCP region (default from env)
        """
        self.model_name = model_name or os.getenv(
            "VERTEX_TEXT_EMBEDDING_MODEL", "text-embedding-004"
        )
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_REGION", "us-central1")

        # Initialize Vertex AI
        initialize_vertex_ai(self.project_id, self.location)

        # Load model
        self._model = TextEmbeddingModel.from_pretrained(self.model_name)

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        embeddings = self._model.get_embeddings([text])
        return embeddings[0].values

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in embeddings])

        return all_embeddings


# Convenience functions for simple usage
def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text."""
    embedder = TextEmbedder()
    return embedder.embed(text)


def embed_documents(
    documents: list[dict],
    text_field: str = "text",
) -> list[dict]:
    """
    Embed documents and add embedding field.

    Args:
        documents: List of document dictionaries
        text_field: Field name containing the text to embed

    Returns:
        Documents with added 'embedding' field
    """
    embedder = TextEmbedder()
    texts = [doc[text_field] for doc in documents]
    embeddings = embedder.embed_batch(texts)

    for doc, emb in zip(documents, embeddings):
        doc["embedding"] = emb

    return documents


if __name__ == "__main__":
    # Example usage
    sample_text = "The chest X-ray shows no acute cardiopulmonary abnormality."
    print(f"Embedding text: {sample_text[:50]}...")

    # Note: Requires GCP credentials to run
    # embedding = embed_text(sample_text)
    # print(f"Embedding dimension: {len(embedding)}")
