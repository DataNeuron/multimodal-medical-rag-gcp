# ============================================
# Image Embeddings Module
# ============================================
"""
Generate multimodal embeddings for medical images using Vertex AI.

This module provides:
    - Single image embedding generation
    - Batch processing for multiple images
    - Support for various image formats (JPEG, PNG, DICOM)
    - Combined text+image embeddings for enhanced retrieval

The multimodal embedding model produces 1408-dimensional vectors
that capture both visual and semantic information.
"""

import base64
import os
from pathlib import Path
from typing import Optional, Union

from google.cloud import aiplatform
from vertexai.vision_models import MultiModalEmbeddingModel, Image

from src.utils.gcp_utils import initialize_vertex_ai


class ImageEmbedder:
    """
    Image embedding generator using Vertex AI multimodal model.

    Supports medical imaging formats including DICOM (converted to PNG).

    Attributes:
        model_name: Name of the multimodal embedding model
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
        Initialize the image embedder.

        Args:
            model_name: Embedding model name (default from env)
            project_id: GCP project ID (default from env)
            location: GCP region (default from env)
        """
        self.model_name = model_name or os.getenv(
            "VERTEX_MULTIMODAL_EMBEDDING_MODEL", "multimodalembedding@001"
        )
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_REGION", "us-central1")

        # Initialize Vertex AI
        initialize_vertex_ai(self.project_id, self.location)

        # Load model
        self._model = MultiModalEmbeddingModel.from_pretrained(self.model_name)

    def embed(
        self,
        image_path: Union[str, Path],
        contextual_text: Optional[str] = None,
    ) -> dict:
        """
        Generate embedding for a single image.

        Args:
            image_path: Path to the image file
            contextual_text: Optional text context to enhance embedding

        Returns:
            Dictionary with 'image_embedding' and optionally 'text_embedding'
        """
        image = Image.load_from_file(str(image_path))

        embeddings = self._model.get_embeddings(
            image=image,
            contextual_text=contextual_text,
        )

        result = {
            "image_embedding": embeddings.image_embedding,
        }

        if contextual_text and embeddings.text_embedding:
            result["text_embedding"] = embeddings.text_embedding

        return result

    def embed_from_bytes(
        self,
        image_bytes: bytes,
        contextual_text: Optional[str] = None,
    ) -> dict:
        """
        Generate embedding from image bytes.

        Args:
            image_bytes: Raw image data
            contextual_text: Optional text context

        Returns:
            Dictionary with embedding vectors
        """
        # Encode as base64 for the API
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        image = Image(image_bytes=encoded)

        embeddings = self._model.get_embeddings(
            image=image,
            contextual_text=contextual_text,
        )

        return {
            "image_embedding": embeddings.image_embedding,
        }


def embed_image(
    image_path: Union[str, Path],
    contextual_text: Optional[str] = None,
) -> list[float]:
    """
    Convenience function to embed a single image.

    Args:
        image_path: Path to the image file
        contextual_text: Optional descriptive text

    Returns:
        Image embedding vector
    """
    embedder = ImageEmbedder()
    result = embedder.embed(image_path, contextual_text)
    return result["image_embedding"]


def embed_images_batch(
    image_paths: list[Union[str, Path]],
    contextual_texts: Optional[list[str]] = None,
) -> list[dict]:
    """
    Embed multiple images with progress tracking.

    Args:
        image_paths: List of image file paths
        contextual_texts: Optional list of contextual texts

    Returns:
        List of embedding dictionaries
    """
    from tqdm import tqdm

    embedder = ImageEmbedder()
    results = []

    texts = contextual_texts or [None] * len(image_paths)

    for path, text in tqdm(zip(image_paths, texts), total=len(image_paths)):
        try:
            embedding = embedder.embed(path, text)
            embedding["source_path"] = str(path)
            embedding["status"] = "success"
        except Exception as e:
            embedding = {
                "source_path": str(path),
                "status": "error",
                "error": str(e),
            }
        results.append(embedding)

    return results


if __name__ == "__main__":
    # Example usage
    print("Image Embedder initialized")
    print("Supported formats: JPEG, PNG, DICOM (requires conversion)")
    print("Embedding dimension: 1408")

    # Note: Requires GCP credentials and actual image file to run
    # embedding = embed_image("sample_xray.png", "Chest X-ray PA view")
    # print(f"Embedding generated: {len(embedding)} dimensions")
