# ============================================
# Vector Search Module
# ============================================
"""
Vertex AI Vector Search integration for efficient similarity search.

This module provides:
    - Index creation and management
    - Document upsert with embeddings
    - K-nearest neighbor search
    - Metadata filtering support

Vector Search uses ScaNN (Scalable Nearest Neighbors) for
sub-millisecond query times on billion-scale datasets.
"""

import os
from typing import Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint

from src.utils.gcp_utils import initialize_vertex_ai


class VectorSearchClient:
    """
    Client for Vertex AI Vector Search operations.

    Handles index management, document indexing, and similarity search.

    Attributes:
        project_id: GCP project ID
        location: GCP region
        index_endpoint: Deployed index endpoint
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        index_endpoint_id: Optional[str] = None,
        deployed_index_id: Optional[str] = None,
    ):
        """
        Initialize the Vector Search client.

        Args:
            project_id: GCP project ID
            location: GCP region
            index_endpoint_id: ID of the deployed index endpoint
            deployed_index_id: ID of the deployed index
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_REGION", "us-central1")
        self.index_endpoint_id = index_endpoint_id or os.getenv(
            "VERTEX_VECTOR_SEARCH_INDEX_ENDPOINT"
        )
        self.deployed_index_id = deployed_index_id or os.getenv(
            "VERTEX_VECTOR_SEARCH_DEPLOYED_INDEX_ID"
        )

        # Initialize Vertex AI
        initialize_vertex_ai(self.project_id, self.location)

        # Load endpoint if configured
        self._endpoint = None
        if self.index_endpoint_id:
            self._load_endpoint()

    def _load_endpoint(self) -> None:
        """Load the index endpoint."""
        self._endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=self.index_endpoint_id
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expression: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_expression: Optional metadata filter

        Returns:
            List of search results with IDs, distances, and metadata

        Raises:
            ValueError: If endpoint is not configured
        """
        if not self._endpoint:
            raise ValueError(
                "Index endpoint not configured. Set VERTEX_VECTOR_SEARCH_INDEX_ENDPOINT"
            )

        # Perform the search
        response = self._endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=top_k,
        )

        # Parse results
        results = []
        for neighbor in response[0]:
            results.append({
                "id": neighbor.id,
                "distance": neighbor.distance,
            })

        return results

    def search_with_text(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search using text query (auto-generates embedding).

        Args:
            query_text: Text query
            top_k: Number of results

        Returns:
            Search results with similarity scores
        """
        from src.embeddings import embed_text

        query_embedding = embed_text(query_text)
        return self.search(query_embedding, top_k)


def create_index(
    display_name: str,
    embedding_dimension: int = 768,
    approximate_neighbors_count: int = 150,
    distance_measure: str = "DOT_PRODUCT_DISTANCE",
) -> MatchingEngineIndex:
    """
    Create a new Vector Search index.

    Args:
        display_name: Human-readable name for the index
        embedding_dimension: Dimension of embedding vectors
        approximate_neighbors_count: ANN algorithm parameter
        distance_measure: Distance metric (DOT_PRODUCT_DISTANCE, COSINE_DISTANCE)

    Returns:
        Created MatchingEngineIndex object
    """
    index = MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        dimensions=embedding_dimension,
        approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=distance_measure,
        description=f"Vector index for {display_name}",
    )

    return index


def search_similar(
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Convenience function for text-based similarity search.

    Args:
        query: Text query
        top_k: Number of results

    Returns:
        List of similar documents
    """
    client = VectorSearchClient()
    return client.search_with_text(query, top_k)


if __name__ == "__main__":
    print("Vector Search Client")
    print("Requires deployed Vertex AI Vector Search index")
    print("Set VERTEX_VECTOR_SEARCH_INDEX_ENDPOINT in environment")
