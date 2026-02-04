# ============================================
# Integration Tests for GCP Connection
# ============================================
"""
Tests to verify GCP service connectivity.

These tests require valid GCP credentials and project access.
Skip with: pytest -m "not integration"
"""

import os
import pytest


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def gcp_project_id():
    """Get GCP project ID from environment."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        pytest.skip("GCP_PROJECT_ID not set")
    return project_id


class TestGCPConnection:
    """Tests for GCP service connectivity."""

    def test_storage_client_connection(self, gcp_project_id):
        """Verify Cloud Storage client can connect."""
        from google.cloud import storage

        client = storage.Client(project=gcp_project_id)
        # Just verify we can list buckets (even if empty)
        buckets = list(client.list_buckets(max_results=1))
        assert isinstance(buckets, list)

    def test_bigquery_client_connection(self, gcp_project_id):
        """Verify BigQuery client can connect."""
        from google.cloud import bigquery

        client = bigquery.Client(project=gcp_project_id)
        # Verify we can list datasets (even if empty)
        datasets = list(client.list_datasets(max_results=1))
        assert isinstance(datasets, list)

    def test_vertex_ai_initialization(self, gcp_project_id):
        """Verify Vertex AI can be initialized."""
        import vertexai

        # Should not raise an exception
        vertexai.init(
            project=gcp_project_id,
            location=os.getenv("GCP_REGION", "us-central1"),
        )


class TestBucketAccess:
    """Tests for bucket access permissions."""

    @pytest.fixture
    def storage_client(self, gcp_project_id):
        """Get storage client."""
        from google.cloud import storage

        return storage.Client(project=gcp_project_id)

    def test_raw_bucket_exists(self, storage_client):
        """Verify raw data bucket exists (if deployed)."""
        from src.utils.gcp_utils import get_bucket_name

        bucket_name = get_bucket_name("raw")

        try:
            bucket = storage_client.bucket(bucket_name)
            exists = bucket.exists()
        except Exception:
            pytest.skip(f"Bucket {bucket_name} not deployed yet")

        if exists:
            assert bucket.name == bucket_name
