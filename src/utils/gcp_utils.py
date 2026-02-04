# ============================================
# GCP Utilities Module
# ============================================
"""
Centralized GCP client management and utility functions.

This module provides:
    - Vertex AI initialization
    - Storage client management
    - BigQuery client management
    - Environment-aware bucket naming
    - Common GCP operations

All clients are lazily initialized and cached for efficiency.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from google.cloud import storage, bigquery
import vertexai


# Load environment variables
load_dotenv()


def get_project_id() -> str:
    """
    Get the current GCP project ID.

    Returns:
        Project ID from environment

    Raises:
        ValueError: If GCP_PROJECT_ID is not set
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError(
            "GCP_PROJECT_ID environment variable is not set. "
            "Please configure your environment."
        )
    return project_id


def get_region() -> str:
    """Get the current GCP region."""
    return os.getenv("GCP_REGION", "us-central1")


def get_environment() -> str:
    """Get the current environment (dev/test/prod)."""
    return os.getenv("ENVIRONMENT", "dev")


def initialize_vertex_ai(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> None:
    """
    Initialize Vertex AI SDK.

    Args:
        project_id: GCP project ID (default from env)
        location: GCP region (default from env)
    """
    project = project_id or get_project_id()
    region = location or get_region()

    vertexai.init(project=project, location=region)


@lru_cache(maxsize=1)
def get_gcs_client() -> storage.Client:
    """
    Get a cached Cloud Storage client.

    Returns:
        Initialized Storage client
    """
    return storage.Client(project=get_project_id())


@lru_cache(maxsize=1)
def get_bigquery_client() -> bigquery.Client:
    """
    Get a cached BigQuery client.

    Returns:
        Initialized BigQuery client
    """
    return bigquery.Client(project=get_project_id())


def get_bucket_name(bucket_type: str) -> str:
    """
    Get environment-specific bucket name.

    Args:
        bucket_type: Type of bucket (raw, processed, embeddings)

    Returns:
        Full bucket name with environment suffix

    Example:
        get_bucket_name("raw") -> "multimodal-medical-rag-raw-data-dev"
    """
    project_id = get_project_id()
    env = get_environment()

    bucket_map = {
        "raw": f"{project_id}-raw-data-{env}",
        "processed": f"{project_id}-processed-{env}",
        "embeddings": f"{project_id}-embeddings-{env}",
    }

    if bucket_type not in bucket_map:
        raise ValueError(f"Unknown bucket type: {bucket_type}")

    return bucket_map[bucket_type]


def upload_to_gcs(
    local_path: str,
    bucket_name: str,
    blob_name: str,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload a file to Cloud Storage.

    Args:
        local_path: Path to local file
        bucket_name: Target bucket name
        blob_name: Target blob path
        content_type: Optional content type

    Returns:
        GCS URI of uploaded file (gs://bucket/path)
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if content_type:
        blob.content_type = content_type

    blob.upload_from_filename(local_path)

    return f"gs://{bucket_name}/{blob_name}"


def download_from_gcs(
    bucket_name: str,
    blob_name: str,
    local_path: str,
) -> str:
    """
    Download a file from Cloud Storage.

    Args:
        bucket_name: Source bucket name
        blob_name: Source blob path
        local_path: Local destination path

    Returns:
        Local path of downloaded file
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(local_path)

    return local_path


def list_blobs(
    bucket_name: str,
    prefix: Optional[str] = None,
    max_results: Optional[int] = None,
) -> list[str]:
    """
    List blobs in a bucket.

    Args:
        bucket_name: Bucket to list
        prefix: Optional path prefix filter
        max_results: Maximum number of results

    Returns:
        List of blob names
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)

    return [blob.name for blob in blobs]


if __name__ == "__main__":
    # Print current configuration
    print("GCP Configuration:")
    print(f"  Project ID: {os.getenv('GCP_PROJECT_ID', 'NOT SET')}")
    print(f"  Region: {get_region()}")
    print(f"  Environment: {get_environment()}")
    print(f"  Raw bucket: {get_bucket_name('raw')}")
