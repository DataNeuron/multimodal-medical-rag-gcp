# ============================================
# Utilities Module
# ============================================
"""
Shared utilities for GCP operations and common functions.

Components:
    - gcp_utils: GCP client initialization and helpers

Features:
    - Centralized GCP client management
    - Environment configuration
    - Logging setup
"""

from .gcp_utils import (
    initialize_vertex_ai,
    get_gcs_client,
    get_bigquery_client,
    get_bucket_name,
    get_project_id,
)

__all__ = [
    "initialize_vertex_ai",
    "get_gcs_client",
    "get_bigquery_client",
    "get_bucket_name",
    "get_project_id",
]
