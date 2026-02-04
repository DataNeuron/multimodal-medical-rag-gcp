# ============================================
# Data Ingestion Module
# ============================================
"""
Handles data acquisition and preprocessing for the RAG system.

Components:
    - download_data: Download medical datasets from public sources
    - generate_synthetic: Generate synthetic medical data for testing

Supported data types:
    - Medical images (DICOM, PNG, JPEG)
    - Clinical documents (PDF, TXT)
    - Structured data (CSV, JSON)
"""

from .download_data import download_dataset, list_available_datasets
from .generate_synthetic import generate_synthetic_reports

__all__ = [
    "download_dataset",
    "list_available_datasets",
    "generate_synthetic_reports",
]
