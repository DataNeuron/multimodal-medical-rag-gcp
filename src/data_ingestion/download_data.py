# ============================================
# Data Download Module
# ============================================
"""
Download medical datasets from public sources and upload to GCS.

This module handles:
    - Downloading datasets from public medical imaging repositories
    - Validating file integrity
    - Uploading to Cloud Storage with proper organization
    - Tracking download progress and metadata

Supported datasets:
    - NIH Chest X-rays (sample)
    - MIMIC-CXR (requires credentials)
    - Custom datasets via URL
"""

import os
from pathlib import Path
from typing import Optional

from google.cloud import storage
from tqdm import tqdm

from src.utils.gcp_utils import get_gcs_client, get_bucket_name


def list_available_datasets() -> list[dict]:
    """
    List available public medical datasets.

    Returns:
        List of dataset metadata dictionaries containing:
            - name: Dataset identifier
            - description: Brief description
            - size: Approximate size
            - url: Download URL
    """
    return [
        {
            "name": "nih-chest-xray-sample",
            "description": "Sample of NIH Chest X-ray dataset (100 images)",
            "size": "~500MB",
            "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        },
        {
            "name": "synthetic-reports",
            "description": "Synthetic radiology reports for testing",
            "size": "~10MB",
            "url": None,  # Generated locally
        },
    ]


def download_dataset(
    dataset_name: str,
    output_dir: Optional[Path] = None,
    upload_to_gcs: bool = True,
) -> Path:
    """
    Download a medical dataset and optionally upload to GCS.

    Args:
        dataset_name: Name of the dataset to download
        output_dir: Local directory to save files (default: data/raw)
        upload_to_gcs: Whether to upload to Cloud Storage

    Returns:
        Path to the downloaded dataset directory

    Raises:
        ValueError: If dataset_name is not recognized
        RuntimeError: If download fails
    """
    if output_dir is None:
        output_dir = Path("data/raw") / dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement actual download logic for each dataset
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")

    if upload_to_gcs:
        _upload_to_gcs(output_dir, dataset_name)

    return output_dir


def _upload_to_gcs(local_dir: Path, dataset_name: str) -> None:
    """
    Upload downloaded files to Cloud Storage.

    Args:
        local_dir: Local directory containing files to upload
        dataset_name: Name prefix for GCS objects
    """
    client = get_gcs_client()
    bucket_name = get_bucket_name("raw")
    bucket = client.bucket(bucket_name)

    files = list(local_dir.glob("**/*"))
    files = [f for f in files if f.is_file()]

    for file_path in tqdm(files, desc="Uploading to GCS"):
        relative_path = file_path.relative_to(local_dir)
        blob_name = f"{dataset_name}/{relative_path}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(file_path))


if __name__ == "__main__":
    # Example usage
    datasets = list_available_datasets()
    print("Available datasets:")
    for ds in datasets:
        print(f"  - {ds['name']}: {ds['description']}")
