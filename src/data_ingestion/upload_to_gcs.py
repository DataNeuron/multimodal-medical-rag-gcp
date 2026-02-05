#!/usr/bin/env python3
# ============================================
# Upload Medical Data to Google Cloud Storage
# ============================================
"""
Upload synthetic medical data to GCS for the Multimodal Medical RAG system.

WHAT THIS SCRIPT DOES:
======================
1. Creates the Cloud Storage bucket if it doesn't exist
2. Uploads local medical data to GCS with organized structure
3. Shows progress during upload (every 10 files)
4. Prints a summary of what was uploaded

DATA MAPPING (LOCAL → GCS):
===========================
This script implements the Bronze layer of our medallion architecture:

    LOCAL DIRECTORY              →  GCS DESTINATION (BRONZE LAYER)
    ─────────────────────────────────────────────────────────────────
    data/raw/images/*            →  gs://bucket/bronze/images/
    data/raw/reports/*           →  gs://bucket/bronze/reports/
    data/raw/patient_vitals.csv  →  gs://bucket/bronze/structured/

WHY BRONZE LAYER?
=================
The bronze layer stores data in its raw, unmodified form:
- Preserves original data for audit and reprocessing
- No transformations = no data loss from processing errors
- Single source of truth for all downstream processing

The data will flow through processing stages:
    BRONZE (raw) → SILVER (cleaned) → GOLD (ML-ready embeddings)

BUCKET ORGANIZATION:
====================
gs://multimodal-medical-rag-data/
├── bronze/
│   ├── images/         # Raw X-ray and medical images
│   │   ├── patient_0001_xray.jpg
│   │   └── ...
│   ├── reports/        # Raw clinical text reports
│   │   ├── patient_0001_report.txt
│   │   └── ...
│   └── structured/     # Raw structured data (CSV, JSON)
│       └── patient_vitals.csv
├── silver/             # (Future: cleaned, validated data)
└── gold/               # (Future: embeddings, indexes)

USAGE:
======
    # From project root:
    python -m src.data_ingestion.upload_to_gcs

    # Or directly:
    python src/data_ingestion/upload_to_gcs.py

PREREQUISITES:
==============
1. Google Cloud SDK installed and configured
2. Authentication set up (gcloud auth application-default login)
3. Required packages: google-cloud-storage
4. Data exists in data/raw/ directory
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path for imports
# This allows running the script from any directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our GCS utilities
# These functions handle the low-level Cloud Storage operations
from src.utils.gcp_storage import (
    create_bucket,
    upload_file,
    upload_directory,
    list_blobs,
)


# ============================================
# Configuration
# ============================================
# Bucket name must be globally unique across ALL of GCP
# Using a descriptive name that indicates the project and purpose
BUCKET_NAME = "multimodal-medical-rag-data"

# GCS region - choose based on:
# - Proximity to users/services for lower latency
# - Data residency requirements (some regulations require data in specific regions)
# - Cost (some regions are cheaper than others)
# - Integration with Vertex AI (should be same region as AI services)
BUCKET_LOCATION = "us-central1"

# Local data directory (relative to project root)
DATA_DIR = project_root / "data" / "raw"

# Mapping of local directories to GCS prefixes
# Each entry maps a local path to its destination in the bronze layer
UPLOAD_MAPPINGS = [
    {
        "name": "Medical Images",
        "local_path": DATA_DIR / "images",
        "gcs_prefix": "bronze/images",
        "extensions": [".jpg", ".jpeg", ".png", ".dcm"],  # Common medical image formats
        "description": "X-ray and medical imaging files"
    },
    {
        "name": "Clinical Reports",
        "local_path": DATA_DIR / "reports",
        "gcs_prefix": "bronze/reports",
        "extensions": [".txt", ".pdf", ".json"],  # Text-based report formats
        "description": "Clinical notes and diagnostic reports"
    },
]

# Single file uploads (for structured data)
SINGLE_FILE_UPLOADS = [
    {
        "name": "Patient Vitals",
        "local_path": DATA_DIR / "patient_vitals.csv",
        "gcs_prefix": "bronze/structured/patient_vitals.csv",
        "description": "Structured patient vital signs data"
    },
]


# ============================================
# Progress Display Functions
# ============================================
def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_progress(current: int, total: int, filename: str) -> None:
    """
    Print upload progress every 10 files.

    This provides feedback during long uploads without flooding
    the console with every file. Shows:
    - Progress percentage
    - Files completed / total
    - Current file being uploaded

    Args:
        current: Number of files uploaded so far
        total: Total number of files to upload
        filename: Name of the current file
    """
    # Print every 10 files, plus first and last
    if current == 1 or current == total or current % 10 == 0:
        pct = (current / total) * 100
        # Extract just the filename from the full path
        short_name = Path(filename).name
        print(f"  [{current:3d}/{total:3d}] {pct:5.1f}% - {short_name}")


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ============================================
# Main Upload Functions
# ============================================
def ensure_bucket_exists() -> bool:
    """
    Create the GCS bucket if it doesn't exist.

    WHY CHECK FIRST?
    - Avoid errors if bucket already exists
    - Provide clear feedback about bucket status
    - Handle the case where bucket exists but in different region

    Returns:
        True if bucket is ready (created or already existed)
    """
    print_header("Step 1: Checking/Creating GCS Bucket")
    print(f"  Bucket name: {BUCKET_NAME}")
    print(f"  Location: {BUCKET_LOCATION}")

    created, message = create_bucket(
        bucket_name=BUCKET_NAME,
        location=BUCKET_LOCATION,
        storage_class="STANDARD",  # Best for frequently accessed data
    )

    if created:
        print(f"  [OK] {message}")
    else:
        # Bucket already exists - this is fine
        print(f"  [OK] {message}")

    return True


def upload_directories() -> Dict[str, Tuple[int, List[str]]]:
    """
    Upload all configured directories to GCS.

    WHY SEPARATE DIRECTORIES?
    - Different data types may have different access patterns
    - Easier to set different permissions per data type
    - Clearer organization for downstream processing
    - Can parallelize processing of different data types

    Returns:
        Dictionary mapping directory name to (file_count, uri_list)
    """
    print_header("Step 2: Uploading Directories")

    results = {}

    for mapping in UPLOAD_MAPPINGS:
        name = mapping["name"]
        local_path = mapping["local_path"]
        gcs_prefix = mapping["gcs_prefix"]
        extensions = mapping.get("extensions")

        print(f"\n  Uploading: {name}")
        print(f"    From: {local_path}")
        print(f"    To:   gs://{BUCKET_NAME}/{gcs_prefix}/")

        # Check if local directory exists
        if not local_path.exists():
            print(f"    [SKIP] Directory not found, skipping")
            results[name] = (0, [])
            continue

        # Upload with progress callback
        try:
            count, uris = upload_directory(
                local_dir=str(local_path),
                bucket_name=BUCKET_NAME,
                gcs_prefix=gcs_prefix,
                file_extensions=extensions,
                progress_callback=print_progress,
            )
            results[name] = (count, uris)
            print(f"    [OK] Uploaded {count} files")
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[name] = (0, [])

    return results


def upload_single_files() -> Dict[str, str]:
    """
    Upload individual files (like CSVs) to GCS.

    WHY SEPARATE SINGLE FILES?
    - Structured data (CSV, JSON) often has different handling
    - May need specific content types set
    - Often a single file per category (unlike many images)

    Returns:
        Dictionary mapping file name to GCS URI (or error message)
    """
    print_header("Step 3: Uploading Structured Data Files")

    results = {}

    for upload in SINGLE_FILE_UPLOADS:
        name = upload["name"]
        local_path = upload["local_path"]
        gcs_prefix = upload["gcs_prefix"]

        print(f"\n  Uploading: {name}")
        print(f"    From: {local_path}")
        print(f"    To:   gs://{BUCKET_NAME}/{gcs_prefix}")

        # Check if file exists
        if not local_path.exists():
            print(f"    [SKIP] File not found, skipping")
            results[name] = "NOT_FOUND"
            continue

        # Upload the file
        try:
            uri = upload_file(
                local_path=str(local_path),
                bucket_name=BUCKET_NAME,
                blob_path=gcs_prefix,
                content_type="text/csv",  # Explicit content type for CSVs
            )
            results[name] = uri
            # Get file size for display
            size = local_path.stat().st_size
            print(f"    [OK] Uploaded ({format_size(size)})")
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[name] = f"ERROR: {e}"

    return results


def verify_uploads() -> Dict[str, int]:
    """
    Verify uploads by listing bucket contents.

    WHY VERIFY?
    - Confirms data actually reached GCS
    - Catches partial upload failures
    - Provides confidence in data availability

    Returns:
        Dictionary of prefix to file count
    """
    print_header("Step 4: Verifying Uploads")

    prefixes = ["bronze/images/", "bronze/reports/", "bronze/structured/"]
    results = {}

    for prefix in prefixes:
        print(f"\n  Checking: gs://{BUCKET_NAME}/{prefix}")
        try:
            blobs = list_blobs(
                bucket_name=BUCKET_NAME,
                prefix=prefix,
            )
            count = len(blobs)
            results[prefix] = count
            print(f"    [OK] Found {count} objects")
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[prefix] = 0

    return results


def print_summary(
    dir_results: Dict[str, Tuple[int, List[str]]],
    file_results: Dict[str, str],
    verify_results: Dict[str, int],
) -> None:
    """
    Print a comprehensive summary of the upload operation.

    Shows:
    - Total files uploaded per category
    - GCS URIs for accessing data
    - Verification status
    - Next steps for processing
    """
    print_header("UPLOAD SUMMARY")

    # Calculate totals
    total_uploaded = sum(count for count, _ in dir_results.values())
    total_files = sum(1 for uri in file_results.values() if not uri.startswith(("NOT_FOUND", "ERROR")))

    print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bucket: gs://{BUCKET_NAME}/")
    print(f"  Region: {BUCKET_LOCATION}")

    print("\n  DIRECTORIES UPLOADED:")
    print("  " + "-" * 40)
    for name, (count, _) in dir_results.items():
        status = "[OK]" if count > 0 else "[SKIP]"
        print(f"    {status} {name}: {count} files")

    print("\n  STRUCTURED FILES UPLOADED:")
    print("  " + "-" * 40)
    for name, uri in file_results.items():
        if uri.startswith("NOT_FOUND"):
            print(f"    [SKIP] {name}: Not found (skipped)")
        elif uri.startswith("ERROR"):
            print(f"    [ERROR] {name}: {uri}")
        else:
            print(f"    [OK] {name}: Uploaded")

    print("\n  VERIFICATION:")
    print("  " + "-" * 40)
    for prefix, count in verify_results.items():
        # Clean up prefix for display
        display_prefix = prefix.rstrip("/")
        print(f"    {display_prefix}: {count} objects")

    total_objects = sum(verify_results.values())
    print(f"\n  TOTAL: {total_uploaded + total_files} files uploaded")
    print(f"         {total_objects} objects in bucket")

    print("\n  GCS PATHS FOR DOWNSTREAM PROCESSING:")
    print("  " + "-" * 40)
    print(f"    Images:     gs://{BUCKET_NAME}/bronze/images/")
    print(f"    Reports:    gs://{BUCKET_NAME}/bronze/reports/")
    print(f"    Structured: gs://{BUCKET_NAME}/bronze/structured/")

    print("\n  NEXT STEPS:")
    print("  " + "-" * 40)
    print("    1. Run data validation on bronze layer")
    print("    2. Process data to silver layer (cleaning, validation)")
    print("    3. Generate embeddings for gold layer")
    print("    4. Build vector search index for RAG retrieval")

    print("\n" + "=" * 60)
    print(" Upload complete!")
    print("=" * 60 + "\n")


# ============================================
# Main Entry Point
# ============================================
def main():
    """
    Main function to orchestrate the upload process.

    Steps:
    1. Check/create bucket
    2. Upload directories (images, reports)
    3. Upload single files (CSVs)
    4. Verify uploads
    5. Print summary
    """
    print("\n" + "=" * 60)
    print(" MULTIMODAL MEDICAL RAG - DATA UPLOAD TO GCS")
    print("=" * 60)

    print(f"\n  Project: multimodal-medical-rag")
    print(f"  Data source: {DATA_DIR}")
    print(f"  Destination: gs://{BUCKET_NAME}/bronze/")

    # Check for local data
    if not DATA_DIR.exists():
        print(f"\n  [ERROR] Data directory not found: {DATA_DIR}")
        print("    Please ensure data exists in data/raw/ before running this script.")
        sys.exit(1)

    # Run upload steps
    try:
        ensure_bucket_exists()
        dir_results = upload_directories()
        file_results = upload_single_files()
        verify_results = verify_uploads()
        print_summary(dir_results, file_results, verify_results)

    except Exception as e:
        print(f"\n  [ERROR] Fatal error: {e}")
        print("\n  TROUBLESHOOTING:")
        print("    1. Run: gcloud auth application-default login")
        print("    2. Ensure you have a GCP project with billing enabled")
        print("    3. Set your project: gcloud config set project YOUR_PROJECT_ID")
        raise


if __name__ == "__main__":
    main()
