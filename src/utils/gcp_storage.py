# ============================================
# Google Cloud Storage Utilities Module
# ============================================
"""
Cloud Storage operations for the Multimodal Medical RAG system.

WHY WE USE GOOGLE CLOUD STORAGE:
================================
Cloud Storage is Google's object storage service that provides:
- Scalability: Virtually unlimited storage capacity (petabytes+)
- Durability: 99.999999999% (11 nines) annual durability
- Global Availability: Data is replicated across multiple data centers
- Cost-Effective: Pay only for what you use, with automatic tiering
- Integration: Native integration with other GCP services (Vertex AI, BigQuery)

For medical data specifically, Cloud Storage offers:
- HIPAA compliance when properly configured
- Encryption at rest and in transit
- Fine-grained IAM access controls
- Audit logging for compliance

MEDALLION ARCHITECTURE (BRONZE/SILVER/GOLD LAYERS):
===================================================
We use a three-tier data lakehouse architecture:

BRONZE LAYER (Raw Data):
    - Raw data exactly as received from source systems
    - No transformations applied
    - Preserves original format and quality
    - Acts as a "single source of truth" for raw data
    - Example: gs://bucket/bronze/images/, gs://bucket/bronze/reports/

SILVER LAYER (Cleaned/Validated):
    - Data has been cleaned, validated, and standardized
    - Duplicates removed, schema enforced
    - Quality checks applied
    - Ready for analysis but not optimized for specific use cases
    - Example: gs://bucket/silver/validated_reports/

GOLD LAYER (Business-Ready):
    - Aggregated, enriched, and optimized for specific use cases
    - Embeddings stored here for RAG retrieval
    - Ready for consumption by ML models and applications
    - Example: gs://bucket/gold/embeddings/, gs://bucket/gold/indexes/

This separation enables:
- Data lineage tracking
- Easier debugging and reprocessing
- Clear data quality boundaries
- Parallel processing of different stages

GCS PATH STRUCTURE:
==================
Cloud Storage uses a flat namespace with "/" delimiters for organization:

    gs://<bucket-name>/<object-path>

Components:
    - bucket-name: Globally unique identifier for your storage bucket
    - object-path: Path-like string that organizes objects (not real folders)

Example: gs://multimodal-medical-rag-data/bronze/images/patient_001.jpg
         ├── bucket: multimodal-medical-rag-data
         └── path: bronze/images/patient_001.jpg

Important: GCS doesn't have real directories - the "/" is part of the object name.
The console shows folders for convenience, but they're just prefix groupings.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
from google.cloud import storage
from google.api_core import exceptions


def get_storage_client(project_id: Optional[str] = None) -> storage.Client:
    """
    Get a Cloud Storage client.

    The client handles authentication automatically using:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Default service account (on GCP)
    3. User credentials from gcloud CLI

    Args:
        project_id: Optional GCP project ID. If not provided,
                   uses default from environment or gcloud config.

    Returns:
        Initialized Storage client
    """
    if project_id:
        return storage.Client(project=project_id)
    return storage.Client()


def create_bucket(
    bucket_name: str,
    location: str = "us-central1",
    storage_class: str = "STANDARD",
    project_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Create a Cloud Storage bucket if it doesn't already exist.

    BUCKET NAMING RULES:
    - Must be globally unique across all of GCP
    - 3-63 characters long
    - Can only contain lowercase letters, numbers, hyphens, underscores
    - Cannot start with "goog" or contain "google"

    STORAGE CLASSES:
    - STANDARD: Best for frequently accessed data (hot data)
    - NEARLINE: For data accessed less than once per month
    - COLDLINE: For data accessed less than once per quarter
    - ARCHIVE: For data accessed less than once per year

    For medical RAG data, we use STANDARD because:
    - Data is actively processed for embedding generation
    - Frequently accessed during model training and inference
    - No minimum storage duration charges

    Args:
        bucket_name: Name for the new bucket (must be globally unique)
        location: GCP region for the bucket (default: us-central1)
                 Choose based on:
                 - User proximity for latency
                 - Data residency requirements (GDPR, HIPAA)
                 - Integration with other GCP services
        storage_class: Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
        project_id: GCP project to create bucket in

    Returns:
        Tuple of (created: bool, message: str)
        - (True, "Created...") if bucket was created
        - (False, "Already exists...") if bucket already exists

    Example:
        >>> created, msg = create_bucket("my-medical-data", "us-central1")
        >>> print(msg)
        "Created bucket my-medical-data in us-central1"
    """
    client = get_storage_client(project_id)

    # Check if bucket already exists
    # We check existence rather than catching exception on create
    # because we want to return a clear status to the caller
    try:
        existing_bucket = client.get_bucket(bucket_name)
        return (
            False,
            f"Bucket '{bucket_name}' already exists in location '{existing_bucket.location}'"
        )
    except exceptions.NotFound:
        # Bucket doesn't exist, we can create it
        pass
    except exceptions.Forbidden:
        # Bucket exists but owned by someone else, or we don't have access
        return (
            False,
            f"Bucket '{bucket_name}' exists but is not accessible (may be owned by another project)"
        )

    # Create the bucket
    # The Bucket object is configured then created via client.create_bucket()
    bucket = storage.Bucket(client, bucket_name)
    bucket.storage_class = storage_class

    # Create the bucket in the specified location
    # Location cannot be changed after creation
    client.create_bucket(bucket, location=location)

    return (
        True,
        f"Created bucket '{bucket_name}' in {location} with storage class {storage_class}"
    )


def upload_file(
    local_path: str,
    bucket_name: str,
    blob_path: str,
    content_type: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    """
    Upload a single file to Cloud Storage.

    UPLOAD BEHAVIOR:
    - If the blob already exists, it will be overwritten
    - Upload is done in chunks for large files (handled automatically)
    - Content type is auto-detected if not specified

    CONTENT TYPES FOR MEDICAL DATA:
    - JPEG images: "image/jpeg"
    - PNG images: "image/png"
    - DICOM files: "application/dicom"
    - PDF reports: "application/pdf"
    - Text reports: "text/plain"
    - CSV files: "text/csv"

    Setting correct content types enables:
    - Proper browser rendering when accessing via signed URLs
    - Correct handling by downstream services
    - Better organization and filtering

    Args:
        local_path: Path to the local file to upload
        bucket_name: Target bucket name
        blob_path: Destination path in the bucket (e.g., "bronze/images/file.jpg")
        content_type: Optional MIME type. Auto-detected if not provided.
        project_id: GCP project ID

    Returns:
        GCS URI of the uploaded file (gs://bucket/path)

    Example:
        >>> uri = upload_file(
        ...     "data/raw/patient_001.jpg",
        ...     "my-bucket",
        ...     "bronze/images/patient_001.jpg"
        ... )
        >>> print(uri)
        "gs://my-bucket/bronze/images/patient_001.jpg"
    """
    client = get_storage_client(project_id)
    bucket = client.bucket(bucket_name)

    # Create a blob object representing the destination
    # The blob name is the full path within the bucket
    blob = bucket.blob(blob_path)

    # Set content type if provided
    # This affects how the file is served and displayed
    if content_type:
        blob.content_type = content_type

    # Upload the file
    # For large files, this automatically uses resumable uploads
    blob.upload_from_filename(local_path)

    # Return the GCS URI for easy reference
    # This format is used by BigQuery, Vertex AI, and other GCP services
    return f"gs://{bucket_name}/{blob_path}"


def upload_directory(
    local_dir: str,
    bucket_name: str,
    gcs_prefix: str,
    file_extensions: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None,
    project_id: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    Upload an entire directory to Cloud Storage, maintaining structure.

    This function recursively uploads all files in a directory,
    preserving the relative path structure under the GCS prefix.

    DIRECTORY MAPPING:
    Local: data/raw/images/patient_001.jpg
    GCS:   gs://bucket/bronze/images/patient_001.jpg
           └── gcs_prefix ──┘└── relative path ──┘

    FILTERING:
    Use file_extensions to upload only specific file types:
    - [".jpg", ".png"] for images only
    - [".txt", ".pdf"] for documents only
    - None for all files

    PROGRESS TRACKING:
    The progress_callback function is called after each file upload:
        callback(files_uploaded: int, total_files: int, current_file: str)

    Args:
        local_dir: Local directory to upload
        bucket_name: Target bucket name
        gcs_prefix: Prefix path in GCS (e.g., "bronze/images")
                   All files will be uploaded under this prefix
        file_extensions: Optional list of extensions to include (e.g., [".jpg", ".txt"])
                        Include the dot! Use None for all files.
        progress_callback: Optional function called after each upload
        project_id: GCP project ID

    Returns:
        Tuple of (count of files uploaded, list of GCS URIs)

    Example:
        >>> count, uris = upload_directory(
        ...     "data/raw/images",
        ...     "my-bucket",
        ...     "bronze/images",
        ...     file_extensions=[".jpg", ".png"]
        ... )
        >>> print(f"Uploaded {count} files")
    """
    client = get_storage_client(project_id)
    bucket = client.bucket(bucket_name)

    local_path = Path(local_dir)

    # Validate the local directory exists
    if not local_path.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")
    if not local_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {local_dir}")

    # Collect all files to upload
    # Using rglob("*") for recursive glob of all files
    files_to_upload = []
    for file_path in local_path.rglob("*"):
        # Skip directories - we only upload files
        # GCS doesn't have real directories, they're implied by path prefixes
        if not file_path.is_file():
            continue

        # Skip hidden files (starting with .)
        if file_path.name.startswith("."):
            continue

        # Filter by extension if specified
        if file_extensions:
            if file_path.suffix.lower() not in [ext.lower() for ext in file_extensions]:
                continue

        files_to_upload.append(file_path)

    total_files = len(files_to_upload)
    uploaded_uris = []

    # Upload each file
    for i, file_path in enumerate(files_to_upload, 1):
        # Calculate the relative path from the source directory
        # This preserves the directory structure in GCS
        relative_path = file_path.relative_to(local_path)

        # Construct the full GCS path
        # Using forward slashes as GCS uses Unix-style paths
        # Normalize the prefix to not have trailing slash
        gcs_path = f"{gcs_prefix.rstrip('/')}/{relative_path.as_posix()}"

        # Create blob and upload
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(file_path))

        gcs_uri = f"gs://{bucket_name}/{gcs_path}"
        uploaded_uris.append(gcs_uri)

        # Call progress callback if provided
        if progress_callback:
            progress_callback(i, total_files, str(file_path))

    return (total_files, uploaded_uris)


def list_blobs(
    bucket_name: str,
    prefix: Optional[str] = None,
    delimiter: Optional[str] = None,
    max_results: Optional[int] = None,
    project_id: Optional[str] = None,
) -> List[str]:
    """
    List objects (blobs) in a Cloud Storage bucket.

    UNDERSTANDING PREFIXES AND DELIMITERS:

    Since GCS doesn't have real directories, we use prefix/delimiter
    to simulate directory-like behavior:

    - prefix: Only return objects whose names start with this string
    - delimiter: Character used to define "directory" boundaries

    Example bucket contents:
        bronze/images/patient_001.jpg
        bronze/images/patient_002.jpg
        bronze/reports/report_001.txt
        bronze/structured/vitals.csv

    list_blobs("bucket", prefix="bronze/images/")
    Returns: ["bronze/images/patient_001.jpg", "bronze/images/patient_002.jpg"]

    list_blobs("bucket", prefix="bronze/", delimiter="/")
    Returns: ["bronze/images/", "bronze/reports/", "bronze/structured/"]
    (Returns "directory" prefixes, not individual files)

    Args:
        bucket_name: Name of the bucket to list
        prefix: Only return blobs with names starting with this prefix
               Use trailing "/" to list a "directory"
        delimiter: Character to use for directory-like listing
                  Usually "/" to get immediate "children" only
        max_results: Maximum number of results to return
        project_id: GCP project ID

    Returns:
        List of blob names (full paths within the bucket)

    Example:
        >>> blobs = list_blobs("my-bucket", prefix="bronze/images/")
        >>> print(blobs[:3])
        ["bronze/images/patient_001.jpg", "bronze/images/patient_002.jpg", ...]
    """
    client = get_storage_client(project_id)
    bucket = client.bucket(bucket_name)

    # List blobs with optional filtering
    # The iterator handles pagination automatically
    blobs = bucket.list_blobs(
        prefix=prefix,
        delimiter=delimiter,
        max_results=max_results,
    )

    # Collect blob names
    # Note: blobs is an iterator, not a list
    blob_names = [blob.name for blob in blobs]

    return blob_names


def get_blob_metadata(
    bucket_name: str,
    blob_path: str,
    project_id: Optional[str] = None,
) -> dict:
    """
    Get metadata for a specific blob.

    Useful for checking:
    - File size (for quota management)
    - Content type
    - Upload timestamp
    - Custom metadata

    Args:
        bucket_name: Bucket containing the blob
        blob_path: Full path to the blob
        project_id: GCP project ID

    Returns:
        Dictionary with blob metadata
    """
    client = get_storage_client(project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.get_blob(blob_path)

    if blob is None:
        raise FileNotFoundError(f"Blob not found: gs://{bucket_name}/{blob_path}")

    return {
        "name": blob.name,
        "size": blob.size,
        "content_type": blob.content_type,
        "created": blob.time_created,
        "updated": blob.updated,
        "md5_hash": blob.md5_hash,
        "storage_class": blob.storage_class,
    }


# ============================================
# Module Self-Test
# ============================================
if __name__ == "__main__":
    print("GCS Storage Utilities Module")
    print("=" * 40)
    print("\nAvailable functions:")
    print("  - create_bucket(bucket_name, location)")
    print("  - upload_file(local_path, bucket_name, blob_path)")
    print("  - upload_directory(local_dir, bucket_name, gcs_prefix)")
    print("  - list_blobs(bucket_name, prefix)")
    print("  - get_blob_metadata(bucket_name, blob_path)")
    print("\nSee docstrings for detailed usage.")
