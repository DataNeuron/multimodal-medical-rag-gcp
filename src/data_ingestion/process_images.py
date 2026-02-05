"""
Bronze → Silver Image Processing Pipeline

This module handles the transformation of raw medical images from the Bronze layer
to the cleaned, standardized Silver layer for ML processing.

Transformations Applied:
    1. Resize to 512x512 pixels (standard size for medical imaging ML models)
    2. Convert to grayscale (medical X-rays are inherently grayscale)
    3. Normalize pixel values to [0, 1] range (required for neural network input)
    4. Optional denoising using Gaussian blur (reduces noise artifacts)

Bronze Layer Structure (input):
    gs://{bucket}/bronze/images/
        ├── case_001.jpeg
        ├── case_002.jpeg
        └── ...

Silver Layer Structure (output):
    gs://{bucket}/silver/images/
        ├── case_001_processed.npy (numpy array, normalized)
        ├── case_001_preview.png (human-viewable preview)
        └── ...

Author: Medical RAG Pipeline
Date: 2025
"""

import os
import io
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Target image dimensions for ML processing
# 512x512 is a common standard for medical imaging models (balance between
# detail retention and computational efficiency)
TARGET_SIZE: Tuple[int, int] = (512, 512)

# Supported image formats for processing
# JPEG and PNG are the most common formats for medical images in research settings
# (Note: Production systems would also handle DICOM)
SUPPORTED_FORMATS: set = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Progress reporting interval - report every N images
PROGRESS_INTERVAL: int = 20

# Denoising parameters
# Gaussian blur radius - higher = more smoothing
# 0.5 is subtle enough to reduce noise without losing diagnostic detail
DENOISE_RADIUS: float = 0.5


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class ImageValidationResult:
    """
    Result of image validation check.

    Attributes:
        is_valid: Whether the image passed all validation checks
        error_message: Description of validation failure (if any)
        original_size: Original image dimensions (width, height)
        original_mode: PIL image mode (RGB, L, etc.)
        file_size_bytes: Size of the image file
    """
    is_valid: bool
    error_message: Optional[str] = None
    original_size: Optional[Tuple[int, int]] = None
    original_mode: Optional[str] = None
    file_size_bytes: int = 0


@dataclass
class TransformationResult:
    """
    Result of image transformation.

    Attributes:
        success: Whether transformation completed successfully
        processed_array: Normalized numpy array (None if failed)
        preview_image: PIL Image for preview (None if failed)
        error_message: Description of error (if any)
    """
    success: bool
    processed_array: Optional[np.ndarray] = None
    preview_image: Optional[Image.Image] = None
    error_message: Optional[str] = None


@dataclass
class ProcessingSummary:
    """
    Summary statistics for the entire processing run.

    Attributes:
        total_images: Total number of images found in Bronze
        processed_successfully: Number of images successfully processed
        failed: Number of images that failed processing
        skipped: Number of images skipped (already processed, etc.)
        average_original_size_kb: Average size of original images
        average_processing_time_ms: Average time to process each image
        failed_images: List of image paths that failed
        start_time: Processing start timestamp
        end_time: Processing end timestamp
    """
    total_images: int = 0
    processed_successfully: int = 0
    failed: int = 0
    skipped: int = 0
    average_original_size_kb: float = 0.0
    average_processing_time_ms: float = 0.0
    failed_images: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# =============================================================================
# GCS UTILITY FUNCTIONS
# =============================================================================

def get_storage_client() -> storage.Client:
    """
    Get or create a Google Cloud Storage client.

    Uses default credentials from:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Application Default Credentials (ADC)
    3. Compute Engine/Cloud Run service account

    Returns:
        storage.Client: Authenticated GCS client
    """
    project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
    return storage.Client(project=project_id)


def get_bucket_names() -> Dict[str, str]:
    """
    Get bucket names for Bronze and Silver layers based on environment.

    Bucket naming convention:
        - Bronze (raw data): multimodal-medical-rag-data
        - Silver (processed): multimodal-medical-rag-processed-dev

    Returns:
        Dict with 'bronze' and 'silver' bucket names
    """
    project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
    environment = os.getenv("ENVIRONMENT", "dev")

    # Bronze bucket uses the upload_to_gcs.py naming convention
    bronze_bucket = os.getenv("GCS_RAW_DATA_BUCKET", f"{project_id}-data")
    # Silver bucket uses environment-specific naming
    silver_bucket = os.getenv("GCS_PROCESSED_BUCKET", f"{project_id}-processed-{environment}")

    return {
        "bronze": bronze_bucket,
        "silver": silver_bucket
    }


# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def download_from_bronze(gcs_path: str) -> Tuple[Optional[bytes], int]:
    """
    Download an image from the Bronze layer in GCS.

    This function retrieves raw image data from the Bronze storage tier,
    which contains unprocessed medical images as received from source systems.

    Args:
        gcs_path: Path within the Bronze bucket (e.g., "bronze/images/case_001.jpeg")

    Returns:
        Tuple of (image_bytes, file_size_bytes)
        Returns (None, 0) if download fails

    Example:
        >>> image_data, size = download_from_bronze("bronze/images/case_001.jpeg")
        >>> if image_data:
        ...     print(f"Downloaded {size} bytes")
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()

        # Access the Bronze bucket
        bucket = client.bucket(bucket_names["bronze"])
        blob = bucket.blob(gcs_path)

        # Check if blob exists before downloading
        if not blob.exists():
            logger.warning(f"Image not found in Bronze: {gcs_path}")
            return None, 0

        # Download as bytes (keeps image in memory for processing)
        image_bytes = blob.download_as_bytes()
        file_size = len(image_bytes)

        logger.debug(f"Downloaded {gcs_path} ({file_size / 1024:.1f} KB)")
        return image_bytes, file_size

    except Exception as e:
        logger.error(f"Failed to download from Bronze: {gcs_path} - {str(e)}")
        return None, 0


def validate_image(image_bytes: bytes) -> ImageValidationResult:
    """
    Validate that image data is not corrupted and in the correct format.

    Validation Checks Performed:
    1. Image can be opened by PIL (not corrupted)
    2. Image has valid dimensions (> 0 x 0)
    3. Image format is supported (JPEG, PNG, etc.)
    4. Image is not truncated (all data present)

    Medical Imaging Context:
    - Corrupted images can cause model training/inference failures
    - Size validation catches placeholder or thumbnail images
    - Format validation ensures consistent processing pipeline

    Args:
        image_bytes: Raw image data as bytes

    Returns:
        ImageValidationResult with validation status and image metadata

    Example:
        >>> result = validate_image(image_bytes)
        >>> if result.is_valid:
        ...     print(f"Valid image: {result.original_size}")
        >>> else:
        ...     print(f"Invalid: {result.error_message}")
    """
    result = ImageValidationResult(is_valid=False, file_size_bytes=len(image_bytes))

    try:
        # Attempt to open image - this catches corrupted files
        # PIL will raise an exception if the image data is invalid
        image = Image.open(io.BytesIO(image_bytes))

        # Verify the image can be fully loaded (catches truncated files)
        # load() forces PIL to decode all pixel data
        image.load()

        # Extract image metadata
        result.original_size = image.size
        result.original_mode = image.mode

        # Validation check 1: Minimum size requirement
        # Very small images are likely placeholders or corrupt
        width, height = image.size
        if width < 64 or height < 64:
            result.error_message = f"Image too small: {width}x{height} (minimum 64x64)"
            return result

        # Validation check 2: Maximum size check (sanity check)
        # Extremely large images may indicate a problem or waste resources
        if width > 10000 or height > 10000:
            result.error_message = f"Image too large: {width}x{height} (maximum 10000x10000)"
            return result

        # Validation check 3: Check for valid color modes
        # Medical images are typically grayscale (L) or RGB
        valid_modes = {'L', 'RGB', 'RGBA', 'P', 'I', 'F'}
        if image.mode not in valid_modes:
            result.error_message = f"Unsupported color mode: {image.mode}"
            return result

        # All checks passed
        result.is_valid = True
        return result

    except IOError as e:
        # PIL couldn't open the file - likely corrupted
        result.error_message = f"Corrupted image file: {str(e)}"
        return result
    except Exception as e:
        # Unexpected error during validation
        result.error_message = f"Validation error: {str(e)}"
        return result


def transform_image(image_bytes: bytes, apply_denoising: bool = True) -> TransformationResult:
    """
    Apply transformations to prepare image for ML processing.

    Transformation Pipeline:

    Step 1: RESIZE TO 512x512
        - Medical imaging models typically expect fixed input dimensions
        - 512x512 balances detail retention with computational efficiency
        - Uses LANCZOS resampling for high-quality downscaling
        - LANCZOS is preferred over BILINEAR for medical images as it
          preserves edges and fine details better

    Step 2: CONVERT TO GRAYSCALE
        - Medical X-rays contain no color diagnostic information
        - Converting to grayscale reduces data dimensions (3 channels → 1)
        - Reduces model complexity and training time
        - Standard practice in radiology AI systems

    Step 3: NORMALIZE PIXEL VALUES
        - Raw pixel values are 0-255 (8-bit integer)
        - Neural networks work better with 0.0-1.0 (float) range
        - Normalization: pixel_normalized = pixel / 255.0
        - This standardizes input distribution for stable training

    Step 4: OPTIONAL DENOISING
        - Applies subtle Gaussian blur to reduce noise
        - Medical images often have noise from acquisition equipment
        - Radius 0.5 removes noise without losing diagnostic edges
        - Can be disabled for images that are already clean

    Args:
        image_bytes: Raw image data as bytes
        apply_denoising: Whether to apply Gaussian blur denoising (default: True)

    Returns:
        TransformationResult containing:
        - processed_array: numpy array of shape (512, 512) with float32 values in [0, 1]
        - preview_image: PIL Image for human viewing (scaled back to 0-255)

    Example:
        >>> result = transform_image(image_bytes, apply_denoising=True)
        >>> if result.success:
        ...     print(f"Array shape: {result.processed_array.shape}")
        ...     print(f"Value range: [{result.processed_array.min():.3f}, {result.processed_array.max():.3f}]")
    """
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # ---------------------------------------------------------------------
        # STEP 1: RESIZE TO TARGET DIMENSIONS
        # ---------------------------------------------------------------------
        # LANCZOS (also known as Lanczos3) is a high-quality resampling filter
        # It uses a sinc function-based kernel that:
        # - Minimizes aliasing artifacts during downscaling
        # - Preserves sharp edges (crucial for detecting lung nodules, fractures)
        # - Produces smoother gradients than nearest-neighbor or bilinear
        resized_image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # ---------------------------------------------------------------------
        # STEP 2: CONVERT TO GRAYSCALE
        # ---------------------------------------------------------------------
        # 'L' mode in PIL = 8-bit grayscale (0-255)
        # The conversion formula (if RGB): L = 0.299*R + 0.587*G + 0.114*B
        # This weighted formula matches human luminance perception
        grayscale_image = resized_image.convert('L')

        # ---------------------------------------------------------------------
        # STEP 3: OPTIONAL DENOISING
        # ---------------------------------------------------------------------
        # Apply subtle Gaussian blur to reduce high-frequency noise
        # This is especially helpful for:
        # - Images with sensor noise from X-ray detectors
        # - Scanned film images with grain
        # - Low-dose CT reconstructions
        if apply_denoising and DENOISE_RADIUS > 0:
            grayscale_image = grayscale_image.filter(
                ImageFilter.GaussianBlur(radius=DENOISE_RADIUS)
            )

        # ---------------------------------------------------------------------
        # STEP 4: NORMALIZE PIXEL VALUES
        # ---------------------------------------------------------------------
        # Convert PIL Image to numpy array
        # dtype=float32 is standard for neural network inputs
        # Using float32 (not float64) saves memory with negligible precision loss
        pixel_array = np.array(grayscale_image, dtype=np.float32)

        # Normalize from [0, 255] to [0.0, 1.0]
        # This is a critical preprocessing step because:
        # - Neural network weights are typically initialized for small input values
        # - Unnormalized inputs cause gradient explosion during training
        # - Standard normalization ensures consistent model behavior
        normalized_array = pixel_array / 255.0

        # Verify normalization succeeded
        assert normalized_array.min() >= 0.0, "Negative values after normalization"
        assert normalized_array.max() <= 1.0, "Values > 1.0 after normalization"

        # Create preview image (rescaled back to 0-255 for human viewing)
        # This allows visual inspection of processing quality
        preview_array = (normalized_array * 255).astype(np.uint8)
        preview_image = Image.fromarray(preview_array, mode='L')

        return TransformationResult(
            success=True,
            processed_array=normalized_array,
            preview_image=preview_image
        )

    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        return TransformationResult(
            success=False,
            error_message=str(e)
        )


def upload_to_silver(
    processed_array: np.ndarray,
    preview_image: Image.Image,
    original_gcs_path: str
) -> bool:
    """
    Upload processed image to the Silver layer in GCS.

    Uploads two files for each processed image:

    1. NUMPY ARRAY (.npy file)
       - Contains the normalized pixel values as float32
       - Ready for direct loading into ML frameworks (PyTorch, TensorFlow)
       - Format: (512, 512) array with values in [0, 1]
       - Why .npy? Fast loading, preserves exact float values, standard format

    2. PREVIEW IMAGE (.png file)
       - Human-viewable version of the processed image
       - Useful for quality assurance and debugging
       - PNG format preserves exact pixels (no JPEG compression artifacts)

    Silver Layer Organization:
        gs://{bucket}/silver/images/
            ├── case_001_processed.npy    (ML-ready array)
            ├── case_001_preview.png      (human preview)
            └── ...

    Args:
        processed_array: Normalized numpy array (512x512, float32, [0,1])
        preview_image: PIL Image for human viewing
        original_gcs_path: Original Bronze path (used to derive Silver path)

    Returns:
        bool: True if upload succeeded, False otherwise

    Example:
        >>> success = upload_to_silver(array, preview, "bronze/images/case_001.jpeg")
        >>> # Creates: silver/images/case_001_processed.npy
        >>> #          silver/images/case_001_preview.png
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        silver_bucket = client.bucket(bucket_names["silver"])

        # Derive Silver paths from Bronze path
        # Extract filename without extension
        # "bronze/images/case_001.jpeg" -> "case_001"
        original_filename = Path(original_gcs_path).stem

        # Define Silver layer paths
        npy_path = f"silver/images/{original_filename}_processed.npy"
        preview_path = f"silver/images/{original_filename}_preview.png"

        # ---------------------------------------------------------------------
        # UPLOAD 1: NUMPY ARRAY
        # ---------------------------------------------------------------------
        # Save numpy array to bytes buffer
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, processed_array)
        npy_buffer.seek(0)  # Reset buffer position for reading

        # Upload to GCS
        npy_blob = silver_bucket.blob(npy_path)
        npy_blob.upload_from_file(
            npy_buffer,
            content_type='application/octet-stream'
        )

        # ---------------------------------------------------------------------
        # UPLOAD 2: PREVIEW PNG
        # ---------------------------------------------------------------------
        # Save PNG to bytes buffer
        png_buffer = io.BytesIO()
        preview_image.save(png_buffer, format='PNG')
        png_buffer.seek(0)

        # Upload to GCS
        preview_blob = silver_bucket.blob(preview_path)
        preview_blob.upload_from_file(
            png_buffer,
            content_type='image/png'
        )

        logger.debug(f"Uploaded to Silver: {npy_path}, {preview_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to Silver: {str(e)}")
        return False


def list_bronze_images() -> List[str]:
    """
    List all images in the Bronze layer.

    Scans the bronze/images/ prefix in the raw data bucket and returns
    paths to all image files with supported extensions.

    Returns:
        List of GCS paths (relative to bucket) for all Bronze images

    Example:
        >>> images = list_bronze_images()
        >>> print(images[:3])
        ['bronze/images/case_001.jpeg', 'bronze/images/case_002.jpeg', ...]
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        bucket = client.bucket(bucket_names["bronze"])

        # List all blobs under bronze/images/
        blobs = bucket.list_blobs(prefix="bronze/images/")

        # Filter to supported image formats
        image_paths = []
        for blob in blobs:
            # Check file extension
            ext = Path(blob.name).suffix.lower()
            if ext in SUPPORTED_FORMATS:
                image_paths.append(blob.name)

        # Sort for consistent processing order
        image_paths.sort()

        logger.info(f"Found {len(image_paths)} images in Bronze layer")
        return image_paths

    except Exception as e:
        logger.error(f"Failed to list Bronze images: {str(e)}")
        return []


def process_all_images(apply_denoising: bool = True) -> ProcessingSummary:
    """
    Orchestrate processing of all images from Bronze to Silver layer.

    This is the main entry point for the image processing pipeline.
    It processes all images in the Bronze layer, applying transformations
    and uploading results to the Silver layer.

    Processing Flow:
        1. List all images in Bronze layer
        2. For each image:
           a. Download from Bronze
           b. Validate image
           c. Apply transformations
           d. Upload to Silver
           e. Track statistics
        3. Generate summary report

    Error Handling Strategy:
        - Skip corrupted/invalid images (don't fail entire batch)
        - Log all failures for later investigation
        - Continue processing remaining images
        - Track failed images in summary

    Progress Reporting:
        - Reports progress every PROGRESS_INTERVAL images (default: 20)
        - Includes current count, success rate, and estimated remaining

    Args:
        apply_denoising: Whether to apply Gaussian blur denoising

    Returns:
        ProcessingSummary with statistics about the processing run

    Example:
        >>> summary = process_all_images(apply_denoising=True)
        >>> print(f"Processed: {summary.processed_successfully}/{summary.total_images}")
        >>> print(f"Failed: {summary.failed}")
    """
    summary = ProcessingSummary(start_time=datetime.now())

    # Timing and statistics tracking
    total_size_kb = 0.0
    total_processing_time_ms = 0.0

    # Get list of all images to process
    image_paths = list_bronze_images()
    summary.total_images = len(image_paths)

    if summary.total_images == 0:
        logger.warning("No images found in Bronze layer")
        summary.end_time = datetime.now()
        return summary

    logger.info(f"Starting image processing: {summary.total_images} images")
    logger.info(f"Denoising: {'enabled' if apply_denoising else 'disabled'}")
    logger.info("-" * 60)

    # Process each image
    for idx, gcs_path in enumerate(image_paths, start=1):
        start_time = time.time()

        try:
            # -----------------------------------------------------------------
            # STEP 1: Download from Bronze
            # -----------------------------------------------------------------
            image_bytes, file_size = download_from_bronze(gcs_path)

            if image_bytes is None:
                summary.failed += 1
                summary.failed_images.append(gcs_path)
                continue

            total_size_kb += file_size / 1024

            # -----------------------------------------------------------------
            # STEP 2: Validate image
            # -----------------------------------------------------------------
            validation_result = validate_image(image_bytes)

            if not validation_result.is_valid:
                logger.warning(f"Skipping invalid image {gcs_path}: {validation_result.error_message}")
                summary.failed += 1
                summary.failed_images.append(gcs_path)
                continue

            # -----------------------------------------------------------------
            # STEP 3: Apply transformations
            # -----------------------------------------------------------------
            transform_result = transform_image(image_bytes, apply_denoising)

            if not transform_result.success:
                logger.warning(f"Transformation failed for {gcs_path}: {transform_result.error_message}")
                summary.failed += 1
                summary.failed_images.append(gcs_path)
                continue

            # -----------------------------------------------------------------
            # STEP 4: Upload to Silver
            # -----------------------------------------------------------------
            upload_success = upload_to_silver(
                transform_result.processed_array,
                transform_result.preview_image,
                gcs_path
            )

            if not upload_success:
                summary.failed += 1
                summary.failed_images.append(gcs_path)
                continue

            # Success!
            summary.processed_successfully += 1

            # Track processing time
            elapsed_ms = (time.time() - start_time) * 1000
            total_processing_time_ms += elapsed_ms

        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected error processing {gcs_path}: {str(e)}")
            summary.failed += 1
            summary.failed_images.append(gcs_path)

        # ---------------------------------------------------------------------
        # PROGRESS REPORTING
        # ---------------------------------------------------------------------
        if idx % PROGRESS_INTERVAL == 0 or idx == summary.total_images:
            success_rate = (summary.processed_successfully / idx) * 100
            logger.info(
                f"Progress: {idx}/{summary.total_images} images | "
                f"Success: {summary.processed_successfully} | "
                f"Failed: {summary.failed} | "
                f"Rate: {success_rate:.1f}%"
            )

    # -------------------------------------------------------------------------
    # FINALIZE SUMMARY
    # -------------------------------------------------------------------------
    summary.end_time = datetime.now()

    # Calculate averages (avoid division by zero)
    if summary.processed_successfully > 0:
        summary.average_original_size_kb = total_size_kb / summary.total_images
        summary.average_processing_time_ms = total_processing_time_ms / summary.processed_successfully

    # Print final summary
    logger.info("=" * 60)
    logger.info("IMAGE PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images:        {summary.total_images}")
    logger.info(f"Successfully processed: {summary.processed_successfully}")
    logger.info(f"Failed:              {summary.failed}")
    logger.info(f"Average size:        {summary.average_original_size_kb:.1f} KB")
    logger.info(f"Average time/image:  {summary.average_processing_time_ms:.1f} ms")

    if summary.failed_images:
        logger.info("\nFailed images:")
        for path in summary.failed_images[:10]:  # Show first 10
            logger.info(f"  - {path}")
        if len(summary.failed_images) > 10:
            logger.info(f"  ... and {len(summary.failed_images) - 10} more")

    duration = summary.end_time - summary.start_time
    logger.info(f"\nTotal duration: {duration.total_seconds():.1f} seconds")
    logger.info("=" * 60)

    return summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run image processing pipeline from command line.

    Usage:
        python process_images.py
        python process_images.py --no-denoise
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process medical images from Bronze to Silver layer"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising filter"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run processing
    summary = process_all_images(apply_denoising=not args.no_denoise)

    # Exit with error code if any failures
    if summary.failed > 0:
        sys.exit(1)
