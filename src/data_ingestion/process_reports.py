"""
Bronze → Silver Text/Reports Processing Pipeline

This module transforms raw clinical radiology reports from the Bronze layer
into structured JSON format in the Silver layer.

Clinical Report Structure (typical radiology report):
    - Patient demographics (ID, age, sex)
    - Clinical indication/symptoms
    - Findings (detailed observations)
    - Impression/diagnosis (summary conclusion)

Processing Steps:
    1. Download raw text from Bronze
    2. Clean and normalize text (whitespace, encoding issues)
    3. Extract structured fields using regex patterns
    4. Validate required fields are present
    5. Upload structured JSON to Silver

Bronze Layer Structure (input):
    gs://{bucket}/bronze/reports/
        ├── case_001_report.txt
        ├── case_002_report.txt
        └── ...

Silver Layer Structure (output):
    gs://{bucket}/silver/reports/
        ├── case_001_structured.json
        ├── case_002_structured.json
        └── ...

Author: Medical RAG Pipeline
Date: 2025
"""

import os
import re
import io
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Progress reporting interval
PROGRESS_INTERVAL: int = 20

# Required fields for a valid structured report
# These are the minimum fields needed for downstream ML processing
REQUIRED_FIELDS: List[str] = ["patient_id", "findings"]


# =============================================================================
# REGEX PATTERNS FOR FIELD EXTRACTION
# =============================================================================
# These patterns are designed to handle variations in clinical report formatting.
# Medical reports often have inconsistent formatting, so patterns are flexible.

# Pattern explanations:
# - (?:...) = non-capturing group (for alternatives)
# - \s* = optional whitespace
# - .+? = non-greedy match (shortest possible)
# - (?=...) = positive lookahead (match position before next section)
# - re.IGNORECASE = case-insensitive matching
# - re.DOTALL = dot matches newlines

PATTERNS = {
    # PATIENT ID PATTERNS
    # Matches: "Patient ID: 12345", "MRN: ABC123", "Patient #: 001"
    # Medical systems use various ID formats (numeric, alphanumeric, with prefixes)
    "patient_id": [
        r"(?:Patient\s*(?:ID|#|Number)?|MRN|Medical\s*Record\s*(?:Number|#)?)\s*[:=]?\s*([A-Za-z0-9_-]+)",
        r"(?:Case\s*(?:ID|#)?)\s*[:=]?\s*([A-Za-z0-9_-]+)",
    ],

    # AGE PATTERNS
    # Matches: "Age: 45", "45 years old", "45yo", "45 y/o", "Age 45"
    # Age is crucial for medical context (normal ranges vary by age)
    "age": [
        r"(?:Age|AGE)\s*[:=]?\s*(\d{1,3})\s*(?:years?|yrs?|y\.?o\.?|y/o)?",
        r"(\d{1,3})\s*(?:years?\s*old|yo|y\.?o\.?|y/o)",
    ],

    # SEX/GENDER PATTERNS
    # Matches: "Sex: Male", "Gender: F", "M/65" (common format: sex/age)
    # Important for gender-specific normal ranges (e.g., heart size)
    "sex": [
        r"(?:Sex|Gender)\s*[:=]?\s*(Male|Female|M|F)\b",
        r"\b(Male|Female)\b",
        r"\b([MF])/\d{1,3}\b",  # M/45 or F/67 format
    ],

    # SYMPTOMS/INDICATION PATTERNS
    # Matches: "Clinical Indication: chest pain", "Symptoms: cough, fever"
    # The reason the imaging study was ordered
    "symptoms": [
        r"(?:Clinical\s*)?(?:Indication|History|Symptoms?|Presenting\s*Complaint|Reason\s*for\s*(?:Study|Exam))\s*[:=]?\s*(.+?)(?=\n\n|\n[A-Z]|Findings?:|Impression:|$)",
        r"(?:Chief\s*Complaint|CC)\s*[:=]?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
    ],

    # FINDINGS PATTERNS
    # Matches the main body of radiologist observations
    # This is the most detailed section describing what was seen in the image
    "findings": [
        r"(?:Findings?|Observations?|Description)\s*[:=]?\s*(.+?)(?=Impression|Conclusion|Diagnosis|Assessment|$)",
        r"(?:Radiological\s*)?(?:Findings?)\s*[:=]?\s*(.+?)(?=\n\n[A-Z]|$)",
    ],

    # IMPRESSION/DIAGNOSIS PATTERNS
    # Matches: "Impression: pneumonia", "Diagnosis: normal"
    # The radiologist's conclusion/interpretation
    "impression": [
        r"(?:Impression|Conclusion|Diagnosis|Assessment|Final\s*Diagnosis)\s*[:=]?\s*(.+?)(?=\n\n|Recommendations?:|$)",
        r"(?:Clinical\s*)?(?:Impression|Interpretation)\s*[:=]?\s*(.+)",
    ],
}


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class StructuredReport:
    """
    Structured representation of a clinical radiology report.

    This format standardizes the varied text formats of clinical reports
    into a consistent JSON structure for downstream processing.

    Attributes:
        patient_id: Unique patient identifier
        age: Patient age in years (None if not found)
        sex: Patient sex (M/F/Male/Female, None if not found)
        symptoms: Clinical indication or presenting symptoms
        findings: Detailed radiological findings
        impression: Radiologist's conclusion/diagnosis
        raw_text: Original unprocessed text (for reference)
        extraction_confidence: How many fields were successfully extracted
        processing_timestamp: When this report was processed
    """
    patient_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: Optional[str] = None
    findings: Optional[str] = None
    impression: Optional[str] = None
    raw_text: Optional[str] = None
    extraction_confidence: float = 0.0
    processing_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ValidationResult:
    """
    Result of report validation.

    Attributes:
        is_valid: Whether the report passed validation
        missing_fields: List of required fields that are missing
        warnings: Non-critical issues found
    """
    is_valid: bool
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingSummary:
    """
    Summary statistics for report processing run.

    Attributes:
        total_reports: Total number of reports found
        processed_successfully: Number successfully processed
        failed: Number that failed processing
        average_confidence: Average field extraction confidence
        field_extraction_stats: How often each field was extracted
        failed_reports: List of report paths that failed
    """
    total_reports: int = 0
    processed_successfully: int = 0
    failed: int = 0
    average_confidence: float = 0.0
    field_extraction_stats: Dict[str, int] = field(default_factory=dict)
    failed_reports: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# =============================================================================
# GCS UTILITY FUNCTIONS
# =============================================================================

def get_storage_client() -> storage.Client:
    """Get authenticated GCS client."""
    project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
    return storage.Client(project=project_id)


def get_bucket_names() -> Dict[str, str]:
    """Get Bronze and Silver bucket names."""
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

def download_report(gcs_path: str) -> Optional[str]:
    """
    Download a clinical report from the Bronze layer.

    Retrieves raw text content from GCS. Handles various text encodings
    that may be present in clinical documents.

    Args:
        gcs_path: Path within Bronze bucket (e.g., "bronze/reports/case_001_report.txt")

    Returns:
        Report text as string, or None if download fails

    Example:
        >>> text = download_report("bronze/reports/case_001_report.txt")
        >>> if text:
        ...     print(f"Downloaded {len(text)} characters")
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        bucket = client.bucket(bucket_names["bronze"])
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            logger.warning(f"Report not found in Bronze: {gcs_path}")
            return None

        # Download as string, handling potential encoding issues
        # Clinical documents may use various encodings (UTF-8, Latin-1, etc.)
        try:
            text = blob.download_as_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to Latin-1 if UTF-8 fails
            logger.debug(f"UTF-8 decode failed for {gcs_path}, trying Latin-1")
            text = blob.download_as_text(encoding='latin-1')

        logger.debug(f"Downloaded report: {gcs_path} ({len(text)} chars)")
        return text

    except Exception as e:
        logger.error(f"Failed to download report: {gcs_path} - {str(e)}")
        return None


def clean_text(text: str) -> str:
    """
    Clean and normalize clinical report text.

    Text Cleaning Operations:
    1. NORMALIZE LINE ENDINGS
       - Convert Windows (\\r\\n) and Mac (\\r) line endings to Unix (\\n)
       - Ensures consistent parsing across different source systems

    2. REMOVE EXCESSIVE WHITESPACE
       - Collapse multiple spaces to single space
       - Remove trailing whitespace from lines
       - Standardizes spacing for regex pattern matching

    3. HANDLE SPECIAL CHARACTERS
       - Remove non-printable characters (except newlines, tabs)
       - Fix common encoding issues (e.g., smart quotes)

    4. NORMALIZE SECTION HEADERS
       - Standardize common variations (e.g., "FINDINGS:" vs "Findings")
       - Makes pattern matching more reliable

    Args:
        text: Raw report text

    Returns:
        Cleaned and normalized text

    Example:
        >>> raw = "FINDINGS:  \\r\\n  Multiple   spaces   here"
        >>> clean = clean_text(raw)
        >>> print(clean)
        "FINDINGS:\\nMultiple spaces here"
    """
    if not text:
        return ""

    # -------------------------------------------------------------------------
    # STEP 1: NORMALIZE LINE ENDINGS
    # -------------------------------------------------------------------------
    # Windows uses \r\n, Mac classic uses \r, Unix uses \n
    # Standardize to Unix-style \n for consistent parsing
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # -------------------------------------------------------------------------
    # STEP 2: REMOVE EXCESSIVE WHITESPACE
    # -------------------------------------------------------------------------
    # Collapse multiple spaces to single space
    # This handles cases like "Patient    ID:    123"
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove trailing whitespace from each line
    # Prevents issues with patterns that anchor to end of line
    text = re.sub(r' +\n', '\n', text)

    # Collapse multiple blank lines to maximum of 2
    # Preserves paragraph structure while removing excessive spacing
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from entire text
    text = text.strip()

    # -------------------------------------------------------------------------
    # STEP 3: HANDLE SPECIAL CHARACTERS
    # -------------------------------------------------------------------------
    # Remove non-printable characters (keep newlines and basic whitespace)
    # This catches control characters that may sneak in from various sources
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)

    # Fix common encoding artifacts (smart quotes, etc.)
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2014': '-',  # Em dash
        '\u2013': '-',  # En dash
        '\u2026': '...', # Ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # -------------------------------------------------------------------------
    # STEP 4: NORMALIZE COMMON SECTION HEADERS
    # -------------------------------------------------------------------------
    # Standardize common variations for more reliable parsing
    # Clinical reports have inconsistent formatting across institutions
    header_normalizations = [
        (r'\bCLINICAL\s+INDICATION\b', 'CLINICAL INDICATION'),
        (r'\bINDICATION\b', 'INDICATION'),
        (r'\bFINDINGS?\b', 'FINDINGS'),
        (r'\bIMPRESSION\b', 'IMPRESSION'),
        (r'\bCONCLUSION\b', 'CONCLUSION'),
    ]

    for pattern, replacement in header_normalizations:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def extract_structured_fields(text: str) -> StructuredReport:
    """
    Parse clinical report text and extract structured fields.

    This function attempts to extract key information from free-text clinical
    reports using regex pattern matching. Medical reports vary significantly
    in format, so multiple patterns are tried for each field.

    Extraction Strategy:
    1. Try each pattern for each field in order
    2. Stop at first successful match
    3. Track which fields were successfully extracted
    4. Calculate overall extraction confidence

    Field Extraction Details:

    PATIENT_ID:
        - Primary identifier for linking reports to other data
        - May be MRN, case number, or custom ID format
        - Critical for data integrity

    AGE:
        - Patient age at time of study
        - Important for clinical context (normal ranges vary by age)
        - Parsed as integer for downstream use

    SEX:
        - Patient biological sex (M/F)
        - Affects normal ranges (e.g., heart size differs by sex)
        - Normalized to single character (M/F)

    SYMPTOMS:
        - Clinical indication or reason for study
        - Helps understand context of findings
        - May be brief or detailed

    FINDINGS:
        - Core radiological observations
        - Most detailed section of report
        - Critical for ML training/inference

    IMPRESSION:
        - Radiologist's conclusion
        - Summary of key findings
        - Often used for classification tasks

    Args:
        text: Cleaned clinical report text

    Returns:
        StructuredReport with extracted fields and confidence score

    Example:
        >>> report = extract_structured_fields(clean_text)
        >>> print(f"Patient: {report.patient_id}, Age: {report.age}")
        >>> print(f"Confidence: {report.extraction_confidence:.1%}")
    """
    report = StructuredReport(
        raw_text=text,
        processing_timestamp=datetime.now().isoformat()
    )

    fields_extracted = 0
    total_fields = len(PATTERNS)

    # -------------------------------------------------------------------------
    # EXTRACT EACH FIELD
    # -------------------------------------------------------------------------

    # PATIENT ID
    for pattern in PATTERNS["patient_id"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            report.patient_id = match.group(1).strip()
            fields_extracted += 1
            break

    # AGE
    for pattern in PATTERNS["age"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                age = int(match.group(1))
                # Sanity check: age should be 0-120
                if 0 <= age <= 120:
                    report.age = age
                    fields_extracted += 1
                    break
            except ValueError:
                continue

    # SEX
    for pattern in PATTERNS["sex"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sex = match.group(1).upper()
            # Normalize to single character
            if sex in ['MALE', 'M']:
                report.sex = 'M'
            elif sex in ['FEMALE', 'F']:
                report.sex = 'F'
            else:
                report.sex = sex
            fields_extracted += 1
            break

    # SYMPTOMS/INDICATION
    for pattern in PATTERNS["symptoms"]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            symptoms = match.group(1).strip()
            # Clean up extracted text
            symptoms = re.sub(r'\s+', ' ', symptoms)  # Collapse whitespace
            if len(symptoms) > 10:  # Avoid very short/empty extractions
                report.symptoms = symptoms[:500]  # Limit length
                fields_extracted += 1
                break

    # FINDINGS
    for pattern in PATTERNS["findings"]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            findings = match.group(1).strip()
            findings = re.sub(r'\s+', ' ', findings)
            if len(findings) > 20:  # Findings should have substance
                report.findings = findings[:2000]  # Limit length
                fields_extracted += 1
                break

    # IMPRESSION
    for pattern in PATTERNS["impression"]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            impression = match.group(1).strip()
            impression = re.sub(r'\s+', ' ', impression)
            if len(impression) > 5:
                report.impression = impression[:1000]  # Limit length
                fields_extracted += 1
                break

    # -------------------------------------------------------------------------
    # FALLBACK: Try to extract patient_id from filename pattern in text
    # -------------------------------------------------------------------------
    if not report.patient_id:
        # Look for case_XXX pattern that might be in the text
        case_match = re.search(r'case[_-]?(\d+)', text, re.IGNORECASE)
        if case_match:
            report.patient_id = f"CASE_{case_match.group(1)}"
            fields_extracted += 1

    # -------------------------------------------------------------------------
    # CALCULATE EXTRACTION CONFIDENCE
    # -------------------------------------------------------------------------
    # Confidence = proportion of fields successfully extracted
    # Higher confidence = more complete structured data
    report.extraction_confidence = fields_extracted / total_fields

    return report


def validate_report(structured_data: StructuredReport) -> ValidationResult:
    """
    Validate that a structured report has required fields.

    Validation ensures data quality for downstream processing:
    - ML models need certain fields for training/inference
    - Missing critical fields should be flagged for review
    - Non-critical missing fields generate warnings

    Required Fields (will fail validation if missing):
        - patient_id: Needed for data linkage
        - findings: Core content for analysis

    Optional Fields (generate warnings if missing):
        - age: Useful but not critical
        - sex: Useful but not critical
        - symptoms: Context, but can be inferred
        - impression: Summary, but findings may suffice

    Args:
        structured_data: StructuredReport to validate

    Returns:
        ValidationResult with is_valid flag and any issues found

    Example:
        >>> result = validate_report(report)
        >>> if not result.is_valid:
        ...     print(f"Missing: {result.missing_fields}")
    """
    result = ValidationResult(is_valid=True)

    # Check required fields
    for field_name in REQUIRED_FIELDS:
        field_value = getattr(structured_data, field_name, None)
        if not field_value or (isinstance(field_value, str) and len(field_value.strip()) == 0):
            result.missing_fields.append(field_name)
            result.is_valid = False

    # Check optional fields and add warnings
    optional_fields = ["age", "sex", "symptoms", "impression"]
    for field_name in optional_fields:
        field_value = getattr(structured_data, field_name, None)
        if not field_value:
            result.warnings.append(f"Optional field '{field_name}' not extracted")

    # Additional quality checks
    if structured_data.findings and len(structured_data.findings) < 50:
        result.warnings.append("Findings section seems unusually short")

    if structured_data.extraction_confidence < 0.4:
        result.warnings.append(f"Low extraction confidence: {structured_data.extraction_confidence:.1%}")

    return result


def upload_to_silver(data: StructuredReport, original_gcs_path: str) -> bool:
    """
    Upload structured report to Silver layer as JSON.

    The Silver layer contains cleaned, validated, structured data
    ready for ML processing or further analysis.

    Output Format:
        - JSON with pretty printing (human-readable)
        - Includes all extracted fields plus metadata
        - UTF-8 encoding

    Args:
        data: StructuredReport to upload
        original_gcs_path: Original Bronze path (used to derive filename)

    Returns:
        bool: True if upload succeeded

    Example:
        >>> success = upload_to_silver(report, "bronze/reports/case_001_report.txt")
        >>> # Creates: silver/reports/case_001_structured.json
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        silver_bucket = client.bucket(bucket_names["silver"])

        # Derive Silver path from Bronze path
        # "bronze/reports/case_001_report.txt" -> "case_001"
        original_stem = Path(original_gcs_path).stem
        # Remove "_report" suffix if present
        if original_stem.endswith("_report"):
            original_stem = original_stem[:-7]

        silver_path = f"silver/reports/{original_stem}_structured.json"

        # Convert to JSON
        json_data = json.dumps(data.to_dict(), indent=2, ensure_ascii=False)

        # Upload to GCS
        blob = silver_bucket.blob(silver_path)
        blob.upload_from_string(json_data, content_type='application/json')

        logger.debug(f"Uploaded structured report to: {silver_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to Silver: {str(e)}")
        return False


def list_bronze_reports() -> List[str]:
    """
    List all reports in the Bronze layer.

    Returns:
        List of GCS paths for all Bronze reports
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        bucket = client.bucket(bucket_names["bronze"])

        # List all blobs under bronze/reports/
        blobs = bucket.list_blobs(prefix="bronze/reports/")

        # Filter to text files
        report_paths = []
        for blob in blobs:
            ext = Path(blob.name).suffix.lower()
            if ext in {'.txt', '.text', '.json'}:
                report_paths.append(blob.name)

        report_paths.sort()
        logger.info(f"Found {len(report_paths)} reports in Bronze layer")
        return report_paths

    except Exception as e:
        logger.error(f"Failed to list Bronze reports: {str(e)}")
        return []


def process_all_reports() -> ProcessingSummary:
    """
    Orchestrate processing of all reports from Bronze to Silver layer.

    This is the main entry point for the report processing pipeline.

    Processing Flow:
        1. List all reports in Bronze layer
        2. For each report:
           a. Download from Bronze
           b. Clean text
           c. Extract structured fields
           d. Validate
           e. Upload to Silver
        3. Generate summary with statistics

    Error Handling:
        - Skip reports that can't be parsed
        - Log all failures
        - Continue with remaining reports

    Returns:
        ProcessingSummary with statistics

    Example:
        >>> summary = process_all_reports()
        >>> print(f"Processed: {summary.processed_successfully}/{summary.total_reports}")
    """
    summary = ProcessingSummary(start_time=datetime.now())

    # Initialize field extraction stats
    for field in ["patient_id", "age", "sex", "symptoms", "findings", "impression"]:
        summary.field_extraction_stats[field] = 0

    total_confidence = 0.0

    # Get list of reports
    report_paths = list_bronze_reports()
    summary.total_reports = len(report_paths)

    if summary.total_reports == 0:
        logger.warning("No reports found in Bronze layer")
        summary.end_time = datetime.now()
        return summary

    logger.info(f"Starting report processing: {summary.total_reports} reports")
    logger.info("-" * 60)

    # Process each report
    for idx, gcs_path in enumerate(report_paths, start=1):
        try:
            # -----------------------------------------------------------------
            # STEP 1: Download from Bronze
            # -----------------------------------------------------------------
            raw_text = download_report(gcs_path)

            if raw_text is None:
                summary.failed += 1
                summary.failed_reports.append(gcs_path)
                continue

            # -----------------------------------------------------------------
            # STEP 2: Clean text
            # -----------------------------------------------------------------
            cleaned_text = clean_text(raw_text)

            if len(cleaned_text) < 50:
                logger.warning(f"Report too short after cleaning: {gcs_path}")
                summary.failed += 1
                summary.failed_reports.append(gcs_path)
                continue

            # -----------------------------------------------------------------
            # STEP 3: Extract structured fields
            # -----------------------------------------------------------------
            structured_report = extract_structured_fields(cleaned_text)

            # Track field extraction stats
            if structured_report.patient_id:
                summary.field_extraction_stats["patient_id"] += 1
            if structured_report.age:
                summary.field_extraction_stats["age"] += 1
            if structured_report.sex:
                summary.field_extraction_stats["sex"] += 1
            if structured_report.symptoms:
                summary.field_extraction_stats["symptoms"] += 1
            if structured_report.findings:
                summary.field_extraction_stats["findings"] += 1
            if structured_report.impression:
                summary.field_extraction_stats["impression"] += 1

            total_confidence += structured_report.extraction_confidence

            # -----------------------------------------------------------------
            # STEP 4: Validate
            # -----------------------------------------------------------------
            validation = validate_report(structured_report)

            if not validation.is_valid:
                logger.warning(
                    f"Validation failed for {gcs_path}: "
                    f"Missing fields: {validation.missing_fields}"
                )
                # Still upload with warnings - let downstream decide
                # Some use cases may tolerate missing fields

            if validation.warnings:
                for warning in validation.warnings:
                    logger.debug(f"  Warning: {warning}")

            # -----------------------------------------------------------------
            # STEP 5: Upload to Silver
            # -----------------------------------------------------------------
            success = upload_to_silver(structured_report, gcs_path)

            if success:
                summary.processed_successfully += 1
            else:
                summary.failed += 1
                summary.failed_reports.append(gcs_path)

        except Exception as e:
            logger.error(f"Unexpected error processing {gcs_path}: {str(e)}")
            summary.failed += 1
            summary.failed_reports.append(gcs_path)

        # ---------------------------------------------------------------------
        # PROGRESS REPORTING
        # ---------------------------------------------------------------------
        if idx % PROGRESS_INTERVAL == 0 or idx == summary.total_reports:
            success_rate = (summary.processed_successfully / idx) * 100
            logger.info(
                f"Progress: {idx}/{summary.total_reports} reports | "
                f"Success: {summary.processed_successfully} | "
                f"Failed: {summary.failed} | "
                f"Rate: {success_rate:.1f}%"
            )

    # -------------------------------------------------------------------------
    # FINALIZE SUMMARY
    # -------------------------------------------------------------------------
    summary.end_time = datetime.now()

    if summary.processed_successfully > 0:
        summary.average_confidence = total_confidence / summary.total_reports

    # Print final summary
    logger.info("=" * 60)
    logger.info("REPORT PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total reports:        {summary.total_reports}")
    logger.info(f"Successfully processed: {summary.processed_successfully}")
    logger.info(f"Failed:               {summary.failed}")
    logger.info(f"Average confidence:   {summary.average_confidence:.1%}")

    logger.info("\nField Extraction Rates:")
    for field, count in summary.field_extraction_stats.items():
        rate = (count / summary.total_reports * 100) if summary.total_reports > 0 else 0
        logger.info(f"  {field:15s}: {count:3d}/{summary.total_reports} ({rate:.1f}%)")

    if summary.failed_reports:
        logger.info("\nFailed reports:")
        for path in summary.failed_reports[:10]:
            logger.info(f"  - {path}")
        if len(summary.failed_reports) > 10:
            logger.info(f"  ... and {len(summary.failed_reports) - 10} more")

    duration = summary.end_time - summary.start_time
    logger.info(f"\nTotal duration: {duration.total_seconds():.1f} seconds")
    logger.info("=" * 60)

    return summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run report processing pipeline from command line.

    Usage:
        python process_reports.py
        python process_reports.py --verbose
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process clinical reports from Bronze to Silver layer"
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
    summary = process_all_reports()

    # Exit with error code if any failures
    if summary.failed > 0:
        sys.exit(1)
