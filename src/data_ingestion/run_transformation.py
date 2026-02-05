"""
Bronze → Silver Transformation Pipeline Orchestrator

This is the main entry point for running the complete data transformation
pipeline that processes raw medical data from Bronze to Silver layer.

Pipeline Components:
    1. Image Processing: Raw images → Normalized arrays + previews
    2. Report Processing: Raw text → Structured JSON
    3. Vitals Processing: Raw CSV → Validated/enriched data + BigQuery

Execution Modes:
    - Sequential: Process images → reports → vitals (safer, easier debugging)
    - Parallel: Process all three simultaneously (faster, uses threading)

Output:
    - Processed data in Silver layer (GCS)
    - Vitals also loaded to BigQuery
    - Comprehensive transformation summary report

Usage:
    python run_transformation.py              # Run all pipelines sequentially
    python run_transformation.py --parallel   # Run pipelines in parallel
    python run_transformation.py --images     # Run only image processing
    python run_transformation.py --reports    # Run only report processing
    python run_transformation.py --vitals     # Run only vitals processing
    python run_transformation.py --verbose    # Enable debug logging

Author: Medical RAG Pipeline
Date: 2025
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from google.cloud import storage

# Import processing modules
from src.data_ingestion.process_images import process_all_images, ProcessingSummary as ImageSummary
from src.data_ingestion.process_reports import process_all_reports, ProcessingSummary as ReportSummary
from src.data_ingestion.process_vitals import process_vitals, ProcessingSummary as VitalsSummary

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("orchestrator")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransformationReport:
    """
    Comprehensive report of the entire transformation pipeline run.

    Attributes:
        pipeline_start_time: When the pipeline started
        pipeline_end_time: When the pipeline completed
        total_duration_seconds: Total runtime
        images_summary: Summary from image processing
        reports_summary: Summary from report processing
        vitals_summary: Summary from vitals processing
        overall_success: True if all components succeeded
        errors: List of any errors encountered
        bucket_structure: Final structure of Silver layer
    """
    pipeline_start_time: Optional[str] = None
    pipeline_end_time: Optional[str] = None
    total_duration_seconds: float = 0.0
    images_summary: Optional[Dict[str, Any]] = None
    reports_summary: Optional[Dict[str, Any]] = None
    vitals_summary: Optional[Dict[str, Any]] = None
    overall_success: bool = False
    errors: List[str] = field(default_factory=list)
    bucket_structure: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def list_bucket_structure(bucket_name: str, prefix: str = "") -> List[str]:
    """
    List contents of a GCS bucket under a prefix.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix to filter objects (e.g., "silver/")

    Returns:
        List of object paths
    """
    try:
        project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix)
        paths = [blob.name for blob in blobs]

        return sorted(paths)

    except Exception as e:
        logger.warning(f"Failed to list bucket contents: {str(e)}")
        return []


def image_summary_to_dict(summary: ImageSummary) -> Dict[str, Any]:
    """Convert ImageSummary dataclass to dictionary."""
    return {
        "total_images": summary.total_images,
        "processed_successfully": summary.processed_successfully,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "average_original_size_kb": round(summary.average_original_size_kb, 2),
        "average_processing_time_ms": round(summary.average_processing_time_ms, 2),
        "failed_images": summary.failed_images[:10],  # First 10 only
        "start_time": summary.start_time.isoformat() if summary.start_time else None,
        "end_time": summary.end_time.isoformat() if summary.end_time else None,
    }


def report_summary_to_dict(summary: ReportSummary) -> Dict[str, Any]:
    """Convert ReportSummary dataclass to dictionary."""
    return {
        "total_reports": summary.total_reports,
        "processed_successfully": summary.processed_successfully,
        "failed": summary.failed,
        "average_confidence": round(summary.average_confidence, 3),
        "field_extraction_stats": summary.field_extraction_stats,
        "failed_reports": summary.failed_reports[:10],
        "start_time": summary.start_time.isoformat() if summary.start_time else None,
        "end_time": summary.end_time.isoformat() if summary.end_time else None,
    }


def vitals_summary_to_dict(summary: VitalsSummary) -> Dict[str, Any]:
    """Convert VitalsSummary dataclass to dictionary."""
    quality_dict = None
    if summary.data_quality:
        quality_dict = {
            "total_rows": summary.data_quality.total_rows,
            "outliers_by_column": summary.data_quality.outliers_by_column,
            "missing_by_column": summary.data_quality.missing_by_column,
            "rows_with_outliers": summary.data_quality.rows_with_outliers,
            "rows_with_missing": summary.data_quality.rows_with_missing,
            "imputed_values": summary.data_quality.imputed_values,
        }

    return {
        "total_records": summary.total_records,
        "records_processed": summary.records_processed,
        "data_quality": quality_dict,
        "derived_features_added": summary.derived_features_added,
        "bigquery_rows_loaded": summary.bigquery_rows_loaded,
        "silver_csv_uploaded": summary.silver_csv_uploaded,
        "start_time": summary.start_time.isoformat() if summary.start_time else None,
        "end_time": summary.end_time.isoformat() if summary.end_time else None,
    }


# =============================================================================
# PIPELINE EXECUTION FUNCTIONS
# =============================================================================

def run_image_processing() -> ImageSummary:
    """
    Run the image processing pipeline.

    Returns:
        ImageSummary with processing statistics
    """
    logger.info("=" * 70)
    logger.info("STARTING IMAGE PROCESSING PIPELINE")
    logger.info("=" * 70)

    try:
        summary = process_all_images(apply_denoising=True)
        return summary
    except Exception as e:
        logger.error(f"Image processing failed with exception: {str(e)}")
        # Return empty summary on failure
        return ImageSummary()


def run_report_processing() -> ReportSummary:
    """
    Run the report processing pipeline.

    Returns:
        ReportSummary with processing statistics
    """
    logger.info("=" * 70)
    logger.info("STARTING REPORT PROCESSING PIPELINE")
    logger.info("=" * 70)

    try:
        summary = process_all_reports()
        return summary
    except Exception as e:
        logger.error(f"Report processing failed with exception: {str(e)}")
        return ReportSummary()


def run_vitals_processing() -> VitalsSummary:
    """
    Run the vitals processing pipeline.

    Returns:
        VitalsSummary with processing statistics
    """
    logger.info("=" * 70)
    logger.info("STARTING VITALS PROCESSING PIPELINE")
    logger.info("=" * 70)

    try:
        summary = process_vitals()
        return summary
    except Exception as e:
        logger.error(f"Vitals processing failed with exception: {str(e)}")
        return VitalsSummary()


def run_sequential(
    process_images: bool = True,
    process_reports: bool = True,
    process_vitals: bool = True
) -> TransformationReport:
    """
    Run processing pipelines sequentially.

    This is the safer option as errors in one pipeline won't affect others,
    and debugging is easier with sequential execution.

    Args:
        process_images: Whether to process images
        process_reports: Whether to process reports
        process_vitals: Whether to process vitals

    Returns:
        TransformationReport with all summaries
    """
    report = TransformationReport()
    report.pipeline_start_time = datetime.now().isoformat()

    start_time = datetime.now()

    # Process images
    if process_images:
        try:
            image_summary = run_image_processing()
            report.images_summary = image_summary_to_dict(image_summary)
            if image_summary.failed > 0:
                report.errors.append(f"Image processing: {image_summary.failed} failures")
        except Exception as e:
            report.errors.append(f"Image processing error: {str(e)}")
            logger.error(f"Image processing error: {str(e)}")

    # Process reports
    if process_reports:
        try:
            report_summary = run_report_processing()
            report.reports_summary = report_summary_to_dict(report_summary)
            if report_summary.failed > 0:
                report.errors.append(f"Report processing: {report_summary.failed} failures")
        except Exception as e:
            report.errors.append(f"Report processing error: {str(e)}")
            logger.error(f"Report processing error: {str(e)}")

    # Process vitals
    if process_vitals:
        try:
            vitals_summary = run_vitals_processing()
            report.vitals_summary = vitals_summary_to_dict(vitals_summary)
            if vitals_summary.bigquery_rows_loaded == 0 and not vitals_summary.silver_csv_uploaded:
                report.errors.append("Vitals processing: Failed to save output")
        except Exception as e:
            report.errors.append(f"Vitals processing error: {str(e)}")
            logger.error(f"Vitals processing error: {str(e)}")

    # Calculate duration
    end_time = datetime.now()
    report.pipeline_end_time = end_time.isoformat()
    report.total_duration_seconds = (end_time - start_time).total_seconds()

    # Determine overall success
    report.overall_success = len(report.errors) == 0

    return report


def run_parallel(
    process_images: bool = True,
    process_reports: bool = True,
    process_vitals: bool = True
) -> TransformationReport:
    """
    Run processing pipelines in parallel using threading.

    This is faster but may be harder to debug if multiple pipelines fail.

    Args:
        process_images: Whether to process images
        process_reports: Whether to process reports
        process_vitals: Whether to process vitals

    Returns:
        TransformationReport with all summaries
    """
    report = TransformationReport()
    report.pipeline_start_time = datetime.now().isoformat()

    start_time = datetime.now()

    # Build list of tasks to run
    tasks = {}
    if process_images:
        tasks["images"] = run_image_processing
    if process_reports:
        tasks["reports"] = run_report_processing
    if process_vitals:
        tasks["vitals"] = run_vitals_processing

    if not tasks:
        logger.warning("No pipelines selected to run")
        report.pipeline_end_time = datetime.now().isoformat()
        return report

    logger.info(f"Running {len(tasks)} pipelines in parallel: {list(tasks.keys())}")

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(func): name
            for name, func in tasks.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()

                if name == "images":
                    report.images_summary = image_summary_to_dict(result)
                    if result.failed > 0:
                        report.errors.append(f"Image processing: {result.failed} failures")

                elif name == "reports":
                    report.reports_summary = report_summary_to_dict(result)
                    if result.failed > 0:
                        report.errors.append(f"Report processing: {result.failed} failures")

                elif name == "vitals":
                    report.vitals_summary = vitals_summary_to_dict(result)
                    if result.bigquery_rows_loaded == 0 and not result.silver_csv_uploaded:
                        report.errors.append("Vitals processing: Failed to save output")

            except Exception as e:
                report.errors.append(f"{name} processing error: {str(e)}")
                logger.error(f"{name} processing error: {str(e)}")

    # Calculate duration
    end_time = datetime.now()
    report.pipeline_end_time = end_time.isoformat()
    report.total_duration_seconds = (end_time - start_time).total_seconds()

    # Determine overall success
    report.overall_success = len(report.errors) == 0

    return report


def get_silver_bucket_structure(report: TransformationReport) -> None:
    """
    Fetch and populate the Silver bucket structure in the report.

    Args:
        report: TransformationReport to update with bucket structure
    """
    bucket_names = get_bucket_names()

    # Get Silver layer contents
    silver_contents = list_bucket_structure(bucket_names["silver"], prefix="silver/")

    # Organize by directory
    structure = {
        "silver/images/": [],
        "silver/reports/": [],
        "silver/structured/": [],
    }

    for path in silver_contents:
        for prefix in structure.keys():
            if path.startswith(prefix):
                # Just show filename, not full path
                filename = path.replace(prefix, "")
                if filename:  # Skip empty (directory markers)
                    structure[prefix].append(filename)
                break

    report.bucket_structure = structure


def print_final_report(report: TransformationReport) -> None:
    """
    Print a comprehensive summary of the transformation pipeline.

    Args:
        report: TransformationReport to display
    """
    print("\n")
    print("=" * 80)
    print("              TRANSFORMATION PIPELINE - FINAL REPORT")
    print("=" * 80)

    # Pipeline timing
    print(f"\n{'Pipeline Timing':-^60}")
    print(f"  Started:  {report.pipeline_start_time}")
    print(f"  Ended:    {report.pipeline_end_time}")
    print(f"  Duration: {report.total_duration_seconds:.1f} seconds")

    # Image processing summary
    if report.images_summary:
        print(f"\n{'Image Processing':-^60}")
        img = report.images_summary
        print(f"  Total images:     {img['total_images']}")
        print(f"  Processed:        {img['processed_successfully']}")
        print(f"  Failed:           {img['failed']}")
        if img['total_images'] > 0:
            success_rate = img['processed_successfully'] / img['total_images'] * 100
            print(f"  Success rate:     {success_rate:.1f}%")
        print(f"  Avg size:         {img['average_original_size_kb']:.1f} KB")
        print(f"  Avg time/image:   {img['average_processing_time_ms']:.1f} ms")

    # Report processing summary
    if report.reports_summary:
        print(f"\n{'Report Processing':-^60}")
        rpt = report.reports_summary
        print(f"  Total reports:    {rpt['total_reports']}")
        print(f"  Processed:        {rpt['processed_successfully']}")
        print(f"  Failed:           {rpt['failed']}")
        if rpt['total_reports'] > 0:
            success_rate = rpt['processed_successfully'] / rpt['total_reports'] * 100
            print(f"  Success rate:     {success_rate:.1f}%")
        print(f"  Avg confidence:   {rpt['average_confidence']:.1%}")
        print(f"  Field extraction rates:")
        for field, count in rpt['field_extraction_stats'].items():
            rate = count / rpt['total_reports'] * 100 if rpt['total_reports'] > 0 else 0
            print(f"    - {field:15s}: {rate:.0f}%")

    # Vitals processing summary
    if report.vitals_summary:
        print(f"\n{'Vitals Processing':-^60}")
        vit = report.vitals_summary
        print(f"  Total records:    {vit['total_records']}")
        print(f"  Processed:        {vit['records_processed']}")
        if vit['data_quality']:
            print(f"  Outliers found:   {vit['data_quality']['rows_with_outliers']}")
            print(f"  Values imputed:   {vit['data_quality']['imputed_values']}")
        print(f"  Features added:   {len(vit['derived_features_added'])}")
        print(f"  BigQuery rows:    {vit['bigquery_rows_loaded']}")
        print(f"  Silver CSV:       {'[OK]' if vit['silver_csv_uploaded'] else '[FAIL]'}")

    # Bucket structure
    if report.bucket_structure:
        print(f"\n{'Silver Layer Structure':-^60}")
        for prefix, files in report.bucket_structure.items():
            print(f"\n  {prefix}")
            if files:
                # Show first 5 files
                for f in files[:5]:
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    - ... and {len(files) - 5} more files")
            else:
                print(f"    - (empty)")

    # Errors
    if report.errors:
        print(f"\n{'Errors Encountered':-^60}")
        for error in report.errors:
            print(f"  [ERROR] {error}")

    # Overall status
    print(f"\n{'Overall Status':-^60}")
    if report.overall_success:
        print("  [SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("  [FAILED] PIPELINE COMPLETED WITH ERRORS")

    print("\n" + "=" * 80)


def save_report_to_file(report: TransformationReport) -> str:
    """
    Save the transformation report to a JSON file.

    Args:
        report: TransformationReport to save

    Returns:
        Path to the saved report file
    """
    # Create reports directory if needed
    reports_dir = project_root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"transformation_report_{timestamp}.json"

    # Save as JSON
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info(f"Report saved to: {report_path}")
    return str(report_path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the transformation pipeline.

    Parses command-line arguments and runs the appropriate pipelines.
    """
    parser = argparse.ArgumentParser(
        description="Run Bronze → Silver data transformation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_transformation.py              # Run all pipelines sequentially
  python run_transformation.py --parallel   # Run all pipelines in parallel
  python run_transformation.py --images     # Run only image processing
  python run_transformation.py --reports    # Run only report processing
  python run_transformation.py --vitals     # Run only vitals processing
  python run_transformation.py --images --reports  # Run images and reports
        """
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run pipelines in parallel (faster but harder to debug)"
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Run image processing pipeline"
    )
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Run report processing pipeline"
    )
    parser.add_argument(
        "--vitals",
        action="store_true",
        help="Run vitals processing pipeline"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Don't save the report to a JSON file"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which pipelines to run
    # If no specific pipelines specified, run all
    run_images = args.images or (not args.images and not args.reports and not args.vitals)
    run_reports = args.reports or (not args.images and not args.reports and not args.vitals)
    run_vitals = args.vitals or (not args.images and not args.reports and not args.vitals)

    logger.info("=" * 80)
    logger.info("          BRONZE → SILVER TRANSFORMATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Mode:     {'Parallel' if args.parallel else 'Sequential'}")
    logger.info(f"Images:   {'Yes' if run_images else 'No'}")
    logger.info(f"Reports:  {'Yes' if run_reports else 'No'}")
    logger.info(f"Vitals:   {'Yes' if run_vitals else 'No'}")
    logger.info("=" * 80)

    # Run the pipeline
    if args.parallel:
        report = run_parallel(
            process_images=run_images,
            process_reports=run_reports,
            process_vitals=run_vitals
        )
    else:
        report = run_sequential(
            process_images=run_images,
            process_reports=run_reports,
            process_vitals=run_vitals
        )

    # Get final bucket structure
    get_silver_bucket_structure(report)

    # Print final report
    print_final_report(report)

    # Save report to file
    if not args.no_save_report:
        report_path = save_report_to_file(report)
        print(f"\nReport saved to: {report_path}")

    # Exit with appropriate code
    if report.overall_success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
