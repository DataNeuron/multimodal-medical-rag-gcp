"""
Bronze → Silver Vitals Processing Pipeline

This module transforms raw patient vitals data from the Bronze layer
into validated, enriched data in the Silver layer and BigQuery.

Data Processing Steps:
    1. Load raw vitals CSV from Bronze
    2. Validate values are within realistic medical ranges
    3. Handle missing values (imputation or flagging)
    4. Add derived clinical features (flags for abnormal values)
    5. Save to Silver layer (CSV) and BigQuery

Medical Context:
    Vital signs are physiological measurements that indicate basic body functions.
    Out-of-range values may indicate data quality issues or critical patient conditions.

Bronze Layer Structure (input):
    gs://{bucket}/bronze/structured/patient_vitals.csv

Silver Layer Structure (output):
    gs://{bucket}/silver/structured/patient_vitals_processed.csv

BigQuery Output:
    {project}.{dataset}.patient_vitals

Author: Medical RAG Pipeline
Date: 2025
"""

import os
import io
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from google.cloud import storage, bigquery
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
# MEDICAL REFERENCE RANGES
# =============================================================================
# These ranges define the bounds for "realistic" vital sign values.
# Values outside these ranges are likely data entry errors or equipment failures.
# Clinical abnormal ranges (for flags) are different and defined separately.

VITAL_RANGES = {
    # -------------------------------------------------------------------------
    # TEMPERATURE (Fahrenheit)
    # -------------------------------------------------------------------------
    # Normal body temperature: 97.8°F - 99.1°F (oral)
    # Hypothermia: < 95°F (severe) - 97°F (mild)
    # Fever: > 100.4°F
    # Hyperpyrexia: > 104°F (medical emergency)
    #
    # Validation range: 95-106°F
    # Below 95°F would indicate severe hypothermia or measurement error
    # Above 106°F is extremely rare (incompatible with life > 108°F)
    "temperature": {
        "min": 95.0,
        "max": 106.0,
        "unit": "deg F",
        "description": "Body temperature (Fahrenheit)"
    },

    # -------------------------------------------------------------------------
    # HEART RATE (beats per minute)
    # -------------------------------------------------------------------------
    # Normal resting HR: 60-100 bpm (adults)
    # Bradycardia: < 60 bpm (may be normal in athletes)
    # Tachycardia: > 100 bpm
    # Athletes may have resting HR as low as 40 bpm
    # Maximum HR = 220 - age (theoretical maximum)
    #
    # Validation range: 40-200 bpm
    # Below 40 bpm would indicate severe bradycardia or cardiac arrest
    # Above 200 bpm is near maximum and suspicious in resting patient
    "heart_rate": {
        "min": 40,
        "max": 200,
        "unit": "bpm",
        "description": "Heart rate (beats per minute)"
    },

    # -------------------------------------------------------------------------
    # SYSTOLIC BLOOD PRESSURE (mmHg)
    # -------------------------------------------------------------------------
    # Normal: 90-120 mmHg
    # Elevated: 120-129 mmHg
    # Hypertension Stage 1: 130-139 mmHg
    # Hypertension Stage 2: >= 140 mmHg
    # Hypertensive Crisis: > 180 mmHg
    # Hypotension: < 90 mmHg
    #
    # Validation range: 70-200 mmHg
    # Below 70 mmHg indicates severe shock (data error likely)
    # Above 200 mmHg is hypertensive emergency (but possible)
    "bp_systolic": {
        "min": 70,
        "max": 200,
        "unit": "mmHg",
        "description": "Systolic blood pressure"
    },

    # -------------------------------------------------------------------------
    # DIASTOLIC BLOOD PRESSURE (mmHg)
    # -------------------------------------------------------------------------
    # Normal: 60-80 mmHg
    # Elevated: 80-89 mmHg
    # Hypertension: >= 90 mmHg
    #
    # Validation range: 40-130 mmHg
    "bp_diastolic": {
        "min": 40,
        "max": 130,
        "unit": "mmHg",
        "description": "Diastolic blood pressure"
    },

    # -------------------------------------------------------------------------
    # OXYGEN SATURATION (SpO2 %)
    # -------------------------------------------------------------------------
    # Normal: 95-100%
    # Mild hypoxemia: 90-94%
    # Moderate hypoxemia: 85-89%
    # Severe hypoxemia: < 85% (medical emergency)
    #
    # Validation range: 70-100%
    # Below 70% is severe hypoxia (usually unconscious)
    # Above 100% is physically impossible (measurement error)
    "oxygen_saturation": {
        "min": 70,
        "max": 100,
        "unit": "%",
        "description": "Oxygen saturation (SpO2)"
    },

    # -------------------------------------------------------------------------
    # RESPIRATORY RATE (breaths per minute)
    # -------------------------------------------------------------------------
    # Normal: 12-20 breaths/min (adults)
    # Tachypnea: > 20 breaths/min
    # Bradypnea: < 12 breaths/min
    #
    # Validation range: 8-40 breaths/min
    "respiratory_rate": {
        "min": 8,
        "max": 40,
        "unit": "breaths/min",
        "description": "Respiratory rate"
    },

    # -------------------------------------------------------------------------
    # WHITE BLOOD CELL COUNT (x1000 cells/uL)
    # -------------------------------------------------------------------------
    # Normal: 4.5-11.0 x1000 cells/uL
    # Leukopenia: < 4.0 x1000 cells/uL (increased infection risk)
    # Leukocytosis: > 11.0 x1000 cells/uL (often indicates infection)
    # Severe leukocytosis: > 30.0 x1000 cells/uL
    #
    # Note: This data uses units of x1000 cells/uL (so 13.8 = 13,800 cells/uL)
    "wbc": {
        "min": 1.0,
        "max": 30.0,
        "unit": "x1000 cells/uL",
        "description": "White blood cell count"
    },
}


# =============================================================================
# CLINICAL THRESHOLDS FOR DERIVED FLAGS
# =============================================================================
# These thresholds define "clinically significant" abnormalities
# Used to create derived features (flags) for ML models

CLINICAL_THRESHOLDS = {
    # -------------------------------------------------------------------------
    # FEVER FLAG
    # -------------------------------------------------------------------------
    # Fever is defined as body temperature >= 100.4F (38C)
    # This is the standard medical definition (CDC, WHO)
    # Low-grade fever: 100.4F - 102.2F
    # High-grade fever: > 102.2F
    "fever_threshold": 100.4,

    # -------------------------------------------------------------------------
    # TACHYCARDIA FLAG
    # -------------------------------------------------------------------------
    # Tachycardia: Heart rate > 100 bpm at rest
    # May indicate: fever, anxiety, anemia, hyperthyroidism, cardiac issues
    # Important prognostic indicator in pneumonia and sepsis
    "tachycardia_threshold": 100,

    # -------------------------------------------------------------------------
    # ELEVATED WBC FLAG
    # -------------------------------------------------------------------------
    # Leukocytosis: WBC > 11.0 x1000 cells/uL (= 11,000 cells/uL)
    # Often indicates infection, inflammation, or stress response
    # Key marker for pneumonia severity
    # Note: Our data uses x1000 cells/uL units
    "elevated_wbc_threshold": 11.0,

    # -------------------------------------------------------------------------
    # HYPOXIA FLAG
    # -------------------------------------------------------------------------
    # Hypoxemia: SpO2 < 94%
    # Indicates inadequate oxygen delivery
    # Critical in pneumonia - indicates respiratory compromise
    "hypoxia_threshold": 94,

    # -------------------------------------------------------------------------
    # HYPERTENSION FLAG
    # -------------------------------------------------------------------------
    # Systolic BP >= 140 mmHg = Stage 2 Hypertension
    "hypertension_systolic_threshold": 140,

    # -------------------------------------------------------------------------
    # TACHYPNEA FLAG
    # -------------------------------------------------------------------------
    # Respiratory rate > 20 breaths/min (adults)
    # Indicates respiratory distress or compensation for hypoxia
    "tachypnea_threshold": 20,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataQualityReport:
    """
    Data quality statistics from validation.

    Attributes:
        total_rows: Total number of records
        outliers_by_column: Count of outliers per column
        missing_by_column: Count of missing values per column
        rows_with_outliers: Number of rows with at least one outlier
        rows_with_missing: Number of rows with at least one missing value
        imputed_values: Count of values that were imputed
    """
    total_rows: int = 0
    outliers_by_column: Dict[str, int] = field(default_factory=dict)
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    rows_with_outliers: int = 0
    rows_with_missing: int = 0
    imputed_values: int = 0


@dataclass
class ProcessingSummary:
    """
    Summary of vitals processing run.
    """
    total_records: int = 0
    records_processed: int = 0
    data_quality: Optional[DataQualityReport] = None
    derived_features_added: List[str] = field(default_factory=list)
    bigquery_rows_loaded: int = 0
    silver_csv_uploaded: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# =============================================================================
# GCS AND BIGQUERY UTILITY FUNCTIONS
# =============================================================================

def get_storage_client() -> storage.Client:
    """Get authenticated GCS client."""
    project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
    return storage.Client(project=project_id)


def get_bigquery_client() -> bigquery.Client:
    """Get authenticated BigQuery client."""
    project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
    return bigquery.Client(project=project_id)


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


def get_bigquery_dataset() -> str:
    """Get BigQuery dataset name."""
    project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
    environment = os.getenv("ENVIRONMENT", "dev")
    # Dataset name: project_id with hyphens replaced by underscores
    dataset_name = f"{project_id.replace('-', '_')}_metadata_{environment}"
    return dataset_name


# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def load_vitals_csv() -> Optional[pd.DataFrame]:
    """
    Load patient vitals CSV from Bronze layer.

    Expects a CSV file at: bronze/structured/patient_vitals.csv

    Expected Columns (from generate_synthetic_data.py):
        - patient_id: Unique identifier for each case
        - age: Patient age
        - sex: Patient sex (M/F)
        - temperature: Body temperature in Fahrenheit
        - heart_rate: Heart rate in beats per minute
        - blood_pressure: BP in "systolic/diastolic" format (e.g., "129/64")
        - oxygen_saturation: SpO2 percentage
        - respiratory_rate: Breaths per minute
        - wbc: White blood cell count (x1000 cells/uL)
        - crp: C-reactive protein level
        - diagnosis: Diagnosis label (pneumonia/normal)

    Returns:
        DataFrame with vitals data, or None if load fails

    Example:
        >>> df = load_vitals_csv()
        >>> if df is not None:
        ...     print(f"Loaded {len(df)} records")
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        bucket = client.bucket(bucket_names["bronze"])

        gcs_path = "bronze/structured/patient_vitals.csv"
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            logger.error(f"Vitals CSV not found at: {gcs_path}")
            return None

        # Download CSV content
        csv_content = blob.download_as_text()

        # Parse CSV with pandas
        df = pd.read_csv(io.StringIO(csv_content))

        logger.info(f"Loaded vitals data: {len(df)} records, {len(df.columns)} columns")
        logger.debug(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"Failed to load vitals CSV: {str(e)}")
        return None


def preprocess_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw vitals data before validation.

    Preprocessing Steps:
    1. Parse blood_pressure column ("129/64") into separate systolic/diastolic columns
    2. Ensure numeric columns are the correct data type

    Args:
        df: Raw DataFrame from CSV

    Returns:
        DataFrame with preprocessed columns ready for validation
    """
    processed_df = df.copy()

    # -------------------------------------------------------------------------
    # PARSE BLOOD PRESSURE COLUMN
    # -------------------------------------------------------------------------
    # Blood pressure is stored as "systolic/diastolic" (e.g., "129/64")
    # We need to split this into two separate numeric columns
    if "blood_pressure" in processed_df.columns:
        def parse_bp(bp_str):
            """Parse blood pressure string into (systolic, diastolic) tuple."""
            try:
                if pd.isna(bp_str):
                    return pd.NA, pd.NA
                parts = str(bp_str).split("/")
                if len(parts) == 2:
                    return float(parts[0]), float(parts[1])
                return pd.NA, pd.NA
            except (ValueError, AttributeError):
                return pd.NA, pd.NA

        # Apply parsing and create new columns
        bp_parsed = processed_df["blood_pressure"].apply(parse_bp)
        processed_df["bp_systolic"] = bp_parsed.apply(lambda x: x[0])
        processed_df["bp_diastolic"] = bp_parsed.apply(lambda x: x[1])

        logger.info("Parsed blood_pressure into bp_systolic and bp_diastolic columns")

    # -------------------------------------------------------------------------
    # ENSURE NUMERIC TYPES
    # -------------------------------------------------------------------------
    numeric_columns = ["temperature", "heart_rate", "respiratory_rate",
                       "oxygen_saturation", "wbc", "bp_systolic", "bp_diastolic"]

    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    return processed_df


def validate_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Validate that vital sign values are within realistic medical ranges.

    Validation Strategy:
    1. For each vital sign column, check if values are within VITAL_RANGES
    2. Mark outliers (values outside range) as NaN
    3. Track outlier counts for data quality reporting

    Why Mark Outliers as NaN?
    - Values outside physiological ranges are likely data entry errors
    - Equipment malfunctions can produce impossible readings
    - Better to treat as missing than use incorrect values
    - Downstream imputation can handle the missing values

    Args:
        df: DataFrame with raw vitals data

    Returns:
        Tuple of (validated DataFrame, DataQualityReport)

    Example:
        >>> validated_df, quality = validate_ranges(raw_df)
        >>> print(f"Found {quality.rows_with_outliers} rows with outliers")
    """
    quality_report = DataQualityReport(total_rows=len(df))

    # Track rows with any outliers
    rows_with_outliers = pd.Series([False] * len(df))

    # Make a copy to avoid modifying original
    validated_df = df.copy()

    # -------------------------------------------------------------------------
    # VALIDATE EACH VITAL SIGN COLUMN
    # -------------------------------------------------------------------------
    for column, range_info in VITAL_RANGES.items():
        if column not in validated_df.columns:
            logger.debug(f"Column '{column}' not found in data, skipping validation")
            continue

        min_val = range_info["min"]
        max_val = range_info["max"]
        unit = range_info["unit"]

        # Find outliers (values outside the valid range)
        # Using | for OR because we're working with pandas Series
        outlier_mask = (validated_df[column] < min_val) | (validated_df[column] > max_val)

        # Count outliers
        outlier_count = outlier_mask.sum()
        quality_report.outliers_by_column[column] = outlier_count

        if outlier_count > 0:
            # Log examples of outliers (first 3)
            outlier_values = validated_df.loc[outlier_mask, column].head(3).tolist()
            logger.warning(
                f"Column '{column}': {outlier_count} outliers "
                f"(valid range: {min_val}-{max_val} {unit}). "
                f"Examples: {outlier_values}"
            )

            # Mark outliers as NaN for later imputation
            validated_df.loc[outlier_mask, column] = np.nan

            # Track which rows have outliers
            rows_with_outliers = rows_with_outliers | outlier_mask

    quality_report.rows_with_outliers = rows_with_outliers.sum()

    # -------------------------------------------------------------------------
    # COUNT MISSING VALUES (including newly marked outliers)
    # -------------------------------------------------------------------------
    for column in VITAL_RANGES.keys():
        if column in validated_df.columns:
            missing_count = validated_df[column].isna().sum()
            quality_report.missing_by_column[column] = missing_count

    # Count rows with any missing values
    quality_report.rows_with_missing = validated_df[list(VITAL_RANGES.keys())].isna().any(axis=1).sum()

    return validated_df, quality_report


def handle_missing_values(df: pd.DataFrame, quality_report: DataQualityReport) -> pd.DataFrame:
    """
    Handle missing values in vital signs data.

    Imputation Strategy:
    For vital signs, we use MEDIAN imputation because:
    1. Median is robust to outliers (which we just removed)
    2. Vital signs have roughly normal distributions
    3. Using mean could be skewed by extreme (but valid) values
    4. Median represents a "typical" patient better

    Alternative strategies considered:
    - Mean: Sensitive to outliers, not used
    - Mode: Not appropriate for continuous values
    - Forward/Backward fill: Not appropriate (no time series order)
    - Model-based imputation: Overkill for this use case

    We also add flags to indicate which values were imputed,
    allowing downstream analysis to account for data quality.

    Args:
        df: DataFrame with validated (but possibly missing) vitals
        quality_report: DataQualityReport to update with imputation stats

    Returns:
        DataFrame with missing values imputed

    Example:
        >>> imputed_df = handle_missing_values(validated_df, quality_report)
        >>> # Now imputed_df has no NaN values in vital sign columns
    """
    imputed_df = df.copy()
    total_imputed = 0

    # -------------------------------------------------------------------------
    # IMPUTE EACH VITAL SIGN COLUMN
    # -------------------------------------------------------------------------
    for column in VITAL_RANGES.keys():
        if column not in imputed_df.columns:
            continue

        missing_mask = imputed_df[column].isna()
        missing_count = missing_mask.sum()

        if missing_count > 0:
            # Calculate median from non-missing values
            median_value = imputed_df[column].median()

            # Impute missing values with median
            imputed_df.loc[missing_mask, column] = median_value

            # Create imputation flag column
            # This allows downstream analysis to know which values are imputed
            flag_column = f"{column}_imputed"
            imputed_df[flag_column] = missing_mask.astype(int)

            logger.info(
                f"Imputed {missing_count} missing values in '{column}' "
                f"with median: {median_value:.1f}"
            )

            total_imputed += missing_count

    quality_report.imputed_values = total_imputed

    return imputed_df


def add_derived_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add derived clinical features based on medical thresholds.

    These derived features (flags) are useful for ML models because:
    1. They encode clinical knowledge into the data
    2. They create binary features that are easy to interpret
    3. They highlight clinically significant abnormalities
    4. They can improve model performance by reducing the feature space

    Derived Features Added:

    FEVER_FLAG (0/1):
        - 1 if temperature ≥ 100.4°F
        - Fever is a key indicator of infection
        - Correlates with pneumonia severity

    TACHYCARDIA_FLAG (0/1):
        - 1 if heart rate > 100 bpm
        - Indicates cardiovascular stress
        - Common in fever, anxiety, or cardiac issues

    ELEVATED_WBC_FLAG (0/1):
        - 1 if WBC > 11,000 cells/µL
        - Indicates infection or inflammation
        - Key marker for bacterial pneumonia

    HYPOXIA_FLAG (0/1):
        - 1 if oxygen saturation < 94%
        - Indicates respiratory compromise
        - Critical prognostic indicator

    HYPERTENSION_FLAG (0/1):
        - 1 if systolic BP ≥ 140 mmHg
        - May indicate underlying cardiovascular disease

    TACHYPNEA_FLAG (0/1):
        - 1 if respiratory rate > 20 breaths/min
        - Indicates respiratory distress

    ABNORMAL_VITAL_COUNT (0-6):
        - Count of how many vitals are abnormal
        - Higher count = more severe illness

    Args:
        df: DataFrame with validated vitals

    Returns:
        Tuple of (DataFrame with new features, list of added column names)

    Example:
        >>> enriched_df, new_cols = add_derived_features(validated_df)
        >>> print(f"Added features: {new_cols}")
    """
    derived_df = df.copy()
    added_features = []

    # -------------------------------------------------------------------------
    # FEVER FLAG
    # -------------------------------------------------------------------------
    # Fever defined as temperature >= 100.4F (standard medical definition)
    # This is the threshold used by CDC and WHO for fever surveillance
    if "temperature" in derived_df.columns:
        threshold = CLINICAL_THRESHOLDS["fever_threshold"]
        derived_df["fever_flag"] = (derived_df["temperature"] >= threshold).astype(int)
        added_features.append("fever_flag")

        fever_count = derived_df["fever_flag"].sum()
        logger.info(f"Fever flag: {fever_count}/{len(derived_df)} patients ({fever_count/len(derived_df)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # TACHYCARDIA FLAG
    # -------------------------------------------------------------------------
    # Tachycardia defined as heart rate > 100 bpm at rest
    # Note: This is for adults; pediatric thresholds are higher
    if "heart_rate" in derived_df.columns:
        threshold = CLINICAL_THRESHOLDS["tachycardia_threshold"]
        derived_df["tachycardia_flag"] = (derived_df["heart_rate"] > threshold).astype(int)
        added_features.append("tachycardia_flag")

        tachy_count = derived_df["tachycardia_flag"].sum()
        logger.info(f"Tachycardia flag: {tachy_count}/{len(derived_df)} patients ({tachy_count/len(derived_df)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # ELEVATED WBC FLAG
    # -------------------------------------------------------------------------
    # Leukocytosis defined as WBC > 11.0 x1000 cells/uL (= 11,000 cells/uL)
    # Indicates infection, inflammation, or stress response
    if "wbc" in derived_df.columns:
        threshold = CLINICAL_THRESHOLDS["elevated_wbc_threshold"]
        derived_df["elevated_wbc_flag"] = (derived_df["wbc"] > threshold).astype(int)
        added_features.append("elevated_wbc_flag")

        wbc_count = derived_df["elevated_wbc_flag"].sum()
        logger.info(f"Elevated WBC flag: {wbc_count}/{len(derived_df)} patients ({wbc_count/len(derived_df)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # HYPOXIA FLAG
    # -------------------------------------------------------------------------
    # Hypoxemia defined as SpO2 < 94%
    # Critical indicator of respiratory compromise
    if "oxygen_saturation" in derived_df.columns:
        threshold = CLINICAL_THRESHOLDS["hypoxia_threshold"]
        derived_df["hypoxia_flag"] = (derived_df["oxygen_saturation"] < threshold).astype(int)
        added_features.append("hypoxia_flag")

        hypoxia_count = derived_df["hypoxia_flag"].sum()
        logger.info(f"Hypoxia flag: {hypoxia_count}/{len(derived_df)} patients ({hypoxia_count/len(derived_df)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # HYPERTENSION FLAG
    # -------------------------------------------------------------------------
    # Hypertension Stage 2: systolic BP >= 140 mmHg
    if "bp_systolic" in derived_df.columns:
        threshold = CLINICAL_THRESHOLDS["hypertension_systolic_threshold"]
        derived_df["hypertension_flag"] = (derived_df["bp_systolic"] >= threshold).astype(int)
        added_features.append("hypertension_flag")

        hyper_count = derived_df["hypertension_flag"].sum()
        logger.info(f"Hypertension flag: {hyper_count}/{len(derived_df)} patients ({hyper_count/len(derived_df)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # TACHYPNEA FLAG
    # -------------------------------------------------------------------------
    # Tachypnea defined as respiratory rate > 20 breaths/min
    if "respiratory_rate" in derived_df.columns:
        threshold = CLINICAL_THRESHOLDS["tachypnea_threshold"]
        derived_df["tachypnea_flag"] = (derived_df["respiratory_rate"] > threshold).astype(int)
        added_features.append("tachypnea_flag")

        tachypnea_count = derived_df["tachypnea_flag"].sum()
        logger.info(f"Tachypnea flag: {tachypnea_count}/{len(derived_df)} patients ({tachypnea_count/len(derived_df)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # ABNORMAL VITAL COUNT
    # -------------------------------------------------------------------------
    # Count how many vital signs are abnormal
    # This is a simple severity score (more abnormal = more severe)
    flag_columns = [col for col in added_features if col.endswith("_flag")]
    if flag_columns:
        derived_df["abnormal_vital_count"] = derived_df[flag_columns].sum(axis=1)
        added_features.append("abnormal_vital_count")

        # Log distribution
        avg_abnormal = derived_df["abnormal_vital_count"].mean()
        max_abnormal = derived_df["abnormal_vital_count"].max()
        logger.info(f"Abnormal vital count: avg={avg_abnormal:.1f}, max={max_abnormal}")

    # -------------------------------------------------------------------------
    # ADD PROCESSING METADATA
    # -------------------------------------------------------------------------
    derived_df["processing_timestamp"] = datetime.now().isoformat()
    added_features.append("processing_timestamp")

    return derived_df, added_features


def save_to_bigquery(df: pd.DataFrame) -> int:
    """
    Load processed vitals data into BigQuery table.

    BigQuery is used for:
    1. SQL-based analysis and reporting
    2. Integration with other GCP services (Looker, Data Studio)
    3. Joining with other metadata tables
    4. Fast aggregation queries

    Table Schema (auto-inferred from DataFrame):
    - All numeric columns become FLOAT64 or INT64
    - String columns become STRING
    - Timestamps become TIMESTAMP

    Args:
        df: Processed DataFrame to load

    Returns:
        Number of rows loaded, or 0 if failed

    Example:
        >>> rows_loaded = save_to_bigquery(processed_df)
        >>> print(f"Loaded {rows_loaded} rows to BigQuery")
    """
    try:
        client = get_bigquery_client()
        project_id = os.getenv("GCP_PROJECT_ID", "multimodal-medical-rag")
        dataset_name = get_bigquery_dataset()
        table_name = "patient_vitals"

        # Full table reference
        table_ref = f"{project_id}.{dataset_name}.{table_name}"

        # Configure load job
        # WRITE_TRUNCATE: Replace existing data (for idempotency)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,  # Auto-detect schema from DataFrame
        )

        # Load DataFrame to BigQuery
        job = client.load_table_from_dataframe(
            df,
            table_ref,
            job_config=job_config
        )

        # Wait for job to complete
        job.result()

        # Get row count
        table = client.get_table(table_ref)
        rows_loaded = table.num_rows

        logger.info(f"Loaded {rows_loaded} rows to BigQuery: {table_ref}")
        return rows_loaded

    except Exception as e:
        logger.error(f"Failed to load to BigQuery: {str(e)}")
        logger.info("Note: BigQuery table may not exist. Ensure Terraform has been applied.")
        return 0


def save_to_silver_csv(df: pd.DataFrame) -> bool:
    """
    Save processed vitals to Silver layer as CSV backup.

    Why CSV Backup?
    1. Provides a portable format for non-GCP systems
    2. Easy to inspect and debug
    3. Can be used if BigQuery is unavailable
    4. Useful for data versioning

    Args:
        df: Processed DataFrame to save

    Returns:
        bool: True if upload succeeded

    Example:
        >>> success = save_to_silver_csv(processed_df)
    """
    try:
        client = get_storage_client()
        bucket_names = get_bucket_names()
        silver_bucket = client.bucket(bucket_names["silver"])

        gcs_path = "silver/structured/patient_vitals_processed.csv"

        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Upload to GCS
        blob = silver_bucket.blob(gcs_path)
        blob.upload_from_string(csv_content, content_type='text/csv')

        logger.info(f"Saved processed vitals to Silver: {gcs_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save to Silver CSV: {str(e)}")
        return False


def process_vitals() -> ProcessingSummary:
    """
    Orchestrate the complete vitals processing pipeline.

    Processing Flow:
        1. Load raw CSV from Bronze
        2. Validate value ranges
        3. Handle missing values
        4. Add derived features
        5. Save to BigQuery
        6. Save CSV backup to Silver

    Returns:
        ProcessingSummary with statistics

    Example:
        >>> summary = process_vitals()
        >>> print(f"Processed {summary.records_processed} records")
        >>> print(f"Quality: {summary.data_quality.rows_with_outliers} outliers")
    """
    summary = ProcessingSummary(start_time=datetime.now())

    logger.info("=" * 60)
    logger.info("VITALS PROCESSING PIPELINE")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD FROM BRONZE
    # -------------------------------------------------------------------------
    logger.info("\n[Step 1/5] Loading vitals from Bronze layer...")
    raw_df = load_vitals_csv()

    if raw_df is None:
        logger.error("Failed to load vitals data. Aborting pipeline.")
        summary.end_time = datetime.now()
        return summary

    summary.total_records = len(raw_df)
    logger.info(f"Loaded {summary.total_records} records")

    # -------------------------------------------------------------------------
    # STEP 1.5: PREPROCESS DATA
    # -------------------------------------------------------------------------
    logger.info("\n[Step 1.5/5] Preprocessing data (parsing blood pressure)...")
    preprocessed_df = preprocess_vitals(raw_df)

    # -------------------------------------------------------------------------
    # STEP 2: VALIDATE RANGES
    # -------------------------------------------------------------------------
    logger.info("\n[Step 2/5] Validating vital sign ranges...")
    validated_df, quality_report = validate_ranges(preprocessed_df)
    summary.data_quality = quality_report

    logger.info(f"Validation complete:")
    logger.info(f"  - Rows with outliers: {quality_report.rows_with_outliers}")
    logger.info(f"  - Total outliers by column:")
    for col, count in quality_report.outliers_by_column.items():
        if count > 0:
            logger.info(f"      {col}: {count}")

    # -------------------------------------------------------------------------
    # STEP 3: HANDLE MISSING VALUES
    # -------------------------------------------------------------------------
    logger.info("\n[Step 3/5] Handling missing values (median imputation)...")
    imputed_df = handle_missing_values(validated_df, quality_report)

    logger.info(f"Imputation complete: {quality_report.imputed_values} values imputed")

    # -------------------------------------------------------------------------
    # STEP 4: ADD DERIVED FEATURES
    # -------------------------------------------------------------------------
    logger.info("\n[Step 4/5] Adding derived clinical features...")
    enriched_df, added_features = add_derived_features(imputed_df)
    summary.derived_features_added = added_features

    logger.info(f"Added {len(added_features)} derived features:")
    for feature in added_features:
        logger.info(f"  - {feature}")

    summary.records_processed = len(enriched_df)

    # -------------------------------------------------------------------------
    # STEP 5: SAVE OUTPUTS
    # -------------------------------------------------------------------------
    logger.info("\n[Step 5/5] Saving processed data...")

    # Save to BigQuery
    rows_loaded = save_to_bigquery(enriched_df)
    summary.bigquery_rows_loaded = rows_loaded

    # Save to Silver CSV
    csv_success = save_to_silver_csv(enriched_df)
    summary.silver_csv_uploaded = csv_success

    # -------------------------------------------------------------------------
    # FINALIZE SUMMARY
    # -------------------------------------------------------------------------
    summary.end_time = datetime.now()

    logger.info("\n" + "=" * 60)
    logger.info("VITALS PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records:       {summary.total_records}")
    logger.info(f"Records processed:   {summary.records_processed}")
    logger.info(f"Outliers found:      {summary.data_quality.rows_with_outliers}")
    logger.info(f"Values imputed:      {summary.data_quality.imputed_values}")
    logger.info(f"Features added:      {len(summary.derived_features_added)}")
    logger.info(f"BigQuery rows:       {summary.bigquery_rows_loaded}")
    logger.info(f"Silver CSV:          {'[OK]' if summary.silver_csv_uploaded else '[FAIL]'}")

    duration = summary.end_time - summary.start_time
    logger.info(f"\nTotal duration: {duration.total_seconds():.1f} seconds")
    logger.info("=" * 60)

    return summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run vitals processing pipeline from command line.

    Usage:
        python process_vitals.py
        python process_vitals.py --verbose
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process patient vitals from Bronze to Silver layer"
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
    summary = process_vitals()

    # Exit with error code if BigQuery or CSV failed
    if summary.bigquery_rows_loaded == 0 and not summary.silver_csv_uploaded:
        sys.exit(1)
