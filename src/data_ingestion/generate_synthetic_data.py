#!/usr/bin/env python3
# ============================================
# Comprehensive Synthetic Medical Data Generator
# ============================================
"""
Generate synthetic multimodal medical data for testing the RAG system.

This module creates realistic but ARTIFICIAL medical data including:
    1. Synthetic chest X-ray images (512x512 grayscale)
    2. Clinical reports (structured text)
    3. Patient vitals and lab results (CSV)

IMPORTANT DISCLAIMER:
    This data is 100% SYNTHETIC and should NEVER be used for:
    - Clinical decisions
    - Patient diagnosis
    - Medical training

    It is intended ONLY for:
    - Testing the RAG pipeline
    - Validating embedding generation
    - Benchmarking retrieval accuracy

Why Synthetic Data?
    - Real medical data requires IRB approval and HIPAA compliance
    - Synthetic data allows rapid iteration without privacy concerns
    - We can control the distribution (60% pneumonia, 40% normal)
    - Ground truth labels are known for evaluation

Data Distribution:
    - 60% Pneumonia cases: Shows infiltrates/consolidation on X-ray
    - 40% Normal cases: Clear lung fields with normal appearance

    This imbalanced distribution reflects real-world scenarios where
    RAG systems often need to handle more pathological queries.

Usage:
    python src/data_ingestion/generate_synthetic_data.py

    Or import and customize:
    from src.data_ingestion.generate_synthetic_data import generate_dataset
    generate_dataset(num_cases=200, pneumonia_ratio=0.7)
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

# ============================================
# Configuration Constants
# ============================================

# Image settings
IMAGE_SIZE = 512  # 512x512 pixels (standard for medical imaging ML)
IMAGE_MODE = "L"  # Grayscale (L = luminance, single channel)

# Dataset distribution
DEFAULT_NUM_CASES = 100
PNEUMONIA_RATIO = 0.60  # 60% pneumonia, 40% normal

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directories (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = OUTPUT_DIR / "images"
REPORTS_DIR = OUTPUT_DIR / "reports"


# ============================================
# Medical Data Templates
# ============================================
"""
These templates are based on common patterns in real radiology reports.
They use standard medical terminology but are NOT real patient data.

Structure of a typical radiology report:
    1. Clinical History: Why the study was ordered
    2. Technique: How the image was acquired
    3. Comparison: Prior studies if available
    4. Findings: Detailed observations
    5. Impression: Summary diagnosis
"""

# Symptoms that might lead to a chest X-ray
PNEUMONIA_SYMPTOMS = [
    "cough and fever for 3 days",
    "productive cough with yellow sputum",
    "shortness of breath and chest pain",
    "fever, chills, and malaise",
    "worsening cough and dyspnea",
    "persistent fever despite antibiotics",
    "cough with hemoptysis",
    "pleuritic chest pain",
]

NORMAL_SYMPTOMS = [
    "routine pre-operative evaluation",
    "annual physical examination",
    "mild cough, rule out pneumonia",
    "chest pain, rule out cardiac etiology",
    "shortness of breath on exertion",
    "follow-up after resolved pneumonia",
    "screening for occupational exposure",
    "immigration medical examination",
]

# Detailed findings for pneumonia cases
# These describe radiographic appearances of lung infection
PNEUMONIA_FINDINGS = [
    "There is a focal area of consolidation in the right lower lobe, "
    "consistent with lobar pneumonia. Air bronchograms are visible. "
    "No pleural effusion identified.",

    "Patchy airspace opacities are seen in the left lower lobe with "
    "associated volume loss. Findings are suggestive of bronchopneumonia. "
    "Heart size is normal.",

    "Bilateral lower lobe infiltrates are present, more prominent on the "
    "right. There is blunting of the right costophrenic angle suggesting "
    "small pleural effusion.",

    "Dense consolidation is noted in the right middle lobe with positive "
    "silhouette sign obscuring the right heart border. No pneumothorax.",

    "Multifocal patchy opacities are seen throughout both lungs, "
    "consistent with multilobar pneumonia. Mild cardiomegaly is noted.",

    "There is a large area of consolidation in the left upper lobe with "
    "air bronchograms. The left hemidiaphragm is obscured. "
    "Consider community-acquired pneumonia.",

    "Retrocardiac opacity is present, likely representing left lower lobe "
    "consolidation. Recommend lateral view for confirmation. "
    "Small left pleural effusion noted.",

    "Extensive bilateral airspace disease is seen, worse on the right. "
    "Pattern is consistent with aspiration pneumonia given clinical history. "
    "No pneumothorax.",
]

# Detailed findings for normal cases
# These describe healthy lung appearances
NORMAL_FINDINGS = [
    "The lungs are clear bilaterally. No focal consolidation, pleural "
    "effusion, or pneumothorax. Heart size is normal. The mediastinal "
    "contours are unremarkable.",

    "Clear lungs without evidence of acute cardiopulmonary disease. "
    "Cardiac silhouette is within normal limits. No hilar adenopathy. "
    "Osseous structures are intact.",

    "No acute pulmonary abnormality. The lungs are well-expanded and clear. "
    "Costophrenic angles are sharp. Heart size is normal. "
    "No pleural effusion.",

    "Normal chest radiograph. Both lungs are clear without focal opacity. "
    "The heart is not enlarged. Pulmonary vasculature appears normal. "
    "No bony abnormalities.",

    "Unremarkable PA and lateral chest radiograph. No evidence of "
    "pneumonia, effusion, or mass. Cardiac and mediastinal contours "
    "are normal.",

    "The lungs are clear and fully expanded. No infiltrates identified. "
    "Heart size and pulmonary vascularity are normal. "
    "No acute osseous abnormality.",
]

# Impression/diagnosis statements
PNEUMONIA_IMPRESSIONS = [
    "Right lower lobe pneumonia.",
    "Left lower lobe consolidation, likely pneumonia.",
    "Bilateral pneumonia, recommend clinical correlation.",
    "Lobar pneumonia in the right middle lobe.",
    "Multilobar pneumonia with small pleural effusion.",
    "Community-acquired pneumonia, left upper lobe.",
    "Probable aspiration pneumonia.",
    "Bronchopneumonia, left lower lobe.",
]

NORMAL_IMPRESSIONS = [
    "No acute cardiopulmonary abnormality.",
    "Normal chest radiograph.",
    "No evidence of pneumonia.",
    "Unremarkable chest X-ray.",
    "Clear lungs, no acute disease.",
    "Normal study, no significant findings.",
]


# ============================================
# Image Generation Functions
# ============================================

def create_base_xray_image() -> np.ndarray:
    """
    Create a base synthetic chest X-ray image.

    Real chest X-rays show:
        - Dark (black) areas: Air-filled lungs
        - Light (white) areas: Bone, heart, soft tissue
        - Gradual transitions between structures

    We simulate this with:
        - Dark background (like lung fields)
        - Lighter central area (mediastinum/heart)
        - Rib-like horizontal patterns
        - Random noise for texture

    Returns:
        numpy.ndarray: 512x512 grayscale image array (0-255)
    """
    # Start with a dark gray background (air in lungs appears dark)
    # Value 30-50 represents healthy aerated lung tissue
    base = np.random.randint(30, 50, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Add Gaussian noise for realistic texture
    # Real X-rays have quantum noise from photon detection
    noise = np.random.normal(0, 10, (IMAGE_SIZE, IMAGE_SIZE))
    base = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Create the mediastinum (central area with heart and great vessels)
    # This appears lighter because it's denser tissue
    center_x, center_y = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]

    # Elliptical shape for heart shadow (slightly left of center, as in anatomy)
    heart_mask = ((x - center_x + 30) ** 2 / 80 ** 2 +
                  (y - center_y + 20) ** 2 / 100 ** 2) < 1
    base[heart_mask] = np.clip(base[heart_mask] + 60, 0, 255)

    # Add rib shadows (horizontal lighter bands)
    # Ribs appear as curved white lines because bone absorbs X-rays
    for i in range(6):
        rib_y = 80 + i * 70  # Spacing between ribs
        rib_thickness = 8
        if rib_y < IMAGE_SIZE - rib_thickness:
            # Curved rib shape
            for x_pos in range(50, IMAGE_SIZE - 50):
                # Parabolic curve for rib shape
                curve = int(15 * np.sin(np.pi * x_pos / IMAGE_SIZE))
                y_pos = rib_y + curve
                if 0 <= y_pos < IMAGE_SIZE - rib_thickness:
                    base[y_pos:y_pos + rib_thickness, x_pos] = np.clip(
                        base[y_pos:y_pos + rib_thickness, x_pos] + 40, 0, 255
                    )

    # Add spine shadow (vertical lighter band in center)
    spine_width = 30
    spine_start = center_x - spine_width // 2
    base[:, spine_start:spine_start + spine_width] = np.clip(
        base[:, spine_start:spine_start + spine_width] + 30, 0, 255
    )

    # Add slight vignette effect (darker at edges)
    # This mimics the natural fall-off of X-ray exposure at image edges
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    vignette = np.clip(distance_from_center / (IMAGE_SIZE * 0.5), 0, 1)
    base = np.clip(base * (1 - 0.3 * vignette), 0, 255).astype(np.uint8)

    return base


def add_pneumonia_infiltrates(image: np.ndarray) -> np.ndarray:
    """
    Add pneumonia-like infiltrates (cloudy patches) to the X-ray.

    In real pneumonia X-rays:
        - Infiltrates appear as white/gray cloudy areas
        - They obscure normal lung markings
        - Can be focal (one area) or diffuse (multiple areas)
        - Often in lower lobes (gravity-dependent)
        - May have air bronchograms (dark air-filled bronchi visible within white consolidation)

    We simulate this with:
        - Random elliptical patches of increased density
        - Gaussian blur for soft edges (infiltrates don't have sharp borders)
        - Predominantly in lower lung zones

    Args:
        image: Base X-ray image as numpy array

    Returns:
        numpy.ndarray: Image with added infiltrates
    """
    result = image.copy()

    # Number of infiltrate patches (1-4, more severe = more patches)
    num_patches = random.randint(1, 4)

    for _ in range(num_patches):
        # Infiltrates are more common in lower lobes
        # So we bias the y-position toward the bottom of the image
        # Avoid the very top (apex) and center (heart)

        # Choose which lung (left or right side of image)
        if random.random() < 0.5:
            # Right lung (left side of image in standard PA view)
            patch_x = random.randint(80, IMAGE_SIZE // 2 - 60)
        else:
            # Left lung (right side of image)
            patch_x = random.randint(IMAGE_SIZE // 2 + 60, IMAGE_SIZE - 80)

        # Lower lobes (bottom half of lungs, below heart level)
        patch_y = random.randint(IMAGE_SIZE // 2, IMAGE_SIZE - 100)

        # Size of infiltrate (varies with severity)
        patch_width = random.randint(40, 120)
        patch_height = random.randint(40, 100)

        # Create elliptical infiltrate patch
        y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
        mask = ((x - patch_x) ** 2 / (patch_width // 2) ** 2 +
                (y - patch_y) ** 2 / (patch_height // 2) ** 2) < 1

        # Infiltrates increase density (appear whiter)
        # Intensity varies (some areas denser than others)
        intensity = random.randint(50, 100)
        result[mask] = np.clip(result[mask] + intensity, 0, 255)

    # Apply Gaussian blur to soften edges
    # Real infiltrates have gradual transitions, not sharp edges
    result_pil = Image.fromarray(result)
    result_pil = result_pil.filter(ImageFilter.GaussianBlur(radius=3))

    return np.array(result_pil)


def generate_xray_image(has_pneumonia: bool) -> Image.Image:
    """
    Generate a complete synthetic chest X-ray image.

    Args:
        has_pneumonia: If True, add infiltrates. If False, keep normal.

    Returns:
        PIL.Image: 512x512 grayscale chest X-ray image
    """
    # Create base anatomical structure
    image = create_base_xray_image()

    # Add pathology if pneumonia case
    if has_pneumonia:
        image = add_pneumonia_infiltrates(image)

    # Convert to PIL Image
    pil_image = Image.fromarray(image, mode=IMAGE_MODE)

    # Apply slight smoothing for more realistic appearance
    pil_image = pil_image.filter(ImageFilter.SMOOTH)

    return pil_image


# ============================================
# Clinical Report Generation
# ============================================

def generate_clinical_report(
    patient_id: str,
    age: int,
    sex: str,
    has_pneumonia: bool,
    vitals: Dict
) -> str:
    """
    Generate a synthetic clinical radiology report.

    Real radiology reports follow a standard structure:
        1. Header: Patient info, study date, ordering physician
        2. Clinical History: Reason for examination
        3. Technique: How the study was performed
        4. Comparison: Prior studies if available
        5. Findings: Detailed description of observations
        6. Impression: Summary diagnosis/conclusion

    This structure is important for RAG because:
        - Users often query by symptoms (clinical history)
        - Or by diagnosis (impression)
        - Or by specific findings
        - The report provides context for image retrieval

    Args:
        patient_id: Unique patient identifier
        age: Patient age in years
        sex: 'M' or 'F'
        has_pneumonia: Determines content of findings/impression
        vitals: Dictionary of vital signs for context

    Returns:
        str: Formatted clinical report text
    """
    # Select appropriate templates based on diagnosis
    if has_pneumonia:
        symptoms = random.choice(PNEUMONIA_SYMPTOMS)
        findings = random.choice(PNEUMONIA_FINDINGS)
        impression = random.choice(PNEUMONIA_IMPRESSIONS)
    else:
        symptoms = random.choice(NORMAL_SYMPTOMS)
        findings = random.choice(NORMAL_FINDINGS)
        impression = random.choice(NORMAL_IMPRESSIONS)

    # Generate study date (within last 30 days)
    study_date = datetime.now() - timedelta(days=random.randint(0, 30))

    # Format the report
    report = f"""
================================================================================
                        RADIOLOGY REPORT
================================================================================

PATIENT ID: {patient_id}
AGE: {age} years
SEX: {'Male' if sex == 'M' else 'Female'}
STUDY DATE: {study_date.strftime('%Y-%m-%d %H:%M')}
STUDY TYPE: Chest X-ray (PA and Lateral)
ORDERING PHYSICIAN: Dr. Synthetic Generator

--------------------------------------------------------------------------------
CLINICAL HISTORY:
--------------------------------------------------------------------------------
{age}-year-old {'male' if sex == 'M' else 'female'} presenting with {symptoms}.

Vital signs at time of study:
- Temperature: {vitals['temperature']}°F
- Heart Rate: {vitals['heart_rate']} bpm
- Respiratory Rate: {vitals['respiratory_rate']} breaths/min
- Blood Pressure: {vitals['blood_pressure']}
- Oxygen Saturation: {vitals['oxygen_saturation']}%

--------------------------------------------------------------------------------
TECHNIQUE:
--------------------------------------------------------------------------------
PA and lateral views of the chest were obtained. Comparison is made to prior
studies when available.

--------------------------------------------------------------------------------
FINDINGS:
--------------------------------------------------------------------------------
{findings}

--------------------------------------------------------------------------------
IMPRESSION:
--------------------------------------------------------------------------------
{impression}

--------------------------------------------------------------------------------
RADIOLOGIST: Dr. AI Synthetic, MD
REPORT STATUS: FINAL
DISCLAIMER: This is SYNTHETIC data for testing purposes only.
================================================================================
"""
    return report.strip()


# ============================================
# Vital Signs and Lab Data Generation
# ============================================

def generate_patient_vitals(has_pneumonia: bool) -> Dict:
    """
    Generate realistic vital signs and lab values for a patient.

    Vital signs in pneumonia vs normal:

    PNEUMONIA patients typically show:
        - Elevated temperature (fever): 100.4-104°F
        - Tachycardia (fast heart rate): 90-120 bpm
        - Tachypnea (fast breathing): 20-30 breaths/min
        - Lower oxygen saturation: 88-95%
        - Elevated WBC (infection): 12,000-25,000
        - Elevated CRP (inflammation): 50-200 mg/L

    NORMAL patients show:
        - Normal temperature: 97.5-99°F
        - Normal heart rate: 60-90 bpm
        - Normal respiratory rate: 12-18 breaths/min
        - Normal oxygen saturation: 96-100%
        - Normal WBC: 4,500-11,000
        - Normal CRP: 0-10 mg/L

    These values are important for RAG because:
        - Users might query by lab values ("high WBC and fever")
        - Vitals provide clinical context for image findings
        - Enables hybrid search (symptoms + image features)

    Args:
        has_pneumonia: Determines the range of values

    Returns:
        dict: Dictionary of vital signs and lab values
    """
    if has_pneumonia:
        # Abnormal values consistent with infection
        vitals = {
            "temperature": round(random.uniform(100.4, 104.0), 1),
            "heart_rate": random.randint(90, 120),
            "respiratory_rate": random.randint(20, 30),
            "systolic_bp": random.randint(100, 140),
            "diastolic_bp": random.randint(60, 90),
            "oxygen_saturation": random.randint(88, 95),
            "wbc": round(random.uniform(12.0, 25.0), 1),  # x10^3/uL
            "crp": round(random.uniform(50, 200), 1),     # mg/L
        }
    else:
        # Normal values
        vitals = {
            "temperature": round(random.uniform(97.5, 99.0), 1),
            "heart_rate": random.randint(60, 90),
            "respiratory_rate": random.randint(12, 18),
            "systolic_bp": random.randint(110, 130),
            "diastolic_bp": random.randint(70, 85),
            "oxygen_saturation": random.randint(96, 100),
            "wbc": round(random.uniform(4.5, 11.0), 1),
            "crp": round(random.uniform(0, 10), 1),
        }

    # Format blood pressure as string
    vitals["blood_pressure"] = f"{vitals['systolic_bp']}/{vitals['diastolic_bp']}"

    return vitals


# ============================================
# Main Dataset Generation Function
# ============================================

def generate_dataset(
    num_cases: int = DEFAULT_NUM_CASES,
    pneumonia_ratio: float = PNEUMONIA_RATIO,
    seed: int = RANDOM_SEED,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Generate a complete synthetic medical dataset.

    This function orchestrates the creation of all three data modalities:
        1. X-ray images (saved as JPEG)
        2. Clinical reports (saved as TXT)
        3. Structured vitals data (saved as CSV)

    The dataset is designed for testing multimodal RAG:
        - Images can be embedded with vision models
        - Reports can be embedded with text models
        - Structured data enables filtered retrieval
        - Ground truth labels allow evaluation

    Args:
        num_cases: Total number of patient cases to generate
        pneumonia_ratio: Fraction of cases with pneumonia (0.0-1.0)
        seed: Random seed for reproducibility
        output_dir: Base directory for output files

    Returns:
        pd.DataFrame: Summary of all generated cases
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("SYNTHETIC MEDICAL DATA GENERATOR")
    print("=" * 60)
    print(f"Generating {num_cases} cases ({pneumonia_ratio*100:.0f}% pneumonia)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Create output directories
    images_dir = output_dir / "images"
    reports_dir = output_dir / "reports"
    images_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cases are pneumonia vs normal
    num_pneumonia = int(num_cases * pneumonia_ratio)
    diagnoses = ["pneumonia"] * num_pneumonia + ["normal"] * (num_cases - num_pneumonia)
    random.shuffle(diagnoses)  # Randomize order

    # Storage for structured data
    all_records = []

    # Generate each case
    for i, diagnosis in enumerate(diagnoses):
        # Generate patient ID (4-digit zero-padded)
        patient_id = f"patient_{i+1:04d}"
        has_pneumonia = (diagnosis == "pneumonia")

        # Generate demographics
        age = random.randint(25, 85)
        sex = random.choice(["M", "F"])

        # Generate vitals and lab values
        vitals = generate_patient_vitals(has_pneumonia)

        # ----------------------------------------
        # Generate and save X-ray image
        # ----------------------------------------
        xray_image = generate_xray_image(has_pneumonia)
        image_filename = f"{patient_id}_xray.jpg"
        image_path = images_dir / image_filename
        xray_image.save(image_path, "JPEG", quality=95)

        # ----------------------------------------
        # Generate and save clinical report
        # ----------------------------------------
        report_text = generate_clinical_report(
            patient_id, age, sex, has_pneumonia, vitals
        )
        report_filename = f"{patient_id}_report.txt"
        report_path = reports_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # ----------------------------------------
        # Store structured data record
        # ----------------------------------------
        record = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "temperature": vitals["temperature"],
            "heart_rate": vitals["heart_rate"],
            "respiratory_rate": vitals["respiratory_rate"],
            "blood_pressure": vitals["blood_pressure"],
            "oxygen_saturation": vitals["oxygen_saturation"],
            "wbc": vitals["wbc"],
            "crp": vitals["crp"],
            "diagnosis": diagnosis,
            "image_file": image_filename,
            "report_file": report_filename,
        }
        all_records.append(record)

        # Progress tracking (every 20 cases)
        if (i + 1) % 20 == 0 or (i + 1) == num_cases:
            print(f"Progress: {i + 1}/{num_cases} cases generated "
                  f"({(i + 1) / num_cases * 100:.0f}%)")

    # ----------------------------------------
    # Save structured data as CSV
    # ----------------------------------------
    df = pd.DataFrame(all_records)
    csv_path = output_dir / "patient_vitals.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics for the generated dataset.

    This helps verify the dataset was generated correctly:
        - Distribution matches expected ratios
        - Value ranges are physiologically plausible
        - Files were created successfully

    Args:
        df: DataFrame containing all patient records
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)

    # Diagnosis distribution
    print("\n--- Diagnosis Distribution ---")
    diagnosis_counts = df["diagnosis"].value_counts()
    for diag, count in diagnosis_counts.items():
        pct = count / len(df) * 100
        print(f"  {diag.capitalize():12s}: {count:4d} ({pct:.1f}%)")

    # Demographics
    print("\n--- Demographics ---")
    print(f"  Age range: {df['age'].min()} - {df['age'].max()} years")
    print(f"  Age mean:  {df['age'].mean():.1f} years")
    sex_counts = df["sex"].value_counts()
    print(f"  Sex:       M={sex_counts.get('M', 0)}, F={sex_counts.get('F', 0)}")

    # Vital signs by diagnosis
    print("\n--- Vital Signs by Diagnosis ---")
    for diag in ["pneumonia", "normal"]:
        subset = df[df["diagnosis"] == diag]
        print(f"\n  {diag.upper()}:")
        print(f"    Temperature:   {subset['temperature'].mean():.1f}°F "
              f"(range: {subset['temperature'].min()}-{subset['temperature'].max()})")
        print(f"    Heart Rate:    {subset['heart_rate'].mean():.0f} bpm "
              f"(range: {subset['heart_rate'].min()}-{subset['heart_rate'].max()})")
        print(f"    O2 Saturation: {subset['oxygen_saturation'].mean():.0f}% "
              f"(range: {subset['oxygen_saturation'].min()}-{subset['oxygen_saturation'].max()})")
        print(f"    WBC:           {subset['wbc'].mean():.1f} x10^3/uL "
              f"(range: {subset['wbc'].min()}-{subset['wbc'].max()})")
        print(f"    CRP:           {subset['crp'].mean():.1f} mg/L "
              f"(range: {subset['crp'].min()}-{subset['crp'].max()})")

    # File counts
    print("\n--- Generated Files ---")
    print(f"  Images:  {len(df)} files in data/raw/images/")
    print(f"  Reports: {len(df)} files in data/raw/reports/")
    print(f"  CSV:     1 file (patient_vitals.csv)")

    # Total size estimate
    avg_image_size = 15  # ~15KB per JPEG
    avg_report_size = 2  # ~2KB per TXT
    total_mb = len(df) * (avg_image_size + avg_report_size) / 1024
    print(f"\n  Estimated total size: ~{total_mb:.1f} MB")

    print("\n" + "=" * 60)


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    """
    Run the synthetic data generator.

    This creates a complete test dataset with:
        - 100 synthetic chest X-ray images
        - 100 clinical reports
        - 1 CSV file with structured vitals data

    Output locations:
        - data/raw/images/patient_XXXX_xray.jpg
        - data/raw/reports/patient_XXXX_report.txt
        - data/raw/patient_vitals.csv
    """
    # Generate the dataset
    df = generate_dataset(
        num_cases=100,
        pneumonia_ratio=0.60,
        seed=42,
    )

    # Print summary statistics
    print_summary_statistics(df)

    # Show sample data
    print("\n--- Sample Records ---")
    print(df[["patient_id", "age", "sex", "diagnosis", "temperature", "wbc"]].head(5))
    print("\nDataset generation complete!")
    print(f"CSV saved to: {OUTPUT_DIR / 'patient_vitals.csv'}")
