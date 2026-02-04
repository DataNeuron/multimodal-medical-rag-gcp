# ============================================
# Synthetic Data Generation Module
# ============================================
"""
Generate synthetic medical data for testing and development.

This module creates realistic but artificial medical data including:
    - Radiology reports with common findings
    - Patient demographics (de-identified)
    - Diagnosis codes and metadata

IMPORTANT: This data is synthetic and should NOT be used for
clinical decisions. It is intended for testing and development only.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4


# Sample findings for synthetic reports
FINDINGS_TEMPLATES = [
    "No acute cardiopulmonary abnormality.",
    "Small bilateral pleural effusions, unchanged from prior study.",
    "Mild cardiomegaly without pulmonary edema.",
    "Linear atelectasis at the lung bases bilaterally.",
    "No focal consolidation, pleural effusion, or pneumothorax.",
    "Stable postoperative changes. No acute findings.",
    "Mild diffuse interstitial thickening, possibly chronic.",
]

IMPRESSIONS = [
    "Normal chest radiograph.",
    "No acute cardiopulmonary disease.",
    "Findings consistent with mild heart failure.",
    "Stable appearance compared to prior examination.",
    "No significant interval change.",
]


def generate_synthetic_reports(
    num_reports: int = 100,
    output_dir: Optional[Path] = None,
    include_metadata: bool = True,
) -> list[dict]:
    """
    Generate synthetic radiology reports for testing.

    Args:
        num_reports: Number of reports to generate
        output_dir: Directory to save reports (optional)
        include_metadata: Include patient metadata

    Returns:
        List of generated report dictionaries
    """
    reports = []

    for i in range(num_reports):
        report = _generate_single_report(include_metadata)
        reports.append(report)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual reports
        for report in reports:
            report_path = output_dir / f"{report['report_id']}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "num_reports": num_reports,
                    "report_ids": [r["report_id"] for r in reports],
                },
                f,
                indent=2,
            )

        print(f"Generated {num_reports} synthetic reports in {output_dir}")

    return reports


def _generate_single_report(include_metadata: bool) -> dict:
    """Generate a single synthetic radiology report."""
    report_id = str(uuid4())[:8]

    report = {
        "report_id": report_id,
        "study_type": "Chest X-ray",
        "study_date": _random_date(),
        "findings": random.choice(FINDINGS_TEMPLATES),
        "impression": random.choice(IMPRESSIONS),
        "is_synthetic": True,  # Always mark as synthetic
    }

    if include_metadata:
        report["metadata"] = {
            "patient_id": f"SYNTH-{random.randint(10000, 99999)}",
            "age": random.randint(25, 85),
            "sex": random.choice(["M", "F"]),
            "modality": "CR",
            "body_part": "CHEST",
        }

    return report


def _random_date() -> str:
    """Generate a random date within the last year."""
    days_ago = random.randint(0, 365)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


if __name__ == "__main__":
    # Example: Generate 10 synthetic reports
    reports = generate_synthetic_reports(
        num_reports=10,
        output_dir=Path("data/raw/synthetic"),
        include_metadata=True,
    )
    print(f"Sample report:\n{json.dumps(reports[0], indent=2)}")
