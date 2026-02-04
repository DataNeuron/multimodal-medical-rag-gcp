# ============================================
# Unit Tests for Synthetic Data Generation
# ============================================
"""
Tests for the synthetic data generation module.
"""

import pytest
from src.data_ingestion.generate_synthetic import (
    generate_synthetic_reports,
    _generate_single_report,
)


class TestSyntheticDataGeneration:
    """Tests for synthetic medical report generation."""

    def test_generate_single_report_has_required_fields(self):
        """Verify single report has all required fields."""
        report = _generate_single_report(include_metadata=False)

        assert "report_id" in report
        assert "study_type" in report
        assert "study_date" in report
        assert "findings" in report
        assert "impression" in report
        assert report["is_synthetic"] is True

    def test_generate_single_report_with_metadata(self):
        """Verify metadata is included when requested."""
        report = _generate_single_report(include_metadata=True)

        assert "metadata" in report
        assert "patient_id" in report["metadata"]
        assert "age" in report["metadata"]
        assert "sex" in report["metadata"]

    def test_generate_multiple_reports(self):
        """Verify batch report generation."""
        num_reports = 5
        reports = generate_synthetic_reports(
            num_reports=num_reports,
            output_dir=None,
            include_metadata=True,
        )

        assert len(reports) == num_reports

        # Verify all reports have unique IDs
        report_ids = [r["report_id"] for r in reports]
        assert len(set(report_ids)) == num_reports

    def test_synthetic_flag_always_set(self):
        """Verify is_synthetic flag is always True."""
        reports = generate_synthetic_reports(num_reports=10)

        for report in reports:
            assert report["is_synthetic"] is True

    def test_patient_id_prefix(self):
        """Verify synthetic patient IDs have correct prefix."""
        report = _generate_single_report(include_metadata=True)

        patient_id = report["metadata"]["patient_id"]
        assert patient_id.startswith("SYNTH-")
