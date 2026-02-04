# ============================================
# Pytest Configuration and Fixtures
# ============================================
"""
Shared fixtures and configuration for all tests.
"""

import os
import sys
import pytest
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running",
    )


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_data_dir(project_root):
    """Return path to sample test data."""
    return project_root / "tests" / "data"


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables."""
    env_vars = {
        "GCP_PROJECT_ID": "test-project",
        "GCP_REGION": "us-central1",
        "ENVIRONMENT": "test",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def sample_report():
    """Return a sample synthetic report for testing."""
    return {
        "report_id": "test-001",
        "study_type": "Chest X-ray",
        "study_date": "2024-01-15",
        "findings": "No acute cardiopulmonary abnormality.",
        "impression": "Normal chest radiograph.",
        "is_synthetic": True,
        "metadata": {
            "patient_id": "SYNTH-12345",
            "age": 45,
            "sex": "M",
            "modality": "CR",
            "body_part": "CHEST",
        },
    }
