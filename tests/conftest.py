"""Shared pytest fixtures for GPCR annotation tools tests."""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdb_data() -> dict:
    """Load the simple (clean) PDB test fixture."""
    with open(FIXTURES_DIR / "sample_pdb_simple.json") as f:
        return json.load(f)


@pytest.fixture
def sample_controversy_data() -> dict:
    """Load the PDB test fixture with controversies."""
    with open(FIXTURES_DIR / "sample_pdb_controversy.json") as f:
        return json.load(f)


@pytest.fixture
def sample_voting_log() -> list:
    """Load the voting log test fixture."""
    with open(FIXTURES_DIR / "sample_voting_log.json") as f:
        return json.load(f)


@pytest.fixture
def sample_controversy_map(sample_voting_log) -> dict:
    """Convert the voting log list into a path-keyed controversy map."""
    return {item["path"]: item for item in sample_voting_log}


@pytest.fixture
def sample_validation_data() -> dict:
    """Load the validation test fixture."""
    with open(FIXTURES_DIR / "sample_validation.json") as f:
        return json.load(f)


@pytest.fixture
def tmp_data_dir(tmp_path, sample_pdb_data) -> Path:
    """Create a temporary data directory mimicking results_aggregated/.

    Populates it with the simple PDB fixture as 'TEST1.json'.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "logs").mkdir()
    (data_dir / "validation_logs").mkdir()

    with open(data_dir / "TEST1.json", "w") as f:
        json.dump(sample_pdb_data, f)

    return data_dir


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory for CSVs."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def configure_paths(tmp_data_dir, tmp_output_dir, monkeypatch):
    """Monkeypatch the config module to use temporary directories.

    Use this in tests that involve data loading or CSV writing.
    """
    monkeypatch.setenv("GPCR_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("GPCR_OUTPUT_DIR", str(tmp_output_dir))

    # Force reimport of config to pick up new env vars
    import importlib

    import gpcr_tools.config

    importlib.reload(gpcr_tools.config)

    # Also reload modules that import from config at module level
    import gpcr_tools.csv_generator.data_loader

    importlib.reload(gpcr_tools.csv_generator.data_loader)
    import gpcr_tools.csv_generator.csv_writer

    importlib.reload(gpcr_tools.csv_generator.csv_writer)
    import gpcr_tools.csv_generator.audit

    importlib.reload(gpcr_tools.csv_generator.audit)

    return tmp_data_dir, tmp_output_dir
