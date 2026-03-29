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
    """Create a temporary aggregated data directory (v3.1 layout).

    Populates it with the simple PDB fixture as 'TEST1.json'.
    """
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    (data_dir / "logs").mkdir()
    (data_dir / "validation_logs").mkdir()

    with open(data_dir / "TEST1.json", "w") as f:
        json.dump(sample_pdb_data, f)

    return data_dir


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory tree (v3.1 layout)."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "csv").mkdir()
    (out_dir / "audit").mkdir()
    return out_dir


def _write_contract(workspace: Path) -> None:
    """Write a valid v1 storage contract into the workspace."""
    contract_dir = workspace / "contract"
    contract_dir.mkdir(exist_ok=True)
    contract_file = contract_dir / "storage_contract.json"
    if not contract_file.exists():
        contract_file.write_text(
            json.dumps(
                {
                    "storage_contract_version": 1,
                    "created_by": "gpcr-tools-test",
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                }
            )
        )


@pytest.fixture
def configure_paths(tmp_data_dir, tmp_output_dir, monkeypatch):
    """Set GPCR_WORKSPACE and reset the config cache.

    All consumers now call get_config() at function-call time, so
    importlib.reload() is no longer necessary.
    """
    workspace = tmp_data_dir.parent  # tmp_path
    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))

    _write_contract(workspace)
    (workspace / "state").mkdir(exist_ok=True)

    from gpcr_tools.config import reset_config

    reset_config()

    return tmp_data_dir, tmp_output_dir


@pytest.fixture
def initialized_workspace(tmp_path, sample_pdb_data, monkeypatch):
    """A fully initialized v3.1 workspace with one PDB fixture (TEST1).

    Uses the real ``init_workspace`` to create the tree and contract,
    then places ``sample_pdb_simple.json`` as ``aggregated/TEST1.json``.

    Returns the workspace root path.
    """
    from gpcr_tools.config import reset_config
    from gpcr_tools.workspace import init_workspace

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    reset_config()

    init_workspace(workspace)

    with open(workspace / "aggregated" / "TEST1.json", "w") as f:
        json.dump(sample_pdb_data, f)

    reset_config()
    yield workspace
    reset_config()
