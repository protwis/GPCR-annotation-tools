"""Shared pytest fixtures for GPCR annotation tools tests."""

import json
import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_PDB_DIR = FIXTURES_DIR / "real_pdbs"

REAL_PDB_IDS: tuple[str, ...] = (
    "5G53",
    "8TII",
    "9AS1",
    "9BLW",
    "9EJZ",
    "9IQS",
    "9M88",
    "9NOR",
    "9O38",
)

REAL_PDB_VOTING_LOG_IDS: frozenset[str] = frozenset({"9M88", "9O38", "9AS1", "9BLW", "9IQS"})

REAL_PDB_VALIDATION_LOG_IDS: frozenset[str] = frozenset(REAL_PDB_IDS)


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
def sample_oligomer_data() -> dict:
    """Load the PDB test fixture with oligomer_analysis data."""
    with open(FIXTURES_DIR / "sample_pdb_oligomer.json") as f:
        return json.load(f)


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


def _populate_real_pdb_workspace(workspace: Path) -> None:
    """Copy committed real PDB fixtures into a temporary workspace layout."""
    aggregated = workspace / "aggregated"
    aggregated.mkdir(exist_ok=True)
    (aggregated / "logs").mkdir(exist_ok=True)
    (aggregated / "validation_logs").mkdir(exist_ok=True)

    for json_file in REAL_PDB_DIR.glob("*.json"):
        shutil.copy2(json_file, aggregated / json_file.name)
    for json_file in (REAL_PDB_DIR / "logs").glob("*.json"):
        shutil.copy2(json_file, aggregated / "logs" / json_file.name)
    for json_file in (REAL_PDB_DIR / "validation_logs").glob("*.json"):
        shutil.copy2(json_file, aggregated / "validation_logs" / json_file.name)


@pytest.fixture
def real_pdb_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a temporary workspace populated with all committed real PDB fixtures.

    Sets ``GPCR_WORKSPACE``, writes a valid storage contract, and resets
    the config cache.  Yields the workspace root.
    """
    from gpcr_tools.config import reset_config

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _populate_real_pdb_workspace(workspace)

    (workspace / "output" / "csv").mkdir(parents=True)
    (workspace / "output" / "audit").mkdir(parents=True)
    (workspace / "state").mkdir()
    _write_contract(workspace)

    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    reset_config()

    yield workspace
    reset_config()
