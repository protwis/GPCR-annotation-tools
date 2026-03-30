"""Tests for workspace initialization, contract validation, and path validation."""

import json

import pytest

from gpcr_tools.config import get_config, reset_config
from gpcr_tools.workspace import (
    SUPPORTED_CONTRACT_VERSION,
    ensure_runtime_dirs,
    init_workspace,
    print_path_summary,
    startup_checks,
    validate_contract,
    validate_paths,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear workspace env vars and config cache between tests."""
    env_vars = [
        "GPCR_WORKSPACE",
        "GPCR_RAW_PATH",
        "GPCR_ENRICHED_PATH",
        "GPCR_PAPERS_PATH",
        "GPCR_AI_RESULTS_PATH",
        "GPCR_AGGREGATED_PATH",
        "GPCR_OUTPUT_PATH",
        "GPCR_CACHE_PATH",
        "GPCR_STATE_PATH",
        "GPCR_TMP_PATH",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    reset_config()
    yield
    reset_config()


# ── init_workspace ───────────────────────────────────────────────────


class TestInitWorkspace:
    def test_creates_full_tree(self, tmp_path):
        init_workspace(tmp_path)

        contract = tmp_path / "contract" / "storage_contract.json"
        assert contract.exists()
        data = json.loads(contract.read_text())
        assert data["storage_contract_version"] == SUPPORTED_CONTRACT_VERSION
        assert data["created_by"] == "gpcr-tools"

        expected_dirs = [
            "raw",
            "raw/pdb_json",
            "raw/structure_files",
            "enriched",
            "papers",
            "ai_results",
            "aggregated",
            "aggregated/logs",
            "aggregated/validation_logs",
            "output",
            "output/csv",
            "output/audit",
            "cache",
            "state",
            "state/pipeline_runs",
            "tmp",
        ]
        for d in expected_dirs:
            assert (tmp_path / d).is_dir(), f"Missing directory: {d}"

    def test_idempotent(self, tmp_path):
        init_workspace(tmp_path)
        contract = tmp_path / "contract" / "storage_contract.json"
        original = contract.read_text()

        init_workspace(tmp_path)
        assert contract.read_text() == original

    def test_refuses_incompatible_version(self, tmp_path):
        contract_dir = tmp_path / "contract"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "storage_contract.json"
        contract_file.write_text(json.dumps({"storage_contract_version": 999}))

        with pytest.raises(SystemExit):
            init_workspace(tmp_path)

    def test_refuses_malformed_json(self, tmp_path):
        contract_dir = tmp_path / "contract"
        contract_dir.mkdir(parents=True)
        (contract_dir / "storage_contract.json").write_text("{bad json")

        with pytest.raises(SystemExit):
            init_workspace(tmp_path)


# ── validate_contract ────────────────────────────────────────────────


class TestValidateContract:
    def _make_contract(self, ws, version=SUPPORTED_CONTRACT_VERSION):
        d = ws / "contract"
        d.mkdir(parents=True, exist_ok=True)
        (d / "storage_contract.json").write_text(json.dumps({"storage_contract_version": version}))

    def test_valid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        self._make_contract(tmp_path)
        validate_contract()  # should not raise

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        with pytest.raises(SystemExit):
            validate_contract()

    def test_malformed_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        d = tmp_path / "contract"
        d.mkdir()
        (d / "storage_contract.json").write_text("not-json")
        with pytest.raises(SystemExit):
            validate_contract()

    def test_unsupported_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        self._make_contract(tmp_path, version=999)
        with pytest.raises(SystemExit):
            validate_contract()

    def test_missing_version_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        d = tmp_path / "contract"
        d.mkdir()
        (d / "storage_contract.json").write_text(json.dumps({"created_by": "test"}))
        with pytest.raises(SystemExit):
            validate_contract()


# ── validate_paths ───────────────────────────────────────────────────


class TestValidatePaths:
    def test_default_config_valid(self):
        validate_paths()  # default /workspace layout has no overlaps

    def test_explicit_override_outside_workspace_allowed(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        external = tmp_path / "external_cache"
        external.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_CACHE_PATH", str(external))
        reset_config()
        validate_paths()  # should not raise

    def test_collision_rejected(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_CACHE_PATH", str(shared))
        monkeypatch.setenv("GPCR_STATE_PATH", str(shared))
        reset_config()
        with pytest.raises(SystemExit):
            validate_paths()

    def test_nesting_rejected(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_CACHE_PATH", str(ws / "state" / "caches"))
        reset_config()
        with pytest.raises(SystemExit):
            validate_paths()

    def test_dir_equals_workspace_rejected(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_OUTPUT_PATH", str(ws))
        reset_config()
        with pytest.raises(SystemExit):
            validate_paths()


# ── print_path_summary ───────────────────────────────────────────────


class TestPrintPathSummary:
    def test_output_to_stderr(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        print_path_summary()
        captured = capsys.readouterr()
        assert "[workspace]" in captured.err
        assert str(tmp_path) in captured.err


# ── ensure_runtime_dirs ──────────────────────────────────────────────


class TestEnsureRuntimeDirs:
    def test_creates_missing_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        ensure_runtime_dirs()
        cfg = get_config()
        assert cfg.raw_dir.is_dir()
        assert cfg.csv_output_dir.is_dir()
        assert cfg.audit_output_dir.is_dir()
        assert cfg.pipeline_runs_dir.is_dir()
        assert cfg.tmp_dir.is_dir()


# ── startup_checks ───────────────────────────────────────────────────


class TestStartupChecks:
    """Test the full startup orchestration: paths → contract → summary → dirs."""

    def test_valid_workspace_passes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        init_workspace(tmp_path)

        config = startup_checks()

        assert config.workspace == tmp_path.resolve()
        assert config.csv_output_dir.is_dir()
        assert config.audit_output_dir.is_dir()
        assert config.state_dir.is_dir()

    def test_creates_operational_dirs(self, tmp_path, monkeypatch):
        """After startup_checks, runtime directories must exist."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        init_workspace(tmp_path)

        config = startup_checks()

        assert config.pipeline_runs_dir.is_dir()
        assert config.tmp_dir.is_dir()
        assert (config.aggregated_dir / "logs").is_dir()
        assert (config.aggregated_dir / "validation_logs").is_dir()

    def test_aborts_without_contract(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        with pytest.raises(SystemExit):
            startup_checks()

    def test_aborts_on_path_overlap(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(ws))
        monkeypatch.setenv("GPCR_CACHE_PATH", str(ws / "state" / "caches"))
        reset_config()
        with pytest.raises(SystemExit):
            startup_checks()
