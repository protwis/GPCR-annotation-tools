"""Integration tests for the curate command and --auto-accept workflow.

Covers:
- ``curate --help`` accessibility
- ``curate --auto-accept`` end-to-end artifact placement
- startup failure paths (missing / bad contract)
- full CLI subprocess flow (init → fixture → auto-accept → verify)
"""

import csv
import json
import os
import subprocess
import sys

import pytest

from gpcr_tools.config import get_config, reset_config

# -- helpers ---------------------------------------------------------------


def _clean_env(workspace: str) -> dict[str, str]:
    """Return a copy of os.environ with only GPCR_WORKSPACE set."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("GPCR_"):
            del env[key]
    env["GPCR_WORKSPACE"] = workspace
    return env


# -- curate --help ---------------------------------------------------------


class TestCurateHelp:
    def test_exits_zero_and_mentions_auto_accept(self):
        result = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "curate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "auto-accept" in result.stdout


# -- auto-accept (direct function calls) ----------------------------------


class TestAutoAccept:
    """Exercise _run_auto_accept via main(auto_accept=True)."""

    def test_writes_all_artifact_categories(self, initialized_workspace):
        """auto-accept must write CSV, audit, and state artifacts in v3.1 locations."""
        from gpcr_tools.csv_generator.app import main

        main(auto_accept=True)

        cfg = get_config()

        # CSV under output/csv/
        assert cfg.csv_output_dir.is_dir()
        structures = cfg.csv_output_dir / "structures.csv"
        assert structures.exists()
        with open(structures) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) == 1
        assert rows[0]["PDB"] == "TEST1"

        # Audit trail under output/audit/
        audit_file = cfg.audit_output_dir / "audit_trail.jsonl"
        assert audit_file.exists()
        entries = [json.loads(line) for line in audit_file.read_text().splitlines()]
        assert any(e["action"] == "auto_accept" for e in entries)

        # Processed log under state/
        assert cfg.processed_log_file.exists()
        log = json.loads(cfg.processed_log_file.read_text())
        assert "TEST1" in log
        assert log["TEST1"]["status"] == "completed"

    def test_targeted_pdb_processes_only_that_pdb(self, initialized_workspace, sample_pdb_data):
        """Targeting a specific PDB should process only that one."""
        with open(initialized_workspace / "aggregated" / "TEST2.json", "w") as f:
            json.dump(sample_pdb_data, f)
        reset_config()

        from gpcr_tools.csv_generator.app import main

        main(target_pdb="TEST1", auto_accept=True)

        log = json.loads(get_config().processed_log_file.read_text())
        assert "TEST1" in log
        assert "TEST2" not in log

    def test_empty_queue_exits_cleanly(self, initialized_workspace):
        """auto-accept with no pending PDBs must not crash."""
        for f in (initialized_workspace / "aggregated").glob("*.json"):
            f.unlink()
        reset_config()

        from gpcr_tools.csv_generator.app import main

        main(auto_accept=True)  # must not raise

        assert not get_config().processed_log_file.exists()


# -- full CLI subprocess flow ---------------------------------------------


class TestAutoAcceptCLI:
    """Ultimate integration: subprocess init → fixture → auto-accept → verify."""

    def test_full_cli_flow(self, tmp_path, sample_pdb_data):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        env = _clean_env(str(workspace))

        r1 = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "init-workspace"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert r1.returncode == 0

        with open(workspace / "aggregated" / "TEST1.json", "w") as f:
            json.dump(sample_pdb_data, f)

        r2 = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "curate", "--auto-accept"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert r2.returncode == 0

        assert (workspace / "output" / "csv" / "structures.csv").exists()
        assert (workspace / "output" / "audit" / "audit_trail.jsonl").exists()
        assert (workspace / "state" / "processed_log.json").exists()

        log = json.loads((workspace / "state" / "processed_log.json").read_text())
        assert "TEST1" in log
        assert log["TEST1"]["status"] == "completed"


# -- startup failure paths -------------------------------------------------


class TestStartupFailures:
    """Curate must abort cleanly if the workspace contract is invalid."""

    def test_missing_contract(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()

        from gpcr_tools.csv_generator.app import main

        with pytest.raises(SystemExit):
            main(auto_accept=True)

    def test_unsupported_contract_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        (tmp_path / "contract").mkdir()
        (tmp_path / "contract" / "storage_contract.json").write_text(
            json.dumps({"storage_contract_version": 999})
        )
        reset_config()

        from gpcr_tools.csv_generator.app import main

        with pytest.raises(SystemExit):
            main(auto_accept=True)

    def test_malformed_contract_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        (tmp_path / "contract").mkdir()
        (tmp_path / "contract" / "storage_contract.json").write_text("{bad json")
        reset_config()

        from gpcr_tools.csv_generator.app import main

        with pytest.raises(SystemExit):
            main(auto_accept=True)
