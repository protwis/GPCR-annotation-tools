"""Integration tests: init-workspace CLI command via subprocess.

These tests exercise the real CLI entry point (``python -m gpcr_tools
init-workspace``) to verify contract creation, idempotency, and
incompatible-version rejection.
"""

import json
import os
import subprocess
import sys


def _clean_env(workspace: str) -> dict[str, str]:
    """Return a copy of os.environ with only GPCR_WORKSPACE set."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("GPCR_"):
            del env[key]
    env["GPCR_WORKSPACE"] = workspace
    return env


class TestInitWorkspaceCLI:
    def test_creates_contract_and_full_tree(self, tmp_path):
        """init-workspace on a blank directory must produce the complete v3.1 tree."""
        result = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "init-workspace"],
            env=_clean_env(str(tmp_path)),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        contract = tmp_path / "contract" / "storage_contract.json"
        assert contract.exists()
        data = json.loads(contract.read_text())
        assert data["storage_contract_version"] == 1
        assert data["created_by"] == "gpcr-tools"

        required_dirs = [
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
        for d in required_dirs:
            assert (tmp_path / d).is_dir(), f"Missing directory: {d}"

    def test_idempotent_run(self, tmp_path):
        """Running init-workspace twice must not corrupt the contract."""
        env = _clean_env(str(tmp_path))

        subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "init-workspace"],
            env=env,
            capture_output=True,
        )
        original = (tmp_path / "contract" / "storage_contract.json").read_text()

        result = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "init-workspace"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (tmp_path / "contract" / "storage_contract.json").read_text() == original

    def test_refuses_incompatible_contract_version(self, tmp_path):
        """init-workspace must exit non-zero if a future contract version exists."""
        (tmp_path / "contract").mkdir()
        (tmp_path / "contract" / "storage_contract.json").write_text(
            json.dumps({"storage_contract_version": 999})
        )
        result = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "init-workspace"],
            env=_clean_env(str(tmp_path)),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "999" in result.stderr
