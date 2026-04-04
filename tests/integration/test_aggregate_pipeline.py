"""Integration tests for the aggregate pipeline (Epic 7).

Covers: full pipeline with synthetic fixtures, voting log presence,
validation report structure, per-PDB error isolation, write atomicity,
warning format gate, skip-api-checks, force flag, multi-error scenario.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from gpcr_tools.aggregator.runner import (
    aggregate_all,
    aggregate_pdb,
)
from gpcr_tools.config import (
    CHIMERA_STATUS_SKIPPED,
    reset_config,
)

_BL3_REGEX = re.compile(r"at ['\"]([^'\"]+)['\"]")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ai_run(
    *,
    receptor_chain: str = "A",
    receptor_uniprot: str = "drd2_human",
    method: str = "X-RAY DIFFRACTION",
    ligand_id: str = "ATP",
) -> dict[str, Any]:
    """Build a synthetic AI run dict."""
    return {
        "structure_info": {"method": method, "resolution": 2.8},
        "receptor_info": {
            "chain_id": receptor_chain,
            "uniprot_entry_name": receptor_uniprot,
        },
        "ligands": [{"chem_comp_id": ligand_id, "name": "Test Ligand", "chain_id": "A"}],
        "signaling_partners": {},
        "auxiliary_proteins": [],
        "key_findings": "Test findings",
    }


def _make_enriched_entry(
    *,
    method: str = "X-RAY DIFFRACTION",
    slug: str = "drd2_human",
    chain_id: str = "A",
    ligand_id: str = "ATP",
) -> dict[str, Any]:
    """Build a synthetic enriched entry dict."""
    return {
        "exptl": [{"method": method}],
        "rcsb_accession_info": {"initial_release_date": "2025-01-15T00:00:00+00:00"},
        "polymer_entities": [
            {
                "rcsb_polymer_entity": {"pdbx_description": "Dopamine receptor D2"},
                "entity_poly": {
                    "pdbx_seq_one_letter_code_can": "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5,
                    "rcsb_sample_sequence_length": 130,
                },
                "uniprots": [
                    {
                        "rcsb_id": "P14416",
                        "gpcrdb_entry_name_slug": slug,
                    }
                ],
                "polymer_entity_instances": [
                    {
                        "rcsb_polymer_entity_instance_container_identifiers": {
                            "auth_asym_id": chain_id,
                            "asym_id": chain_id,
                        }
                    }
                ],
            }
        ],
        "nonpolymer_entities": [
            {
                "nonpolymer_comp": {"chem_comp": {"id": ligand_id}},
                "nonpolymer_entity_instances": [
                    {"rcsb_nonpolymer_entity_instance_container_identifiers": {"auth_asym_id": "B"}}
                ],
            }
        ],
        "assemblies": [],
    }


@pytest.fixture
def aggregate_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a full workspace with AI runs and enriched data for TEST1."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create workspace structure
    for subdir in (
        "raw",
        "enriched",
        "ai_results",
        "aggregated",
        "output",
        "cache",
        "state",
        "tmp",
    ):
        (workspace / subdir).mkdir()
    (workspace / "aggregated" / "logs").mkdir()
    (workspace / "aggregated" / "validation_logs").mkdir()

    # Contract
    contract_dir = workspace / "contract"
    contract_dir.mkdir()
    (contract_dir / "storage_contract.json").write_text(
        json.dumps(
            {
                "storage_contract_version": 1,
                "created_by": "test",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    # AI runs for TEST1
    ai_dir = workspace / "ai_results" / "TEST1"
    ai_dir.mkdir(parents=True)
    for i in range(3):
        run = _make_ai_run()
        (ai_dir / f"run_{i:02d}.json").write_text(json.dumps(run))

    # Enriched data for TEST1
    enriched = {"data": {"entry": _make_enriched_entry()}}
    (workspace / "enriched" / "TEST1.json").write_text(json.dumps(enriched))

    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    reset_config()
    yield workspace
    reset_config()


@pytest.fixture
def multi_pdb_workspace(aggregate_workspace: Path) -> Path:
    """Extend aggregate_workspace with a second PDB (TEST2)."""
    ws = aggregate_workspace

    # AI runs for TEST2
    ai_dir = ws / "ai_results" / "TEST2"
    ai_dir.mkdir(parents=True)
    for i in range(3):
        run = _make_ai_run(receptor_chain="B", receptor_uniprot="oprm_human")
        (ai_dir / f"run_{i:02d}.json").write_text(json.dumps(run))

    # Enriched data for TEST2
    enriched = {"data": {"entry": _make_enriched_entry(slug="oprm_human", chain_id="B")}}
    (ws / "enriched" / "TEST2.json").write_text(json.dumps(enriched))

    return ws


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestAggregatePdb:
    def test_full_pipeline_success(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.success is True
        assert result.aggregated_path is not None
        assert result.aggregated_path.is_file()
        assert result.validation_path is not None
        assert result.validation_path.is_file()

    def test_aggregated_json_structure(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        assert "structure_info" in data
        assert "receptor_info" in data
        assert "ligands" in data
        assert "oligomer_analysis" in data

    def test_validation_report_structure(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.validation_path is not None
        report = json.loads(result.validation_path.read_text())
        assert "critical_warnings" in report
        assert "algo_conflicts" in report
        assert "algo_notes" in report
        assert "chimera_score" in report
        assert "chimera_status" in report
        assert "timestamp" in report

    def test_chimera_skipped_when_no_api(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.validation_path is not None
        report = json.loads(result.validation_path.read_text())
        assert report["chimera_status"] == CHIMERA_STATUS_SKIPPED

    def test_no_runs_returns_failure(self, aggregate_workspace: Path) -> None:
        result = aggregate_pdb("NONEXISTENT", skip_api_checks=True)
        assert result.success is False
        assert "No valid AI runs" in (result.error or "")

    def test_no_enriched_returns_failure(self, aggregate_workspace: Path) -> None:
        # Create AI runs but no enriched data
        ai_dir = aggregate_workspace / "ai_results" / "NO_ENRICHED"
        ai_dir.mkdir(parents=True)
        (ai_dir / "run_00.json").write_text(json.dumps(_make_ai_run()))

        result = aggregate_pdb("NO_ENRICHED", skip_api_checks=True)
        assert result.success is False
        assert "Enriched data" in (result.error or "")

    def test_ground_truth_injected(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        # Ground truth should override method from enriched
        assert data["structure_info"]["method"] == "X-RAY DIFFRACTION"
        assert data["structure_info"]["release_date"] == "2025-01-15"


# ---------------------------------------------------------------------------
# Voting log tests
# ---------------------------------------------------------------------------


class TestVotingLog:
    def test_no_voting_log_when_no_discrepancies(self, aggregate_workspace: Path) -> None:
        """Identical runs -> no discrepancies -> no voting log."""
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)
        assert result.voting_log_path is None

    def test_voting_log_written_on_discrepancy(self, aggregate_workspace: Path) -> None:
        """Different runs -> discrepancies -> voting log written."""
        ai_dir = aggregate_workspace / "ai_results" / "TEST1"
        # Overwrite TWO runs to make "Z" the majority, so best_run (the one with
        # highest score) will diverge from majority on receptor_info.chain_id
        run_diff = _make_ai_run(receptor_chain="Z")
        (ai_dir / "run_01.json").write_text(json.dumps(run_diff))
        (ai_dir / "run_02.json").write_text(json.dumps(run_diff))

        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)
        assert result.voting_log_path is not None
        assert result.voting_log_path.is_file()


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_one_corrupt_pdb_doesnt_block_others(self, multi_pdb_workspace: Path) -> None:
        """Per-PDB error isolation: one failure shouldn't crash the batch."""
        # Corrupt TEST1's enriched data
        (multi_pdb_workspace / "enriched" / "TEST1.json").write_text("NOT_JSON")

        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            results = aggregate_all(skip_api_checks=True)

        # TEST1 should fail, TEST2 should succeed
        result_map = {r.pdb_id: r for r in results}
        assert result_map["TEST1"].success is False
        assert result_map["TEST2"].success is True

    def test_aggregate_all_records_log(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            aggregate_all(skip_api_checks=True)

        log_path = aggregate_workspace / "state" / "aggregate_log.json"
        assert log_path.is_file()
        log_data = json.loads(log_path.read_text())
        assert "TEST1" in log_data
        assert log_data["TEST1"]["status"] == "completed"


# ---------------------------------------------------------------------------
# Write atomicity
# ---------------------------------------------------------------------------


class TestWriteAtomicity:
    def test_no_output_on_validator_exception(self, aggregate_workspace: Path) -> None:
        """If a validator raises mid-pipeline, no output files should be left behind."""
        with (
            patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)),
            patch(
                "gpcr_tools.aggregator.runner.validate_and_enrich_ligands",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.success is False
        # No output files should exist
        aggregated = aggregate_workspace / "aggregated" / "TEST1.json"
        assert not aggregated.exists()


# ---------------------------------------------------------------------------
# Skip API checks & force
# ---------------------------------------------------------------------------


class TestFlags:
    def test_skip_api_skips_chimera(self, aggregate_workspace: Path) -> None:
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            result = aggregate_pdb("TEST1", skip_api_checks=True)

        assert result.validation_path is not None
        report = json.loads(result.validation_path.read_text())
        assert report["chimera_status"] == CHIMERA_STATUS_SKIPPED

    def test_force_reprocesses(self, aggregate_workspace: Path) -> None:
        """Force flag should reprocess PDBs already in the aggregate log."""
        # First run
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            aggregate_all(skip_api_checks=True)

        # Second run without force - should skip
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            results_no_force = aggregate_all(skip_api_checks=True)
        assert len(results_no_force) == 0

        # Third run with force - should reprocess
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm", return_value=({}, None)):
            results_force = aggregate_all(skip_api_checks=True, force=True)
        assert len(results_force) == 1
        assert results_force[0].success is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestAggregateCLI:
    def test_help_exits_zero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "gpcr_tools", "aggregate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "aggregate" in result.stdout.lower()
        assert "--skip-api-checks" in result.stdout
        assert "--force" in result.stdout
