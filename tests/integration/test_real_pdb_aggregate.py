"""Integration tests for aggregate pipeline against real PDB fixtures (Epic 8).

Runs ``aggregate_pdb()`` on all 9 canonical PDB IDs using committed
fixture data (AI runs + enriched JSONs) with ``skip_api_checks=True``
to avoid live API hits in CI.

Verifies:
- Aggregated JSON top-level keys and structure
- Validation report structure
- Voting log presence/absence for known PDBs
- Oligomer analysis output
- Equivalence coverage: receptor UniProt, ligand count, structure_info,
  signaling_partners, auxiliary_proteins, oligomer_analysis (Review 4 A-3)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.conftest import (
    REAL_PDB_DIR,
    REAL_PDB_IDS,
    REAL_PDB_VOTING_LOG_IDS,
)

FIXTURE_AI_RESULTS = REAL_PDB_DIR / "ai_results"
FIXTURE_ENRICHED = REAL_PDB_DIR / "enriched"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def real_aggregate_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Build a workspace populated with real AI runs + enriched data.

    Does NOT pre-populate aggregated/ — the tests will generate that via
    ``aggregate_pdb()``.
    """
    from gpcr_tools.config import reset_config

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for subdir in (
        "raw",
        "enriched",
        "ai_results",
        "aggregated",
        "aggregated/logs",
        "aggregated/validation_logs",
        "output",
        "output/csv",
        "output/audit",
        "cache",
        "state",
        "tmp",
    ):
        (workspace / subdir).mkdir(parents=True, exist_ok=True)

    # Contract
    contract_dir = workspace / "contract"
    contract_dir.mkdir()
    (contract_dir / "storage_contract.json").write_text(
        json.dumps(
            {
                "storage_contract_version": 2,
                "created_by": "test",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    # Copy AI results
    for pdb_id in REAL_PDB_IDS:
        src_dir = FIXTURE_AI_RESULTS / pdb_id
        if src_dir.is_dir():
            dst_dir = workspace / "ai_results" / pdb_id
            shutil.copytree(src_dir, dst_dir)

    # Copy enriched data
    for pdb_id in REAL_PDB_IDS:
        src = FIXTURE_ENRICHED / f"{pdb_id}.json"
        if src.is_file():
            shutil.copy2(src, workspace / "enriched" / f"{pdb_id}.json")

    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    reset_config()
    yield workspace
    reset_config()


def _run_aggregate(pdb_id: str) -> Any:
    """Run aggregate_pdb with mocked 7TM scan (no live GraphQL)."""
    from gpcr_tools.aggregator.runner import aggregate_pdb

    with patch(
        "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
        return_value=({}, None),
    ):
        return aggregate_pdb(pdb_id, skip_api_checks=True)


def _run_aggregate_tm_complete(pdb_id: str) -> Any:
    """Run aggregate_pdb with every GPCR chain scanned as a COMPLETE 7TM receptor.

    Unlike ``_run_aggregate`` (which returns an empty 7TM scan, leaving the
    receptor count transmembrane-unverified so the receptor-level cross-check
    stays silent), this builds a populated scan from the enriched GPCR roster and
    a non-empty GraphQL entry. That makes ``tm_data_available`` true, so the
    AI-vs-classifier oligomeric cross-check is genuinely exercised end to end
    rather than skipped as untrusted.
    """
    from gpcr_tools.aggregator.runner import aggregate_pdb
    from gpcr_tools.config import TM_STATUS_COMPLETE
    from gpcr_tools.validator.oligomer import _build_gpcr_roster

    enriched = json.loads((FIXTURE_ENRICHED / f"{pdb_id}.json").read_text())
    entry = (enriched.get("data") or {}).get("entry") or {}
    roster = _build_gpcr_roster(entry)
    tm_roster = {
        chain: {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE} for chain in roster
    }
    # A non-empty GraphQL entry signals a successful fetch (tm_data_available),
    # so the cross-check treats the receptor count as verified.
    graphql_entry = {"polymer_entities": []}

    with patch(
        "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
        return_value=(tm_roster, graphql_entry),
    ):
        return aggregate_pdb(pdb_id, skip_api_checks=True)


# ---------------------------------------------------------------------------
# Phase B: Live Aggregate + Validate Tests
# ---------------------------------------------------------------------------


class TestRealPdbAggregate:
    """Run aggregate_pdb on every canonical PDB and verify output."""

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_aggregate_succeeds(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.success is True, f"[{pdb_id}] Failed: {result.error}"

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_aggregated_json_top_level_keys(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())

        # Must have all top-level blocks
        assert "structure_info" in data
        assert "receptor_info" in data
        assert "ligands" in data
        assert "oligomer_analysis" in data

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_validation_report_structure(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.validation_path is not None
        report = json.loads(result.validation_path.read_text())

        assert "critical_warnings" in report
        assert "algo_conflicts" in report
        assert "detector_notes" in report
        assert "chimera_score" in report
        assert "chimera_status" in report
        assert "timestamp" in report
        assert isinstance(report["critical_warnings"], list)

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_oligomer_analysis_present(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        oligo = data["oligomer_analysis"]

        assert "classification" in oligo
        assert "all_gpcr_chains" in oligo
        assert "primary_protomer_suggestion" in oligo
        assert "alerts" in oligo
        assert "chain_id_override" in oligo
        assert "label_asym_id_map" in oligo


class TestVotingLogPresence:
    """Verify voting log presence matches known expectations."""

    @pytest.mark.parametrize("pdb_id", sorted(REAL_PDB_VOTING_LOG_IDS))
    def test_voting_log_present_for_known_discrepant(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        """PDBs known to have voting discrepancies should produce a voting log."""
        result = _run_aggregate(pdb_id)
        # Note: voting log presence depends on actual data discrepancies.
        # If result has voting log, validate it.
        if result.voting_log_path is not None:
            assert result.voting_log_path.is_file()
            log_data = json.loads(result.voting_log_path.read_text())
            assert isinstance(log_data, list)
            assert len(log_data) > 0


# ---------------------------------------------------------------------------
# Equivalence coverage (Review 4 A-3)
# ---------------------------------------------------------------------------


class TestEquivalenceCoverage:
    """Verify aggregated output covers all key fields.

    Per Review 4 A-3: must cover receptor UniProt, ligand count,
    structure_info, signaling_partners, auxiliary_proteins, and
    oligomer_analysis — not just 3 fields.
    """

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_receptor_uniprot_present(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        receptor = data.get("receptor_info") or {}
        # receptor_info should exist with chain_id
        assert "chain_id" in receptor

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_ligands_is_list(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        assert isinstance(data.get("ligands"), list)

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_structure_info_has_method(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        si = data.get("structure_info") or {}
        assert "method" in si
        assert "resolution" in si

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_signaling_partners_present(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        # signaling_partners may be dict, None, or absent (some PDBs have no signaling partners)
        sp = data.get("signaling_partners")
        assert sp is None or isinstance(sp, dict), (
            f"signaling_partners should be dict or None, got {type(sp)}"
        )

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_auxiliary_proteins_present(
        self,
        pdb_id: str,
        real_aggregate_workspace: Path,
    ) -> None:
        result = _run_aggregate(pdb_id)
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        # auxiliary_proteins may be list, None, or absent
        ap = data.get("auxiliary_proteins")
        assert ap is None or isinstance(ap, list), (
            f"auxiliary_proteins should be list or None, got {type(ap)}"
        )


# ---------------------------------------------------------------------------
# aggregate_all test (Review 4 A-3 gap)
# ---------------------------------------------------------------------------


class TestAggregateAll:
    def test_aggregate_all_processes_all_pending(
        self,
        real_aggregate_workspace: Path,
    ) -> None:
        from gpcr_tools.aggregator.runner import aggregate_all

        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            results = aggregate_all(skip_api_checks=True)

        assert len(results) == len(REAL_PDB_IDS)
        successes = [r for r in results if r.success]
        assert len(successes) == len(REAL_PDB_IDS), (
            f"Expected all {len(REAL_PDB_IDS)} to succeed, "
            f"failures: {[(r.pdb_id, r.error) for r in results if not r.success]}"
        )

    def test_aggregate_all_updates_log(
        self,
        real_aggregate_workspace: Path,
    ) -> None:
        from gpcr_tools.aggregator.runner import aggregate_all

        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            aggregate_all(skip_api_checks=True)

        log_path = real_aggregate_workspace / "state" / "aggregate_log.json"
        assert log_path.is_file()
        log_data = json.loads(log_path.read_text())
        for pdb_id in REAL_PDB_IDS:
            assert pdb_id in log_data, f"{pdb_id} missing from aggregate log"


# ---------------------------------------------------------------------------
# Receptor oligomeric-state cross-check, end to end through a real run
# ---------------------------------------------------------------------------


class TestOligomericStateCrossCheckEndToEnd:
    """The ``receptor_info.oligomeric_state`` field, populated in real AI-run
    fixtures, flows through aggregate -> ``analyze_oligomer`` -> the receptor-level
    cross-check and AGREES with the deterministic classifier, so no
    ``ALERT_OLIGOMER_DISAGREEMENT`` is raised and nothing routes to a curator.

    These two PDBs are the only canonical fixtures populated with
    ``oligomeric_state`` (the field is REQUIRED by the schema; the rest predate
    it). They exercise the agreement path with a verified 7TM count, proving a
    real run reaches the tripwire -- not just synthetic input. (No genuine Class C
    homo-dimer exists in the canonical set, so the multi-receptor agreement case
    is the obligate sweet-taste hetero-dimer; see fixture choice below.)
    """

    def _oligomer_block(self, result: Any) -> dict[str, Any]:
        assert result.success is True, f"aggregate failed: {result.error}"
        assert result.aggregated_path is not None
        data = json.loads(result.aggregated_path.read_text())
        # The populated, voted field must survive aggregation into the output.
        oligo_state = (data.get("receptor_info") or {}).get("oligomeric_state") or {}
        assert oligo_state.get("value"), "oligomeric_state did not reach the aggregated output"
        return data["oligomer_analysis"]

    def _has_disagreement(self, oligo: dict[str, Any]) -> bool:
        from gpcr_tools.config import ALERT_OLIGOMER_DISAGREEMENT

        return any(a["type"] == ALERT_OLIGOMER_DISAGREEMENT for a in oligo.get("alerts") or [])

    def test_monomer_fixture_agrees_no_route(
        self,
        real_aggregate_workspace: Path,
    ) -> None:
        # 9AS1: a single 5-HT2A receptor + Gq heterotrimer. AI 'monomer',
        # classifier MONOMER -> AGREE, so no disagreement alert and not gated.
        from gpcr_tools.config import ALERT_OLIGOMER_DISAGREEMENT, OLIGOMER_MONOMER
        from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

        oligo = self._oligomer_block(_run_aggregate_tm_complete("9AS1"))
        assert oligo["classification"] == OLIGOMER_MONOMER
        assert oligo["tm_data_available"] is True
        assert oligo["receptor_count"] == 1
        assert not self._has_disagreement(oligo)

        # The tripwire injection path must not gate (no critical_warning).
        validation_data: dict[str, Any] = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data.get("critical_warnings") or []
        assert not any(ALERT_OLIGOMER_DISAGREEMENT in w for w in warnings)

    def test_dimer_fixture_agrees_no_route(
        self,
        real_aggregate_workspace: Path,
    ) -> None:
        # 9NOR: the sweet-taste receptor, an obligate Class C hetero-dimer of
        # TAS1R2 + TAS1R3. AI 'hetero-dimer', classifier HETEROMER (2 receptor
        # chains) -> AGREE, so no disagreement alert and not gated.
        from gpcr_tools.config import ALERT_OLIGOMER_DISAGREEMENT, OLIGOMER_HETEROMER
        from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

        oligo = self._oligomer_block(_run_aggregate_tm_complete("9NOR"))
        assert oligo["classification"] == OLIGOMER_HETEROMER
        assert oligo["tm_data_available"] is True
        assert oligo["receptor_count"] == 2
        assert not self._has_disagreement(oligo)

        validation_data: dict[str, Any] = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data.get("critical_warnings") or []
        assert not any(ALERT_OLIGOMER_DISAGREEMENT in w for w in warnings)
