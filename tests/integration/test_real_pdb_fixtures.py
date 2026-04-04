"""Fixture integrity tests for committed real PDB data.

Verifies that all required fixture files exist in the repository, that
expected sidecars are present or explicitly absent, and that the
real_pdb_workspace fixture loads every selected PDB successfully.
"""

from pathlib import Path

import pytest

from tests.conftest import (
    REAL_PDB_DIR,
    REAL_PDB_IDS,
    REAL_PDB_VALIDATION_LOG_IDS,
    REAL_PDB_VOTING_LOG_IDS,
)


class TestFixtureFilesExist:
    """Verify committed fixture file existence and structure."""

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_main_json_exists(self, pdb_id: str) -> None:
        path = REAL_PDB_DIR / f"{pdb_id}.json"
        assert path.exists(), f"Missing main fixture: {path}"
        assert path.stat().st_size > 0, f"Empty main fixture: {path}"

    @pytest.mark.parametrize("pdb_id", sorted(REAL_PDB_VOTING_LOG_IDS))
    def test_voting_log_exists_where_expected(self, pdb_id: str) -> None:
        path = REAL_PDB_DIR / "logs" / f"{pdb_id}_voting_log.json"
        assert path.exists(), f"Missing expected voting log: {path}"

    @pytest.mark.parametrize(
        "pdb_id",
        sorted(set(REAL_PDB_IDS) - REAL_PDB_VOTING_LOG_IDS),
    )
    def test_voting_log_absent_where_expected(self, pdb_id: str) -> None:
        path = REAL_PDB_DIR / "logs" / f"{pdb_id}_voting_log.json"
        assert not path.exists(), f"Unexpected voting log found: {path}"

    @pytest.mark.parametrize("pdb_id", sorted(REAL_PDB_VALIDATION_LOG_IDS))
    def test_validation_log_exists(self, pdb_id: str) -> None:
        path = REAL_PDB_DIR / "validation_logs" / f"{pdb_id}_validation.json"
        assert path.exists(), f"Missing validation log: {path}"

    def test_no_unexpected_main_fixtures(self) -> None:
        """Ensure no stale or unaudited fixtures are checked in."""
        found_ids = {f.stem for f in REAL_PDB_DIR.glob("*.json")}
        expected_ids = set(REAL_PDB_IDS)
        assert found_ids == expected_ids, f"Unexpected fixture files: {found_ids - expected_ids}"

    def test_directory_structure(self) -> None:
        assert REAL_PDB_DIR.is_dir()
        assert (REAL_PDB_DIR / "logs").is_dir()
        assert (REAL_PDB_DIR / "validation_logs").is_dir()
        assert (REAL_PDB_DIR / "ai_results").is_dir()
        assert (REAL_PDB_DIR / "enriched").is_dir()
        assert (REAL_PDB_DIR / "cache").is_dir()

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_ai_results_exist(self, pdb_id: str) -> None:
        ai_dir = REAL_PDB_DIR / "ai_results" / pdb_id
        assert ai_dir.is_dir(), f"Missing AI results dir: {ai_dir}"
        runs = list(ai_dir.glob("run_*.json"))
        assert len(runs) >= 1, f"No run files for {pdb_id}"

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_enriched_exists(self, pdb_id: str) -> None:
        path = REAL_PDB_DIR / "enriched" / f"{pdb_id}.json"
        assert path.exists(), f"Missing enriched fixture: {path}"
        assert path.stat().st_size > 0, f"Empty enriched fixture: {path}"


class TestWorkspaceFixtureLoading:
    """Verify that the real_pdb_workspace fixture enables load_pdb_data."""

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_load_pdb_data_succeeds(self, pdb_id: str, real_pdb_workspace: Path) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        main_data, controversies, validation_data = load_pdb_data(pdb_id)

        assert main_data is not None, f"load_pdb_data returned None for {pdb_id}"
        assert isinstance(main_data, dict)
        assert isinstance(controversies, dict)
        assert isinstance(validation_data, dict)

    @pytest.mark.parametrize("pdb_id", sorted(REAL_PDB_VOTING_LOG_IDS))
    def test_controversy_map_nonempty_for_fixtures_with_voting_logs(
        self, pdb_id: str, real_pdb_workspace: Path
    ) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data(pdb_id)
        assert len(controversies) > 0, (
            f"{pdb_id} has a voting log but produced an empty controversy map"
        )

    @pytest.mark.parametrize(
        "pdb_id",
        sorted(set(REAL_PDB_IDS) - REAL_PDB_VOTING_LOG_IDS),
    )
    def test_controversy_map_empty_for_fixtures_without_voting_logs(
        self, pdb_id: str, real_pdb_workspace: Path
    ) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, controversies, _ = load_pdb_data(pdb_id)
        assert controversies == {}, (
            f"{pdb_id} has no voting log but produced a non-empty controversy map"
        )

    @pytest.mark.parametrize("pdb_id", REAL_PDB_IDS)
    def test_validation_data_has_expected_buckets(
        self, pdb_id: str, real_pdb_workspace: Path
    ) -> None:
        from gpcr_tools.csv_generator.data_loader import load_pdb_data

        _, _, validation_data = load_pdb_data(pdb_id)
        assert "critical_warnings" in validation_data or "algo_conflicts" in validation_data, (
            f"{pdb_id} validation_data missing both buckets"
        )
