"""Integration tests for fetch pipeline — LIVE API calls.

These tests actually call RCSB, UniProt, PubChem, and RCSB Search APIs.
They use the canonical 9 PDB IDs from the test fixture set.

Run with:
    pytest tests/integration/test_fetch_live.py -v

These tests require network access and may take 2-5 minutes due to
API rate limiting (1s sleep per RCSB request).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import REAL_PDB_IDS


@pytest.fixture()
def fetch_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a workspace for live fetch testing."""
    from gpcr_tools.config import reset_config
    from gpcr_tools.workspace import init_workspace

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    reset_config()
    init_workspace(workspace)
    reset_config()

    yield workspace
    reset_config()


# ---------------------------------------------------------------------------
# Single-PDB live fetch + enrich
# ---------------------------------------------------------------------------


class TestFetchSingleLive:
    """Fetch a single PDB via live API and verify output structure."""

    def test_rcsb_download(self, fetch_workspace: Path) -> None:
        """Download raw JSON for one PDB from RCSB."""
        from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

        data = fetch_single_pdb("7W55")

        assert data is not None
        assert "data" in data
        assert "entry" in data["data"]
        assert data["data"]["entry"]["rcsb_id"] == "7W55"

        raw_file = fetch_workspace / "raw" / "pdb_json" / "7W55.json"
        assert raw_file.exists()

    def test_rcsb_download_invalid_pdb(self, fetch_workspace: Path) -> None:
        """Invalid PDB ID should return None (not crash)."""
        from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

        data = fetch_single_pdb("ZZZZ")
        # RCSB returns data with entry=null for invalid IDs
        # This should not crash
        assert data is None or data.get("data", {}).get("entry") is None


class TestEnrichSingleLive:
    """Enrich a single PDB via live APIs and verify enriched output."""

    def test_enrich_produces_enriched_json(self, fetch_workspace: Path) -> None:
        """Full fetch + enrich for one PDB."""
        from gpcr_tools.fetcher.enricher import enrich_single_pdb
        from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

        # Step 1: Download
        raw_data = fetch_single_pdb("7W55")
        assert raw_data is not None

        # Step 2: Enrich
        success = enrich_single_pdb("7W55")
        assert success is True

        enriched_file = fetch_workspace / "enriched" / "7W55.json"
        assert enriched_file.exists()

        data = json.loads(enriched_file.read_text())
        entry = data["data"]["entry"]

        # UniProt slug should be present on at least one polymer
        polymers = entry.get("polymer_entities") or []
        assert len(polymers) > 0
        slugs_found = []
        for poly in polymers:
            for uni in poly.get("uniprots") or []:
                slug = uni.get("gpcrdb_entry_name_slug")
                if slug:
                    slugs_found.append(slug)
        assert len(slugs_found) > 0, "No UniProt slugs found after enrichment"

        # Sibling PDBs should be a list
        assert isinstance(entry.get("sibling_pdbs"), list)

        # Nonpolymers should have gpcrdb_determined_type
        non_polymers = entry.get("nonpolymer_entities") or []
        for np_entity in non_polymers:
            comp = np_entity.get("nonpolymer_comp") or {}
            assert "gpcrdb_determined_type" in comp


# ---------------------------------------------------------------------------
# Full canonical set: fetch + enrich all 9 PDBs
# ---------------------------------------------------------------------------


class TestFetchCanonicalSetLive:
    """Run fetch + enrich on all 9 canonical PDB IDs via live APIs.

    This test writes a targets.txt with all 9 IDs and uses the
    runner to process them. Verifies output structure for each.
    """

    def test_full_pipeline_all_9(self, fetch_workspace: Path) -> None:
        """Fetch + enrich all 9 canonical PDBs."""
        from gpcr_tools.fetcher.runner import run_fetch

        # Write targets file
        targets_file = fetch_workspace / "targets.txt"
        targets_file.write_text(
            "\n".join(REAL_PDB_IDS) + "\n",
            encoding="utf-8",
        )

        # Run fetch pipeline
        run_fetch()

        # Verify all outputs exist
        for pdb_id in REAL_PDB_IDS:
            raw_file = fetch_workspace / "raw" / "pdb_json" / f"{pdb_id}.json"
            enriched_file = fetch_workspace / "enriched" / f"{pdb_id}.json"

            assert raw_file.exists(), f"Missing raw JSON for {pdb_id}"
            assert enriched_file.exists(), f"Missing enriched JSON for {pdb_id}"

            # Basic structure check
            data = json.loads(enriched_file.read_text())
            entry = (data.get("data") or {}).get("entry") or {}
            assert entry.get("rcsb_id") == pdb_id, (
                f"[{pdb_id}] rcsb_id mismatch: {entry.get('rcsb_id')}"
            )
            assert isinstance(entry.get("sibling_pdbs"), list), (
                f"[{pdb_id}] sibling_pdbs not a list"
            )

    def test_resumability(self, fetch_workspace: Path) -> None:
        """Running fetch twice should skip already-done PDBs."""
        from gpcr_tools.fetcher.runner import run_fetch

        targets_file = fetch_workspace / "targets.txt"
        targets_file.write_text("7W55\n", encoding="utf-8")

        # First run
        run_fetch()
        enriched = fetch_workspace / "enriched" / "7W55.json"
        assert enriched.exists()
        mtime1 = enriched.stat().st_mtime

        # Second run — should skip
        run_fetch()
        mtime2 = enriched.stat().st_mtime
        assert mtime1 == mtime2, "File was rewritten despite already existing"

    def test_caches_populated(self, fetch_workspace: Path) -> None:
        """Caches should be created after fetching."""
        from gpcr_tools.fetcher.runner import run_fetch

        targets_file = fetch_workspace / "targets.txt"
        targets_file.write_text("7W55\n", encoding="utf-8")

        run_fetch()

        cache_dir = fetch_workspace / "cache"
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) > 0, "No cache files created"


# ---------------------------------------------------------------------------
# Enriched fixture equivalence test
# ---------------------------------------------------------------------------


class TestEnrichedEquivalence:
    """Compare live-fetched enriched data with the committed fixture.

    This ensures the fetch pipeline produces output that is structurally
    compatible with the enriched fixtures used by the aggregation tests.
    We compare KEY FIELDS only (not the full JSON, since API data evolves).
    """

    def test_enriched_keys_match_fixture(self, fetch_workspace: Path) -> None:
        """Key enrichment fields should match the committed fixtures."""
        from gpcr_tools.fetcher.enricher import enrich_single_pdb
        from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb
        from tests.conftest import REAL_PDB_DIR

        pdb_id = "9IQS"  # Simpler PDB — fewer API calls

        # Fetch + enrich live
        fetch_single_pdb(pdb_id)
        enrich_single_pdb(pdb_id)

        live_file = fetch_workspace / "enriched" / f"{pdb_id}.json"
        fixture_file = REAL_PDB_DIR / "enriched" / f"{pdb_id}.json"

        assert live_file.exists()
        assert fixture_file.exists()

        live = json.loads(live_file.read_text())
        fixture = json.loads(fixture_file.read_text())

        live_entry = (live.get("data") or {}).get("entry") or {}
        fix_entry = (fixture.get("data") or {}).get("entry") or {}

        # Same PDB ID
        assert live_entry.get("rcsb_id") == fix_entry.get("rcsb_id")

        # Same polymer count
        live_polys = live_entry.get("polymer_entities") or []
        fix_polys = fix_entry.get("polymer_entities") or []
        assert len(live_polys) == len(fix_polys), (
            f"Polymer count mismatch: live={len(live_polys)}, fixture={len(fix_polys)}"
        )

        # Same nonpolymer count
        live_nps = live_entry.get("nonpolymer_entities") or []
        fix_nps = fix_entry.get("nonpolymer_entities") or []
        assert len(live_nps) == len(fix_nps), (
            f"Nonpolymer count mismatch: live={len(live_nps)}, fixture={len(fix_nps)}"
        )

        # Assembly count
        live_asm = live_entry.get("assemblies") or []
        fix_asm = fix_entry.get("assemblies") or []
        assert len(live_asm) == len(fix_asm), (
            f"Assembly count mismatch: live={len(live_asm)}, fixture={len(fix_asm)}"
        )
