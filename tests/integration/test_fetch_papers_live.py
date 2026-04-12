"""Integration tests for fetch-papers — LIVE API calls.

These tests actually call CrossRef, Unpaywall, and NCBI PMC APIs.
They use data from the canonical 9 PDB enriched fixtures.

Requires ``GPCR_EMAIL_FOR_APIS`` environment variable to be set.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests.conftest import REAL_PDB_DIR, REAL_PDB_IDS

_EMAIL = os.environ.get("GPCR_EMAIL_FOR_APIS", "")
_SKIP_REASON = "GPCR_EMAIL_FOR_APIS not set"


@pytest.fixture()
def papers_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up workspace with enriched data for papers testing."""
    from gpcr_tools.config import reset_config
    from gpcr_tools.workspace import init_workspace

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    if _EMAIL:
        monkeypatch.setenv("GPCR_EMAIL_FOR_APIS", _EMAIL)
    reset_config()
    init_workspace(workspace)

    # Copy enriched fixtures
    for pdb_id in REAL_PDB_IDS:
        src = REAL_PDB_DIR / "enriched" / f"{pdb_id}.json"
        if src.exists():
            shutil.copy2(src, workspace / "enriched" / f"{pdb_id}.json")

    reset_config()
    yield workspace
    reset_config()


class TestFetchPapersLive:
    """Live API test: download papers for canonical PDB set."""

    @pytest.mark.skipif(not _EMAIL, reason=_SKIP_REASON)
    def test_auto_only_downloads_some_papers(self, papers_workspace: Path) -> None:
        """Run fetch-papers --auto-only on all 9 PDBs."""
        from gpcr_tools.papers.runner import run_fetch_papers

        run_fetch_papers(auto_only=True)

        papers_dir = papers_workspace / "papers"
        pdfs = list(papers_dir.glob("*.pdf"))
        # We expect at least SOME PDFs to be downloaded (OA papers)
        # Not all 9 will succeed (some are paywalled)
        assert len(pdfs) > 0, "No PDFs downloaded at all"

        # Download log should exist
        log_path = papers_workspace / "state" / "download_log.json"
        assert log_path.exists()
        log = json.loads(log_path.read_text())
        assert len(log) == len(REAL_PDB_IDS), (
            f"Download log has {len(log)} entries, expected {len(REAL_PDB_IDS)}"
        )

        # Every entry should have a valid status
        valid_statuses = {
            "success_pdf_downloaded",
            "skipped_already_downloaded",
            "failed_no_doi",
            "failed_no_data",
            "fallback_paywalled",
        }
        for pdb_id, entry in log.items():
            assert entry["status"] in valid_statuses, (
                f"[{pdb_id}] unexpected status: {entry['status']}"
            )

    @pytest.mark.skipif(not _EMAIL, reason=_SKIP_REASON)
    def test_resumability(self, papers_workspace: Path) -> None:
        """Running fetch-papers twice should skip already-downloaded."""
        from gpcr_tools.papers.runner import run_fetch_papers

        # First run — single PDB
        run_fetch_papers(pdb_id="9IQS", auto_only=True)

        log_path = papers_workspace / "state" / "download_log.json"
        log1 = json.loads(log_path.read_text())

        # Second run — same PDB
        run_fetch_papers(pdb_id="9IQS", auto_only=True)

        log2 = json.loads(log_path.read_text())
        # Status should be skipped on second run
        if log1.get("9IQS", {}).get("status") == "success_pdf_downloaded":
            assert log2["9IQS"]["status"] == "skipped_already_downloaded"

    @pytest.mark.skipif(not _EMAIL, reason=_SKIP_REASON)
    def test_auto_discover_finds_missing(self, papers_workspace: Path) -> None:
        """Auto-discover should only find PDBs without papers."""
        from gpcr_tools.papers.runner import _discover_missing_papers

        missing = _discover_missing_papers()
        assert len(missing) == len(REAL_PDB_IDS)  # None have papers yet

        # Fake one paper existing
        (papers_workspace / "papers" / "9IQS.pdf").write_bytes(b"%PDF-1.4 fake")
        missing2 = _discover_missing_papers()
        assert "9IQS" not in missing2
        assert len(missing2) == len(REAL_PDB_IDS) - 1
