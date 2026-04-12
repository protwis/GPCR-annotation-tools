"""Tests for papers/downloader.py — multi-tier PDF download logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gpcr_tools.papers.downloader import (
    _fetch_crossref_metadata,
    _fetch_unpaywall_pdf_url,
    _read_download_log,
    _update_download_log,
    download_paper_for_pdb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENRICHED_DATA: dict[str, Any] = {
    "data": {
        "entry": {
            "rcsb_id": "7W55",
            "rcsb_primary_citation": {
                "pdbx_database_id_DOI": "10.1038/s41586-022-04958-8",
            },
            "rcsb_entry_container_identifiers": {"pubmed_id": 12345},
            "pubmed": {"rcsb_pubmed_central_id": "PMC789"},
        }
    }
}


@pytest.fixture()
def papers_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a workspace for papers testing."""
    from gpcr_tools.config import reset_config

    workspace = tmp_path
    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    monkeypatch.setenv("GPCR_EMAIL_FOR_APIS", "test@example.com")
    reset_config()

    # Create enriched data
    enriched_dir = workspace / "enriched"
    enriched_dir.mkdir(parents=True)
    (enriched_dir / "7W55.json").write_text(json.dumps(_ENRICHED_DATA))

    # Create necessary dirs
    (workspace / "papers").mkdir()
    (workspace / "state").mkdir()
    (workspace / "cache").mkdir()

    yield workspace
    reset_config()


# ---------------------------------------------------------------------------
# Download log
# ---------------------------------------------------------------------------


class TestDownloadLog:
    def test_read_empty_log(self, papers_workspace: Path) -> None:
        log = _read_download_log()
        assert log == {}

    def test_write_and_read_log(self, papers_workspace: Path) -> None:
        _update_download_log("7W55", {"status": "success_pdf_downloaded"})
        log = _read_download_log()
        assert "7W55" in log
        assert log["7W55"]["status"] == "success_pdf_downloaded"

    def test_atomic_update_preserves_existing(self, papers_workspace: Path) -> None:
        _update_download_log("7W55", {"status": "success_pdf_downloaded"})
        _update_download_log("8ABC", {"status": "fallback_paywalled"})
        log = _read_download_log()
        assert "7W55" in log
        assert "8ABC" in log


# ---------------------------------------------------------------------------
# Tier API functions
# ---------------------------------------------------------------------------


class TestCrossRefMetadata:
    def test_extracts_pmid(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"PMID": "12345", "link": []}}
        mock_session.get.return_value = mock_resp

        result = _fetch_crossref_metadata("10.1038/test", mock_session)
        assert result["pmid"] == "12345"

    def test_handles_api_failure(self) -> None:
        mock_session = MagicMock()
        import requests

        mock_session.get.side_effect = requests.exceptions.ConnectionError("timeout")

        result = _fetch_crossref_metadata("10.1038/test", mock_session)
        assert result["pmid"] is None


class TestUnpaywallPdfUrl:
    def test_returns_pdf_url(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}
        }
        mock_session.get.return_value = mock_resp

        result = _fetch_unpaywall_pdf_url("10.1038/test", mock_session)
        assert result == "https://example.com/paper.pdf"

    def test_returns_none_when_no_oa(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"best_oa_location": None}
        mock_session.get.return_value = mock_resp

        result = _fetch_unpaywall_pdf_url("10.1038/test", mock_session)
        assert result is None


# ---------------------------------------------------------------------------
# download_paper_for_pdb
# ---------------------------------------------------------------------------


class TestDownloadPaperForPdb:
    def test_skips_if_enriched_missing(self, papers_workspace: Path) -> None:
        result = download_paper_for_pdb("XXXX", email="test@example.com")
        assert result["status"] == "skipped_no_enriched_data"

    def test_skips_if_pdf_exists(self, papers_workspace: Path) -> None:
        (papers_workspace / "papers" / "7W55.pdf").write_text("fake pdf")
        result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "skipped_already_downloaded"

    @patch(
        "gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url",
        return_value="https://example.com/7W55.pdf",
    )
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": "12345", "pmcid": "PMC789"},
    )
    def test_success_downloads_pdf(
        self,
        _mock_cr: MagicMock,
        _mock_up: MagicMock,
        papers_workspace: Path,
    ) -> None:
        # Mock _download_file to actually create the temp file
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.4 fake content")
            return True

        with patch(
            "gpcr_tools.papers.downloader._download_file",
            side_effect=fake_download,
        ):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "unpaywall_pdf"

    @patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None)
    @patch("gpcr_tools.papers.downloader._fetch_pmc_oa_pdf_url", return_value=None)
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": None, "pmcid": None},
    )
    @patch(
        "gpcr_tools.papers.downloader._fetch_abstract_from_ncbi",
        return_value=None,
    )
    def test_fallback_paywalled(
        self,
        _mock_abs: MagicMock,
        _mock_cr: MagicMock,
        _mock_pmc: MagicMock,
        _mock_up: MagicMock,
        papers_workspace: Path,
    ) -> None:
        result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "fallback_paywalled"
