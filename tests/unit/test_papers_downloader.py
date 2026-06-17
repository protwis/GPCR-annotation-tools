"""Tests for papers/downloader.py — multi-tier PDF download logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gpcr_tools.papers.downloader import (
    _fetch_crossref_metadata,
    _fetch_pmc_s3_pdf_url,
    _fetch_unpaywall_pdf_url,
    _read_download_log,
    _resolve_pmcid,
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

    def test_extracts_direct_pdf_link(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "PMID": "999",
                "link": [
                    {
                        "URL": "https://www.nature.com/articles/x.pdf",
                        "content-type": "application/pdf",
                    }
                ],
            }
        }
        mock_session.get.return_value = mock_resp
        result = _fetch_crossref_metadata("10.1038/x", mock_session)
        assert result["crossref_pdf_url"] == "https://www.nature.com/articles/x.pdf"

    def test_does_not_extract_pmcid_from_link(self) -> None:
        # Regression: a PMC URL in CrossRef link[] (often a *reference's* PMC
        # article) must NOT be harvested as this paper's PMCID -- that injected a
        # wrong PMCID that could fetch the WRONG paper. PMCID now comes only from
        # the authoritative ID Converter.
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "PMID": "999",
                "link": [{"URL": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5864293/"}],
            }
        }
        mock_session.get.return_value = mock_resp
        result = _fetch_crossref_metadata("10.1038/x", mock_session)
        assert "pmcid" not in result  # no PMCID is harvested from links anymore


class TestPmcS3PdfUrl:
    def test_returns_url_when_metadata_has_pdf(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"pdf_url": "PMC789.1/PMC789.1.pdf"}
        mock_session.get.return_value = mock_resp
        result = _fetch_pmc_s3_pdf_url("PMC789", mock_session)
        assert result == "https://pmc-oa-opendata.s3.amazonaws.com/PMC789.1/PMC789.1.pdf"

    def test_none_when_metadata_has_no_pdf(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"pdf_url": None}
        mock_session.get.return_value = mock_resp
        assert _fetch_pmc_s3_pdf_url("789", mock_session) is None

    def test_none_on_404(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_session.get.return_value = mock_resp
        assert _fetch_pmc_s3_pdf_url("PMC404", mock_session) is None


class TestResolvePmcid:
    def test_resolves_from_doi_first(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": [{"requested-id": "10.1/x", "pmcid": "PMC987"}]}
        mock_session.get.return_value = mock_resp
        with patch("gpcr_tools.papers.downloader.time.sleep"):
            assert _resolve_pmcid("10.1/x", "12345", mock_session) == "PMC987"
        # DOI is queried first; a hit means PMID is never tried.
        assert mock_session.get.call_count == 1

    def test_falls_back_to_pmid(self) -> None:
        mock_session = MagicMock()
        no_pmc = MagicMock(status_code=200)
        no_pmc.json.return_value = {"records": [{"requested-id": "10.1/x"}]}  # no pmcid
        hit = MagicMock(status_code=200)
        hit.json.return_value = {"records": [{"requested-id": "12345", "pmcid": "PMC987"}]}
        mock_session.get.side_effect = [no_pmc, hit]
        with patch("gpcr_tools.papers.downloader.time.sleep"):
            assert _resolve_pmcid("10.1/x", "12345", mock_session) == "PMC987"
        assert mock_session.get.call_count == 2

    def test_none_when_neither_resolves(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"records": [{"requested-id": "10.1/x"}]}
        mock_session.get.return_value = mock_resp
        with patch("gpcr_tools.papers.downloader.time.sleep"):
            assert _resolve_pmcid("10.1/x", "12345", mock_session) is None

    def test_ignores_pmcid_when_requested_id_mismatches(self) -> None:
        # Echo guard: a record whose requested-id does NOT match the queried id is
        # not trusted (defends against a future batched query cross-binding).
        mock_session = MagicMock()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "records": [{"requested-id": "10.9/other", "pmcid": "PMC987"}]
        }
        mock_session.get.return_value = mock_resp
        with patch("gpcr_tools.papers.downloader.time.sleep"):
            assert _resolve_pmcid("10.1/x", None, mock_session) is None

    def test_none_when_no_identifiers(self) -> None:
        mock_session = MagicMock()
        assert _resolve_pmcid(None, None, mock_session) is None
        mock_session.get.assert_not_called()


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

    def test_skip_exists_preserves_prior_identifiers(self, papers_workspace: Path) -> None:
        """Re-running when the PDF already exists must keep the DOI a prior run
        resolved -- same-paper sibling grouping in the manual workflow depends on
        it surviving across runs (the log is full-replace)."""
        from gpcr_tools.papers.downloader import _read_download_log, _update_download_log

        _update_download_log(
            "7W55",
            {"status": "success_pdf_downloaded", "doi": "10.1/keep", "pmid": "999", "pmcid": None},
        )
        (papers_workspace / "papers" / "7W55.pdf").write_text("fake pdf")
        result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "skipped_already_downloaded"
        assert result["doi"] == "10.1/keep"  # not nulled
        assert _read_download_log()["7W55"]["doi"] == "10.1/keep"

    @patch(
        "gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url",
        return_value="https://example.com/7W55.pdf",
    )
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": "12345", "crossref_pdf_url": None},
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
        # The PDF is saved to the canonical DOI-named file (one per paper).
        from gpcr_tools.config import sanitize_doi

        canonical = (
            papers_workspace
            / "papers"
            / f"{sanitize_doi(_ENRICHED_DATA['data']['entry']['rcsb_primary_citation']['pdbx_database_id_DOI'])}.pdf"
        )
        assert canonical.is_file()
        assert result["file_path"] == str(canonical)

    @patch(
        "gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url",
        return_value="https://example.com/unpaywall.pdf",
    )
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={
            "pmid": None,
            "crossref_pdf_url": "https://publisher.example/challenge.pdf",
        },
    )
    def test_chain_continues_when_resolved_url_is_not_a_pdf(
        self, _mock_cr: MagicMock, _mock_up: MagicMock, papers_workspace: Path
    ) -> None:
        # The first candidate (crossref_pdf) resolves to a URL but yields a
        # non-PDF (HTML bot challenge); the chain must NOT stop there -- it falls
        # through to Unpaywall and succeeds. This is the load-bearing behaviour
        # the rewrite exists for.
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            if "challenge" in url:
                return False
            output_path.write_bytes(b"%PDF-1.5 real content")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "unpaywall_pdf"

    @patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None)
    @patch("gpcr_tools.papers.downloader._resolve_pmcid", return_value=None)
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": None, "crossref_pdf_url": None},
    )
    def test_fallback_paywalled(
        self,
        _mock_cr: MagicMock,
        _mock_resolve: MagicMock,
        _mock_up: MagicMock,
        papers_workspace: Path,
    ) -> None:
        result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "fallback_paywalled"

    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={
            "pmid": None,
            "crossref_pdf_url": "https://www.nature.com/articles/x.pdf",
        },
    )
    def test_crossref_pdf_link_is_first_choice(
        self, _mock_cr: MagicMock, papers_workspace: Path
    ) -> None:
        # The direct CrossRef PDF link is tried before Unpaywall/PMC and, when it
        # yields a real PDF, wins -- this is the largest DOI-only recovery path.
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.5 nature")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "crossref_pdf"

    @patch(
        "gpcr_tools.papers.downloader._fetch_pmc_s3_pdf_url",
        return_value="https://pmc-oa-opendata.s3.amazonaws.com/PMC789.1/PMC789.1.pdf",
    )
    @patch("gpcr_tools.papers.downloader._resolve_pmcid", return_value="PMC789")
    @patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None)
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": "12345", "crossref_pdf_url": None},
    )
    def test_falls_through_to_pmc_s3(
        self,
        _mock_cr: MagicMock,
        _mock_up: MagicMock,
        _mock_resolve: MagicMock,
        _mock_s3: MagicMock,
        papers_workspace: Path,
    ) -> None:
        # crossref-pdf absent, Unpaywall empty -> the chain reaches the PMC S3 tier,
        # whose PMCID is resolved authoritatively (not from an unverified field).
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.7 pmc")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "pmc_s3_pdf"
        assert result["pmcid"] == "PMC789"  # the authoritative, resolved PMCID is logged

    @patch(
        "gpcr_tools.papers.downloader._fetch_pmc_s3_pdf_url",
        return_value="https://pmc-oa-opendata.s3.amazonaws.com/PMC789.1/PMC789.1.pdf",
    )
    @patch("gpcr_tools.papers.downloader._resolve_pmcid", return_value="PMC789")
    @patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None)
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": "12345", "crossref_pdf_url": None},
    )
    def test_enriched_pmcid_is_ignored_only_resolved_one_used(
        self,
        _mock_cr: MagicMock,
        _mock_up: MagicMock,
        _mock_resolve: MagicMock,
        _mock_s3: MagicMock,
        papers_workspace: Path,
    ) -> None:
        # Regression: the (possibly WRONG) enriched rcsb_pubmed_central_id must NOT
        # be used -- only the authoritatively-resolved PMCID. Here enriched carries
        # a WRONG PMCID; the logged + used PMCID must be the resolved PMC789.
        enriched = {
            "data": {
                "entry": {
                    "rcsb_id": "9WRG",
                    "rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1038/real"},
                    "pubmed": {"rcsb_pubmed_central_id": "PMC999999"},  # WRONG / stale
                }
            }
        }
        (papers_workspace / "enriched" / "9WRG.json").write_text(json.dumps(enriched))

        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.7 right paper")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("9WRG", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["pmcid"] == "PMC789"  # NOT PMC999999 from the enriched field

    def test_citation_table_doi_is_not_consulted(self, papers_workspace: Path) -> None:
        # We trust ONLY the DOI RCSB tags on rcsb_primary_citation. A DOI sitting
        # in the citation[] table (but absent from the primary citation) is NOT
        # recovered -- it falls through to failed_no_doi, hand-droppable later.
        enriched = {
            "data": {
                "entry": {
                    "rcsb_id": "9NOD",
                    "rcsb_primary_citation": {},
                    "citation": [{"id": "primary", "pdbx_database_id_DOI": "10.1038/ignored"}],
                }
            }
        }
        (papers_workspace / "enriched" / "9NOD.json").write_text(json.dumps(enriched))
        result = download_paper_for_pdb("9NOD", email="test@example.com")
        assert result["status"] == "failed_no_doi"
        assert result["doi"] is None

    def test_genuinely_no_doi_stays_failed_no_doi(self, papers_workspace: Path) -> None:
        enriched = {
            "data": {"entry": {"rcsb_id": "9XUP", "rcsb_primary_citation": {}, "citation": []}}
        }
        (papers_workspace / "enriched" / "9XUP.json").write_text(json.dumps(enriched))
        result = download_paper_for_pdb("9XUP", email="test@example.com")
        assert result["status"] == "failed_no_doi"


class TestRunFetchPapersIsolation:
    def test_per_pdb_error_does_not_abort_batch(self, papers_workspace: Path) -> None:
        # One PDB raising an unexpected error must not abort the batch nor escape
        # the log -- it is recorded as failed_no_data and the run continues.
        from gpcr_tools.papers.runner import run_fetch_papers

        with patch(
            "gpcr_tools.papers.runner.download_paper_for_pdb",
            side_effect=RuntimeError("boom"),
        ):
            run_fetch_papers(pdb_id="7W55", auto_only=True)  # must not raise
        log = _read_download_log()
        assert log["7W55"]["status"] == "failed_no_data"
