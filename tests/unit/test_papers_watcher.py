"""Tests for papers/watcher.py — the manual paper workflow (DOI-grouped)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from gpcr_tools.papers.storage import resolve_pdf_path
from gpcr_tools.papers.watcher import (
    _build_doi_groups,
    _clickable,
    _detect_new_pdf,
    _get_pending_paywalled,
    _is_valid_pdf,
    _replicate_existing,
    _wait_for_stability,
    run_watcher,
)


class TestClickable:
    def test_plain_url_when_not_a_tty(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.stderr", io.StringIO())  # StringIO.isatty() -> False
        assert _clickable("https://doi.org/10.1/x") == "https://doi.org/10.1/x"

    def test_osc8_hyperlink_when_tty(self, monkeypatch) -> None:
        class _Tty(io.StringIO):
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr("sys.stderr", _Tty())
        out = _clickable("https://doi.org/10.1/x")
        assert out != "https://doi.org/10.1/x"  # wrapped, not plain
        assert "\033]8;;https://doi.org/10.1/x" in out  # OSC 8 hyperlink open
        assert out.endswith("\033]8;;\033\\")  # OSC 8 hyperlink close


def _sandbox(tmp_path: Path, monkeypatch) -> Any:
    """Point the workspace at tmp_path so the download log is written there."""
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    from gpcr_tools.config import get_config, reset_config

    reset_config()
    cfg = get_config()
    cfg.papers_dir.mkdir(parents=True, exist_ok=True)
    cfg.download_log_file.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def _pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4" + b"x" * 200)


class TestIsValidPdf:
    def test_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "a.pdf"
        _pdf(f)
        assert _is_valid_pdf(f) is True

    def test_invalid(self, tmp_path: Path) -> None:
        f = tmp_path / "a.pdf"
        f.write_bytes(b"<html>nope</html>")
        assert _is_valid_pdf(f) is False

    def test_missing(self, tmp_path: Path) -> None:
        assert _is_valid_pdf(tmp_path / "nope.pdf") is False


class TestWaitForStability:
    def test_stable(self, tmp_path: Path) -> None:
        f = tmp_path / "a.pdf"
        _pdf(f)
        assert _wait_for_stability(f) is True

    def test_missing(self, tmp_path: Path) -> None:
        assert _wait_for_stability(tmp_path / "nope.pdf") is False


class TestGetPendingPaywalled:
    def test_filters(self) -> None:
        log: dict[str, Any] = {
            "7W55": {"status": "success_pdf_downloaded"},
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1038/test"},
            "9XYZ": {"status": "fallback_paywalled", "doi": None},
        }
        pending = _get_pending_paywalled(log)
        assert set(pending) == {"8ABC", "9XYZ"}

    def test_empty(self) -> None:
        assert _get_pending_paywalled({}) == {}


class TestBuildDoiGroups:
    def test_groups_by_doi_and_marks_pdf_exists(self, tmp_path: Path) -> None:
        papers = tmp_path / "papers"
        papers.mkdir()
        _pdf(papers / "7W55.pdf")  # exists
        log: dict[str, Any] = {
            "7W55": {"status": "success_pdf_downloaded", "doi": "10.1/shared"},
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1/shared"},  # sibling
            "9XYZ": {"status": "fallback_paywalled", "doi": "10.2/other"},
            "NODOI": {"status": "fallback_paywalled", "doi": None},  # excluded
        }
        groups = _build_doi_groups(log, papers)
        assert set(groups) == {"10.1/shared", "10.2/other"}
        shared = {p["pdb_id"]: p for p in groups["10.1/shared"]}
        assert shared["7W55"]["pdf_exists"] is True
        assert shared["8ABC"]["pdf_exists"] is False


class TestEnrichedDoiFallback:
    def test_doi_recovered_from_enriched_when_log_nulled(self, tmp_path: Path, monkeypatch) -> None:
        """An entry whose log DOI was nulled (e.g. a prior skip-exists run) is still
        grouped, by recovering the DOI from the enriched metadata."""
        cfg = _sandbox(tmp_path, monkeypatch)
        cfg.enriched_dir.mkdir(parents=True, exist_ok=True)
        (cfg.enriched_dir / "7W55.json").write_text(
            json.dumps(
                {
                    "data": {
                        "entry": {"rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1/shared"}}
                    }
                }
            )
        )
        _pdf(cfg.papers_dir / "7W55.pdf")
        log: dict[str, Any] = {
            "7W55": {"status": "skipped_already_downloaded", "doi": None},  # DOI lost
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1/shared"},  # sibling
        }
        groups = _build_doi_groups(log, cfg.papers_dir)
        assert set(groups) == {"10.1/shared"}
        # 7W55 (DOI recovered from enriched) rejoins the group as the PDF source.
        n = _replicate_existing(groups, cfg.papers_dir)
        assert n == 1
        # Under DOI storage the sibling resolves to the shared canonical file
        # (no per-PDB copy); resolve_pdf_path finds it for 8ABC.
        assert resolve_pdf_path("8ABC", log) is not None


class TestReplicateExisting:
    def test_fills_sibling_from_existing_pdf(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _sandbox(tmp_path, monkeypatch)
        _pdf(cfg.papers_dir / "7W55.pdf")  # one structure already downloaded
        log: dict[str, Any] = {
            "7W55": {"status": "success_pdf_downloaded", "doi": "10.1/shared"},
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1/shared"},  # paywalled sibling
        }
        groups = _build_doi_groups(log, cfg.papers_dir)
        n = _replicate_existing(groups, cfg.papers_dir)
        assert n == 1
        # The sibling is covered by the shared canonical paper (DOI storage).
        assert resolve_pdf_path("8ABC", log) is not None
        written = json.loads(cfg.download_log_file.read_text())
        assert written["8ABC"]["status"] == "manual_user_provided"

    def test_no_existing_pdf_replicates_nothing(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _sandbox(tmp_path, monkeypatch)
        log: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1/shared"},
            "8DEF": {"status": "fallback_paywalled", "doi": "10.1/shared"},
        }
        groups = _build_doi_groups(log, cfg.papers_dir)
        assert _replicate_existing(groups, cfg.papers_dir) == 0

    def test_doi_storage_skips_sibling_when_canonical_missing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """DOI storage: a sibling is NOT marked covered if the canonical paper was
        never established (e.g. an upstream copy failed) — the manifest must not
        over-report coverage for a file that does not exist."""
        from gpcr_tools.papers import watcher as w

        cfg = _sandbox(tmp_path, monkeypatch)
        siblings = [
            {"pdb_id": "8ABC", "doi": "10.1/x", "entry": {"doi": "10.1/x"}, "pdf_exists": False}
        ]
        n = w._replicate_to_siblings(cfg.papers_dir / "nonexistent.pdf", siblings, cfg.papers_dir)
        assert n == 0
        assert siblings[0]["pdf_exists"] is False
        assert not cfg.download_log_file.exists() or "8ABC" not in json.loads(
            cfg.download_log_file.read_text()
        )


class TestDetectNewPdf:
    def test_target_already_present(self, tmp_path: Path) -> None:
        target = tmp_path / "7W55.pdf"
        _pdf(target)
        assert _detect_new_pdf(tmp_path, before={"7W55.pdf"}, target=target) is True

    def test_new_arbitrary_pdf_renamed_to_target(self, tmp_path: Path) -> None:
        # A newly-dropped, arbitrarily-named PDF is renamed to the active target.
        _pdf(tmp_path / "downloaded_paper.pdf")
        target = tmp_path / "7W55.pdf"
        assert _detect_new_pdf(tmp_path, before=set(), target=target) is True
        assert target.exists()
        assert not (tmp_path / "downloaded_paper.pdf").exists()

    def test_no_new_pdf(self, tmp_path: Path) -> None:
        _pdf(tmp_path / "old.pdf")
        assert _detect_new_pdf(tmp_path, before={"old.pdf"}, target=tmp_path / "7W55.pdf") is False


class TestRunWatcherPhase1Resolves:
    def test_all_resolved_by_replication_returns_without_prompting(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When Phase-1 replication fills every paywalled sibling from an existing
        PDF, there is no manual work left, so run_watcher returns without entering
        the interactive loop (no input/poll hang)."""
        cfg = _sandbox(tmp_path, monkeypatch)
        _pdf(cfg.papers_dir / "7W55.pdf")
        log = {
            "7W55": {"status": "success_pdf_downloaded", "doi": "10.1/shared"},
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1/shared"},  # sibling, filled
        }
        provided = run_watcher(log)
        assert provided == 0  # nothing needed manual fetching
        # Sibling covered in Phase 1 via the shared canonical paper (DOI storage).
        assert resolve_pdf_path("8ABC", log) is not None
