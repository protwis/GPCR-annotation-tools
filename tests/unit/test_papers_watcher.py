"""Tests for papers/watcher.py — filesystem watcher for paywalled papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpcr_tools.papers.watcher import (
    _get_pending_paywalled,
    _is_valid_pdf,
    _match_pdf_to_pdb,
    _wait_for_stability,
)


class TestIsValidPdf:
    def test_valid_pdf(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 some content")
        assert _is_valid_pdf(f) is True

    def test_invalid(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"not a pdf")
        assert _is_valid_pdf(f) is False

    def test_missing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.pdf"
        assert _is_valid_pdf(f) is False


class TestGetPendingPaywalled:
    def test_filters_paywalled(self) -> None:
        log: dict[str, Any] = {
            "7W55": {"status": "success_pdf_downloaded"},
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1038/test"},
            "9XYZ": {"status": "fallback_paywalled", "doi": None},
        }
        pending = _get_pending_paywalled(log)
        assert "7W55" not in pending
        assert "8ABC" in pending
        assert "9XYZ" in pending

    def test_empty_log(self) -> None:
        assert _get_pending_paywalled({}) == {}


class TestMatchPdfToPdb:
    def test_exact_filename_match(self, tmp_path: Path) -> None:
        pdf = tmp_path / "8ABC.pdf"
        pending: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled"},
            "9XYZ": {"status": "fallback_paywalled"},
        }
        assert _match_pdf_to_pdb(pdf, pending) == "8ABC"

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        pdf = tmp_path / "8abc.pdf"
        pending: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled"},
        }
        assert _match_pdf_to_pdb(pdf, pending) == "8ABC"

    def test_no_match(self, tmp_path: Path) -> None:
        pdf = tmp_path / "random_paper.pdf"
        pending: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled"},
        }
        assert _match_pdf_to_pdb(pdf, pending) is None


class TestWaitForStability:
    def test_stable_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4" + b"x" * 1000)
        # File is already stable (not being written)
        assert _wait_for_stability(f) is True

    def test_missing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.pdf"
        assert _wait_for_stability(f) is False
