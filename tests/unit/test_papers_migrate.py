"""Tests for the DOI-named paper-storage migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpcr_tools.config import get_config, reset_config, sanitize_doi
from gpcr_tools.papers.migrate import migrate_papers_to_doi_storage
from gpcr_tools.papers.storage import resolve_pdf_path


@pytest.fixture()
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    cfg = get_config()
    cfg.papers_dir.mkdir(parents=True, exist_ok=True)
    cfg.enriched_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4" + b"x" * 100)


def _log(cfg: Any, mapping: dict[str, str]) -> None:
    cfg.download_log_file.write_text(
        json.dumps(
            {pdb: {"status": "success_pdf_downloaded", "doi": doi} for pdb, doi in mapping.items()}
        )
    )


def test_consolidates_siblings_into_one_canonical(ws: Any) -> None:
    """Two per-PDB copies of one paper collapse to a single canonical DOI file;
    the per-PDB copies are removed and both PDBs still resolve."""
    _pdf(ws.papers_dir / "AAA.pdf")
    _pdf(ws.papers_dir / "BBB.pdf")
    _log(ws, {"AAA": "10.1/shared", "BBB": "10.1/shared"})

    result = migrate_papers_to_doi_storage()

    log = json.loads(ws.download_log_file.read_text())
    canonical = ws.papers_dir / f"{sanitize_doi('10.1/shared')}.pdf"
    assert canonical.is_file()
    assert not (ws.papers_dir / "AAA.pdf").exists()
    assert not (ws.papers_dir / "BBB.pdf").exists()
    assert resolve_pdf_path("AAA", log) == canonical
    assert resolve_pdf_path("BBB", log) == canonical
    # One promoted to canonical, the other removed as redundant.
    assert result.consolidated == 1
    assert result.redundant_removed == 1


def test_no_doi_pdb_kept_as_is(ws: Any) -> None:
    """A no-DOI PDB keeps its {pdb}.pdf (no canonical to collapse to)."""
    _pdf(ws.papers_dir / "NODOI.pdf")
    _log(ws, {})  # empty log, no enriched -> no DOI

    result = migrate_papers_to_doi_storage()

    assert (ws.papers_dir / "NODOI.pdf").exists()
    assert result.no_doi_kept == 1


def test_migration_is_idempotent(ws: Any) -> None:
    """A second run after a complete migration changes nothing further."""
    _pdf(ws.papers_dir / "AAA.pdf")
    _log(ws, {"AAA": "10.1/shared"})

    migrate_papers_to_doi_storage()
    canonical = ws.papers_dir / f"{sanitize_doi('10.1/shared')}.pdf"
    assert canonical.is_file()

    # Re-run: canonical already present; nothing consolidated, source already gone.
    result2 = migrate_papers_to_doi_storage()
    assert result2.consolidated == 0
    assert canonical.is_file()


def test_never_deletes_source_when_canonical_invalid(
    ws: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the canonical fails to validate after the copy, the source is kept."""
    _pdf(ws.papers_dir / "AAA.pdf")
    _log(ws, {"AAA": "10.1/shared"})

    # Make the written canonical look invalid so the safety gate keeps the source.
    import gpcr_tools.papers.migrate as migrate_mod

    real_is_valid = migrate_mod._is_valid_pdf

    def _is_valid(path: Path) -> bool:
        # The source is valid; a freshly-written canonical is treated as invalid.
        if path.name.startswith("10.1"):
            return False
        return real_is_valid(path)

    monkeypatch.setattr(migrate_mod, "_is_valid_pdf", _is_valid)

    result = migrate_papers_to_doi_storage()

    assert (ws.papers_dir / "AAA.pdf").exists()  # source never lost
    assert result.consolidated == 0


def test_invalid_source_left_in_place(ws: Any) -> None:
    """A per-PDB file that is not a valid PDF is left alone (not consolidated)."""
    (ws.papers_dir / "AAA.pdf").write_bytes(b"<html>not a pdf</html>")
    _log(ws, {"AAA": "10.1/shared"})

    result = migrate_papers_to_doi_storage()

    assert (ws.papers_dir / "AAA.pdf").exists()
    assert result.skipped_invalid == 1


def test_disabled_by_kill_switch(ws: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _pdf(ws.papers_dir / "AAA.pdf")
    _log(ws, {"AAA": "10.1/shared"})
    monkeypatch.setattr("gpcr_tools.papers.migrate.DOI_FILENAME_STORAGE", False)

    result = migrate_papers_to_doi_storage()

    assert (ws.papers_dir / "AAA.pdf").exists()  # untouched
    assert not (ws.papers_dir / f"{sanitize_doi('10.1/shared')}.pdf").exists()
    assert result.consolidated == 0
