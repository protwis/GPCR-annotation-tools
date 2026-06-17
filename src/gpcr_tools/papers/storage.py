"""Canonical paper-PDF storage resolution — the single source of truth.

A paper is shared by many PDB structures (one DOI, up to dozens of PDBs). This
module resolves, for a given PDB, the DOI of its paper and the on-disk path of
the paper's PDF, so every reader (annotator, watcher, fetch-papers, reports)
addresses the same file rather than each keeping a private copy.

Two storage layouts are supported during the transition (gated by
``DOI_FILENAME_STORAGE``):

* DOI-named canonical: ``papers/{sanitized_doi}.pdf`` — one file per paper, shared
  by every same-DOI sibling. A no-DOI PDB keeps ``papers/{pdb}.pdf``.
* Legacy per-PDB: ``papers/{pdb}.pdf`` — one physical copy per structure.

``resolve_pdf_path`` is tolerant of BOTH: it prefers the canonical DOI-named file
when present, then falls back to the per-PDB file, so a partially-migrated
workspace keeps working.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from gpcr_tools.config import DOI_FILENAME_STORAGE, get_config, sanitize_doi

logger = logging.getLogger(__name__)


def _enriched_doi(pdb_id: str) -> str:
    """Read the primary-citation DOI from ``enriched/{pdb_id}.json`` (or "")."""
    cfg = get_config()
    path = cfg.enriched_dir / f"{pdb_id.upper()}.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    entry = (data.get("data") or {}).get("entry") or {}
    doi = (entry.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI")
    return (doi or "").strip()


def resolve_doi(pdb_id: str, download_log: dict[str, Any] | None = None) -> str:
    """The DOI for *pdb_id*: the download-log entry's DOI, else the enriched
    metadata, else "" (no DOI).

    Mirrors the watcher's resolution order so sibling grouping, dedup keying, and
    canonical-filename resolution all agree on a PDB's paper.
    """
    if download_log is not None:
        entry = download_log.get(pdb_id.upper()) or download_log.get(pdb_id)
        if isinstance(entry, dict):
            doi = (entry.get("doi") or "").strip()
            if doi:
                return doi
    return _enriched_doi(pdb_id)


def canonical_pdf_name(pdb_id: str, doi: str) -> str:
    """The canonical PDF filename for a PDB given its *doi*.

    DOI-named (``{sanitized_doi}.pdf``) when storage is DOI-keyed and a DOI is
    present; otherwise the per-PDB ``{pdb}.pdf``.
    """
    if DOI_FILENAME_STORAGE and doi:
        token = sanitize_doi(doi)
        if token:
            return f"{token}.pdf"
    return f"{pdb_id.upper()}.pdf"


def canonical_pdf_path(pdb_id: str, doi: str) -> Path:
    """Absolute path of the canonical PDF for *pdb_id* given its *doi*."""
    return get_config().papers_dir / canonical_pdf_name(pdb_id, doi)


def resolve_pdf_path(pdb_id: str, download_log: dict[str, Any] | None = None) -> Path | None:
    """The on-disk PDF for *pdb_id*, tolerant of both storage layouts.

    Resolution order: the DOI-named canonical file (when DOI storage is on and the
    file exists), then the legacy per-PDB ``{pdb}.pdf``. Returns ``None`` when
    neither exists, so a caller can skip a PDB whose paper is not present.
    """
    cfg = get_config()
    per_pdb = cfg.papers_dir / f"{pdb_id.upper()}.pdf"
    if DOI_FILENAME_STORAGE:
        doi = resolve_doi(pdb_id, download_log)
        if doi:
            canonical = cfg.papers_dir / canonical_pdf_name(pdb_id, doi)
            if canonical.is_file():
                return canonical
    if per_pdb.is_file():
        return per_pdb
    return None


def content_hash(path: Path) -> str | None:
    """SHA-256 of a file's bytes, or ``None`` if it cannot be read.

    The cheap byte-equality safety gate for upload dedup: a shared fileUri is
    reused across siblings only when their PDFs hash identically (or, failing a
    hash, when they carry the same DOI — see the annotator dedup path).
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return None
