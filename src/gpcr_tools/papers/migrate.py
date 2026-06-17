"""One-time, idempotent, resumable migration to DOI-named canonical PDFs.

Consolidates the legacy per-PDB ``papers/{pdb}.pdf`` copies into one canonical
``papers/{sanitized_doi}.pdf`` per paper, then removes the now-redundant per-PDB
copies. Safety invariant: a source per-PDB copy is NEVER deleted until its
canonical file exists AND is a valid PDF, so an interrupted run can be re-run
without losing any paper.

A no-DOI PDB keeps its ``{pdb}.pdf`` (there is no canonical name to collapse to).
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from gpcr_tools.config import DOI_FILENAME_STORAGE, get_config
from gpcr_tools.papers.downloader import _read_download_log
from gpcr_tools.papers.storage import canonical_pdf_name, resolve_doi

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


def _is_valid_pdf(path: Path) -> bool:
    """True if *path* exists and starts with the ``%PDF`` magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(len(_PDF_MAGIC)) == _PDF_MAGIC
    except OSError:
        return False


@dataclass
class MigrationResult:
    """Summary of one migration pass."""

    consolidated: int = 0  # per-PDB copies promoted to a new canonical file
    redundant_removed: int = 0  # per-PDB copies deleted (canonical already valid)
    no_doi_kept: int = 0  # per-PDB files with no DOI, left as-is
    skipped_invalid: int = 0  # per-PDB files that are not valid PDFs, left as-is


def migrate_papers_to_doi_storage() -> MigrationResult:
    """Consolidate legacy per-PDB PDFs into DOI-named canonical files.

    Idempotent and resumable: a per-PDB copy is only deleted once its canonical
    exists and is a valid PDF; a re-run after an interruption simply finishes the
    job. No-op (returns an empty result) when DOI storage is disabled.
    """
    result = MigrationResult()
    if not DOI_FILENAME_STORAGE:
        logger.info("DOI filename storage is disabled; nothing to migrate.")
        return result

    cfg = get_config()
    papers_dir = cfg.papers_dir
    if not papers_dir.is_dir():
        return result

    log = _read_download_log()
    # Canonical files already established for some PDB's DOI are NOT stray per-PDB
    # copies — skip them so a re-run over a migrated corpus doesn't reprocess them
    # or miscount each canonical as a no-DOI file.
    known_canonicals = {
        canonical_pdf_name(pid, doi)
        for pid in {str(k).upper() for k in log}
        if (doi := resolve_doi(pid, log))
    }

    # Iterate the legacy per-PDB files: a 4-char-ish PDB stem (uppercased) whose
    # DOI resolves to a different canonical name. A file already named like a
    # canonical (sanitized DOI) won't resolve to a PDB DOI and is left alone.
    for pdf in sorted(papers_dir.glob("*.pdf")):
        if pdf.name in known_canonicals:
            continue  # already an established canonical, not a stray per-PDB copy
        pdb_id = pdf.stem.upper()
        doi = resolve_doi(pdb_id, log)
        if not doi:
            result.no_doi_kept += 1
            continue
        canonical = papers_dir / canonical_pdf_name(pdb_id, doi)
        if canonical == pdf:
            # Already the canonical file for this PDB's DOI.
            continue
        if not _is_valid_pdf(pdf):
            logger.warning("[%s] %s is not a valid PDF; leaving it in place.", pdb_id, pdf.name)
            result.skipped_invalid += 1
            continue

        if _is_valid_pdf(canonical):
            # Canonical already present and valid -> the per-PDB copy is redundant.
            with contextlib.suppress(OSError):
                pdf.unlink()
                result.redundant_removed += 1
                logger.info("[%s] removed redundant %s (canonical exists)", pdb_id, pdf.name)
            continue

        # Promote this per-PDB copy to the canonical file, then remove the source
        # ONLY after the canonical exists and validates (never lose the only copy).
        try:
            shutil.copyfile(pdf, canonical)
        except OSError as exc:
            logger.warning("[%s] could not write canonical %s: %s", pdb_id, canonical.name, exc)
            continue
        if not _is_valid_pdf(canonical):
            logger.warning(
                "[%s] canonical %s failed validation; keeping source.", pdb_id, canonical.name
            )
            with contextlib.suppress(OSError):
                canonical.unlink()
            continue
        result.consolidated += 1
        with contextlib.suppress(OSError):
            pdf.unlink()
        logger.info("[%s] consolidated %s -> %s", pdb_id, pdf.name, canonical.name)

    logger.info(
        "Paper migration: %d consolidated, %d redundant removed, %d no-DOI kept, %d invalid skipped.",
        result.consolidated,
        result.redundant_removed,
        result.no_doi_kept,
        result.skipped_invalid,
    )
    return result
