"""Fetch-papers runner — orchestrate PDF download + optional watch mode.

Resolves targets, downloads papers via the multi-tier downloader,
then optionally enters watch mode for paywalled papers.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from gpcr_tools.config import DL_STATUS_FAILED_NO_DATA, get_config
from gpcr_tools.fetcher.targets import read_targets
from gpcr_tools.papers.downloader import (
    _build_session,
    _read_download_log,
    _update_download_log,
    download_paper_for_pdb,
)
from gpcr_tools.papers.storage import resolve_pdf_path
from gpcr_tools.papers.watcher import _get_pending_paywalled, run_watcher

logger = logging.getLogger(__name__)


def _discover_missing_papers() -> list[str]:
    """Scan enriched/ for PDB IDs with no resolvable paper PDF.

    A PDB is NOT missing when its canonical DOI-named file (shared by same-DOI
    siblings) or its legacy per-PDB file is present — :func:`resolve_pdf_path`
    tolerates both layouts, so a paper downloaded once for a sibling already
    covers this PDB.
    """
    cfg = get_config()
    enriched_dir = cfg.enriched_dir

    if not enriched_dir.exists():
        return []

    log = _read_download_log()
    pdb_ids: list[str] = []
    for f in sorted(enriched_dir.glob("*.json")):
        pdb_id = f.stem.upper()
        if resolve_pdf_path(pdb_id, log) is None:
            pdb_ids.append(pdb_id)
    return pdb_ids


def run_fetch_papers(
    *,
    pdb_id: str | None = None,
    targets_file: str | None = None,
    auto_only: bool = False,
    watch_only: bool = False,
    force: bool = False,
) -> None:
    """Execute the fetch-papers pipeline.

    Input resolution (highest priority wins):
      1. ``pdb_id`` — single PDB
      2. ``targets_file`` — explicit file path
      3. Default — scan ``enriched/`` for PDBs missing papers

    *watch_only* skips the auto-download retry entirely and goes straight to watch
    mode for the papers already marked paywalled in the download log -- useful when
    a prior run already established which papers are paywalled and you only want to
    drop the manually-fetched PDFs (no email or network needed).
    """
    if watch_only:
        log = _read_download_log()
        if not _get_pending_paywalled(log):
            print(
                "No paywalled papers recorded in the download log "
                "(run a normal fetch-papers first).",
                file=sys.stderr,
            )
            return
        run_watcher(log)
        return

    # Fail fast: require email
    email = os.environ.get("GPCR_EMAIL_FOR_APIS")
    if not email:
        print(
            "Error: Please set the GPCR_EMAIL_FOR_APIS environment variable.\n\n"
            '    export GPCR_EMAIL_FOR_APIS="your.email@example.com"\n',
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve target list
    if pdb_id:
        pdb_ids = [pdb_id.upper()]
    elif targets_file:
        pdb_ids = read_targets(Path(targets_file))
        if not pdb_ids:
            print("No PDB IDs found in the specified targets file.", file=sys.stderr)
            return
    else:
        pdb_ids = _discover_missing_papers()
        if not pdb_ids:
            print(
                "No PDBs with missing papers found in enriched/.",
                file=sys.stderr,
            )
            return

    # Build shared session
    session = _build_session(email)

    from tqdm import tqdm

    ok = 0
    fail = 0
    for pid in tqdm(pdb_ids, desc="Fetching papers"):
        try:
            result = download_paper_for_pdb(
                pid,
                session=session,
                email=email,
                force=force,
            )
        except Exception as exc:
            # One bad record must not abort the batch: log it and record a
            # terminal entry so it is never silently dropped.
            logger.warning("[%s] Unexpected error, recording as failed: %s", pid, exc)
            result = {
                "status": DL_STATUS_FAILED_NO_DATA,
                "source": None,
                "file_path": None,
                "doi": None,
                "pmid": None,
                "pmcid": None,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            _update_download_log(pid, result)
        status = result.get("status") or ""
        if status.startswith("success") or status.startswith("skipped"):
            ok += 1
        else:
            fail += 1

    print(
        f"Paper fetch: {ok} succeeded/skipped, {fail} failed/paywalled.",
        file=sys.stderr,
    )

    # Manual paper workflow (unless --auto-only)
    if not auto_only:
        log = _read_download_log()
        if _get_pending_paywalled(log):
            run_watcher(log)
