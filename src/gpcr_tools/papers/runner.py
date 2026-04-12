"""Fetch-papers runner — orchestrate PDF download + optional watch mode.

Resolves targets, downloads papers via the multi-tier downloader,
then optionally enters watch mode for paywalled papers.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from gpcr_tools.config import get_config
from gpcr_tools.fetcher.targets import read_targets
from gpcr_tools.papers.downloader import (
    _build_session,
    _read_download_log,
    download_paper_for_pdb,
)
from gpcr_tools.papers.watcher import _get_pending_paywalled, run_watcher

logger = logging.getLogger(__name__)


def _discover_missing_papers() -> list[str]:
    """Scan enriched/ for PDB IDs without a corresponding papers/{pdb_id}.pdf."""
    cfg = get_config()
    enriched_dir = cfg.enriched_dir
    papers_dir = cfg.papers_dir

    if not enriched_dir.exists():
        return []

    pdb_ids: list[str] = []
    for f in sorted(enriched_dir.glob("*.json")):
        pdb_id = f.stem.upper()
        if not (papers_dir / f"{pdb_id}.pdf").exists():
            pdb_ids.append(pdb_id)
    return pdb_ids


def run_fetch_papers(
    *,
    pdb_id: str | None = None,
    targets_file: str | None = None,
    auto_only: bool = False,
    force: bool = False,
) -> None:
    """Execute the fetch-papers pipeline.

    Input resolution (highest priority wins):
      1. ``pdb_id`` — single PDB
      2. ``targets_file`` — explicit file path
      3. Default — scan ``enriched/`` for PDBs missing papers
    """
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
        result = download_paper_for_pdb(
            pid,
            session=session,
            email=email,
            force=force,
        )
        status = result.get("status") or ""
        if status.startswith("success") or status.startswith("skipped"):
            ok += 1
        else:
            fail += 1

    print(
        f"Paper fetch: {ok} succeeded/skipped, {fail} failed/paywalled.",
        file=sys.stderr,
    )

    # Watch mode (unless --auto-only)
    if not auto_only:
        log = _read_download_log()
        paywalled = _get_pending_paywalled(log)
        if paywalled:
            run_watcher(paywalled)
