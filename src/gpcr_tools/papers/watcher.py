"""Filesystem watcher for manually-downloaded paywalled papers.

After the auto-download phase, watches ``papers/`` for new PDFs dropped
by the user, validates them, auto-renames to ``{pdb_id}.pdf``, and
updates the download log.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    WATCHER_POLL_INTERVAL,
    WATCHER_STABILITY_CHECKS,
    WATCHER_STABILITY_INTERVAL,
    get_config,
)
from gpcr_tools.papers.downloader import _update_download_log

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


def _is_valid_pdf(path: Path) -> bool:
    """Check if a file starts with the ``%PDF`` magic bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header == _PDF_MAGIC
    except OSError:
        return False


def _get_pending_paywalled(download_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return entries from the log that are still paywalled."""
    return {
        pdb_id: entry
        for pdb_id, entry in download_log.items()
        if isinstance(entry, dict) and entry.get("status") == "fallback_paywalled"
    }


def _match_pdf_to_pdb(
    pdf_path: Path,
    pending: dict[str, dict[str, Any]],
) -> str | None:
    """Try to match a PDF filename to a pending paywalled PDB ID.

    Strategy:
      1. If the stem matches a pending PDB ID (e.g., ``7W55.pdf``), use it.
      2. Otherwise return None (user will need to rename).
    """
    stem = pdf_path.stem.upper()
    if stem in pending:
        return stem
    return None


def _wait_for_stability(path: Path) -> bool:
    """Wait until the file size stops changing (download complete)."""
    prev_size = -1
    stable_count = 0
    for _ in range(WATCHER_STABILITY_CHECKS + 5):  # max ~7 iterations
        try:
            current_size = path.stat().st_size
        except OSError:
            return False
        if current_size == prev_size and current_size > 0:
            stable_count += 1
            if stable_count >= WATCHER_STABILITY_CHECKS:
                return True
        else:
            stable_count = 0
        prev_size = current_size
        time.sleep(WATCHER_STABILITY_INTERVAL)
    return False


def run_watcher(paywalled_entries: dict[str, dict[str, Any]]) -> int:
    """Watch ``papers/`` for new PDFs and match to paywalled entries.

    Return the number of papers successfully matched.
    """
    cfg = get_config()
    papers_dir = cfg.papers_dir
    papers_dir.mkdir(parents=True, exist_ok=True)

    pending = dict(paywalled_entries)
    if not pending:
        return 0

    # Phase 1: Print instructions
    _print_instructions(pending, papers_dir)

    matched = 0
    known_files: set[str] = {f.name for f in papers_dir.iterdir() if f.suffix.lower() == ".pdf"}

    try:
        while pending:
            time.sleep(WATCHER_POLL_INTERVAL)
            current_files = {f.name for f in papers_dir.iterdir() if f.suffix.lower() == ".pdf"}
            new_files = current_files - known_files

            for filename in sorted(new_files):
                pdf_path = papers_dir / filename
                pdb_id = _match_pdf_to_pdb(pdf_path, pending)

                if pdb_id is None:
                    # Can't match — skip this file
                    logger.info(
                        "New PDF %s doesn't match any pending PDB, ignoring",
                        filename,
                    )
                    known_files.add(filename)
                    continue

                # Wait for file stability
                if not _wait_for_stability(pdf_path):
                    logger.warning("File %s not stable, skipping", filename)
                    known_files.add(filename)
                    continue

                # Validate PDF
                if not _is_valid_pdf(pdf_path):
                    logger.warning("File %s is not a valid PDF, skipping", filename)
                    known_files.add(filename)
                    continue

                # Rename to canonical form
                canonical = papers_dir / f"{pdb_id}.pdf"
                if pdf_path != canonical:
                    os.replace(str(pdf_path), str(canonical))

                # Update log
                _update_download_log(
                    pdb_id,
                    {
                        "status": "manual_user_provided",
                        "source": "user_manual",
                        "file_path": str(canonical),
                        "doi": pending[pdb_id].get("doi"),
                        "pmid": pending[pdb_id].get("pmid"),
                        "pmcid": pending[pdb_id].get("pmcid"),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                del pending[pdb_id]
                matched += 1
                remaining = len(pending)
                print(
                    f"  ✅ {pdb_id}.pdf — matched and saved ({remaining} remaining)",
                    file=sys.stderr,
                )

                known_files.add(f"{pdb_id}.pdf")
                known_files.add(filename)

    except KeyboardInterrupt:
        # Clean exit — log remaining as skipped
        for pdb_id, entry in pending.items():
            _update_download_log(
                pdb_id,
                {
                    "status": "skipped_no_paper",
                    "source": None,
                    "file_path": None,
                    "doi": entry.get("doi"),
                    "pmid": entry.get("pmid"),
                    "pmcid": entry.get("pmcid"),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    # Phase 3: Summary
    total = len(paywalled_entries)
    skipped = len(pending)
    skipped_ids = ", ".join(sorted(pending.keys()))
    if skipped:
        print(
            f"\nDone. {matched}/{total} paywalled papers provided. "
            f"{skipped} skipped ({skipped_ids}).",
            file=sys.stderr,
        )
    else:
        print(
            f"\nDone. {matched}/{total} paywalled papers provided. All resolved.",
            file=sys.stderr,
        )

    return matched


def _print_instructions(pending: dict[str, dict[str, Any]], papers_dir: Path) -> None:
    """Print the paywalled paper instructions box."""
    count = len(pending)
    print(
        "\n╭─ Papers Needing Manual Download ─────────────────────────────╮",
        file=sys.stderr,
    )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        f"│  {count} paper(s) could not be auto-downloaded (paywalled).   │",
        file=sys.stderr,
    )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        "│  Download them in your browser and save to:                  │",
        file=sys.stderr,
    )
    print(
        f"│    📂  {papers_dir!s:<50s}│",
        file=sys.stderr,
    )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        f"│  {'PDB':<6s} {'DOI':<42s} {'PMID':<10s}│",
        file=sys.stderr,
    )
    for pdb_id, entry in sorted(pending.items()):
        doi = entry.get("doi") or "(none)"
        if len(doi) > 40:
            doi = doi[:37] + "..."
        pmid = str(entry.get("pmid") or "(none)")
        print(
            f"│  {pdb_id:<6s} {doi:<42s} {pmid:<10s}│",
            file=sys.stderr,
        )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        "│  ⏳ Watching for new PDFs... (Ctrl+C to stop)                │",
        file=sys.stderr,
    )
    print(
        "╰──────────────────────────────────────────────────────────────╯\n",
        file=sys.stderr,
    )
