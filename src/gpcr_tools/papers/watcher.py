"""Manual paper workflow for paywalled papers (Docker-friendly).

After the auto-download phase, the papers the open-access tiers could not fetch are
handled here, mirroring the original process_manual.py design:

  Phase 1 -- group every PDB by its paper DOI (from the download-log metadata, NOT
  by reading PDFs) and copy an already-downloaded paper to its same-DOI sibling
  structures. One paper deposits several structures; downloading it once fills them
  all.

  Phase 2 -- walk the remaining DOIs ONE AT A TIME: print the DOI link, watch
  ``papers/`` for the PDF the user drops, and rename it to ``{PDB}.pdf`` by context
  (the DOI currently being processed -- so the filename never matters and no content
  matching is needed), then replicate it to that paper's other structures.

Docker note: the code runs inside the container, which has no browser, so the DOI
link is printed (clickable in most terminals) rather than auto-opened. The user
downloads on the host into the mounted ``papers/`` folder; the watcher renames it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    DOI_FILENAME_STORAGE,
    WATCHER_POLL_INTERVAL,
    WATCHER_STABILITY_CHECKS,
    WATCHER_STABILITY_INTERVAL,
    get_config,
)
from gpcr_tools.papers.downloader import _update_download_log
from gpcr_tools.papers.storage import canonical_pdf_name

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"
# Seconds of silent auto-detection per paper before falling back to an Enter prompt.
_MANUAL_DETECT_POLLS = 60  # x WATCHER_POLL_INTERVAL (2s) = ~120s, matching the original.


def _is_valid_pdf(path: Path) -> bool:
    """True if the file starts with the ``%PDF`` magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _PDF_MAGIC
    except OSError:
        return False


def _wait_for_stability(path: Path) -> bool:
    """Wait until the file size stops changing (download finished)."""
    prev_size = -1
    stable = 0
    for _ in range(WATCHER_STABILITY_CHECKS + 5):
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == prev_size and size > 0:
            stable += 1
            if stable >= WATCHER_STABILITY_CHECKS:
                return True
        else:
            stable = 0
        prev_size = size
        time.sleep(WATCHER_STABILITY_INTERVAL)
    return False


def _get_pending_paywalled(download_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Entries still marked paywalled (used by the --watch-only gate)."""
    return {
        pdb_id: entry
        for pdb_id, entry in download_log.items()
        if isinstance(entry, dict) and entry.get("status") == "fallback_paywalled"
    }


def _enriched_doi(pdb_id: str, enriched_dir: Path) -> str:
    """Read the primary-citation DOI from ``enriched/{pdb_id}.json`` (or "")."""
    path = enriched_dir / f"{pdb_id}.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    entry = (data.get("data") or {}).get("entry") or {}
    doi = (entry.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI")
    return (doi or "").strip()


def _build_doi_groups(
    download_log: dict[str, Any], papers_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    """Group PDBs by paper DOI from the log metadata (no PDF reading).

    Returns ``{doi: [{pdb_id, status, entry, pdf_exists}, ...]}``. ``pdf_exists`` is
    a live filesystem check for ``papers/{pdb_id}.pdf``. When a log entry has no DOI
    (e.g. an older run nulled it on the skip-exists path), the DOI is recovered from
    the enriched metadata so sibling grouping stays reliable across re-runs.
    """
    enriched_dir = get_config().enriched_dir
    groups: dict[str, list[dict[str, Any]]] = {}
    for pdb_id, entry in download_log.items():
        if not isinstance(entry, dict):
            continue
        pid = pdb_id.upper()
        doi = (entry.get("doi") or "").strip() or _enriched_doi(pid, enriched_dir)
        if not doi:
            continue
        # Under DOI storage a single canonical file covers every same-DOI sibling,
        # so existence resolves through the shared name (with a per-PDB fallback).
        groups.setdefault(doi, []).append(
            {
                "pdb_id": pid,
                "doi": doi,
                "status": entry.get("status"),
                "entry": entry,
                "pdf_exists": (papers_dir / canonical_pdf_name(pid, doi)).exists()
                or (papers_dir / f"{pid}.pdf").exists(),
            }
        )
    return groups


def _log_manual(pdb_id: str, path: Path, src_entry: dict[str, Any], source: str) -> None:
    _update_download_log(
        pdb_id,
        {
            "status": "manual_user_provided",
            "source": source,
            "file_path": str(path),
            "doi": src_entry.get("doi"),
            "pmid": src_entry.get("pmid"),
            "pmcid": src_entry.get("pmcid"),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


def _replicate_to_siblings(
    source_path: Path, siblings: list[dict[str, Any]], papers_dir: Path
) -> int:
    """Cover each same-DOI sibling from *source_path*.

    Under DOI storage the siblings already resolve to the SAME canonical file, so
    no bytes are copied — each sibling is just recorded as covered (the curator
    sees it filled, and every reader resolves it via the shared canonical name).
    Under the legacy layout the file is physically copied to ``{pdb_id}.pdf``.
    """
    n = 0
    for sib in siblings:
        if sib["pdf_exists"]:
            continue
        if DOI_FILENAME_STORAGE and sib.get("doi"):
            # Siblings share the canonical file; record coverage without copying.
            target = papers_dir / canonical_pdf_name(sib["pdb_id"], sib["doi"])
            if not target.exists():
                # The canonical paper was never established (e.g. an upstream copy
                # failed). Don't record the sibling as covered when no file exists.
                logger.warning(
                    "[%s] canonical paper %s missing; not marking covered",
                    sib["pdb_id"],
                    target.name,
                )
                continue
        else:
            target = papers_dir / f"{sib['pdb_id']}.pdf"
            try:
                shutil.copyfile(source_path, target)
            except OSError as exc:
                logger.warning("[%s] could not replicate sibling PDF: %s", sib["pdb_id"], exc)
                continue
        sib["pdf_exists"] = True
        _log_manual(sib["pdb_id"], target, sib["entry"], "replicated_sibling")
        print(f"    → also covered {sib['pdb_id']} (same paper)", file=sys.stderr)
        n += 1
    return n


def _replicate_existing(groups: dict[str, list[dict[str, Any]]], papers_dir: Path) -> int:
    """Phase 1: for every DOI that already has a PDF, cover its missing siblings.

    Under DOI storage the paper is established ONCE as the canonical
    ``{sanitized_doi}.pdf`` (consolidating a legacy per-PDB source if that is all
    that exists yet), and every sibling then resolves to that one file — no
    physical per-sibling copies. Under the legacy layout it copies per sibling.
    """
    replicated = 0
    for plist in groups.values():
        source = next((p for p in plist if p["pdf_exists"]), None)
        if source is None:
            continue
        if DOI_FILENAME_STORAGE and source.get("doi"):
            canonical = papers_dir / canonical_pdf_name(source["pdb_id"], source["doi"])
            legacy = papers_dir / f"{source['pdb_id']}.pdf"
            # Establish the canonical file once, from a legacy source if needed,
            # so every sibling resolves to it.
            if not canonical.exists() and legacy.exists():
                try:
                    shutil.copyfile(legacy, canonical)
                except OSError as exc:
                    logger.warning(
                        "[%s] could not establish canonical PDF: %s", source["pdb_id"], exc
                    )
            source_path = canonical if canonical.exists() else legacy
        else:
            source_path = papers_dir / f"{source['pdb_id']}.pdf"
        siblings = [p for p in plist if not p["pdf_exists"]]
        replicated += _replicate_to_siblings(source_path, siblings, papers_dir)
    return replicated


def _detect_new_pdf(papers_dir: Path, before: set[str], target: Path) -> bool:
    """One detection pass for the paper currently being fetched.

    Succeeds if *target* already exists (user pre-named it) or a NEW, stable, valid
    PDF appeared since *before* -- in which case it is renamed to *target*. Because
    only one paper is being processed, any new PDF is unambiguously this one, so no
    content matching is needed.
    """
    if target.exists() and _is_valid_pdf(target):
        return True
    current = {f.name for f in papers_dir.glob("*.pdf")}
    for name in sorted(current - before):
        candidate = papers_dir / name
        if not _wait_for_stability(candidate) or not _is_valid_pdf(candidate):
            continue
        if candidate != target:
            os.replace(str(candidate), str(target))
        return True
    return False


def _clickable(url: str) -> str:
    """Render *url* as an OSC 8 terminal hyperlink (Cmd/Ctrl-click to open in
    iTerm2, VS Code, and modern Terminal.app). Falls back to the plain URL when
    stderr is not an interactive terminal, so redirected logs/pipes stay clean.
    The visible text is the URL itself, so it stays copy-pasteable even on the
    rare terminal/multiplexer (old tmux/screen) that doesn't pass OSC 8 through.
    """
    if not sys.stderr.isatty():
        return url
    esc = "\033"
    return f"{esc}]8;;{url}{esc}\\{url}{esc}]8;;{esc}\\"


def run_watcher(download_log: dict[str, Any]) -> int:
    """Run the two-phase manual paper workflow. Returns papers provided this session.

    *download_log* is the FULL download log (every PDB), so PDBs can be grouped by
    DOI for sibling replication.
    """
    cfg = get_config()
    papers_dir = cfg.papers_dir
    papers_dir.mkdir(parents=True, exist_ok=True)

    groups = _build_doi_groups(download_log, papers_dir)

    # Phase 1: fill same-paper siblings from PDFs already present (metadata-only).
    replicated = _replicate_existing(groups, papers_dir)
    print(
        f"\nFilled {replicated} structure(s) from papers you already have (same DOI).",
        file=sys.stderr,
    )

    # Worklist: DOIs with a paywalled PDB still missing its PDF.
    todo = sorted(
        (
            (doi, plist)
            for doi, plist in groups.items()
            if any(p["status"] == "fallback_paywalled" and not p["pdf_exists"] for p in plist)
        ),
        key=lambda item: item[0],
    )
    if not todo:
        print("All paywalled papers are resolved. Nothing to download.", file=sys.stderr)
        return 0

    print(
        f"\n{len(todo)} paywalled paper(s) need a manual download — one at a time.\n"
        "For each: open the link, save the PDF into papers/ (any filename), and it is\n"
        "renamed automatically. Press Ctrl+C anytime to stop (resume with --watch-only).",
        file=sys.stderr,
    )

    provided = 0
    try:
        for index, (doi, plist) in enumerate(todo, start=1):
            missing = [p for p in plist if not p["pdf_exists"]]
            if not missing:
                continue
            primary = missing[0]
            # Save the dropped PDF to the canonical DOI-named file (one per paper)
            # so every same-DOI sibling resolves to it without a physical copy.
            target = (
                papers_dir / canonical_pdf_name(primary["pdb_id"], doi)
                if (DOI_FILENAME_STORAGE and doi)
                else papers_dir / f"{primary['pdb_id']}.pdf"
            )
            siblings = missing[1:]
            also = f"  (also covers {', '.join(s['pdb_id'] for s in siblings)})" if siblings else ""
            doi_url = f"https://doi.org/{doi}"
            print(f"\n[{index}/{len(todo)}] {primary['pdb_id']}{also}", file=sys.stderr)
            print(f"   open:  {_clickable(doi_url)}", file=sys.stderr)
            print("   then save the PDF into papers/ (any filename) — watching…", file=sys.stderr)

            before = {f.name for f in papers_dir.glob("*.pdf")}
            detected = False
            for _ in range(_MANUAL_DETECT_POLLS):
                if _detect_new_pdf(papers_dir, before, target):
                    detected = True
                    break
                time.sleep(WATCHER_POLL_INTERVAL)

            if not detected:
                with contextlib.suppress(EOFError):
                    input("   not auto-detected — save it then press Enter (or Ctrl+C to stop)… ")
                detected = _detect_new_pdf(papers_dir, before, target)

            if detected and target.exists():
                _log_manual(primary["pdb_id"], target, primary["entry"], "user_manual")
                print(f"   ✅ saved {primary['pdb_id']}.pdf", file=sys.stderr)
                _replicate_to_siblings(target, siblings, papers_dir)
                provided += 1
            else:
                print(f"   ⏭  skipped {primary['pdb_id']} (no PDF provided).", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped. Re-run `fetch-papers --watch-only` to resume.", file=sys.stderr)

    print(f"\nDone. {provided}/{len(todo)} papers provided this session.", file=sys.stderr)
    return provided
