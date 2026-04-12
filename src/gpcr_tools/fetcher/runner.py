"""Fetch runner — orchestrate RCSB download + enrichment for PDB IDs.

Provides ``run_fetch()`` which resolves targets, downloads raw metadata,
enriches each entry, and saves all caches at the end.
"""

from __future__ import annotations

import logging
import sys

from gpcr_tools.config import get_config
from gpcr_tools.fetcher.cache import JsonCache
from gpcr_tools.fetcher.enricher import _build_session, enrich_single_pdb
from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb
from gpcr_tools.fetcher.targets import read_targets

logger = logging.getLogger(__name__)


def run_fetch(
    *,
    pdb_id: str | None = None,
    targets_file: str | None = None,
    force: bool = False,
) -> None:
    """Execute the fetch + enrich pipeline.

    Input resolution (highest priority wins):
      1. ``pdb_id`` — single PDB
      2. ``targets_file`` — explicit file path
      3. Default — ``config.targets_file`` (workspace ``targets.txt``)
    """
    cfg = get_config()

    # Resolve target list
    if pdb_id:
        pdb_ids = [pdb_id.upper()]
    elif targets_file:
        from pathlib import Path

        pdb_ids = read_targets(Path(targets_file))
        if not pdb_ids:
            print("No PDB IDs found in the specified targets file.", file=sys.stderr)
            return
    else:
        if not cfg.targets_file.exists():
            print(
                f"Error: targets.txt not found at {cfg.targets_file}\n"
                "Please initialize the workspace first:\n\n"
                "    gpcr-tools init-workspace\n",
                file=sys.stderr,
            )
            sys.exit(1)
        pdb_ids = read_targets(cfg.targets_file)
        if not pdb_ids:
            print(
                "targets.txt is empty (no uncommented PDB IDs). Add PDB IDs and try again.",
                file=sys.stderr,
            )
            return

    # Build shared session and caches
    session = _build_session()
    cache_dir = cfg.cache_dir
    uniprot_cache = JsonCache(cache_dir / "uniprot_cache.json")
    pubchem_cache = JsonCache(cache_dir / "pubchem_cache.json")
    synonyms_cache = JsonCache(cache_dir / "pubchem_synonyms_cache.json")
    doi_cache = JsonCache(cache_dir / "doi_siblings_cache.json")
    smiles_cache = JsonCache(cache_dir / "smiles_cache.json")

    from tqdm import tqdm

    ok = 0
    fail = 0
    for pid in tqdm(pdb_ids, desc="Fetching"):
        # Step 1: Download raw metadata
        raw_data = fetch_single_pdb(pid, force=force)
        if raw_data is None:
            logger.error("[%s] RCSB download failed, skipping enrichment", pid)
            fail += 1
            continue

        # Step 2: Enrich
        success = enrich_single_pdb(
            pid,
            force=force,
            session=session,
            uniprot_cache=uniprot_cache,
            pubchem_cache=pubchem_cache,
            synonyms_cache=synonyms_cache,
            doi_cache=doi_cache,
            smiles_cache=smiles_cache,
        )
        if success:
            ok += 1
        else:
            fail += 1

    # Save all caches (best-effort, after all PDBs processed)
    for cache in (uniprot_cache, pubchem_cache, synonyms_cache, doi_cache, smiles_cache):
        try:
            cache.save()
        except OSError as exc:
            logger.warning("Failed to save cache: %s", exc)

    print(f"Fetch complete: {ok} succeeded, {fail} failed.", file=sys.stderr)
    if fail > 0:
        sys.exit(1)
