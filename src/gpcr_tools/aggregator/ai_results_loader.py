"""Load multi-run AI annotation results for a PDB ID.

Reads ``run_*.json`` files from the workspace ``ai_results/{pdb_id}/``
directory.  Pure I/O — no validation or transformation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from gpcr_tools.config import get_config

logger = logging.getLogger(__name__)

_RUN_GLOB_PATTERN = "run_*.json"


def load_ai_runs(pdb_id: str) -> list[dict[str, Any]]:
    """Load all AI annotation runs for *pdb_id*, sorted by filename.

    Skips corrupt/non-parseable JSON files with a warning.
    Returns an empty list if the PDB directory does not exist or contains
    no valid run files.
    """
    cfg = get_config()
    pdb_dir = cfg.ai_results_dir / pdb_id
    if not pdb_dir.is_dir():
        logger.warning("[%s] AI results directory not found: %s", pdb_id, pdb_dir)
        return []

    run_files = sorted(pdb_dir.glob(_RUN_GLOB_PATTERN))
    if not run_files:
        logger.warning("[%s] No run files found in %s", pdb_id, pdb_dir)
        return []

    runs: list[dict[str, Any]] = []
    for run_file in run_files:
        try:
            with run_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                runs.append(data)
            else:
                logger.warning(
                    "[%s] Skipped %s: top-level is not a dict",
                    pdb_id,
                    run_file.name,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[%s] Skipped corrupt file %s: %s", pdb_id, run_file.name, exc)

    return runs


def get_pending_pdb_ids() -> list[str]:
    """Return PDB IDs that have AI results but are not yet aggregated.

    Cross-references subdirectories under ``ai_results/`` against
    ``state/aggregate_log.json`` to skip already-processed PDBs.
    """
    cfg = get_config()
    ai_dir = cfg.ai_results_dir
    if not ai_dir.is_dir():
        return []

    all_ids = sorted(
        d.name for d in ai_dir.iterdir() if d.is_dir() and list(d.glob(_RUN_GLOB_PATTERN))
    )

    aggregate_log = _load_aggregate_log(cfg.state_dir / "aggregate_log.json")
    return [pid for pid in all_ids if pid not in aggregate_log]


def _load_aggregate_log(path: Path) -> set[str]:
    """Load the set of already-processed PDB IDs from the aggregate log."""
    if not path.is_file():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return set(data.keys())
        return set()
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read aggregate log %s: %s", path, exc)
        return set()
