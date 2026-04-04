"""Load enriched PDB metadata from the workspace.

Returns the ``data.entry`` dict from the RCSB enriched JSON, using
explicit ``isinstance`` checks rather than chained ``.get()`` calls
(Blood Lesson 1, Review 8 improvement).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from gpcr_tools.config import get_config

logger = logging.getLogger(__name__)


def load_enriched_data(pdb_id: str) -> dict[str, Any] | None:
    """Load enriched data for *pdb_id* and return the ``data.entry`` dict.

    Returns ``None`` when the file is missing, unreadable, or the JSON
    structure is unexpected.

    Blood Lesson 5 — Truthiness:
        Callers MUST check ``if enriched is None:`` — NOT ``if not enriched:``.
        An empty dict ``{}`` is valid enriched data.
    """
    cfg = get_config()
    source_path = cfg.enriched_dir / f"{pdb_id}.json"

    if not source_path.is_file():
        logger.warning("[%s] Enriched file not found: %s", pdb_id, source_path)
        return None

    try:
        with source_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[%s] Failed to read enriched file: %s", pdb_id, exc)
        return None

    # Explicit isinstance checks — not chained .get()
    if not isinstance(raw, dict):
        logger.warning("[%s] Enriched JSON top-level is not a dict", pdb_id)
        return None

    data_block = raw.get("data")
    if not isinstance(data_block, dict):
        logger.warning("[%s] Enriched JSON missing or invalid 'data' key", pdb_id)
        return None

    entry = data_block.get("entry")
    if not isinstance(entry, dict):
        logger.warning("[%s] Enriched JSON missing or invalid 'data.entry' key", pdb_id)
        return None

    return entry
