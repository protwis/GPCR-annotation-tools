"""Load enriched PDB metadata from the workspace.

Returns the ``data.entry`` dict from the RCSB enriched JSON, using
explicit ``isinstance`` checks rather than chained ``.get()`` calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from gpcr_tools.config import get_config
from gpcr_tools.fetcher.enricher import INCOMPLETE_MARKER_KEY

logger = logging.getLogger(__name__)


def enriched_is_incomplete(pdb_id: str) -> bool:
    """True if the enriched record was written during a transient API outage.

    Such a record carries a top-level incomplete marker (set when a UniProt /
    PubChem / RCSB lookup transiently failed mid-enrichment) and must be
    re-enriched before it is trusted — consuming it would bake a transient gap
    (e.g. an unresolved receptor slug) into an affirmative answer. A missing or
    unreadable file returns ``False`` (``load_enriched_data`` reports those as
    ``None``). The marker lives at the top level, OUTSIDE ``data.entry``.
    """
    source_path = get_config().enriched_dir / f"{pdb_id}.json"
    if not source_path.is_file():
        return False
    try:
        with source_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return bool(isinstance(raw, dict) and raw.get(INCOMPLETE_MARKER_KEY))


def load_enriched_data(pdb_id: str) -> dict[str, Any] | None:
    """Load enriched data for *pdb_id* and return the ``data.entry`` dict.

    Returns ``None`` when the file is missing, unreadable, or the JSON
    structure is unexpected.

    Truthiness:
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
