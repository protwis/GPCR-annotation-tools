"""Ground truth injection — overwrite objective metadata fields.

Replaces ``structure_info.method``, ``structure_info.resolution``, and
``structure_info.release_date`` in the best-run data with authoritative
values from the enriched PDB entry.

The caller is responsible for ``deepcopy``-ing the best-run data before
calling this function.  This function mutates *best_run_data* in-place.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def first_list_entry(container: Any, key: str) -> dict[str, Any]:
    """Return the first element of *container[key]* if it is a non-empty list,
    otherwise an empty dict.  *container* itself may be any type — if it is not
    a dict the function returns ``{}`` without raising.
    """
    if not isinstance(container, dict):
        return {}
    value = container.get(key)
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, dict) else {}
    return {}


def inject_ground_truth(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
) -> None:
    """Overwrite objective metadata in *best_run_data* from *enriched_entry*.

    Blood Lesson 1 — None-safety:
        Every extraction uses ``(x.get(k) or {})`` / ``or []`` / ``or ""``.
        Never ``.get(k, {})`` on external data.

    Blood Lesson 5 — Truthiness:
        ``enriched_entry`` validity is checked with ``is None`` by the caller
        (enriched_loader).  Here we trust it is a dict.
    """
    if not isinstance(best_run_data, dict):
        logger.error("[%s] best_run_data is not a dict; cannot inject ground truth", pdb_id)
        return

    structure_info: dict[str, Any] = best_run_data.setdefault("structure_info", {})

    # --- Method ---
    exptl_entry = first_list_entry(enriched_entry, "exptl")
    method: str | None = (
        (exptl_entry.get("method") or None) if isinstance(exptl_entry, dict) else None
    )

    # --- Resolution (EM preferred, X-ray fallback) ---
    em_entry = first_list_entry(enriched_entry, "em_3d_reconstruction")
    em_resolution: float | None = em_entry.get("resolution") if isinstance(em_entry, dict) else None

    refine_entry = first_list_entry(enriched_entry, "refine")
    xray_resolution: float | None = (
        refine_entry.get("ls_d_res_high") if isinstance(refine_entry, dict) else None
    )

    resolution = em_resolution if em_resolution is not None else xray_resolution

    # --- Release date ---
    rcsb_accession = enriched_entry.get("rcsb_accession_info") or {}
    release_date_raw: str | None = rcsb_accession.get("initial_release_date")
    release_date: str | None = None
    if release_date_raw:
        date_str = release_date_raw.split("T")[0]
        try:
            release_date = datetime.fromisoformat(date_str).strftime("%Y-%m-%d")
        except ValueError:
            release_date = date_str[:10]

    # --- Inject (only when authoritative value is present) ---
    if method:
        structure_info["method"] = method
    if resolution is not None:
        structure_info["resolution"] = resolution
    if release_date:
        structure_info["release_date"] = release_date
