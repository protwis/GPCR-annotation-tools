"""Core voting engine — majority voting, run scoring, and discrepancy detection.

All functions are **pure**: data in, data out.  No file I/O, no mutations of
input data.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from gpcr_tools.config import (
    GROUND_TRUTH_PATHS,
    LIST_ITEM_KEY_FIELDS,
    SOFT_FIELD_KEYS,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _first_list_entry(container: Any, key: str) -> dict[str, Any]:
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


def extract_ai_g_protein(data: dict[str, Any]) -> str | None:
    """Safely extract the G-protein alpha-subunit UniProt entry name.

    Uses the None-safe ``(x.get(k) or {})`` chain at every level
    (Blood Lesson 1).
    """
    signaling: dict[str, Any] = data.get("signaling_partners") or {}
    g_protein: dict[str, Any] = signaling.get("g_protein") or {}
    alpha: dict[str, Any] = g_protein.get("alpha_subunit") or {}
    name: str | None = alpha.get("uniprot_entry_name")
    return name


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------


def _resolve_key_field(path: str) -> str | None:
    """Return the grouping key field for list-of-dict voting, or ``None``."""
    for segment, key_field in LIST_ITEM_KEY_FIELDS.items():
        if segment in path:
            return key_field
    return None


def get_majority_votes(
    values: list[Any],
    path: str = "",
) -> tuple[Any, Any]:
    """Recursively compute majority votes across *values* (one entry per run).

    Returns ``(majority_data, all_votes_data)`` where *all_votes_data*
    preserves the per-value vote counts for downstream discrepancy reporting.

    Blood Lesson 5 — Truthiness:
        When checking if a majority item should be appended we use
        ``if maj_item is not None`` — an empty dict ``{}`` is a valid vote.
    """
    if not values:
        return None, {}

    first_item = values[0]

    # --- Soft-field exclusion ---
    key_name = path.rsplit(".", maxsplit=1)[-1]
    if key_name in SOFT_FIELD_KEYS:
        return None, {}

    # --- List-of-dict branch (e.g. ligands, auxiliary_proteins) ---
    if isinstance(first_item, list) and first_item and isinstance(first_item[0], dict):
        key_field = _resolve_key_field(path)

        if key_field:
            grouped_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
            index_items: list[tuple[int, dict[str, Any]]] = []

            for run_list in values:
                if not isinstance(run_list, list):
                    continue
                for idx, item in enumerate(run_list):
                    if not isinstance(item, dict):
                        continue
                    group_key = item.get(key_field)
                    if group_key:
                        grouped_items[group_key].append(item)
                    else:
                        # Fallback: index-based matching for items missing
                        # the key field (e.g. hallucinated ligands without
                        # chem_comp_id).  This matches legacy behaviour and
                        # was explicitly mandated — do NOT remove.
                        index_items.append((idx, item))

            majority_list: list[Any] = []
            counts_list: list[Any] = []

            for group_key in sorted(grouped_items):
                items = grouped_items[group_key]
                group_path = f"{path}[{group_key}]"
                maj_item, counts_item = get_majority_votes(items, group_path)
                # Blood Lesson 5: empty dict {} is valid
                if maj_item is not None:
                    majority_list.append(maj_item)
                    counts_list.append(counts_item)

            return majority_list, counts_list

    # --- Dict branch ---
    if isinstance(first_item, dict):
        all_keys: set[str] = set()
        for v in values:
            if isinstance(v, dict):
                all_keys.update(v.keys())

        majority_dict: dict[str, Any] = {}
        counts_dict: dict[str, Any] = {}
        for key in sorted(all_keys):
            child_path = f"{path}.{key}" if path else key
            child_values = [v.get(key) for v in values if isinstance(v, dict)]
            majority_dict[key], counts_dict[key] = get_majority_votes(child_values, child_path)
        return majority_dict, counts_dict

    # --- Scalar branch ---
    try:
        counter = Counter(values)
        most_common = counter.most_common(1)[0][0]
        return most_common, dict(counter)
    except TypeError:
        # Unhashable values — serialise to JSON for comparison
        try:
            str_values = [json.dumps(v, sort_keys=True) for v in values]
            counter = Counter(str_values)
            most_common_str = counter.most_common(1)[0][0]
            return json.loads(most_common_str), dict(counter)
        except (TypeError, json.JSONDecodeError):
            return values[0], {"fallback": len(values)}


# ---------------------------------------------------------------------------
# Run scoring
# ---------------------------------------------------------------------------


def score_run(
    run_data: Any,
    majority_votes: Any,
) -> int:
    """Score *run_data* against *majority_votes*.

    One point for each scalar leaf that matches the majority value.
    """
    if isinstance(majority_votes, dict):
        if not isinstance(run_data, dict):
            return 0
        score = 0
        for key, maj_val in majority_votes.items():
            run_val = run_data.get(key)
            score += score_run(run_val, maj_val)
        return score

    if isinstance(majority_votes, list):
        if not isinstance(run_data, list):
            return 0
        return sum(1 for item in run_data if item in majority_votes)

    if majority_votes is not None:
        return 1 if run_data == majority_votes else 0

    return 0


def select_best_run(
    runs: list[dict[str, Any]],
    majority_votes: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Return ``(index, data)`` for the run that best matches *majority_votes*.

    Ties are broken by lowest index (earliest run).
    """
    if not runs:
        msg = "Cannot select best run from an empty list"
        raise ValueError(msg)

    scores = [score_run(run, majority_votes) for run in runs]
    best_index = scores.index(max(scores))
    return best_index, runs[best_index]


# ---------------------------------------------------------------------------
# Discrepancy detection
# ---------------------------------------------------------------------------


def find_discrepancies(
    best_run_data: Any,
    majority_data: Any,
    all_votes_data: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    """Find paths where *best_run_data* diverges from *majority_data*.

    Respects ``SOFT_FIELD_KEYS`` and ``GROUND_TRUTH_PATHS`` exclusions.
    """
    discrepancies: list[dict[str, Any]] = []

    current_key = path.rsplit(".", maxsplit=1)[-1]
    if current_key in SOFT_FIELD_KEYS:
        return []
    if path in GROUND_TRUTH_PATHS:
        return []

    if isinstance(majority_data, dict):
        if not isinstance(best_run_data, dict):
            return []
        for key, maj_val in majority_data.items():
            new_path = f"{path}.{key}" if path else key
            run_val = best_run_data.get(key) if best_run_data else None
            # None-safe: all_votes_data may itself be None
            all_votes_val = (
                (all_votes_data.get(key) or {}) if isinstance(all_votes_data, dict) else {}
            )
            discrepancies.extend(find_discrepancies(run_val, maj_val, all_votes_val, new_path))
        return discrepancies

    if isinstance(majority_data, list):
        if not isinstance(best_run_data, list):
            return []
        key_field = _resolve_key_field(path)

        if key_field:
            best_run_map: dict[str | None, dict[str, Any]] = {
                item.get(key_field): item for item in best_run_data if isinstance(item, dict)
            }
            for i, maj_item in enumerate(majority_data):
                if not isinstance(maj_item, dict):
                    continue
                item_key = maj_item.get(key_field)
                if not item_key:
                    continue
                run_item = best_run_map.get(item_key)
                votes_item = (
                    all_votes_data[i]
                    if isinstance(all_votes_data, list) and i < len(all_votes_data)
                    else {}
                )
                new_path = f"{path}[{item_key}]"
                discrepancies.extend(find_discrepancies(run_item, maj_item, votes_item, new_path))
        return discrepancies

    if majority_data is not None:
        if best_run_data != majority_data:
            discrepancies.append(
                {
                    "path": path,
                    "best_run_value": best_run_data,
                    "majority_vote_value": majority_data,
                    "all_votes": all_votes_data,
                }
            )
        return discrepancies

    return []
