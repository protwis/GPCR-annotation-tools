"""Core voting engine — majority voting, run scoring, and discrepancy detection.

All functions are **pure**: data in, data out.  No file I/O, no mutations of
input data.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from gpcr_tools.config import (
    CASE_FOLD_NAME_FIELDS,
    GROUND_TRUTH_PATHS,
    LIST_ITEM_KEY_FIELDS,
    SOFT_FIELD_KEYS,
    VOTE_NEAR_TIE_MARGIN,
)
from gpcr_tools.config import (
    list_item_identity as _list_item_identity,
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

    Uses the None-safe ``(x.get(k) or {})`` chain at every level so a missing
    or null intermediate node never raises.
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


def _case_only_canonical(values: list[Any]) -> str | None:
    """Return one canonical casing when *values* are a pure case-only variation.

    The runs are a pure-case variation when every value is a string and they
    all ``casefold()`` to a single bucket — i.e. the runs agree on the entity
    and disagree only on capitalisation ("Nanobody" vs "nanobody"). Returns
    ``None`` when that does not hold, so a genuine spelling difference (one run
    saying "CholesterolBase" against "Cholesterol") still folds to more than one
    bucket and is left to surface as a real disagreement.

    The chosen casing is deterministic so the displayed name does not drift
    between runs: most-common original spelling, then fewest uppercase letters,
    then lexicographic.
    """
    if not values or not all(isinstance(v, str) for v in values):
        return None
    if len({v.casefold() for v in values}) != 1:
        return None
    counts = Counter(values)
    return str(
        min(
            counts,
            key=lambda s: (-counts[s], sum(1 for ch in s if ch.isupper()), s),
        )
    )


def get_majority_votes(
    values: list[Any],
    path: str = "",
) -> tuple[Any, Any]:
    """Recursively compute majority votes across *values* (one entry per run).

    Returns ``(majority_data, all_votes_data)`` where *all_votes_data*
    preserves the per-value vote counts for downstream discrepancy reporting.

    Truthiness:
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

            for run_list in values:
                if not isinstance(run_list, list):
                    continue
                for idx, item in enumerate(run_list):
                    if not isinstance(item, dict):
                        continue
                    # Items lacking a usable key must neither be silently
                    # dropped nor collapsed under a placeholder like "None":
                    # group them under a stable fallback identity so distinct
                    # entities stay separate and still survive aggregation.
                    grouped_items[_list_item_identity(item, key_field, idx)].append(item)

            majority_list: list[Any] = []
            counts_list: list[Any] = []

            for group_key in sorted(grouped_items):
                items = grouped_items[group_key]
                group_path = f"{path}[{group_key}]"
                maj_item, counts_item = get_majority_votes(items, group_path)
                # Use `is not None` — an empty dict {} is a valid vote
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
    # A name field whose values differ only by letter case is one entity the
    # runs agree on. Collapse the vote to a single canonical-casing candidate
    # so the pure-case split is never read as a near-tie. A genuine spelling
    # difference folds to more than one bucket and is left to vote normally.
    if key_name in CASE_FOLD_NAME_FIELDS:
        canonical = _case_only_canonical(values)
        if canonical is not None:
            return canonical, {canonical: len(values)}

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
    path: str = "",
) -> int:
    """Score *run_data* against *majority_votes*.

    One point for each scalar leaf that matches the majority value.  Object
    lists (ligands, auxiliary_proteins) are matched item-by-item on their key
    field and scored field-by-field, instead of requiring whole-object
    equality — which never holds once soft fields are normalised to ``None``,
    so those lists would otherwise contribute nothing to best-run selection.
    """
    if isinstance(majority_votes, dict):
        if not isinstance(run_data, dict):
            return 0
        score = 0
        for key, maj_val in majority_votes.items():
            child_path = f"{path}.{key}" if path else key
            score += score_run(run_data.get(key), maj_val, child_path)
        return score

    if isinstance(majority_votes, list):
        if not isinstance(run_data, list):
            return 0
        key_field = _resolve_key_field(path)
        if key_field and majority_votes and isinstance(majority_votes[0], dict):
            run_map = {
                _list_item_identity(item, key_field, idx): item
                for idx, item in enumerate(run_data)
                if isinstance(item, dict)
            }
            score = 0
            for idx, maj_item in enumerate(majority_votes):
                if not isinstance(maj_item, dict):
                    continue
                run_item = run_map.get(_list_item_identity(maj_item, key_field, idx))
                if run_item is not None:
                    score += score_run(run_item, maj_item, f"{path}[item]")
            return score
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


def _vote_margin(all_votes: Any) -> int | None:
    """Return ``top1 - top2`` vote counts for a scalar vote-count mapping.

    Returns ``None`` when there are fewer than two candidates or the counts are
    unusable, so a single-candidate vote is never treated as a near-tie.
    """
    if not isinstance(all_votes, dict):
        return None
    counts = sorted((c for c in all_votes.values() if isinstance(c, int)), reverse=True)
    if len(counts) < 2:
        return None
    return counts[0] - counts[1]


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
            # Group by the same stable identity as get_majority_votes so
            # keyless items (placeholder "None") map 1:1 instead of collapsing.
            best_run_map: dict[str, dict[str, Any]] = {
                _list_item_identity(item, key_field, idx): item
                for idx, item in enumerate(best_run_data)
                if isinstance(item, dict)
            }
            for i, maj_item in enumerate(majority_data):
                if not isinstance(maj_item, dict):
                    continue
                item_key = _list_item_identity(maj_item, key_field, i)
                run_item = best_run_map.get(item_key)
                votes_item = (
                    all_votes_data[i]
                    if isinstance(all_votes_data, list) and i < len(all_votes_data)
                    else {}
                )
                new_path = f"{path}[{item_key}]"
                # A voted/majority list entity that the best run does not carry is
                # dropped silently by the deepcopy aggregation: recursing on a None
                # run item against a dict majority falls straight through and the
                # omission leaves no trace. Surface it as an advisory record (a
                # review marker, like the near-tie case below) so a curator can see
                # an entity some runs reported but the chosen run omitted -- e.g. a
                # ligand only a minority of runs found, or a minority site-label
                # variant. Advisory only: this records but does not gate, and it
                # never enters the validation report's critical_warnings.
                #
                # gating=False distinguishes this minority-omission advisory from a
                # near-tie disagreement: both carry needs_review=True (so both stay
                # visible to the curator), but only the near-tie record should
                # disable one-click accept-all. The curator-side gate filters on this
                # marker so the advisory is shown without blocking accept-all.
                if run_item is None:
                    discrepancies.append(
                        {
                            "path": new_path,
                            "best_run_value": None,
                            "majority_vote_value": maj_item,
                            "all_votes": votes_item,
                            "needs_review": True,
                            "gating": False,
                        }
                    )
                    continue
                discrepancies.extend(find_discrepancies(run_item, maj_item, votes_item, new_path))
        return discrepancies

    if majority_data is not None:
        if best_run_data != majority_data:
            # A name leaf whose best-run value differs from the majority value
            # ONLY by letter case is not a real disagreement: the best run was
            # selected on the whole structure and may carry a minority
            # capitalisation of a name it otherwise agrees on. Skip it. This is
            # independent of how the best run is chosen, so it catches the
            # selected-value-differs case that the vote-collapse alone does not.
            # The skip is confined to this differing-value branch on purpose:
            # when the values are equal the vote shape must still be checked
            # below, so a genuine near-tie between distinct name wordings (which
            # do not fold to one bucket) keeps being surfaced for review.
            if (
                current_key in CASE_FOLD_NAME_FIELDS
                and isinstance(best_run_data, str)
                and isinstance(majority_data, str)
                and best_run_data.casefold() == majority_data.casefold()
            ):
                return discrepancies
            discrepancies.append(
                {
                    "path": path,
                    "best_run_value": best_run_data,
                    "majority_vote_value": majority_data,
                    "all_votes": all_votes_data,
                }
            )
        else:
            margin = _vote_margin(all_votes_data)
            if margin is not None and margin <= VOTE_NEAR_TIE_MARGIN:
                discrepancies.append(
                    {
                        "path": path,
                        "best_run_value": best_run_data,
                        "majority_vote_value": majority_data,
                        "all_votes": all_votes_data,
                        "needs_review": True,
                        "vote_margin": margin,
                    }
                )
        return discrepancies

    return []


def flag_low_confidence_consensus(
    best_run_data: Any,
    low_levels: frozenset[str],
) -> list[dict[str, Any]]:
    """Flag decision units whose self-reported confidence is in *low_levels*.

    Cross-run agreement is not correctness: a unanimous but low-confidence call
    is still a guess, so surface it for human review.  Records share the shape
    and ``.value`` path of discrepancies so the curate UI picks them up with no
    extra wiring.
    """
    flags: list[dict[str, Any]] = []
    if not isinstance(best_run_data, dict):
        return flags

    def _is_low(node: Any) -> bool:
        return isinstance(node, dict) and node.get("confidence") in low_levels

    def _record(node_path: str, node: Any) -> dict[str, Any]:
        value = node.get("value")
        return {
            "path": node_path,
            "best_run_value": value,
            "majority_vote_value": value,
            "all_votes": {},
            "needs_review": True,
            "low_confidence": node.get("confidence"),
        }

    state = (best_run_data.get("structure_info") or {}).get("state")
    if _is_low(state):
        flags.append(_record("structure_info.state.value", state))

    oligomeric_state = (best_run_data.get("receptor_info") or {}).get("oligomeric_state")
    if _is_low(oligomeric_state):
        flags.append(_record("receptor_info.oligomeric_state.value", oligomeric_state))

    for idx, lig in enumerate(best_run_data.get("ligands") or []):
        if isinstance(lig, dict) and _is_low(lig.get("role")):
            key = _list_item_identity(lig, "chem_comp_id", idx)
            flags.append(_record(f"ligands[{key}].role.value", lig["role"]))

    for idx, aux in enumerate(best_run_data.get("auxiliary_proteins") or []):
        if isinstance(aux, dict) and _is_low(aux.get("type")):
            key = _list_item_identity(aux, "name", idx)
            flags.append(_record(f"auxiliary_proteins[{key}].type.value", aux["type"]))

    return flags
