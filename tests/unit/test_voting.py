"""Tests for the core voting engine (Epic 1).

Covers: majority voting (scalar, dict, list-of-dict), soft-field exclusion,
run scoring, best-run selection, discrepancy detection, ground-truth path
exclusion, missing grouping key fallback, edge cases, and utility helpers.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpcr_tools.aggregator.voting import (
    _first_list_entry,
    extract_ai_g_protein,
    find_discrepancies,
    get_majority_votes,
    score_run,
    select_best_run,
)

# ===================================================================
# Helpers / fixtures
# ===================================================================


def _make_runs(*values: Any) -> list[Any]:
    """Wrap scalar values into a list for voting."""
    return list(values)


# ===================================================================
# Scalar voting
# ===================================================================


class TestScalarVoting:
    def test_unanimous(self) -> None:
        majority, votes = get_majority_votes(["X-RAY", "X-RAY", "X-RAY"])
        assert majority == "X-RAY"
        assert votes == {"X-RAY": 3}

    def test_split_vote(self) -> None:
        majority, votes = get_majority_votes(["X-RAY", "X-RAY", "EM"])
        assert majority == "X-RAY"
        assert votes["X-RAY"] == 2
        assert votes["EM"] == 1

    def test_tie_returns_most_common(self) -> None:
        """Counter.most_common(1) returns the first element — deterministic."""
        majority, votes = get_majority_votes(["A", "B", "A", "B"])
        assert majority in ("A", "B")
        assert votes["A"] == 2
        assert votes["B"] == 2

    def test_single_value(self) -> None:
        majority, votes = get_majority_votes(["only"])
        assert majority == "only"
        assert votes == {"only": 1}

    def test_empty_returns_none(self) -> None:
        majority, votes = get_majority_votes([])
        assert majority is None
        assert votes == {}

    def test_none_values(self) -> None:
        majority, votes = get_majority_votes([None, None, None])
        assert majority is None
        assert votes == {None: 3}

    def test_numeric_values(self) -> None:
        majority, _ = get_majority_votes([1.5, 1.5, 2.0])
        assert majority == 1.5

    def test_boolean_values(self) -> None:
        majority, _ = get_majority_votes([True, True, False])
        assert majority is True


# ===================================================================
# Dict voting
# ===================================================================


class TestDictVoting:
    def test_nested_fields(self) -> None:
        runs = [
            {"method": "X-RAY", "resolution": 2.5},
            {"method": "X-RAY", "resolution": 2.5},
            {"method": "EM", "resolution": 3.0},
        ]
        majority, _votes = get_majority_votes(runs)
        assert majority["method"] == "X-RAY"
        assert majority["resolution"] == 2.5

    def test_deeply_nested(self) -> None:
        runs = [
            {"a": {"b": {"c": "val1"}}},
            {"a": {"b": {"c": "val1"}}},
            {"a": {"b": {"c": "val2"}}},
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["a"]["b"]["c"] == "val1"

    def test_keys_union(self) -> None:
        """All keys across all runs are collected."""
        runs = [
            {"a": 1, "b": 2},
            {"a": 1, "c": 3},
        ]
        majority, _ = get_majority_votes(runs)
        assert set(majority.keys()) == {"a", "b", "c"}


# ===================================================================
# List-of-dict voting (grouping by key field)
# ===================================================================


class TestListOfDictVoting:
    def test_group_by_chem_comp_id(self) -> None:
        runs = [
            [{"chem_comp_id": "ATP", "role": "agonist"}],
            [{"chem_comp_id": "ATP", "role": "agonist"}],
            [{"chem_comp_id": "ATP", "role": "antagonist"}],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        assert len(majority) == 1
        assert majority[0]["chem_comp_id"] == "ATP"
        assert majority[0]["role"] == "agonist"

    def test_group_by_name(self) -> None:
        runs = [
            [{"name": "Nanobody", "type": "Nb"}],
            [{"name": "Nanobody", "type": "Nb"}],
            [{"name": "Nanobody", "type": "Ab"}],
        ]
        majority, _ = get_majority_votes(runs, path="auxiliary_proteins")
        assert len(majority) == 1
        assert majority[0]["name"] == "Nanobody"
        assert majority[0]["type"] == "Nb"

    def test_multiple_groups(self) -> None:
        runs = [
            [
                {"chem_comp_id": "ATP", "role": "agonist"},
                {"chem_comp_id": "GTP", "role": "cofactor"},
            ],
            [
                {"chem_comp_id": "ATP", "role": "agonist"},
                {"chem_comp_id": "GTP", "role": "cofactor"},
            ],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        assert len(majority) == 2
        ids = [m["chem_comp_id"] for m in majority]
        assert "ATP" in ids
        assert "GTP" in ids

    def test_missing_grouping_key_fallback(self) -> None:
        """Items without the key field fall back to index-based matching.

        This is explicitly mandated by the migration plan — hallucinated
        ligands may lack chem_comp_id.  They should not crash voting.
        """
        runs = [
            [{"chem_comp_id": "ATP", "role": "agonist"}, {"role": "unknown"}],
            [{"chem_comp_id": "ATP", "role": "agonist"}],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        # ATP should still appear — the keyless item is silently skipped
        assert any(isinstance(m, dict) and m.get("chem_comp_id") == "ATP" for m in majority)


# ===================================================================
# Soft-field exclusion
# ===================================================================


class TestSoftFieldExclusion:
    def test_soft_fields_excluded(self) -> None:
        runs = [
            {"method": "X-RAY", "reasoning": "text1", "confidence": "high"},
            {"method": "X-RAY", "reasoning": "text2", "confidence": "low"},
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["method"] == "X-RAY"
        assert majority["reasoning"] is None
        assert majority["confidence"] is None

    def test_nested_soft_field(self) -> None:
        runs = [
            {"info": {"value": "A", "note": "n1"}},
            {"info": {"value": "A", "note": "n2"}},
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["info"]["value"] == "A"
        assert majority["info"]["note"] is None


# ===================================================================
# Truthiness — Blood Lesson 5
# ===================================================================


class TestTruthiness:
    def test_empty_dict_is_valid_majority(self) -> None:
        """An empty dict {} must NOT be dropped by truthiness check."""
        runs = [
            [{"chem_comp_id": "ATP", "extra": {}}],
            [{"chem_comp_id": "ATP", "extra": {}}],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        assert len(majority) == 1
        # The maj_item (which contains extra={}) must be present
        assert majority[0]["chem_comp_id"] == "ATP"

    def test_empty_list_majority(self) -> None:
        """Empty list [] is a valid scalar vote result."""
        majority, _ = get_majority_votes([[], [], [1]])
        # Counter can't hash lists — falls back to JSON serialisation
        assert majority == []


# ===================================================================
# Run scoring
# ===================================================================


class TestScoring:
    def test_perfect_match(self) -> None:
        majority = {"a": 1, "b": 2}
        run = {"a": 1, "b": 2}
        assert score_run(run, majority) == 2

    def test_partial_match(self) -> None:
        majority = {"a": 1, "b": 2}
        run = {"a": 1, "b": 99}
        assert score_run(run, majority) == 1

    def test_no_match(self) -> None:
        majority = {"a": 1}
        run = {"a": 99}
        assert score_run(run, majority) == 0

    def test_nested_scoring(self) -> None:
        majority = {"x": {"y": "val"}}
        run = {"x": {"y": "val"}}
        assert score_run(run, majority) == 1

    def test_list_scoring(self) -> None:
        majority = {"items": [1, 2, 3]}
        run = {"items": [1, 2, 99]}
        assert score_run(run, majority) == 2

    def test_none_majority_scores_zero(self) -> None:
        assert score_run("anything", None) == 0

    def test_type_mismatch_dict_vs_scalar(self) -> None:
        assert score_run("string", {"a": 1}) == 0

    def test_type_mismatch_list_vs_scalar(self) -> None:
        assert score_run("string", [1, 2]) == 0


# ===================================================================
# Best-run selection
# ===================================================================


class TestSelectBestRun:
    def test_selects_highest_score(self) -> None:
        runs = [
            {"a": 1, "b": 99},  # score 1
            {"a": 1, "b": 2},  # score 2 (best)
            {"a": 99, "b": 99},  # score 0
        ]
        majority = {"a": 1, "b": 2}
        idx, best = select_best_run(runs, majority)
        assert idx == 1
        assert best["b"] == 2

    def test_tie_breaks_by_lowest_index(self) -> None:
        runs = [
            {"a": 1},
            {"a": 1},
        ]
        majority = {"a": 1}
        idx, _ = select_best_run(runs, majority)
        assert idx == 0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            select_best_run([], {"a": 1})


# ===================================================================
# Discrepancy detection
# ===================================================================


class TestDiscrepancies:
    def test_no_discrepancy_on_match(self) -> None:
        data = {"a": 1, "b": 2}
        assert find_discrepancies(data, data, {}) == []

    def test_scalar_discrepancy(self) -> None:
        best = {"a": "X"}
        majority = {"a": "Y"}
        votes = {"a": {"X": 1, "Y": 2}}
        discs = find_discrepancies(best, majority, votes)
        assert len(discs) == 1
        assert discs[0]["path"] == "a"
        assert discs[0]["best_run_value"] == "X"
        assert discs[0]["majority_vote_value"] == "Y"

    def test_nested_path(self) -> None:
        best = {"x": {"y": "A"}}
        majority = {"x": {"y": "B"}}
        votes = {"x": {"y": {"A": 1, "B": 2}}}
        discs = find_discrepancies(best, majority, votes)
        assert discs[0]["path"] == "x.y"

    def test_soft_field_excluded(self) -> None:
        best = {"reasoning": "A"}
        majority = {"reasoning": "B"}
        discs = find_discrepancies(best, majority, {})
        assert discs == []

    def test_ground_truth_path_excluded(self) -> None:
        best = {"structure_info": {"method": "EM"}}
        majority = {"structure_info": {"method": "X-RAY"}}
        votes = {"structure_info": {"method": {"EM": 1, "X-RAY": 2}}}
        discs = find_discrepancies(best, majority, votes)
        assert discs == []

    def test_ground_truth_resolution_excluded(self) -> None:
        best = {"structure_info": {"resolution": 2.5}}
        majority = {"structure_info": {"resolution": 3.0}}
        discs = find_discrepancies(best, majority, {})
        assert discs == []

    def test_list_with_key_field(self) -> None:
        best = {"ligands": [{"chem_comp_id": "ATP", "role": "agonist"}]}
        majority = {"ligands": [{"chem_comp_id": "ATP", "role": "antagonist"}]}
        votes = {"ligands": [{"role": {"agonist": 1, "antagonist": 2}}]}
        discs = find_discrepancies(best, majority, votes)
        assert len(discs) == 1
        assert discs[0]["path"] == "ligands[ATP].role"

    def test_type_mismatch_returns_empty(self) -> None:
        """If best_run is not a dict but majority is, return []."""
        discs = find_discrepancies("string", {"a": 1}, {})
        assert discs == []


# ===================================================================
# Utility: _first_list_entry
# ===================================================================


class TestFirstListEntry:
    def test_non_dict_container(self) -> None:
        assert _first_list_entry("not a dict", "key") == {}

    def test_missing_key(self) -> None:
        assert _first_list_entry({"a": 1}, "b") == {}

    def test_empty_list(self) -> None:
        assert _first_list_entry({"items": []}, "items") == {}

    def test_non_list_value(self) -> None:
        assert _first_list_entry({"items": "string"}, "items") == {}

    def test_returns_first_dict(self) -> None:
        container = {"items": [{"a": 1}, {"a": 2}]}
        assert _first_list_entry(container, "items") == {"a": 1}

    def test_first_non_dict_returns_empty(self) -> None:
        assert _first_list_entry({"items": [42]}, "items") == {}


# ===================================================================
# Utility: extract_ai_g_protein
# ===================================================================


class TestExtractAiGProtein:
    def test_full_path_present(self) -> None:
        data: dict[str, Any] = {
            "signaling_partners": {
                "g_protein": {"alpha_subunit": {"uniprot_entry_name": "gnas2_human"}}
            }
        }
        assert extract_ai_g_protein(data) == "gnas2_human"

    def test_missing_signaling_partners(self) -> None:
        assert extract_ai_g_protein({}) is None

    def test_null_signaling_partners(self) -> None:
        """Blood Lesson 1: explicit null must not crash."""
        assert extract_ai_g_protein({"signaling_partners": None}) is None

    def test_null_g_protein(self) -> None:
        data: dict[str, Any] = {"signaling_partners": {"g_protein": None}}
        assert extract_ai_g_protein(data) is None

    def test_null_alpha_subunit(self) -> None:
        data: dict[str, Any] = {"signaling_partners": {"g_protein": {"alpha_subunit": None}}}
        assert extract_ai_g_protein(data) is None

    def test_null_entry_name(self) -> None:
        data: dict[str, Any] = {
            "signaling_partners": {"g_protein": {"alpha_subunit": {"uniprot_entry_name": None}}}
        }
        assert extract_ai_g_protein(data) is None


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_run(self) -> None:
        runs = [{"method": "EM", "resolution": 3.0}]
        majority, _ = get_majority_votes(runs)
        assert majority["method"] == "EM"
        assert majority["resolution"] == 3.0

    def test_unhashable_values(self) -> None:
        """Lists are unhashable — should fall back to JSON serialisation."""
        runs = [
            {"tags": [1, 2]},
            {"tags": [1, 2]},
            {"tags": [3, 4]},
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["tags"] == [1, 2]

    def test_mixed_types_in_values(self) -> None:
        """Gracefully handle mixed types across runs."""
        majority, _ = get_majority_votes([1, 1, "1"])
        assert majority == 1

    def test_empty_runs_list(self) -> None:
        majority, votes = get_majority_votes([])
        assert majority is None
        assert votes == {}

    def test_all_none_values(self) -> None:
        majority, votes = get_majority_votes([None, None])
        assert majority is None
        assert votes == {None: 2}
