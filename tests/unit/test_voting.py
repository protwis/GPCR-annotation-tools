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

    def test_pharmacological_role_check_is_voted(self) -> None:
        # The incidental_candidate-fork field is voted by the generic nested recursion.
        runs = [
            [{"chem_comp_id": "CLR", "pharmacological_role_check": {"is_functional_ligand": True}}],
            [{"chem_comp_id": "CLR", "pharmacological_role_check": {"is_functional_ligand": True}}],
            [
                {
                    "chem_comp_id": "CLR",
                    "pharmacological_role_check": {"is_functional_ligand": False},
                }
            ],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        assert majority[0]["pharmacological_role_check"]["is_functional_ligand"] is True

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
        """Items without the key field fall back to identity/index grouping.

        Hallucinated ligands may lack chem_comp_id.  They must not be silently
        dropped — they survive through the fallback path.
        """
        runs = [
            [{"chem_comp_id": "ATP", "role": "agonist"}, {"role": "unknown"}],
            [{"chem_comp_id": "ATP", "role": "agonist"}],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        assert any(isinstance(m, dict) and m.get("chem_comp_id") == "ATP" for m in majority)
        # the keyless item must survive, not be silently skipped
        assert any(isinstance(m, dict) and m.get("role") == "unknown" for m in majority)


class TestEmptyAndPlaceholderKeys:
    """Placeholder/empty grouping keys must neither collapse distinct entities
    nor silently drop keyless items.
    """

    def test_placeholder_none_string_does_not_collapse_protein_ligands(self) -> None:
        # The schema fills chem_comp_id="None" (a string) for protein ligands.
        # Treating the literal "None" as a real key would collapse every
        # protein ligand into one bogus group, even when they are distinct.
        runs = [
            [
                {"chem_comp_id": "None", "name": "R-spondin-2", "type": "protein"},
                {"chem_comp_id": "None", "name": "ZNRF3", "type": "protein"},
            ],
            [
                {"chem_comp_id": "None", "name": "R-spondin-2", "type": "protein"},
                {"chem_comp_id": "None", "name": "ZNRF3", "type": "protein"},
            ],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        names = {m.get("name") for m in majority if isinstance(m, dict)}
        assert names == {"R-spondin-2", "ZNRF3"}

    def test_empty_value_variants_not_used_as_key(self) -> None:
        # Every EMPTY_VALUES variant must be treated as empty, so two distinct
        # items are not merged under it.
        runs = [
            [
                {"chem_comp_id": "n/a", "name": "Alpha", "type": "protein"},
                {"chem_comp_id": "null", "name": "Beta", "type": "protein"},
            ],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        names = {m.get("name") for m in majority if isinstance(m, dict)}
        assert names == {"Alpha", "Beta"}

    def test_keyless_item_not_silently_dropped(self) -> None:
        # An item without chem_comp_id must survive voting, not vanish.
        runs = [
            [
                {"chem_comp_id": "ATP", "role": "agonist"},
                {"name": "mystery-ligand", "role": "unknown"},
            ],
            [
                {"chem_comp_id": "ATP", "role": "agonist"},
                {"name": "mystery-ligand", "role": "unknown"},
            ],
        ]
        majority, _ = get_majority_votes(runs, path="ligands")
        assert any(m.get("chem_comp_id") == "ATP" for m in majority if isinstance(m, dict))
        assert any(m.get("name") == "mystery-ligand" for m in majority if isinstance(m, dict))


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

    def test_source_excluded(self) -> None:
        # The evidence "source" field is explanatory provenance (paper vs PDB
        # metadata), not an ingested decision value; like reasoning/quote it
        # must not drive cross-run voting.
        runs = [
            {"value": "X", "source": "Paper"},
            {"value": "X", "source": "Both Paper and PDB Metadata"},
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["value"] == "X"
        assert majority["source"] is None

    def test_source_not_a_discrepancy(self) -> None:
        best = {"value": "X", "source": "Paper"}
        majority = {"value": "X", "source": "Both Paper and PDB Metadata"}
        votes = {
            "value": {"X": 2},
            "source": {"Paper": 1, "Both Paper and PDB Metadata": 1},
        }
        discs = find_discrepancies(best, majority, votes)
        assert all(not d["path"].endswith("source") for d in discs)

    def test_provenance_excluded(self) -> None:
        # The per-run _provenance block (model/prompt/run metadata) must never
        # enter cross-run voting or produce discrepancies.
        runs = [
            {"value": "X", "_provenance": {"model_served": "a", "run": 1}},
            {"value": "X", "_provenance": {"model_served": "b", "run": 2}},
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["value"] == "X"
        assert majority["_provenance"] is None
        discs = find_discrepancies(runs[0], majority, {})
        assert all(not d["path"].endswith("_provenance") for d in discs)

    def test_justification_and_evidence_excluded(self) -> None:
        # Per-run free-text justification prose (why a site was called, the
        # evidence cited) is phrased differently every run; it must not drive
        # voting or surface as a discrepancy.
        runs = [
            {
                "site_ref": "orthosteric",
                "site_ref_justification": "contacts the canonical pocket residues",
                "evidence": "Fig 2 of the paper shows the bound pose",
            },
            {
                "site_ref": "orthosteric",
                "site_ref_justification": "sits in the conserved binding cleft",
                "evidence": "described in the results section",
            },
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["site_ref"] == "orthosteric"
        assert majority["site_ref_justification"] is None
        assert majority["evidence"] is None
        discs = find_discrepancies(runs[0], majority, {})
        assert all(not d["path"].endswith(("site_ref_justification", "evidence")) for d in discs)

    def test_structured_evidence_subtree_fully_excluded(self) -> None:
        # The structured evidence object ({source, quote_or_path, reasoning})
        # is the justification block on an inference. Its children are all
        # explanatory: source is a provenance label, the other two are prose.
        # None of them is an ingested decision value, so the whole subtree is
        # excluded from voting and never surfaces a discrepancy — even when the
        # runs disagree on every sub-field. The decided value lives in `value`.
        runs = [
            {
                "value": "active",
                "evidence": {
                    "source": "Paper",
                    "quote_or_path": "stabilised by the bound agonist",
                    "reasoning": "agonist-bound is the active conformation",
                },
            },
            {
                "value": "active",
                "evidence": {
                    "source": "PDB Metadata",
                    "quote_or_path": "annotated active in the entry",
                    "reasoning": "metadata records an active state",
                },
            },
        ]
        majority, _ = get_majority_votes(runs)
        assert majority["value"] == "active"
        assert majority["evidence"] is None
        discs = find_discrepancies(runs[0], majority, {})
        # The disagreeing source/quote/reasoning must NOT produce discrepancies.
        assert all(".evidence" not in d["path"] for d in discs)


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

    def test_near_tie_flagged_even_when_best_matches_majority(self) -> None:
        # A 6:5 majority is a near coin-flip — surface it for review even
        # though the best run agrees with the majority.
        best = {"a": "Y"}
        majority = {"a": "Y"}
        votes = {"a": {"Y": 6, "X": 5}}
        discs = find_discrepancies(best, majority, votes)
        flagged = [d for d in discs if d["path"] == "a"]
        assert flagged and flagged[0].get("needs_review") is True
        assert flagged[0].get("vote_margin") == 1

    def test_exact_tie_flagged(self) -> None:
        best = {"a": "Y"}
        majority = {"a": "Y"}
        votes = {"a": {"Y": 5, "X": 5}}
        discs = find_discrepancies(best, majority, votes)
        assert any(d.get("needs_review") for d in discs)

    def test_clear_majority_not_flagged(self) -> None:
        best = {"a": "Y"}
        majority = {"a": "Y"}
        votes = {"a": {"Y": 9, "X": 1}}
        assert find_discrepancies(best, majority, votes) == []

    def test_moderate_margin_not_flagged(self) -> None:
        # 6:4 (margin 2) is a clear enough majority; conservative — no flag.
        best = {"a": "Y"}
        majority = {"a": "Y"}
        votes = {"a": {"Y": 6, "X": 4}}
        assert find_discrepancies(best, majority, votes) == []

    def test_single_candidate_no_crash_no_flag(self) -> None:
        best = {"a": "Y"}
        majority = {"a": "Y"}
        votes = {"a": {"Y": 3}}
        assert find_discrepancies(best, majority, votes) == []

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

    def test_majority_list_item_absent_from_best_run_is_advisory(self) -> None:
        # A voted/majority ligand the best run does not carry must not vanish: the
        # omission is surfaced as an advisory review marker (e.g. a ligand only a
        # minority of runs reported, or a minority-only entity that the chosen run
        # dropped). best_run_value is None and the majority item is preserved.
        best = {"ligands": [{"chem_comp_id": "ATP", "role": "agonist"}]}
        majority = {
            "ligands": [
                {"chem_comp_id": "ATP", "role": "agonist"},
                {"chem_comp_id": "RET", "role": "agonist"},
            ]
        }
        votes = {
            "ligands": [
                {"role": {"agonist": 10}},
                {"role": {"agonist": 2}},
            ]
        }
        discs = find_discrepancies(best, majority, votes)
        advisory = [d for d in discs if d["path"] == "ligands[RET]"]
        assert len(advisory) == 1
        assert advisory[0]["best_run_value"] is None
        assert advisory[0]["majority_vote_value"] == {"chem_comp_id": "RET", "role": "agonist"}
        assert advisory[0]["all_votes"] == {"role": {"agonist": 2}}
        assert advisory[0]["needs_review"] is True
        # Tagged gating=False: this minority-omission advisory is shown to the
        # curator but must not block one-click accept-all (unlike a near-tie).
        assert advisory[0]["gating"] is False

    def test_omission_advisory_is_review_marker_not_critical_warning(self) -> None:
        # The omission advisory is a discrepancy record (a structured review
        # marker), never a critical-warning string. critical_warnings is built
        # independently in the validation report, so these records cannot gate
        # one-click accept through that channel.
        best = {"ligands": []}
        majority = {"ligands": [{"chem_comp_id": "RET", "role": "agonist"}]}
        votes = {"ligands": [{"role": {"agonist": 3}}]}
        discs = find_discrepancies(best, majority, votes)
        assert len(discs) == 1
        # A discrepancy record is a dict, not a warning string.
        assert isinstance(discs[0], dict)
        assert discs[0]["path"] == "ligands[RET]"
        assert discs[0]["best_run_value"] is None


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


class TestKeylessDiscrepancyDetection:
    def test_keyless_ligands_not_cross_wired(self) -> None:
        # Two protein ligands both with chem_comp_id="None" must stay distinct
        # in discrepancy detection (they collapsed under "None" before).
        best = {
            "ligands": [
                {"chem_comp_id": "None", "name": "Alpha", "role": "X"},
                {"chem_comp_id": "None", "name": "Beta", "role": "Y"},
            ]
        }
        majority = {
            "ligands": [
                {"chem_comp_id": "None", "name": "Alpha", "role": "Z"},
                {"chem_comp_id": "None", "name": "Beta", "role": "Y"},
            ]
        }
        votes = {"ligands": [{}, {}]}
        discs = find_discrepancies(best, majority, votes)
        paths = [d["path"] for d in discs]
        # Alpha's role disagreement surfaces on Alpha's own (normalized) path,
        # never collapsed under ligands[None].
        assert any("alpha" in p.lower() and p.endswith(".role") for p in paths)
        assert not any(p == "ligands[None].role" for p in paths)


class TestLowConfidenceConsensus:
    def test_low_confidence_state_flagged(self) -> None:
        from gpcr_tools.aggregator.voting import flag_low_confidence_consensus

        best = {"structure_info": {"state": {"value": "active", "confidence": "Low"}}}
        flags = flag_low_confidence_consensus(best, frozenset({"Low"}))
        assert any(
            f["path"] == "structure_info.state.value" and f.get("needs_review") for f in flags
        )

    def test_high_confidence_not_flagged(self) -> None:
        from gpcr_tools.aggregator.voting import flag_low_confidence_consensus

        best = {"structure_info": {"state": {"value": "active", "confidence": "High"}}}
        assert flag_low_confidence_consensus(best, frozenset({"Low"})) == []

    def test_low_confidence_oligomeric_state_flagged(self) -> None:
        from gpcr_tools.aggregator.voting import flag_low_confidence_consensus

        best = {"receptor_info": {"oligomeric_state": {"value": "homo-dimer", "confidence": "Low"}}}
        flags = flag_low_confidence_consensus(best, frozenset({"Low"}))
        assert any(
            f["path"] == "receptor_info.oligomeric_state.value" and f.get("needs_review")
            for f in flags
        )

    def test_high_confidence_oligomeric_state_not_flagged(self) -> None:
        from gpcr_tools.aggregator.voting import flag_low_confidence_consensus

        best = {"receptor_info": {"oligomeric_state": {"value": "monomer", "confidence": "High"}}}
        assert flag_low_confidence_consensus(best, frozenset({"Low"})) == []

    def test_low_confidence_ligand_role_flagged(self) -> None:
        from gpcr_tools.aggregator.voting import flag_low_confidence_consensus

        best = {
            "ligands": [{"chem_comp_id": "ATP", "role": {"value": "agonist", "confidence": "Low"}}]
        }
        flags = flag_low_confidence_consensus(best, frozenset({"Low"}))
        assert any(f["path"] == "ligands[ATP].role.value" for f in flags)

    def test_low_confidence_aux_type_flagged(self) -> None:
        from gpcr_tools.aggregator.voting import flag_low_confidence_consensus

        best = {
            "auxiliary_proteins": [
                {
                    "name": "Nb35",
                    "chain_id": "B",
                    "type": {"value": "nanobody", "confidence": "Low"},
                }
            ]
        }
        flags = flag_low_confidence_consensus(best, frozenset({"Low"}))
        # Identity is the normalized name plus the chain-set suffix.
        assert any(f["path"] == "auxiliary_proteins[nb35|ch:b].type.value" for f in flags)


class TestNameCaseFolding:
    """A name field whose values differ only by letter case is one entity the
    runs agree on; a pure-case split must not be surfaced for review. Genuine
    spelling differences, and any non-name field, must still surface.
    """

    def test_pure_case_split_collapses_to_single_candidate(self) -> None:
        # Five "Nanobody Nb52" vs five "nanobody Nb52" differ only by case:
        # the vote collapses to one canonical candidate so the margin logic
        # never sees a near-tie.
        runs = [{"name": "Nanobody Nb52"}] * 5 + [{"name": "nanobody Nb52"}] * 5
        majority, votes = get_majority_votes(runs)
        assert isinstance(majority["name"], str)
        assert majority["name"].casefold() == "nanobody nb52"
        # Collapsed to a single candidate carrying the full vote count.
        assert votes["name"] == {majority["name"]: 10}

    def test_collapsed_pure_case_vote_not_flagged_as_near_tie(self) -> None:
        # End to end: the pure-case 5:5 split must not produce a needs_review
        # near-tie discrepancy once the vote is collapsed.
        runs = [{"name": "Palmitic Acid"}] * 5 + [{"name": "Palmitic acid"}] * 5
        majority, votes = get_majority_votes(runs)
        best = runs[0]
        discs = find_discrepancies(best, majority, votes)
        assert discs == []

    def test_canonical_casing_is_deterministic(self) -> None:
        # Most-common original spelling wins; an even split breaks to fewest
        # uppercase then lexicographic — stable across reorderings of the input.
        forward = [{"name": "CHOLESTEROL"}] * 5 + [{"name": "Cholesterol"}] * 5
        reverse = [{"name": "Cholesterol"}] * 5 + [{"name": "CHOLESTEROL"}] * 5
        maj_f, _ = get_majority_votes(forward)
        maj_r, _ = get_majority_votes(reverse)
        # Tie on count -> fewest uppercase letters -> "Cholesterol".
        assert maj_f["name"] == "Cholesterol"
        assert maj_r["name"] == "Cholesterol"

    def test_canonical_prefers_most_common(self) -> None:
        runs = [{"name": "Cholesterol"}] * 7 + [{"name": "CHOLESTEROL"}] * 3
        majority, _ = get_majority_votes(runs)
        assert majority["name"] == "Cholesterol"

    def test_selected_value_differs_only_by_case_suppressed(self) -> None:
        # Regression guard for the second blocking path: the best run carries a
        # minority capitalisation while the majority value is the canonical one.
        # The vote collapse cannot catch this (it is about which run was chosen,
        # not the vote shape), so the discrepancy stage must skip it directly.
        best = {"name": "CALCIUM ION"}
        majority = {"name": "Calcium Ion"}
        votes = {"name": {"Calcium Ion": 10}}
        discs = find_discrepancies(best, majority, votes)
        assert discs == []

    def test_name_near_tie_still_flagged_when_best_matches_majority(self) -> None:
        # The case-only skip must stay confined to the selected-value-differs
        # path. When the best run carries the majority value but the votes are a
        # genuine multi-bucket near-tie between distinct name wordings, the
        # near-tie must still surface for review (regression guard: an earlier
        # version short-circuited any case-equal name leaf, swallowing this).
        best = {"name": "mTOR fragment"}
        majority = {"name": "mTOR fragment"}
        votes = {"name": {"mTOR fragment": 5, "Rapamycin binding fragment": 5}}
        discs = find_discrepancies(best, majority, votes)
        flagged = [d for d in discs if d["path"] == "name"]
        assert flagged and flagged[0].get("needs_review") is True
        assert flagged[0].get("vote_margin") == 0

    def test_mixed_spelling_singleton_still_flagged(self) -> None:
        # A lone genuine spelling variant ("CholesterolBase") against a
        # case-only cluster does NOT fold to one bucket, so it votes normally
        # and the best run carrying it surfaces as a real disagreement.
        runs = (
            [{"name": "Cholesterol"}] * 5
            + [{"name": "CHOLESTEROL"}] * 4
            + [{"name": "CholesterolBase"}]
        )
        majority, votes = get_majority_votes(runs)
        # Multi-bucket: the strict single-bucket gate does not fire, so the
        # name still votes normally and the majority is the cholesterol cluster.
        assert majority["name"].casefold() == "cholesterol"
        best = {"name": "CholesterolBase"}
        discs = find_discrepancies(best, majority, votes)
        assert any(d["path"] == "name" for d in discs)

    def test_true_wording_difference_still_flagged(self) -> None:
        # Different wording (not a case variant) must still vote and surface.
        runs = [{"name": "mTOR fragment"}] * 5 + [{"name": "Rapamycin binding fragment"}] * 5
        majority, votes = get_majority_votes(runs)
        best = {"name": "Rapamycin binding fragment"}
        discs = find_discrepancies(best, majority, votes)
        # The best run's value differs from the majority in more than case.
        assert any(d["path"] == "name" for d in discs)

    def test_chain_id_case_not_folded(self) -> None:
        # Scope guard: chain_id is case-sensitive (A and a can be distinct
        # chains), so a pure-case split there must NOT collapse.
        runs = [{"chain_id": "A"}] * 5 + [{"chain_id": "a"}] * 5
        _majority, votes = get_majority_votes(runs)
        # Two competing candidates remain — not collapsed to one.
        assert len(votes["chain_id"]) == 2
        # And a chain_id whose selected value differs is still a discrepancy.
        best = {"chain_id": "a"}
        discs = find_discrepancies(best, {"chain_id": "A"}, {"chain_id": {"A": 5, "a": 5}})
        assert any(d["path"] == "chain_id" for d in discs)

    def test_non_name_scalar_field_not_folded(self) -> None:
        # A non-name scalar leaf (here a role value) keeps case-sensitive
        # voting and reports when the selected value differs from the majority.
        best = {"role": "Agonist"}
        majority = {"role": "agonist"}
        votes = {"role": {"agonist": 6, "Agonist": 4}}
        discs = find_discrepancies(best, majority, votes)
        assert any(d["path"] == "role" for d in discs)

    def test_name_with_none_value_votes_normally(self) -> None:
        # A run reporting None for the name alongside string values is not a
        # pure-case variation: the non-string guard leaves the votes to be
        # counted normally rather than collapsed to one casing candidate.
        runs = [{"name": "Cholesterol"}] * 5 + [{"name": "CHOLESTEROL"}] * 4 + [{"name": None}]
        majority, votes = get_majority_votes(runs)
        assert majority["name"] == "Cholesterol"
        # Not collapsed: the None run keeps a distinct, separately counted vote.
        assert votes["name"].get(None) == 1
        assert votes["name"].get("Cholesterol") == 5


class TestObjectListScoring:
    def test_object_list_scored_by_structured_match(self) -> None:
        # An object list (ligands) must contribute to the score via per-item
        # match; whole-object equality fails once soft fields are None.
        majority = {"ligands": [{"chem_comp_id": "ATP", "role": "agonist", "reasoning": None}]}
        run = {"ligands": [{"chem_comp_id": "ATP", "role": "agonist", "reasoning": "text"}]}
        assert score_run(run, majority) >= 1

    def test_scalar_list_membership_preserved(self) -> None:
        # Plain scalar lists keep whole-item membership scoring.
        assert score_run([1, 2, 99], [1, 2, 3]) == 2

    def test_best_run_counts_ligand_match(self) -> None:
        # The run whose ligand role matches the majority is selected, even
        # though both runs tie on scalar fields.
        majority = {"ligands": [{"chem_comp_id": "ATP", "role": "agonist", "reasoning": None}]}
        runs = [
            {"ligands": [{"chem_comp_id": "ATP", "role": "antagonist", "reasoning": "a"}]},
            {"ligands": [{"chem_comp_id": "ATP", "role": "agonist", "reasoning": "b"}]},
        ]
        idx, _ = select_best_run(runs, majority)
        assert idx == 1
