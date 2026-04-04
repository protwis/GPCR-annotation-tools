"""Tests for oligomer analysis suite (Epic 6).

Covers: is_gpcr_slug, get_sequence_length, _build_gpcr_roster,
classification (MONOMER/HOMOMER/HETEROMER/NO_GPCR), protomer suggestion
(5-rank framework), alert generation (hallucination/missed protomer/confirmed),
chain override (HALLUCINATION/7TM_UPGRADE triggers), 7TM analysis,
label_asym_id mapping, assembly cross-check, and warning format compliance.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

from gpcr_tools.config import (
    ALERT_7TM_UPGRADE,
    ALERT_CHAIN_ID_OVERRIDDEN,
    ALERT_CONFIRMED_OLIGOMER,
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    ALERT_SUSPICIOUS_7TM,
    OLIGOMER_HETEROMER,
    OLIGOMER_HOMOMER,
    OLIGOMER_MONOMER,
    OLIGOMER_NO_GPCR,
    TM_STATUS_COMPLETE,
    TM_STATUS_INCOMPLETE,
    TM_STATUS_UNKNOWN,
)
from gpcr_tools.validator.oligomer import (
    _analyze_tm_for_entity_instance,
    _apply_chain_override,
    _build_gpcr_roster,
    _build_label_asym_id_map,
    _generate_alerts,
    _get_assembly_cross_check,
    _suggest_primary_protomer,
    analyze_oligomer,
    get_sequence_length,
    is_gpcr_slug,
    map_uniprot_to_entity,
    scan_all_chains_7tm,
)

# ===================================================================
# is_gpcr_slug
# ===================================================================


class TestIsGpcrSlug:
    def test_positive_drd2(self) -> None:
        assert is_gpcr_slug("drd2_human") is True

    def test_positive_5ht2a(self) -> None:
        assert is_gpcr_slug("5HT2A_HUMAN") is True

    def test_negative_gnai1(self) -> None:
        assert is_gpcr_slug("gnai1_human") is False

    def test_negative_gbb1(self) -> None:
        assert is_gpcr_slug("gbb1_human") is False

    def test_negative_arr2(self) -> None:
        assert is_gpcr_slug("arr2_human") is False

    def test_negative_gnas2(self) -> None:
        assert is_gpcr_slug("gnas2_human") is False

    def test_negative_ramp(self) -> None:
        assert is_gpcr_slug("ramp1_human") is False

    def test_negative_grk(self) -> None:
        assert is_gpcr_slug("grk2_human") is False

    def test_negative_mtor(self) -> None:
        assert is_gpcr_slug("mtor_human") is False

    def test_negative_gbg(self) -> None:
        assert is_gpcr_slug("gbg1_human") is False

    def test_empty_string(self) -> None:
        assert is_gpcr_slug("") is False

    def test_none_string(self) -> None:
        # Empty string guard
        assert is_gpcr_slug("") is False


# ===================================================================
# get_sequence_length
# ===================================================================


class TestGetSequenceLength:
    def test_from_rcsb_sample(self) -> None:
        entity: dict[str, Any] = {"entity_poly": {"rcsb_sample_sequence_length": 350}}
        assert get_sequence_length(entity) == 350

    def test_from_sequence(self) -> None:
        entity: dict[str, Any] = {"entity_poly": {"pdbx_seq_one_letter_code_can": "ABCDEF"}}
        assert get_sequence_length(entity) == 6

    def test_none_entity_poly(self) -> None:
        """BL1: null entity_poly must not crash."""
        entity: dict[str, Any] = {"entity_poly": None}
        assert get_sequence_length(entity) == 0

    def test_missing_entity_poly(self) -> None:
        assert get_sequence_length({}) == 0

    def test_null_rcsb_sample_length(self) -> None:
        """BL1: rcsb_sample_sequence_length explicitly null."""
        entity: dict[str, Any] = {
            "entity_poly": {
                "rcsb_sample_sequence_length": None,
                "pdbx_seq_one_letter_code_can": "ABCD",
            }
        }
        assert get_sequence_length(entity) == 4


# ===================================================================
# _build_gpcr_roster
# ===================================================================


def _make_enriched_with_entities(
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"polymer_entities": entities}


def _make_entity(
    slug: str,
    auth_asym_id: str,
    length: int = 300,
    asym_id: str | None = None,
) -> dict[str, Any]:
    return {
        "uniprots": [{"gpcrdb_entry_name_slug": slug}],
        "entity_poly": {"rcsb_sample_sequence_length": length},
        "polymer_entity_instances": [
            {
                "rcsb_polymer_entity_instance_container_identifiers": {
                    "auth_asym_id": auth_asym_id,
                    "asym_id": asym_id or auth_asym_id,
                }
            }
        ],
    }


class TestBuildGpcrRoster:
    def test_single_gpcr(self) -> None:
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        roster = _build_gpcr_roster(enriched)
        assert "A" in roster
        assert roster["A"]["slug"] == "drd2_human"

    def test_filters_g_protein(self) -> None:
        enriched = _make_enriched_with_entities(
            [
                _make_entity("drd2_human", "A"),
                _make_entity("gnai1_human", "B"),
                _make_entity("gbb1_human", "C"),
            ]
        )
        roster = _build_gpcr_roster(enriched)
        assert "A" in roster
        assert "B" not in roster
        assert "C" not in roster

    def test_empty_entities(self) -> None:
        roster = _build_gpcr_roster({"polymer_entities": []})
        assert roster == {}

    def test_null_polymer_entities(self) -> None:
        roster = _build_gpcr_roster({})
        assert roster == {}

    def test_null_uniprots(self) -> None:
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "uniprots": None,
                    "entity_poly": {"rcsb_sample_sequence_length": 100},
                    "polymer_entity_instances": [
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A",
                                "asym_id": "A",
                            }
                        }
                    ],
                }
            ]
        }
        roster = _build_gpcr_roster(enriched)
        assert roster == {}


# ===================================================================
# Classification
# ===================================================================


class TestClassification:
    def _run_analysis(
        self,
        entities: list[dict[str, Any]],
        ai_chain: str = "A",
    ) -> dict[str, Any]:
        enriched = _make_enriched_with_entities(entities)
        data: dict[str, Any] = {"receptor_info": {"chain_id": ai_chain}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        return data["oligomer_analysis"]

    def test_no_gpcr(self) -> None:
        result = self._run_analysis([])
        assert result["classification"] == OLIGOMER_NO_GPCR

    def test_monomer(self) -> None:
        result = self._run_analysis([_make_entity("drd2_human", "A")])
        assert result["classification"] == OLIGOMER_MONOMER

    def test_homomer(self) -> None:
        result = self._run_analysis(
            [
                _make_entity("drd2_human", "A"),
                _make_entity("drd2_human", "B"),
            ]
        )
        assert result["classification"] == OLIGOMER_HOMOMER

    def test_heteromer(self) -> None:
        result = self._run_analysis(
            [
                _make_entity("drd2_human", "A"),
                _make_entity("oprm_human", "B"),
            ]
        )
        assert result["classification"] == OLIGOMER_HETEROMER


# ===================================================================
# _suggest_primary_protomer
# ===================================================================


class TestSuggestPrimaryProtomer:
    def test_no_gpcr_returns_none(self) -> None:
        result = _suggest_primary_protomer({}, {}, OLIGOMER_NO_GPCR, None, {}, [])
        assert result["chain_id"] is None

    def test_rank1_gprotein_bound(self) -> None:
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        result = _suggest_primary_protomer(roster, {}, OLIGOMER_MONOMER, "A", {"g_protein": {}}, [])
        assert result["chain_id"] == "A"
        assert result["rank_used"] == 1

    def test_rank2_exclusive_ligand(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        ligands: list[dict[str, Any]] = [{"chain_id": "A"}]
        result = _suggest_primary_protomer(roster, {}, OLIGOMER_HETEROMER, None, {}, ligands)
        assert result["chain_id"] == "A"
        assert result["rank_used"] == 2

    def test_rank3_best_7tm(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        tm_roster: dict[str, dict[str, Any]] = {
            "A": {"resolved_tms": 3, "total_tms": 7, "status": TM_STATUS_INCOMPLETE},
            "B": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
        }
        result = _suggest_primary_protomer(roster, tm_roster, OLIGOMER_HETEROMER, None, {}, [])
        assert result["chain_id"] == "B"
        assert result["rank_used"] == 3

    def test_rank4_valid_ai_choice(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        result = _suggest_primary_protomer(roster, {}, OLIGOMER_HETEROMER, "A", {}, [])
        assert result["chain_id"] == "A"
        assert result["rank_used"] == 4

    def test_rank4_longest_sequence(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 500, "asym_id": "B"},
        }
        result = _suggest_primary_protomer(roster, {}, OLIGOMER_HETEROMER, "X", {}, [])
        assert result["chain_id"] == "B"
        assert result["rank_used"] == 4

    def test_homomer_rank0(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "drd2_human", "length": 300, "asym_id": "B"},
        }
        result = _suggest_primary_protomer(roster, {}, OLIGOMER_HOMOMER, "A", {"g_protein": {}}, [])
        assert result["rank_used"] == 0
        assert "Homomer" in result["reason"]


# ===================================================================
# _generate_alerts
# ===================================================================


class TestGenerateAlerts:
    def test_hallucination(self) -> None:
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        alerts = _generate_alerts(roster, OLIGOMER_MONOMER, "X", {})
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_HALLUCINATION

    def test_missed_protomer(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        alerts = _generate_alerts(roster, OLIGOMER_HETEROMER, "A", {})
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_MISSED_PROTOMER

    def test_confirmed_oligomer(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        alerts = _generate_alerts(roster, OLIGOMER_HETEROMER, "A,B", {})
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_CONFIRMED_OLIGOMER

    def test_no_ai_chain(self) -> None:
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        alerts = _generate_alerts(roster, OLIGOMER_MONOMER, None, {})
        assert alerts == []

    def test_no_roster(self) -> None:
        alerts = _generate_alerts({}, OLIGOMER_NO_GPCR, "A", {})
        assert alerts == []


# ===================================================================
# _apply_chain_override
# ===================================================================


class TestApplyChainOverride:
    def test_hallucination_trigger(self) -> None:
        receptor_info: dict[str, Any] = {"chain_id": "X", "uniprot_entry_name": "bad_slug"}
        suggestion: dict[str, Any] = {"chain_id": "A"}
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        alerts: list[dict[str, str]] = [
            {"type": ALERT_HALLUCINATION, "message": "test hallucination"}
        ]
        result = _apply_chain_override(receptor_info, "X", suggestion, roster, {}, alerts)
        assert result["applied"] is True
        assert result["trigger"] == ALERT_HALLUCINATION
        assert result["original_chain_id"] == "X"
        assert result["corrected_chain_id"] == "A"
        assert receptor_info["chain_id"] == "A"
        assert receptor_info["uniprot_entry_name"] == "drd2_human"

    def test_7tm_upgrade_trigger(self) -> None:
        receptor_info: dict[str, Any] = {"chain_id": "A", "uniprot_entry_name": "drd2_human"}
        suggestion: dict[str, Any] = {"chain_id": "B"}
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        tm_roster: dict[str, dict[str, Any]] = {
            "A": {"resolved_tms": 3, "total_tms": 7, "status": TM_STATUS_INCOMPLETE},
            "B": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
        }
        result = _apply_chain_override(receptor_info, "A", suggestion, roster, tm_roster, [])
        assert result["applied"] is True
        assert result["trigger"] == ALERT_7TM_UPGRADE
        assert receptor_info["chain_id"] == "B"

    def test_no_override_when_ai_correct(self) -> None:
        receptor_info: dict[str, Any] = {"chain_id": "A"}
        suggestion: dict[str, Any] = {"chain_id": "A"}
        result = _apply_chain_override(receptor_info, "A", suggestion, {}, {}, [])
        assert result["applied"] is False

    def test_no_override_no_trigger(self) -> None:
        receptor_info: dict[str, Any] = {"chain_id": "A"}
        suggestion: dict[str, Any] = {"chain_id": "B"}
        tm_roster: dict[str, dict[str, Any]] = {
            "A": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
            "B": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
        }
        result = _apply_chain_override(receptor_info, "A", suggestion, {}, tm_roster, [])
        assert result["applied"] is False
        assert receptor_info["chain_id"] == "A"  # Not changed

    def test_no_chain_data(self) -> None:
        result = _apply_chain_override({}, None, {"chain_id": "A"}, {}, {}, [])
        assert result["applied"] is False

    def test_override_adds_alert(self) -> None:
        receptor_info: dict[str, Any] = {"chain_id": "X", "uniprot_entry_name": "bad"}
        alerts: list[dict[str, str]] = [{"type": ALERT_HALLUCINATION, "message": "test"}]
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        _apply_chain_override(receptor_info, "X", {"chain_id": "A"}, roster, {}, alerts)
        override_alerts = [a for a in alerts if a["type"] == ALERT_CHAIN_ID_OVERRIDDEN]
        assert len(override_alerts) == 1

    def test_override_result_keys(self) -> None:
        """BL7: return dict MUST include original_chain_id and corrected_chain_id."""
        receptor_info: dict[str, Any] = {"chain_id": "X", "uniprot_entry_name": "bad"}
        alerts: list[dict[str, str]] = [{"type": ALERT_HALLUCINATION, "message": "test"}]
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        result = _apply_chain_override(receptor_info, "X", {"chain_id": "A"}, roster, {}, alerts)
        assert "original_chain_id" in result
        assert "corrected_chain_id" in result
        assert "original_uniprot" in result
        assert "corrected_uniprot" in result


# ===================================================================
# 7TM Analysis
# ===================================================================


class TestAnalyzeTm:
    def _make_entity_with_tm(
        self,
        *,
        entity_features: list[dict[str, Any]] | None = None,
        instance_features: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        entity: dict[str, Any] = {
            "rcsb_polymer_entity_feature": entity_features or [],
            "rcsb_polymer_entity_align": [],
            "uniprots": [],
        }
        instance: dict[str, Any] = {
            "rcsb_polymer_instance_feature": instance_features or [],
        }
        return entity, instance

    def test_complete_7tm(self) -> None:
        tm_features = [
            {
                "type": "TRANSMEMBRANE",
                "name": "TM",
                "feature_positions": [
                    {"beg_seq_id": i * 30, "end_seq_id": i * 30 + 20} for i in range(1, 8)
                ],
            }
        ]
        entity, instance = self._make_entity_with_tm(entity_features=tm_features)
        result = _analyze_tm_for_entity_instance(entity, instance)
        assert result["status"] == TM_STATUS_COMPLETE
        assert result["resolved_tms"] == 7
        assert result["total_tms"] == 7

    def test_incomplete_7tm(self) -> None:
        tm_features = [
            {
                "type": "TRANSMEMBRANE",
                "name": "TM",
                "feature_positions": [
                    {"beg_seq_id": i * 30, "end_seq_id": i * 30 + 20} for i in range(1, 4)
                ],
            }
        ]
        entity, instance = self._make_entity_with_tm(entity_features=tm_features)
        result = _analyze_tm_for_entity_instance(entity, instance)
        assert result["status"] == TM_STATUS_INCOMPLETE
        assert result["resolved_tms"] == 3

    def test_unknown_no_features(self) -> None:
        entity, instance = self._make_entity_with_tm()
        result = _analyze_tm_for_entity_instance(entity, instance)
        assert result["status"] == TM_STATUS_UNKNOWN

    def test_unmodeled_reduces_coverage(self) -> None:
        """TM region fully overlapping with unmodeled -> not resolved."""
        tm_features = [
            {
                "type": "TRANSMEMBRANE",
                "name": "TM",
                "feature_positions": [{"beg_seq_id": 10, "end_seq_id": 30}],
            }
        ]
        unmodeled = [
            {
                "type": "UNOBSERVED_RESIDUE_XYZ",
                "name": "unmodeled",
                "feature_positions": [{"beg_seq_id": 10, "end_seq_id": 30}],
            }
        ]
        entity, instance = self._make_entity_with_tm(
            entity_features=tm_features, instance_features=unmodeled
        )
        result = _analyze_tm_for_entity_instance(entity, instance)
        assert result["resolved_tms"] == 0


class TestScanAllChains7tm:
    def test_with_mock_graphql(self) -> None:
        gql_entry: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity_feature": [
                        {
                            "type": "TRANSMEMBRANE",
                            "name": "TM",
                            "feature_positions": [
                                {"beg_seq_id": i * 30, "end_seq_id": i * 30 + 20}
                                for i in range(1, 8)
                            ],
                        }
                    ],
                    "rcsb_polymer_entity_align": [],
                    "uniprots": [],
                    "polymer_entity_instances": [
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A"
                            },
                            "rcsb_polymer_instance_feature": [],
                        }
                    ],
                }
            ]
        }
        results, _entry = scan_all_chains_7tm("TEST", {"A"}, graphql_entry=gql_entry)
        assert "A" in results
        assert results["A"]["status"] == TM_STATUS_COMPLETE

    def test_chain_not_in_gpcr_set(self) -> None:
        gql_entry: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity_feature": [],
                    "rcsb_polymer_entity_align": [],
                    "uniprots": [],
                    "polymer_entity_instances": [
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "B"
                            },
                            "rcsb_polymer_instance_feature": [],
                        }
                    ],
                }
            ]
        }
        results, _ = scan_all_chains_7tm("TEST", {"A"}, graphql_entry=gql_entry)
        assert "B" not in results

    def test_null_graphql(self) -> None:
        with patch(
            "gpcr_tools.validator.oligomer.fetch_polymer_features",
            return_value=None,
        ):
            results, entry = scan_all_chains_7tm("TEST", {"A"})
        assert results == {}
        assert entry is None


# ===================================================================
# map_uniprot_to_entity
# ===================================================================


class TestMapUniprotToEntity:
    def test_direct_mapping(self) -> None:
        alignments = [{"ref_beg_seq_id": 1, "entity_beg_seq_id": 1, "length": 100}]
        mapped = map_uniprot_to_entity(10, 20, alignments)
        assert mapped == [(10, 20)]

    def test_offset_mapping(self) -> None:
        alignments = [{"ref_beg_seq_id": 100, "entity_beg_seq_id": 1, "length": 50}]
        mapped = map_uniprot_to_entity(110, 120, alignments)
        assert mapped == [(11, 21)]

    def test_no_overlap(self) -> None:
        alignments = [{"ref_beg_seq_id": 1, "entity_beg_seq_id": 1, "length": 10}]
        mapped = map_uniprot_to_entity(20, 30, alignments)
        assert mapped == []


# ===================================================================
# _build_label_asym_id_map
# ===================================================================


class TestBuildLabelAsymIdMap:
    def test_basic(self) -> None:
        enriched = _make_enriched_with_entities(
            [
                _make_entity("drd2_human", "A", asym_id="X"),
            ]
        )
        mapping = _build_label_asym_id_map(enriched)
        assert mapping == {"A": "X"}

    def test_empty(self) -> None:
        mapping = _build_label_asym_id_map({})
        assert mapping == {}


# ===================================================================
# _get_assembly_cross_check
# ===================================================================


class TestAssemblyCrossCheck:
    def test_with_symmetry(self) -> None:
        enriched: dict[str, Any] = {
            "assemblies": [
                {
                    "rcsb_struct_symmetry": [
                        {
                            "oligomeric_state": "Homo 2-mer",
                            "stoichiometry": "A2",
                            "kind": "Global Symmetry",
                            "type": "Cyclic",
                        }
                    ]
                }
            ]
        }
        result = _get_assembly_cross_check(enriched)
        assert result["oligomeric_state"] == "Homo 2-mer"

    def test_empty(self) -> None:
        result = _get_assembly_cross_check({})
        assert result == {}


# ===================================================================
# Full analyze_oligomer
# ===================================================================


class TestAnalyzeOligomer:
    def test_output_keys(self) -> None:
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        result = data["oligomer_analysis"]
        assert "classification" in result
        assert "all_gpcr_chains" in result
        assert "primary_protomer_suggestion" in result
        assert "assembly_cross_check" in result
        assert "alerts" in result
        assert "chain_id_override" in result
        assert "label_asym_id_map" in result

    def test_monomer_no_override(self) -> None:
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        result = data["oligomer_analysis"]
        assert result["classification"] == OLIGOMER_MONOMER
        assert result["chain_id_override"]["applied"] is False
        assert data["receptor_info"]["chain_id"] == "A"

    def test_hallucination_override(self) -> None:
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        data: dict[str, Any] = {
            "receptor_info": {"chain_id": "X", "uniprot_entry_name": "bad_slug"}
        }
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        result = data["oligomer_analysis"]
        assert result["chain_id_override"]["applied"] is True
        assert result["chain_id_override"]["trigger"] == ALERT_HALLUCINATION
        assert data["receptor_info"]["chain_id"] == "A"
        assert data["receptor_info"]["uniprot_entry_name"] == "drd2_human"

    def test_empty_enriched(self) -> None:
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        analyze_oligomer("TEST", data, {})
        assert data["oligomer_analysis"]["classification"] == OLIGOMER_NO_GPCR


# ===================================================================
# Warning format compliance (Blood Lesson 3)
# ===================================================================


_BL3_REGEX = re.compile(r"at ['\"]([^'\"]+)['\"]")


class TestWarningFormat:
    """All alert messages must match the UI regex ``at '...'``."""

    def test_hallucination_format(self) -> None:
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        alerts = _generate_alerts(roster, OLIGOMER_MONOMER, "X", {})
        assert len(alerts) == 1
        assert _BL3_REGEX.search(alerts[0]["message"]), (
            f"Alert message does not match BL3 format: {alerts[0]['message']}"
        )

    def test_missed_protomer_format(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        alerts = _generate_alerts(roster, OLIGOMER_HETEROMER, "A", {})
        assert _BL3_REGEX.search(alerts[0]["message"])

    def test_confirmed_oligomer_format(self) -> None:
        roster = {
            "A": {"slug": "drd2_human", "length": 300, "asym_id": "A"},
            "B": {"slug": "oprm_human", "length": 350, "asym_id": "B"},
        }
        alerts = _generate_alerts(roster, OLIGOMER_HETEROMER, "A,B", {})
        assert _BL3_REGEX.search(alerts[0]["message"])

    def test_override_format(self) -> None:
        receptor_info: dict[str, Any] = {"chain_id": "X", "uniprot_entry_name": "bad"}
        alerts: list[dict[str, str]] = [{"type": ALERT_HALLUCINATION, "message": "test"}]
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        _apply_chain_override(receptor_info, "X", {"chain_id": "A"}, roster, {}, alerts)
        override_alerts = [a for a in alerts if a["type"] == ALERT_CHAIN_ID_OVERRIDDEN]
        assert len(override_alerts) == 1
        assert _BL3_REGEX.search(override_alerts[0]["message"])

    def test_suspicious_7tm_format(self) -> None:
        from gpcr_tools.validator.oligomer import analyze_oligomer

        # Mock enriched entry that produces an UNKNOWN tm_status
        enriched = {
            "polymer_entities": [
                {
                    "uniprots": [{"gpcrdb_entry_name_slug": "test_human"}],
                    "polymer_entity_instances": [
                        {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "A"}}
                    ],
                }
            ]
        }
        best_run_data: dict[str, Any] = {
            "receptor_info": {"chain_id": "A", "uniprot_entry_name": "test_human"}
        }

        # We need a quick mock to let the analysis run and reach alert injection
        with patch("gpcr_tools.validator.oligomer.scan_all_chains_7tm") as mock_scan:
            mock_scan.return_value = (
                {"A": {"status": TM_STATUS_UNKNOWN, "resolved_tms": 0, "total_tms": 0}},
                {},
            )
            with patch("gpcr_tools.validator.oligomer.is_gpcr_slug", return_value=True):
                analyze_oligomer("1XYZ", best_run_data, enriched)

        alerts: list[dict[str, str]] = best_run_data.get("oligomer_analysis", {}).get("alerts", [])
        suspicious_alerts = [a for a in alerts if a["type"] == ALERT_SUSPICIOUS_7TM]

        assert len(suspicious_alerts) == 1
        assert _BL3_REGEX.search(suspicious_alerts[0]["message"])
