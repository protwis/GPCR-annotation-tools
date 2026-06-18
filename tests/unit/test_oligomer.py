"""Tests for oligomer analysis suite (Epic 6).

Covers: is_gpcr_slug, get_sequence_length, _build_gpcr_roster,
classification (MONOMER/HOMOMER/HETEROMER/NO_GPCR), protomer suggestion
(5-rank framework), alert generation (hallucination/missed protomer/confirmed),
chain override (HALLUCINATION/7TM_UPGRADE triggers), 7TM analysis,
label_asym_id mapping, assembly cross-check, and warning format compliance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

from gpcr_tools.config import (
    AI_OLIGOMER_HETERO_DIMER,
    AI_OLIGOMER_HOMO_DIMER,
    AI_OLIGOMER_HOMO_TETRAMER,
    AI_OLIGOMER_HOMO_TRIMER,
    AI_OLIGOMER_MONOMER,
    AI_OLIGOMER_UNKNOWN,
    ALERT_7TM_UPGRADE,
    ALERT_ASSEMBLY_MISMATCH,
    ALERT_CHAIN_ID_OVERRIDDEN,
    ALERT_CONFIRMED_OLIGOMER,
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    ALERT_MULTI_COPY_LIGAND,
    ALERT_NO_GPCR,
    ALERT_OLIGOMER_DISAGREEMENT,
    ALERT_PROTOMER_IN_AUXILIARY,
    ALERT_SUSPICIOUS_7TM,
    ALERT_TM_DATA_UNAVAILABLE,
    OLIGOMER_HETEROMER,
    OLIGOMER_HOMOMER,
    OLIGOMER_MONOMER,
    OLIGOMER_NO_GPCR,
    TM_STATUS_COMPLETE,
    TM_STATUS_INCOMPLETE,
    TM_STATUS_UNKNOWN,
)
from gpcr_tools.csv_generator.logic import resolve_partner_protomer
from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts
from gpcr_tools.validator.oligomer import (
    _analyze_tm_for_entity_instance,
    _apply_chain_override,
    _build_gpcr_roster,
    _build_label_asym_id_map,
    _generate_alerts,
    _get_assembly_cross_check,
    _parse_oligomeric_count,
    _reconcile_ai_oligomer,
    _reconcile_assembly_consistency,
    _suggest_primary_protomer,
    analyze_oligomer,
    build_nonpolymer_instance_index,
    find_multi_copy_components,
    format_7tm_status,
    get_sequence_length,
    is_gpcr_slug,
    map_uniprot_to_entity,
    reconcile_gpcr_in_auxiliary,
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


class TestRosterTmGating:
    """A chain mis-mapped to a GPCR slug but whose annotation is not 7TM (a
    peptide ligand, single-pass partner, soluble protein) must not count as a
    protomer — no false HETEROMER, no false MISSED_PROTOMER."""

    def _analyze(self, entities, tm_roster, ai_chain="A"):
        enriched = _make_enriched_with_entities(entities)
        data: dict[str, Any] = {"receptor_info": {"chain_id": ai_chain}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=(tm_roster, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        return data["oligomer_analysis"]

    def test_zero_tm_peptide_ligand_chain_excluded(self) -> None:
        # 8XGR-shape: endothelin-1 peptide (0 TM) sharing the roster with EDNRB.
        o = self._analyze(
            [_make_entity("ednrb_human", "R"), _make_entity("edn1_human", "L")],
            {
                "R": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
                "L": {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN},
            },
            ai_chain="R",
        )
        assert o["classification"] == OLIGOMER_MONOMER
        assert not any(a["type"] == ALERT_MISSED_PROTOMER for a in o.get("alerts") or [])

    def test_single_pass_partner_excluded(self) -> None:
        # 8XFS-shape: single-pass E3 ligase ZNRF3 (1 TM) alongside 7TM LGR4.
        o = self._analyze(
            [_make_entity("lgr4_human", "A"), _make_entity("znrf3_human", "C")],
            {
                "A": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
                "C": {"resolved_tms": 1, "total_tms": 1, "status": TM_STATUS_INCOMPLETE},
            },
        )
        assert o["classification"] == OLIGOMER_MONOMER
        assert not any(a["type"] == ALERT_MISSED_PROTOMER for a in o.get("alerts") or [])

    def test_two_real_gpcrs_stay_heteromer(self) -> None:
        # Guard against over-pruning: two genuine 7TM GPCRs remain a heteromer.
        o = self._analyze(
            [_make_entity("drd2_human", "A"), _make_entity("oprm_human", "B")],
            {
                "A": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
                "B": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
            },
        )
        assert o["classification"] == OLIGOMER_HETEROMER


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

    def test_rank0_coupling_override_beats_ai_and_ligand(self) -> None:
        # GABA-B shape: agonist on GABBR1 (chain A), but the G protein couples GABBR2
        # (chain B). The geometric coupling protomer must win over the AI's chain
        # (Rank 1) and the ligand-binding chain (Rank 2).
        roster = {
            "A": {"slug": "gabr1_human", "length": 900, "asym_id": "A"},
            "B": {"slug": "gabr2_human", "length": 900, "asym_id": "B"},
        }
        ligands: list[dict[str, Any]] = [{"chain_id": "A"}]
        result = _suggest_primary_protomer(
            roster, {}, OLIGOMER_HETEROMER, "A", {"g_protein": {}}, ligands, coupling_chain="B"
        )
        assert result["chain_id"] == "B"
        assert result["rank_used"] == 0
        assert "coupling" in result["reason"].lower()

    def test_coupling_chain_absent_from_roster_falls_back(self) -> None:
        # A coupling chain not in the roster is ignored; lower ranks decide.
        roster = {"A": {"slug": "drd2_human", "length": 300, "asym_id": "A"}}
        result = _suggest_primary_protomer(
            roster, {}, OLIGOMER_MONOMER, "A", {"g_protein": {}}, [], coupling_chain="Z"
        )
        assert result["chain_id"] == "A"
        assert result["rank_used"] == 1

    def test_homomer_with_coupling_chain(self) -> None:
        # Both rank-0 paths active: the coupling block picks the chain, then the
        # homomer relabel prefixes the reason. rank stays 0; reason carries both.
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm2_human", "length": 800, "asym_id": "B"},
        }
        result = _suggest_primary_protomer(
            roster, {}, OLIGOMER_HOMOMER, "A", {"g_protein": {}}, [], coupling_chain="B"
        )
        assert result["chain_id"] == "B"
        assert result["rank_used"] == 0
        assert "Homomer" in result["reason"] and "coupling" in result["reason"].lower()


def test_map_uniprot_to_entity_skips_incomplete_region():
    """An alignment region missing a coordinate field is skipped, not a KeyError
    that would fail the whole PDB's oligomer analysis."""
    regions = [{"ref_beg_seq_id": 1, "entity_beg_seq_id": 1}]  # no 'length'
    assert map_uniprot_to_entity(1, 10, regions) == []


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
        assert "nonpolymer_instance_index" in result

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
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A"
                            }
                        }
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


# ===================================================================
# build_nonpolymer_instance_index / find_multi_copy_components
# ===================================================================

_ENRICHED_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "real_pdbs" / "enriched"


def _load_enriched_entry(pdb_id: str) -> dict[str, Any]:
    """Load a real enriched fixture and return its ``data.entry`` structure dict.

    This is the same shape the pipeline hands to ``analyze_oligomer``
    (``load_enriched_data`` returns the ``data.entry`` dict).
    """
    text = (_ENRICHED_FIXTURES / f"{pdb_id}.json").read_text(encoding="utf-8")
    return json.loads(text)["data"]["entry"]


class TestBuildNonpolymerInstanceIndex:
    def test_indexes_real_multi_and_single_copies(self) -> None:
        index = build_nonpolymer_instance_index(_load_enriched_entry("5G53"))
        # Two copies of NEC on different author chains (A->E, B->H).
        assert index["NEC"] == [
            {"auth_asym_id": "A", "label_asym_id": "E", "auth_seq_id": "400"},
            {"auth_asym_id": "B", "label_asym_id": "H", "auth_seq_id": "400"},
        ]
        # A single GDP copy.
        assert index["GDP"] == [
            {"auth_asym_id": "C", "label_asym_id": "I", "auth_seq_id": "400"},
        ]

    def test_distinguishes_copies_sharing_one_author_chain(self) -> None:
        # SOG has two modelled copies, BOTH on author chain A; only the
        # label_asym_id (F vs G) tells them apart -- the reason the author chain
        # alone cannot distinguish copies.
        index = build_nonpolymer_instance_index(_load_enriched_entry("5G53"))
        sog = index["SOG"]
        assert [r["auth_asym_id"] for r in sog] == ["A", "A"]
        assert [r["label_asym_id"] for r in sog] == ["F", "G"]
        assert [r["auth_seq_id"] for r in sog] == ["501", "502"]

    def test_instances_sorted_by_label_asym_id(self) -> None:
        index = build_nonpolymer_instance_index(_load_enriched_entry("9NOR"))
        labels = [r["label_asym_id"] for r in index["NAG"]]
        assert labels == sorted(labels)
        assert len(labels) == 16

    def test_empty_when_no_nonpolymer_entities(self) -> None:
        assert build_nonpolymer_instance_index(_load_enriched_entry("8TII")) == {}

    def test_handles_missing_and_null_keys(self) -> None:
        assert build_nonpolymer_instance_index({}) == {}
        assert build_nonpolymer_instance_index({"nonpolymer_entities": None}) == {}
        assert build_nonpolymer_instance_index({"nonpolymer_entities": []}) == {}

    def test_skips_entity_without_component_id(self) -> None:
        enriched: dict[str, Any] = {
            "nonpolymer_entities": [
                {
                    "rcsb_nonpolymer_entity_container_identifiers": {},
                    "nonpolymer_entity_instances": [
                        {
                            "rcsb_nonpolymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A",
                                "asym_id": "D",
                                "auth_seq_id": "201",
                            }
                        }
                    ],
                }
            ]
        }
        assert build_nonpolymer_instance_index(enriched) == {}

    def test_skips_instance_without_label_asym_id(self) -> None:
        enriched: dict[str, Any] = {
            "nonpolymer_entities": [
                {
                    "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": "NA"},
                    "nonpolymer_entity_instances": [
                        {
                            "rcsb_nonpolymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A",
                                "auth_seq_id": "201",
                            }
                        },
                        {
                            "rcsb_nonpolymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A",
                                "asym_id": "D",
                                "auth_seq_id": "202",
                            }
                        },
                    ],
                }
            ]
        }
        # Only the instance carrying a label_asym_id survives.
        assert build_nonpolymer_instance_index(enriched) == {
            "NA": [{"auth_asym_id": "A", "label_asym_id": "D", "auth_seq_id": "202"}]
        }


class TestFindMultiCopyComponents:
    def test_reports_only_repeated_components(self) -> None:
        index = build_nonpolymer_instance_index(_load_enriched_entry("5G53"))
        # NEC and SOG each have two copies; the single GDP copy is excluded.
        assert find_multi_copy_components(index) == {"NEC": 2, "SOG": 2}

    def test_counts_every_repeated_component(self) -> None:
        index = build_nonpolymer_instance_index(_load_enriched_entry("9M88"))
        assert find_multi_copy_components(index) == {
            "5YM": 2,
            "1DO": 8,
            "A1EM3": 4,
            "A1EQ8": 4,
            "A1EM2": 2,
            "A1EM1": 2,
        }

    def test_counts_repeated_glycosylation(self) -> None:
        index = build_nonpolymer_instance_index(_load_enriched_entry("9NOR"))
        assert find_multi_copy_components(index) == {"NAG": 16}

    def test_empty_index_has_no_multi_copies(self) -> None:
        assert find_multi_copy_components({}) == {}

    def test_single_copy_excluded(self) -> None:
        index = {"GDP": [{"auth_asym_id": "C", "label_asym_id": "I", "auth_seq_id": "400"}]}
        assert find_multi_copy_components(index) == {}


class TestAnalyzeOligomerAttachesInstanceIndex:
    def test_index_attached_and_sorted(self) -> None:
        # No GPCR polymer chains -> no 7TM scan -> the test stays offline.
        enriched: dict[str, Any] = {
            "polymer_entities": [],
            "nonpolymer_entities": [
                {
                    "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": "NA"},
                    "nonpolymer_entity_instances": [
                        {
                            "rcsb_nonpolymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A",
                                "asym_id": "E",
                                "auth_seq_id": "202",
                            }
                        },
                        {
                            "rcsb_nonpolymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A",
                                "asym_id": "D",
                                "auth_seq_id": "201",
                            }
                        },
                    ],
                }
            ],
        }
        data: dict[str, Any] = {"receptor_info": {}}
        analyze_oligomer("TEST", data, enriched)
        assert data["oligomer_analysis"]["nonpolymer_instance_index"] == {
            "NA": [
                {"auth_asym_id": "A", "label_asym_id": "D", "auth_seq_id": "201"},
                {"auth_asym_id": "A", "label_asym_id": "E", "auth_seq_id": "202"},
            ]
        }


class TestMultiCopyLigandAlert:
    """analyze_oligomer raises a review alert when a component the model
    annotated as a ligand is modelled in more than one copy."""

    def _enriched(self, comps: dict[str, list[tuple[str, str, str]]]) -> dict[str, Any]:
        nonpolymer = [
            {
                "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": comp_id},
                "nonpolymer_entity_instances": [
                    {
                        "rcsb_nonpolymer_entity_instance_container_identifiers": {
                            "auth_asym_id": auth,
                            "asym_id": label,
                            "auth_seq_id": seq,
                        }
                    }
                    for (auth, label, seq) in insts
                ],
            }
            for comp_id, insts in comps.items()
        ]
        # No GPCR polymer chains -> no 7TM scan -> the test stays offline.
        return {"polymer_entities": [], "nonpolymer_entities": nonpolymer}

    def _multi_copy_alerts(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            a for a in data["oligomer_analysis"]["alerts"] if a["type"] == ALERT_MULTI_COPY_LIGAND
        ]

    def test_annotated_multi_copy_emits_alert(self) -> None:
        enriched = self._enriched({"CA": [("A", "D", "201"), ("A", "E", "202")]})
        data: dict[str, Any] = {
            "receptor_info": {},
            "ligands": [{"chem_comp_id": "CA", "name": "Calcium ion"}],
        }
        analyze_oligomer("TEST", data, enriched)
        alerts = self._multi_copy_alerts(data)
        assert len(alerts) == 1
        msg = alerts[0]["message"]
        assert "ligands[CA]" in msg
        assert "2 copies" in msg

    def test_unannotated_multi_copy_stays_silent(self) -> None:
        # NAG is modelled twice but the model did not annotate it as a ligand.
        enriched = self._enriched({"NAG": [("A", "D", "301"), ("A", "E", "302")]})
        data: dict[str, Any] = {
            "receptor_info": {},
            "ligands": [{"chem_comp_id": "CA", "name": "Calcium ion"}],
        }
        analyze_oligomer("TEST", data, enriched)
        assert self._multi_copy_alerts(data) == []

    def test_single_instance_no_alert(self) -> None:
        enriched = self._enriched({"CA": [("A", "D", "201")]})
        data: dict[str, Any] = {
            "receptor_info": {},
            "ligands": [{"chem_comp_id": "CA", "name": "Calcium ion"}],
        }
        analyze_oligomer("TEST", data, enriched)
        assert self._multi_copy_alerts(data) == []


# ===================================================================
# reconcile_gpcr_in_auxiliary — GPCR protomer mis-filed as auxiliary
# ===================================================================


def _aux(name: str, chain_id: Any, type_value: str = "Other") -> dict[str, Any]:
    """Build an auxiliary_proteins entry mirroring the schema shape."""
    return {"name": name, "type": {"value": type_value}, "chain_id": chain_id}


class TestReconcileGpcrInAuxiliary:
    """A Class C obligate-dimer partner protomer mis-filed under
    auxiliary_proteins must be evicted (with an alert) and never reach
    other_aux_proteins.csv, while genuine non-GPCR auxiliaries stay put."""

    def test_protomer_evicted_with_alert(self) -> None:
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        data: dict[str, Any] = {
            "auxiliary_proteins": [_aux("Metabotropic glutamate receptor 7", "B")],
        }
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == []
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_PROTOMER_IN_AUXILIARY
        msg = alerts[0]["message"]
        assert "auxiliary_proteins" in msg
        assert "chain B" in msg
        assert "grm7_human" in msg
        assert "Metabotropic glutamate receptor 7" in msg

    def test_evicted_partner_still_resolvable(self) -> None:
        # The partner protomer lives in all_gpcr_chains independently of the
        # auxiliary block, so eviction loses no data: resolve_partner_protomer
        # still returns it.
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        data: dict[str, Any] = {
            "auxiliary_proteins": [_aux("mGlu7 partner", "B")],
        }
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, [])
        assert data["auxiliary_proteins"] == []

        oligo = {
            "all_gpcr_chains": [{"chain_id": c, "slug": info["slug"]} for c, info in roster.items()]
        }
        partner_uniprot, partner_chains = resolve_partner_protomer(oligo, "A")
        assert partner_uniprot == "grm7_human"
        assert partner_chains == "B"

    def test_non_gpcr_auxiliary_not_evicted(self) -> None:
        # A genuine non-GPCR auxiliary (nanobody chain not in the validated
        # roster) must survive — no false eviction, even in a dimer.
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        nanobody = _aux("Nanobody-35", "N", type_value="Nanobody")
        data: dict[str, Any] = {"auxiliary_proteins": [nanobody]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == [nanobody]
        assert alerts == []

    def test_empty_chain_id_not_evicted(self) -> None:
        # An entry with empty/missing chain_id parses to no chains and is left
        # untouched (fail-safe to current behaviour).
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        empty = _aux("T4-Lysozyme", "")
        missing = _aux("BRIL", None)
        data: dict[str, Any] = {"auxiliary_proteins": [empty, missing]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == [empty, missing]
        assert alerts == []

    def test_mixed_list_only_protomer_evicted(self) -> None:
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        nanobody = _aux("Nanobody-6", "N", type_value="Nanobody")
        partner = _aux("mGlu7 partner", "B")
        data: dict[str, Any] = {"auxiliary_proteins": [nanobody, partner]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == [nanobody]
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_PROTOMER_IN_AUXILIARY

    def test_empty_roster_no_op(self) -> None:
        partner = _aux("mGlu7 partner", "B")
        data: dict[str, Any] = {"auxiliary_proteins": [partner]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, {}, OLIGOMER_HETEROMER, alerts)
        assert data["auxiliary_proteins"] == [partner]
        assert alerts == []

    def test_no_auxiliary_proteins_no_op(self) -> None:
        roster = {"A": {"slug": "grm2_human", "length": 800, "asym_id": "A"}}
        data: dict[str, Any] = {}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_MONOMER, alerts)
        assert alerts == []

    def test_monomer_fusion_on_receptor_chain_not_evicted(self) -> None:
        # A single-receptor structure has no second protomer to recover: a
        # crystallization fusion (BRIL / cytochrome b562) modelled on the lone
        # receptor chain is a sub-domain of that chain and must be kept.
        roster = {"A": {"slug": "drd2_human", "length": 400, "asym_id": "A"}}
        bril = _aux("BRIL", "A", type_value="Fusion protein")
        data: dict[str, Any] = {"auxiliary_proteins": [bril]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_MONOMER, alerts)

        assert data["auxiliary_proteins"] == [bril]
        assert alerts == []

    def test_non_transmembrane_partner_not_evicted(self) -> None:
        # An E3 ligase / R-spondin ectodomain mis-mapped to a receptor slug is
        # filtered out of the transmembrane-gated validated roster, so its chain
        # never matches and the entry survives even in a heteromer.
        validated_roster = {"A": {"slug": "lgr4_human", "length": 900, "asym_id": "A"}}
        znrf3 = _aux("E3 ubiquitin-protein ligase ZNRF3", "C, E", type_value="Other")
        data: dict[str, Any] = {"auxiliary_proteins": [znrf3]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, validated_roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == [znrf3]
        assert alerts == []

    def test_fusion_typed_aux_on_protomer_chain_not_evicted(self) -> None:
        # In a dimer, a chain that is itself a protomer can also carry a fusion
        # sub-domain (e.g. an FRB / mTOR fragment fused onto a protomer chain). An
        # entry typed "Fusion protein" on that chain is the model's own claim of a
        # fusion, so it is kept rather than evicted as a mis-filed protomer.
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        frb = _aux("FRB fragment of mTOR", "B", type_value="Fusion protein")
        data: dict[str, Any] = {"auxiliary_proteins": [frb]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == [frb]
        assert alerts == []

    def test_fusion_named_aux_on_protomer_chain_not_evicted(self) -> None:
        # Same protection by name when the type is generic: a green fluorescent
        # protein / T4 lysozyme on a protomer chain is a fusion partner, kept.
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        gfp = _aux("Green fluorescent protein", "B", type_value="Other")
        data: dict[str, Any] = {"auxiliary_proteins": [gfp]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == [gfp]
        assert alerts == []

    def test_class_c_partner_typed_other_still_evicted(self) -> None:
        # The case that must still fire: in a Class C dimer the partner protomer
        # is mis-filed as a receptor-named "Other" auxiliary on the partner chain.
        roster = {
            "A": {"slug": "grm2_human", "length": 800, "asym_id": "A"},
            "B": {"slug": "grm7_human", "length": 850, "asym_id": "B"},
        }
        partner = _aux("Metabotropic glutamate receptor 7", "B", type_value="Other")
        data: dict[str, Any] = {"auxiliary_proteins": [partner]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HETEROMER, alerts)

        assert data["auxiliary_proteins"] == []
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_PROTOMER_IN_AUXILIARY

    def test_homomer_protomer_in_aux_evicted(self) -> None:
        # A symmetric homodimer (two copies of the same receptor) can also hide a
        # mis-filed second protomer: the model files the chain-B copy as a
        # receptor-named "Other" auxiliary. Eviction must fire here just as for a
        # heteromer — both protomers are real GPCR chains.
        roster = {
            "A": {"slug": "drd2_human", "length": 400, "asym_id": "A"},
            "B": {"slug": "drd2_human", "length": 400, "asym_id": "B"},
        }
        partner = _aux("Dopamine receptor D2", "B", type_value="Other")
        data: dict[str, Any] = {"auxiliary_proteins": [partner]}
        alerts: list[dict[str, str]] = []
        reconcile_gpcr_in_auxiliary(data, roster, OLIGOMER_HOMOMER, alerts)

        assert data["auxiliary_proteins"] == []
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_PROTOMER_IN_AUXILIARY

    def test_monomer_fusion_kept_end_to_end(self) -> None:
        # Through analyze_oligomer: a single 7TM receptor (chain A) carrying a
        # cytochrome b562 fusion the model filed under auxiliary_proteins. Being a
        # monomer, the fusion is kept and no eviction alert is raised.
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        bril = _aux("Soluble cytochrome b562", "A", type_value="Fusion protein")
        data: dict[str, Any] = {
            "receptor_info": {"chain_id": "A"},
            "auxiliary_proteins": [bril],
        }
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)

        assert data["auxiliary_proteins"] == [bril]
        analysis = data["oligomer_analysis"]
        assert analysis["classification"] == OLIGOMER_MONOMER
        assert not any(
            a["type"] == ALERT_PROTOMER_IN_AUXILIARY for a in analysis.get("alerts") or []
        )

    def test_non_transmembrane_partner_kept_end_to_end(self) -> None:
        # Through analyze_oligomer: a 7TM receptor (chain A) plus a chain B that
        # carries a receptor slug but whose annotation is not transmembrane (a
        # single-pass / soluble partner). The transmembrane gate drops B from the
        # validated roster, leaving a single validated protomer, so the structure
        # is classified as a monomer; the monomer guard then returns early and the
        # auxiliary entry on chain B is kept. (The roster-membership guard on its
        # own, with a genuine second protomer present, is isolated in the unit
        # test test_non_transmembrane_partner_not_evicted.)
        enriched = _make_enriched_with_entities(
            [
                _make_entity("lgr4_human", "A"),
                _make_entity("grm7_human", "B"),
            ]
        )
        partner = _aux("E3 ubiquitin-protein ligase ZNRF3", "B", type_value="Other")
        data: dict[str, Any] = {
            "receptor_info": {"chain_id": "A"},
            "auxiliary_proteins": [partner],
        }
        tm_roster = {
            "A": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
            "B": {"resolved_tms": 1, "total_tms": 1, "status": TM_STATUS_INCOMPLETE},
        }
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=(tm_roster, None),
        ):
            analyze_oligomer("TEST", data, enriched)

        assert data["auxiliary_proteins"] == [partner]
        analysis = data["oligomer_analysis"]
        assert analysis["classification"] == OLIGOMER_MONOMER
        assert not any(
            a["type"] == ALERT_PROTOMER_IN_AUXILIARY for a in analysis.get("alerts") or []
        )

    def test_end_to_end_via_analyze_oligomer(self) -> None:
        # Through analyze_oligomer: a heterodimer (grm2 chain A + grm7 chain B)
        # where the model mis-filed chain B's protomer as an "Other" auxiliary.
        # After analysis the auxiliary block is clean and the alert is recorded.
        enriched = _make_enriched_with_entities(
            [
                _make_entity("grm2_human", "A"),
                _make_entity("grm7_human", "B"),
            ]
        )
        data: dict[str, Any] = {
            "receptor_info": {"chain_id": "A"},
            "auxiliary_proteins": [_aux("Metabotropic glutamate receptor 7", "B")],
        }
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)

        assert data["auxiliary_proteins"] == []
        analysis = data["oligomer_analysis"]
        evictions = [a for a in analysis["alerts"] if a["type"] == ALERT_PROTOMER_IN_AUXILIARY]
        assert len(evictions) == 1
        # The partner is still recorded via all_gpcr_chains -> Partner columns.
        partner_uniprot, partner_chains = resolve_partner_protomer(analysis, "A")
        assert partner_uniprot == "grm7_human"
        assert partner_chains == "B"


# ===================================================================
# _get_assembly_cross_check — candidate-assembly preference
# ===================================================================


class TestAssemblyCrossCheckCandidatePreference:
    """The cross-check prefers the assembly RCSB flags as the biological
    candidate and surfaces the candidate flag + modeled-monomer count."""

    def test_prefers_candidate_assembly(self) -> None:
        # First assembly is NOT the candidate; the second one is. The candidate's
        # symmetry block must win even though it is not first.
        enriched: dict[str, Any] = {
            "assemblies": [
                {
                    "pdbx_struct_assembly": {"rcsb_candidate_assembly": "N"},
                    "rcsb_assembly_info": {"modeled_polymer_monomer_count": 100},
                    "rcsb_struct_symmetry": [
                        {"oligomeric_state": "Monomer", "stoichiometry": ["A1"]}
                    ],
                },
                {
                    "pdbx_struct_assembly": {"rcsb_candidate_assembly": "Y"},
                    "rcsb_assembly_info": {"modeled_polymer_monomer_count": 600},
                    "rcsb_struct_symmetry": [
                        {
                            "oligomeric_state": "Hetero 6-mer",
                            "stoichiometry": ["A2", "B2", "C1", "D1"],
                        }
                    ],
                },
            ]
        }
        result = _get_assembly_cross_check(enriched)
        assert result["oligomeric_state"] == "Hetero 6-mer"
        assert result["rcsb_candidate_assembly"] == "Y"
        assert result["modeled_polymer_monomer_count"] == 600

    def test_falls_back_to_first_when_none_flagged(self) -> None:
        enriched: dict[str, Any] = {
            "assemblies": [
                {
                    "pdbx_struct_assembly": {"rcsb_candidate_assembly": "N"},
                    "rcsb_struct_symmetry": [
                        {"oligomeric_state": "Monomer", "stoichiometry": ["A1"]}
                    ],
                },
                {
                    "pdbx_struct_assembly": {"rcsb_candidate_assembly": "N"},
                    "rcsb_struct_symmetry": [
                        {"oligomeric_state": "Homo 2-mer", "stoichiometry": ["A2"]}
                    ],
                },
            ]
        }
        result = _get_assembly_cross_check(enriched)
        assert result["oligomeric_state"] == "Monomer"

    def test_none_safe_on_missing_struct_assembly(self) -> None:
        # No pdbx_struct_assembly key at all -> falls back to first, no crash.
        enriched: dict[str, Any] = {
            "assemblies": [
                {"rcsb_struct_symmetry": [{"oligomeric_state": "Monomer", "stoichiometry": ["A1"]}]}
            ]
        }
        result = _get_assembly_cross_check(enriched)
        assert result["oligomeric_state"] == "Monomer"
        assert result["rcsb_candidate_assembly"] is None
        assert result["modeled_polymer_monomer_count"] is None

    def test_real_fixture_8xfs_prefers_candidate(self) -> None:
        result = _get_assembly_cross_check(_load_enriched_entry("8XFS"))
        assert result["oligomeric_state"] == "Hetero 6-mer"
        assert result["rcsb_candidate_assembly"] == "Y"

    def test_real_fixture_5g53(self) -> None:
        result = _get_assembly_cross_check(_load_enriched_entry("5G53"))
        assert result["oligomeric_state"] == "Hetero 2-mer"
        assert result["rcsb_candidate_assembly"] == "Y"

    def test_empty_assemblies(self) -> None:
        assert _get_assembly_cross_check({"assemblies": []}) == {}
        assert _get_assembly_cross_check({"assemblies": None}) == {}
        assert _get_assembly_cross_check({}) == {}


# ===================================================================
# _get_assembly_cross_check — derived has_homo_symmetry (all blocks scanned)
# ===================================================================


class TestAssemblyCrossCheckHasHomoSymmetry:
    """The first block still supplies the displayed state, but ALL symmetry blocks
    of the chosen assembly are scanned for the derived has_homo_symmetry flag."""

    @staticmethod
    def _entry(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "assemblies": [
                {
                    "pdbx_struct_assembly": {"rcsb_candidate_assembly": "Y"},
                    "rcsb_struct_symmetry": blocks,
                }
            ]
        }

    def test_homo_first_block_true(self) -> None:
        result = self._entry([{"oligomeric_state": "Homo 2-mer", "kind": "Global Symmetry"}])
        out = _get_assembly_cross_check(result)
        assert out["oligomeric_state"] == "Homo 2-mer"
        assert out["has_homo_symmetry"] is True

    def test_homo_in_later_local_block_true(self) -> None:
        # 9AYF-style: first (Global) block is a hetero-complex, a later (Local) block
        # declares the Homo oligomer. Displayed state stays the first block's value;
        # the derived flag still picks up the later Homo block.
        out = _get_assembly_cross_check(
            self._entry(
                [
                    {"oligomeric_state": "Hetero 6-mer", "kind": "Global Symmetry"},
                    {"oligomeric_state": "Homo 2-mer", "kind": "Local Symmetry"},
                ]
            )
        )
        assert out["oligomeric_state"] == "Hetero 6-mer"
        assert out["has_homo_symmetry"] is True

    def test_no_homo_block_false(self) -> None:
        # 5G53-style: only a hetero-complex block, no Homo block anywhere.
        out = _get_assembly_cross_check(
            self._entry([{"oligomeric_state": "Hetero 2-mer", "kind": "Global Symmetry"}])
        )
        assert out["oligomeric_state"] == "Hetero 2-mer"
        assert out["has_homo_symmetry"] is False

    def test_case_insensitive_and_none_safe(self) -> None:
        out = _get_assembly_cross_check(
            self._entry(
                [
                    {"oligomeric_state": None},
                    {"oligomeric_state": "homo 3-mer"},
                ]
            )
        )
        assert out["has_homo_symmetry"] is True

    def test_real_5g53_no_homo(self) -> None:
        # The real 5G53 candidate assembly carries only a Hetero 2-mer block.
        out = _get_assembly_cross_check(_load_enriched_entry("5G53"))
        assert out["has_homo_symmetry"] is False


# ===================================================================
# _parse_oligomeric_count
# ===================================================================


class TestParseOligomericCount:
    def test_monomer(self) -> None:
        assert _parse_oligomeric_count("Monomer") == 1

    def test_homo_mer(self) -> None:
        assert _parse_oligomeric_count("Homo 2-mer") == 2

    def test_hetero_mer(self) -> None:
        assert _parse_oligomeric_count("Hetero 6-mer") == 6

    def test_double_digit(self) -> None:
        assert _parse_oligomeric_count("Hetero 12-mer") == 12

    def test_none(self) -> None:
        assert _parse_oligomeric_count(None) is None

    def test_empty(self) -> None:
        assert _parse_oligomeric_count("") is None

    def test_unparseable(self) -> None:
        assert _parse_oligomeric_count("garbage") is None

    def test_non_string(self) -> None:
        assert _parse_oligomeric_count(2) is None


# ===================================================================
# _reconcile_assembly_consistency — parallel advisory (classification untouched)
# ===================================================================


class TestReconcileAssemblyConsistency:
    """Pure, None-safe reconcile of the GPCR-centric classification against the
    biological assembly. Fires only on the two clear contradictions; silent (and
    byte-identical to before) in every ordinary case."""

    def test_monomer_higher_order_complex_fires(self) -> None:
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_MONOMER,
            {"oligomeric_state": "Hetero 6-mer", "stoichiometry": ["A2", "B2", "C1", "D1"]},
        )
        assert consistency["agrees"] is False
        assert "higher-order" in consistency["note"]
        assert alert is not None
        assert alert["type"] == ALERT_ASSEMBLY_MISMATCH

    def test_monomer_homo_assembly_fires(self) -> None:
        # MONOMER but the biological assembly is a homo-oligomer of the receptor
        # (rare, but a real contradiction: two copies in the biological unit while
        # the GPCR-only count read MONOMER) -> the higher-order branch fires.
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_MONOMER,
            {"oligomeric_state": "Homo 2-mer", "stoichiometry": ["A2"]},
        )
        assert consistency["agrees"] is False
        assert "higher-order" in consistency["note"]
        assert alert is not None
        assert alert["type"] == ALERT_ASSEMBLY_MISMATCH

    def test_homomer_not_corroborated_fires(self) -> None:
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_HOMOMER,
            {"oligomeric_state": "Hetero 2-mer", "stoichiometry": ["A1", "B1"]},
        )
        assert consistency["agrees"] is False
        assert "crystallographic copies" in consistency["note"]
        assert alert is not None
        assert alert["type"] == ALERT_ASSEMBLY_MISMATCH

    def test_homomer_monomer_assembly_fires(self) -> None:
        # HOMOMER but the biological assembly is a monomer -> not corroborated.
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_HOMOMER,
            {"oligomeric_state": "Monomer", "stoichiometry": ["A1"]},
        )
        assert consistency["agrees"] is False
        assert alert is not None

    def test_monomer_monomer_assembly_silent(self) -> None:
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_MONOMER, {"oligomeric_state": "Monomer", "stoichiometry": ["A1"]}
        )
        assert consistency["agrees"] is True
        assert alert is None

    def test_homomer_homodimer_corroborated_silent(self) -> None:
        # Corroboration is driven by the derived has_homo_symmetry flag (any Homo
        # block across the assembly), not the first block's displayed state.
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_HOMOMER,
            {"oligomeric_state": "Homo 2-mer", "stoichiometry": ["A2"], "has_homo_symmetry": True},
        )
        assert consistency["agrees"] is True
        assert alert is None

    def test_homomer_homo_block_in_later_block_silent(self) -> None:
        # The first block reads as a hetero-complex, but a later Local/Pseudo block
        # declares a Homo oligomer (has_homo_symmetry True) -> a real homodimer is
        # corroborated and the advisory stays silent (9AYF-style).
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_HOMOMER,
            {
                "oligomeric_state": "Hetero 6-mer",
                "stoichiometry": ["A2", "B2", "C2"],
                "has_homo_symmetry": True,
            },
        )
        assert consistency["agrees"] is True
        assert alert is None

    def test_absent_assembly_silent(self) -> None:
        consistency, alert = _reconcile_assembly_consistency(OLIGOMER_MONOMER, {})
        assert consistency["agrees"] is True
        assert alert is None

    def test_none_state_silent(self) -> None:
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_MONOMER, {"oligomeric_state": None}
        )
        assert consistency["agrees"] is True
        assert alert is None

    def test_heteromer_never_fires(self) -> None:
        # The advisory only watches MONOMER and HOMOMER; a HETEROMER is left alone.
        consistency, alert = _reconcile_assembly_consistency(
            OLIGOMER_HETEROMER, {"oligomeric_state": "Hetero 6-mer", "stoichiometry": ["A2", "B2"]}
        )
        assert consistency["agrees"] is True
        assert alert is None


# ===================================================================
# analyze_oligomer — assembly-consistency advisory end-to-end
# ===================================================================


class TestAnalyzeOligomerAssemblyConsistency:
    """End-to-end: the advisory rides alongside the classification (never changing
    it) on the two real edge-case fixtures, and stays silent on a clean monomer."""

    def _mismatch_alerts(self, oligo: dict[str, Any]) -> list[dict[str, Any]]:
        return [a for a in oligo["alerts"] if a["type"] == ALERT_ASSEMBLY_MISMATCH]

    def test_8xfs_monomer_flags_higher_order_complex(self) -> None:
        # 8XFS is a 2:2:2 hetero-6-mer, but only LGR4 (chain A) is a 7TM GPCR;
        # the other roster chains (R-spondin-2, ZNRF3) carry no 7TM annotation and
        # are TM-gated out, so the GPCR-centric classification is MONOMER. The
        # biological assembly contradicts that -> advisory fires, label unchanged.
        enriched = _load_enriched_entry("8XFS")
        tm_roster = {
            "A": {"resolved_tms": 7, "total_tms": 7, "status": TM_STATUS_COMPLETE},
            "B": {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN},
            "C": {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN},
            "D": {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN},
            "E": {"resolved_tms": 0, "total_tms": 0, "status": TM_STATUS_UNKNOWN},
        }
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=(tm_roster, None),
        ):
            analyze_oligomer("8XFS", data, enriched)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_MONOMER
        assert oligo["assembly_consistency"]["agrees"] is False
        alerts = self._mismatch_alerts(oligo)
        assert len(alerts) == 1
        assert "higher-order" in alerts[0]["message"]

    def test_5g53_homomer_flags_crystallographic_copies(self) -> None:
        # 5G53 has two crystallographic copies of A2A (chains A, B -> HOMOMER), but
        # the biological assembly has the receptor only once (Hetero 2-mer). The
        # advisory flags possible crystallographic copies; classification stays HOMOMER.
        enriched = _load_enriched_entry("5G53")
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("5G53", data, enriched)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_HOMOMER
        assert oligo["assembly_consistency"]["agrees"] is False
        alerts = self._mismatch_alerts(oligo)
        assert len(alerts) == 1
        assert "crystallographic copies" in alerts[0]["message"]

    def test_clean_monomer_no_advisory(self) -> None:
        # A single-GPCR structure whose biological assembly is a monomer: the
        # overwhelming normal case. No advisory alert, classification unchanged.
        enriched = _load_enriched_entry("9AS1")
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("9AS1", data, enriched)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_MONOMER
        assert oligo["assembly_consistency"]["agrees"] is True
        assert self._mismatch_alerts(oligo) == []

    def test_no_assembly_no_alert_no_crash(self) -> None:
        # None / empty assemblies -> None-safe, no advisory, no crash.
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        enriched["assemblies"] = None
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_MONOMER
        assert oligo["assembly_cross_check"] == {}
        assert oligo["assembly_consistency"]["agrees"] is True
        assert self._mismatch_alerts(oligo) == []


# ===================================================================
# ASSEMBLY_MISMATCH stays a NON-gating advisory
# ===================================================================


class TestAssemblyMismatchNonGating:
    """A HOMOMER with no Homo symmetry block fires ASSEMBLY_MISMATCH; one with a
    Homo block (even a later Local block) does not. Either way the advisory never
    reaches the curator's critical_warnings (one-click accept stays enabled)."""

    @staticmethod
    def _homomer_with_symmetry(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        enriched = _make_enriched_with_entities(
            [_make_entity("drd2_human", "A"), _make_entity("drd2_human", "B")]
        )
        enriched["assemblies"] = [
            {
                "pdbx_struct_assembly": {"rcsb_candidate_assembly": "Y"},
                "rcsb_struct_symmetry": blocks,
            }
        ]
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A, B"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        return data["oligomer_analysis"]

    def test_no_homo_block_fires_mismatch(self) -> None:
        oligo = self._homomer_with_symmetry(
            [{"oligomeric_state": "Hetero 2-mer", "kind": "Global Symmetry"}]
        )
        assert oligo["classification"] == OLIGOMER_HOMOMER
        assert any(a["type"] == ALERT_ASSEMBLY_MISMATCH for a in oligo["alerts"])

    def test_later_homo_block_silent(self) -> None:
        oligo = self._homomer_with_symmetry(
            [
                {"oligomeric_state": "Hetero 6-mer", "kind": "Global Symmetry"},
                {"oligomeric_state": "Homo 2-mer", "kind": "Local Symmetry"},
            ]
        )
        assert oligo["classification"] == OLIGOMER_HOMOMER
        assert not any(a["type"] == ALERT_ASSEMBLY_MISMATCH for a in oligo["alerts"])

    def test_mismatch_not_promoted_to_critical(self) -> None:
        # The mismatch advisory is NOT in inject_oligomer_alerts's promoted set, so
        # it never disables one-click accept -- it stays a non-gating advisory.
        oligo = self._homomer_with_symmetry(
            [{"oligomeric_state": "Hetero 2-mer", "kind": "Global Symmetry"}]
        )
        assert any(a["type"] == ALERT_ASSEMBLY_MISMATCH for a in oligo["alerts"])
        validation_data: dict[str, Any] = {}
        inject_oligomer_alerts(oligo, validation_data)
        assert not any(
            ALERT_ASSEMBLY_MISMATCH in w for w in validation_data.get("critical_warnings") or []
        )


# ===================================================================
# NO_GPCR honesty — the empty roster must alarm the curator (GATING)
# ===================================================================


class TestNoGpcrGating:
    """A GPCR-annotation tool finding no GPCR must alarm the curator: analyze_oligomer
    emits an ALERT_NO_GPCR that inject_oligomer_alerts promotes to a critical warning
    (one-click accept disabled)."""

    def _no_gpcr_oligo(self) -> dict[str, Any]:
        # An empty roster: no polymer entity carried a resolved GPCRdb slug
        # (e.g. an antibody/fusion-only entry, or an unresolved UniProt mapping).
        enriched = _make_enriched_with_entities([])
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        return data["oligomer_analysis"]

    def test_no_gpcr_emits_alert(self) -> None:
        oligo = self._no_gpcr_oligo()
        assert oligo["classification"] == OLIGOMER_NO_GPCR
        assert any(a["type"] == ALERT_NO_GPCR for a in oligo["alerts"])

    def test_no_gpcr_alert_promoted_to_critical(self) -> None:
        oligo = self._no_gpcr_oligo()
        validation_data: dict[str, Any] = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data.get("critical_warnings") or []
        assert any(ALERT_NO_GPCR in w for w in warnings)

    def test_gpcr_present_emits_no_no_gpcr_alert(self) -> None:
        # A normal monomer with a real GPCR must not raise the NO_GPCR alarm.
        enriched = _make_enriched_with_entities([_make_entity("drd2_human", "A")])
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("TEST", data, enriched)
        oligo = data["oligomer_analysis"]
        assert not any(a["type"] == ALERT_NO_GPCR for a in oligo["alerts"])


# ===================================================================
# format_7tm_status
# ===================================================================


class TestFormat7tmStatus:
    def test_complete(self) -> None:
        assert (
            format_7tm_status({"status": TM_STATUS_COMPLETE, "resolved_tms": 7, "total_tms": 7})
            == "COMPLETE 7/7"
        )

    def test_none_renders_unknown(self) -> None:
        assert format_7tm_status(None) == "UNKNOWN 0/0"

    def test_incomplete(self) -> None:
        assert (
            format_7tm_status({"status": TM_STATUS_INCOMPLETE, "resolved_tms": 4, "total_tms": 7})
            == "INCOMPLETE_7TM 4/7"
        )


# ===================================================================
# Polymer-chain 7TM scan (item B data path) + TM-fetch reliability
# ===================================================================


def _gql_entity_with_tm(auth_id: str, *, tm_count: int) -> dict[str, Any]:
    """A GraphQL polymer entity carrying *tm_count* transmembrane helices.

    ``tm_count == 0`` models a peptide / partner chain (no TM annotation), which
    reads UNKNOWN; a full receptor uses 7.
    """
    features = []
    if tm_count > 0:
        features = [
            {
                "type": "TRANSMEMBRANE",
                "name": "TM",
                "feature_positions": [
                    {"beg_seq_id": i * 30, "end_seq_id": i * 30 + 20}
                    for i in range(1, tm_count + 1)
                ],
            }
        ]
    return {
        "rcsb_polymer_entity_feature": features,
        "rcsb_polymer_entity_align": [],
        "uniprots": [],
        "polymer_entity_instances": [
            {
                "rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": auth_id},
                "rcsb_polymer_instance_feature": [],
            }
        ],
    }


class _FakePolymerFeaturesCache:
    """Dict-valued cache double; ``preload`` seeds a fresh entry."""

    def __init__(self) -> None:
        self._d: dict[str, dict[str, Any]] = {}

    def preload(self, key: str, value: dict[str, Any]) -> None:
        self._d[key] = value

    def get(self, key: str) -> dict[str, Any] | None:
        return self._d.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._d[key] = value


class TestTmFetchReliability:
    """The receptor count stays clean when the live TM-helix fetch fails."""

    def _enriched_receptor_plus_peptide(self) -> dict[str, Any]:
        # Both chains carry a GPCR slug in the roster; only the real receptor is a
        # 7TM bundle. The peptide (a short agonist) must NOT count as a receptor.
        return _make_enriched_with_entities(
            [
                _make_entity("pth1r_human", "A", length=420),
                _make_entity("pthy_human", "P", length=34),
            ]
        )

    def test_cached_tm_data_keeps_peptide_out_and_yields_monomer(self) -> None:
        # With TM data available (cache hit), the transmembrane gate drops the
        # peptide so the receptor count is 1 -> MONOMER, and no route alert fires.
        enriched = self._enriched_receptor_plus_peptide()
        cache = _FakePolymerFeaturesCache()
        cache.preload(
            "9JR3",
            {
                "polymer_entities": [
                    _gql_entity_with_tm("A", tm_count=7),
                    _gql_entity_with_tm("P", tm_count=0),
                ]
            },
        )
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        analyze_oligomer("9JR3", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_MONOMER
        assert oligo["receptor_count"] == 1
        assert oligo["tm_data_available"] is True
        assert not any(a["type"] == ALERT_TM_DATA_UNAVAILABLE for a in oligo["alerts"])
        # The peptide is shown as a non-7TM chain in the per-chain facts.
        by_chain = {c["chain_id"]: c for c in oligo["all_gpcr_chains"]}
        assert by_chain["A"]["7tm_status"] == TM_STATUS_COMPLETE
        assert by_chain["P"]["7tm_status"] == TM_STATUS_UNKNOWN

    def test_fetch_failure_routes_to_human_not_failed_open(self) -> None:
        # The live fetch fails and nothing is cached: the peptide must NOT be
        # silently counted as a receptor. Instead the classification is flagged
        # low-confidence and a route-to-human alert fires.
        enriched = self._enriched_receptor_plus_peptide()
        cache = _FakePolymerFeaturesCache()  # empty -> miss
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=None,
        ):
            analyze_oligomer("9JR3", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["tm_data_available"] is False
        assert any(a["type"] == ALERT_TM_DATA_UNAVAILABLE for a in oligo["alerts"])

    def test_route_alert_is_gating(self) -> None:
        # The route-to-human alert must disable one-click accept via the gating path.
        enriched = self._enriched_receptor_plus_peptide()
        cache = _FakePolymerFeaturesCache()
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=None,
        ):
            analyze_oligomer("9JR3", data, enriched, polymer_features_cache=cache)
        validation_data: dict[str, Any] = {}
        inject_oligomer_alerts(data["oligomer_analysis"], validation_data)
        assert any(
            ALERT_TM_DATA_UNAVAILABLE in w for w in validation_data.get("critical_warnings") or []
        )

    def test_no_cache_preserves_legacy_no_route(self) -> None:
        # Without a cache wired (e.g. offline single-PDB path), behavior is the
        # legacy fallback: no route alert, no failure flag.
        enriched = self._enriched_receptor_plus_peptide()
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.oligomer.scan_all_chains_7tm",
            return_value=({}, None),
        ):
            analyze_oligomer("9JR3", data, enriched)
        oligo = data["oligomer_analysis"]
        assert oligo["tm_data_available"] is True
        assert not any(a["type"] == ALERT_TM_DATA_UNAVAILABLE for a in oligo["alerts"])

    def test_fetch_failure_does_not_falsely_accuse_real_receptors(self) -> None:
        # On a fetch failure every GPCR chain falls back to UNKNOWN status, but
        # that is a data gap -- NOT evidence the chains are not GPCRs. The
        # "are you sure these are GPCRs?" alert must NOT fire; only the
        # data-unavailable route alert is correct.
        enriched = self._enriched_receptor_plus_peptide()
        cache = _FakePolymerFeaturesCache()  # empty -> miss
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=None,
        ):
            analyze_oligomer("9JR3", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert not any(a["type"] == ALERT_SUSPICIOUS_7TM for a in oligo["alerts"])
        assert any(a["type"] == ALERT_TM_DATA_UNAVAILABLE for a in oligo["alerts"])

    def test_suspicious_7tm_still_fires_when_fetch_succeeded(self) -> None:
        # Sanity guard for the FIX 4 condition: when the fetch SUCCEEDS but a
        # GPCR-slug chain genuinely has no TM helices, the suspicious-7TM alert
        # still fires (the data gap was the only reason to suppress it).
        enriched = _make_enriched_with_entities([_make_entity("rec_human", "A", length=350)])
        cache = _FakePolymerFeaturesCache()
        cache.preload("9JR3", {"polymer_entities": [_gql_entity_with_tm("A", tm_count=0)]})
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        analyze_oligomer("9JR3", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["tm_data_available"] is True
        assert any(a["type"] == ALERT_SUSPICIOUS_7TM for a in oligo["alerts"])

    def test_receptor_count_under_failure_is_unfiltered_roster(self) -> None:
        # FIX 7 documentation guard: when tm_data_available is False the
        # receptor_count is the UNFILTERED roster (the TM gate could not run), so
        # a peptide carrying a GPCR slug is still counted. A consumer (e.g. a
        # future receptor-level cross-check) MUST check tm_data_available before
        # trusting receptor_count. Here both the receptor and the peptide carry a
        # slug, so the unverified count is 2 -- not the true receptor count of 1.
        enriched = self._enriched_receptor_plus_peptide()
        cache = _FakePolymerFeaturesCache()  # empty -> miss
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=None,
        ):
            analyze_oligomer("9JR3", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["tm_data_available"] is False
        assert oligo["receptor_count"] == 2  # unfiltered: receptor + peptide both have slugs


# ===================================================================
# _reconcile_ai_oligomer — receptor-level AI-vs-classifier cross-check
# ===================================================================


class TestReconcileAiOligomer:
    """Pure receptor-level cross-check: the AI's receptor oligomeric state vs the
    deterministic classifier. Compares receptor count + homo/hetero kind ONLY --
    never the whole RCSB assembly -- so a receptor + G-protein complex never
    routes. Routes only on a true receptor-level disagreement."""

    # --- The explicit mapping table: AI enum <-> classifier => agree (no alert) ---

    def test_monomer_vs_monomer_agrees(self) -> None:
        # Receptor + Gabg complex: classifier MONOMER (count 1), AI 'monomer'.
        assert (
            _reconcile_ai_oligomer(AI_OLIGOMER_MONOMER, OLIGOMER_MONOMER, 1, tm_data_available=True)
            is None
        )

    def test_homo_dimer_vs_homomer_count_2_agrees(self) -> None:
        # A Class C receptor dimer (mGlu2 / CaSR): classifier HOMOMER count 2.
        assert (
            _reconcile_ai_oligomer(
                AI_OLIGOMER_HOMO_DIMER, OLIGOMER_HOMOMER, 2, tm_data_available=True
            )
            is None
        )

    def test_hetero_dimer_vs_heteromer_count_2_agrees(self) -> None:
        # GABA-B (GBR1 + GBR2): classifier HETEROMER count 2.
        assert (
            _reconcile_ai_oligomer(
                AI_OLIGOMER_HETERO_DIMER, OLIGOMER_HETEROMER, 2, tm_data_available=True
            )
            is None
        )

    def test_homo_trimer_vs_homomer_count_3_agrees(self) -> None:
        assert (
            _reconcile_ai_oligomer(
                AI_OLIGOMER_HOMO_TRIMER, OLIGOMER_HOMOMER, 3, tm_data_available=True
            )
            is None
        )

    def test_homo_tetramer_vs_homomer_count_4_agrees(self) -> None:
        assert (
            _reconcile_ai_oligomer(
                AI_OLIGOMER_HOMO_TETRAMER, OLIGOMER_HOMOMER, 4, tm_data_available=True
            )
            is None
        )

    # --- Disagreements that MUST route ---

    def test_monomer_but_classifier_counts_two_routes(self) -> None:
        # The crystallographic-copy case: AI says 'monomer' but the classifier
        # resolved 2 receptor chains -> a real ambiguity, route it.
        alert = _reconcile_ai_oligomer(
            AI_OLIGOMER_MONOMER, OLIGOMER_HOMOMER, 2, tm_data_available=True
        )
        assert alert is not None
        assert alert["type"] == ALERT_OLIGOMER_DISAGREEMENT
        assert "receptor_info" in alert["message"]

    def test_monomer_but_classifier_counts_four_routes(self) -> None:
        # Four crystallographic copies of one receptor read as a monomer in biology.
        alert = _reconcile_ai_oligomer(
            AI_OLIGOMER_MONOMER, OLIGOMER_HOMOMER, 4, tm_data_available=True
        )
        assert alert is not None
        assert alert["type"] == ALERT_OLIGOMER_DISAGREEMENT

    def test_count_mismatch_routes(self) -> None:
        # AI homo-dimer (2) vs classifier homomer count 3.
        alert = _reconcile_ai_oligomer(
            AI_OLIGOMER_HOMO_DIMER, OLIGOMER_HOMOMER, 3, tm_data_available=True
        )
        assert alert is not None
        assert alert["type"] == ALERT_OLIGOMER_DISAGREEMENT

    def test_kind_mismatch_same_count_routes(self) -> None:
        # Same count (2) but the AI says hetero-dimer while the classifier read
        # two copies of the SAME slug (HOMOMER) -> a real receptor-level conflict.
        alert = _reconcile_ai_oligomer(
            AI_OLIGOMER_HETERO_DIMER, OLIGOMER_HOMOMER, 2, tm_data_available=True
        )
        assert alert is not None
        assert alert["type"] == ALERT_OLIGOMER_DISAGREEMENT

    # --- Guards that MUST stay silent (never route) ---

    def test_unknown_never_routes(self) -> None:
        # The AI makes no receptor-level claim -> nothing to contradict.
        assert (
            _reconcile_ai_oligomer(AI_OLIGOMER_UNKNOWN, OLIGOMER_HOMOMER, 4, tm_data_available=True)
            is None
        )

    def test_tm_data_unavailable_never_routes(self) -> None:
        # The classifier count is itself unverified (it already routes via
        # TM_DATA_UNAVAILABLE); a second disagreement off it would be spurious.
        assert (
            _reconcile_ai_oligomer(
                AI_OLIGOMER_MONOMER, OLIGOMER_HOMOMER, 2, tm_data_available=False
            )
            is None
        )

    def test_no_gpcr_never_routes(self) -> None:
        # The empty roster already routes via its own NO_GPCR alert.
        assert (
            _reconcile_ai_oligomer(AI_OLIGOMER_MONOMER, OLIGOMER_NO_GPCR, 0, tm_data_available=True)
            is None
        )

    def test_missing_ai_value_never_routes(self) -> None:
        # A null / unrecognised AI value asserts no count -> silent.
        assert _reconcile_ai_oligomer(None, OLIGOMER_HOMOMER, 2, tm_data_available=True) is None
        assert (
            _reconcile_ai_oligomer("garbage", OLIGOMER_HOMOMER, 2, tm_data_available=True) is None
        )


class TestAnalyzeOligomerAiCrossCheck:
    """End-to-end: the AI's receptor oligomeric state flows through analyze_oligomer
    and the receptor-level cross-check attaches (or withholds) a routing alert."""

    def _data_with_ai_oligomer(self, chain_id: str, value: str) -> dict[str, Any]:
        return {
            "receptor_info": {
                "chain_id": chain_id,
                "oligomeric_state": {"value": value, "confidence": "High"},
            }
        }

    def _has_disagreement(self, oligo: dict[str, Any]) -> bool:
        return any(a["type"] == ALERT_OLIGOMER_DISAGREEMENT for a in oligo["alerts"])

    def test_receptor_plus_gprotein_monomer_does_not_route(self) -> None:
        # The dominant corpus case: one receptor + Gabg. Classifier MONOMER, AI
        # 'monomer' -> agree, no route. (This is the ~90% that MUST NOT route.)
        enriched = _make_enriched_with_entities(
            [
                _make_entity("drd2_human", "A", length=400),
                _make_entity("gnai1_human", "B", length=350),
                _make_entity("gbb1_human", "C", length=340),
                _make_entity("gbg2_human", "D", length=70),
            ]
        )
        cache = _FakePolymerFeaturesCache()
        cache.preload(
            "TEST",
            {
                "polymer_entities": [
                    _gql_entity_with_tm("A", tm_count=7),
                    _gql_entity_with_tm("B", tm_count=0),
                    _gql_entity_with_tm("C", tm_count=0),
                    _gql_entity_with_tm("D", tm_count=0),
                ]
            },
        )
        data = self._data_with_ai_oligomer("A", AI_OLIGOMER_MONOMER)
        analyze_oligomer("TEST", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_MONOMER
        assert oligo["receptor_count"] == 1
        assert not self._has_disagreement(oligo)

    def test_homodimer_agreement_does_not_route(self) -> None:
        # Class C receptor homodimer; AI 'homo-dimer', classifier HOMOMER count 2.
        enriched = _make_enriched_with_entities(
            [
                _make_entity("grm2_human", "A", length=400),
                _make_entity("grm2_human", "B", length=400),
            ]
        )
        cache = _FakePolymerFeaturesCache()
        cache.preload(
            "TEST",
            {
                "polymer_entities": [
                    _gql_entity_with_tm("A", tm_count=7),
                    _gql_entity_with_tm("B", tm_count=7),
                ]
            },
        )
        data = self._data_with_ai_oligomer("A", AI_OLIGOMER_HOMO_DIMER)
        analyze_oligomer("TEST", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_HOMOMER
        assert oligo["receptor_count"] == 2
        assert not self._has_disagreement(oligo)

    def test_monomer_vs_two_receptor_chains_routes(self) -> None:
        # The crystallographic-copy case: two real 7TM receptor chains of the same
        # slug, but the AI read the biology as a single monomer -> route.
        enriched = _make_enriched_with_entities(
            [
                _make_entity("opsd_bovin", "A", length=350),
                _make_entity("opsd_bovin", "B", length=350),
            ]
        )
        cache = _FakePolymerFeaturesCache()
        cache.preload(
            "TEST",
            {
                "polymer_entities": [
                    _gql_entity_with_tm("A", tm_count=7),
                    _gql_entity_with_tm("B", tm_count=7),
                ]
            },
        )
        data = self._data_with_ai_oligomer("A", AI_OLIGOMER_MONOMER)
        analyze_oligomer("TEST", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["classification"] == OLIGOMER_HOMOMER
        assert oligo["receptor_count"] == 2
        assert self._has_disagreement(oligo)

    def test_tm_data_unavailable_does_not_spuriously_route(self) -> None:
        # receptor + peptide (both slug-bearing); the TM fetch fails so the
        # classifier count is the unfiltered 2. The AI says 'monomer'. The count
        # is untrustworthy and already routes via TM_DATA_UNAVAILABLE, so the
        # cross-check must NOT add a second (spurious) disagreement.
        enriched = _make_enriched_with_entities(
            [
                _make_entity("pth1r_human", "A", length=420),
                _make_entity("pthy_human", "P", length=34),
            ]
        )
        cache = _FakePolymerFeaturesCache()  # empty -> miss
        data = self._data_with_ai_oligomer("A", AI_OLIGOMER_MONOMER)
        with patch(
            "gpcr_tools.validator.api_clients.fetch_polymer_features",
            return_value=None,
        ):
            analyze_oligomer("TEST", data, enriched, polymer_features_cache=cache)
        oligo = data["oligomer_analysis"]
        assert oligo["tm_data_available"] is False
        assert not self._has_disagreement(oligo)
        # The TM_DATA_UNAVAILABLE alert is the correct (and only) routing signal here.
        assert any(a["type"] == ALERT_TM_DATA_UNAVAILABLE for a in oligo["alerts"])

    def test_disagreement_promotes_to_gating_warning(self) -> None:
        # The routing alert must reach critical_warnings (block one-click accept)
        # through the same inject path the other oligomer alerts use.
        oligo = {
            "alerts": [
                {
                    "type": ALERT_OLIGOMER_DISAGREEMENT,
                    "message": (
                        f"[{ALERT_OLIGOMER_DISAGREEMENT}] at 'receptor_info': "
                        "receptor oligomeric state disagrees with the classifier."
                    ),
                }
            ],
            "all_gpcr_chains": [],
        }
        validation_data: dict[str, Any] = {}
        inject_oligomer_alerts(oligo, validation_data)
        warnings = validation_data["critical_warnings"]
        assert any(ALERT_OLIGOMER_DISAGREEMENT in w for w in warnings)
        assert any("receptor_info" in w for w in warnings)
