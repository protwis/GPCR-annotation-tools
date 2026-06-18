"""Tests for the missed non-GPCR polymer reconciliation.

A real polymer chain (nanobody, scFv, RAMP, peptide ligand, G-protein subunit,
second receptor) that the model never annotates should be surfaced for review.
GPCR chains are the oligomer missed-protomer check's responsibility and are not
re-reported here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpcr_tools.csv_generator.validation_display import get_relevant_validation_warnings
from gpcr_tools.validator.oligomer import (
    collect_ai_claimed_chains,
    reconcile_missed_polymers,
)

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "real_pdbs"


def _poly_entity(auth: str, slug: str | None, description: str) -> dict[str, Any]:
    return {
        "rcsb_polymer_entity": {"pdbx_description": description},
        "uniprots": [{"gpcrdb_entry_name_slug": slug}] if slug else [],
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": auth}}
        ],
    }


def _enriched(*entities: dict[str, Any]) -> dict[str, Any]:
    return {"polymer_entities": list(entities)}


def _flagged_chains(warnings: list[str]) -> set[str]:
    return {w.split("chain '")[1].split("'")[0] for w in warnings}


class TestCollectClaimedChains:
    def test_unions_every_slot(self) -> None:
        best = {
            "receptor_info": {"chain_id": "R"},
            "signaling_partners": {
                "g_protein": {
                    "alpha_subunit": {"chain_id": "A"},
                    "beta_subunit": {"chain_id": "B"},
                    "gamma_subunit": {"chain_id": "G"},
                },
                "arrestin": {"chain_id": "X"},
            },
            "auxiliary_proteins": [{"chain_id": "N,M"}],
            "ligands": [{"chain_id": "P"}, {"chain_id": "None"}],
        }
        assert collect_ai_claimed_chains(best) == {"R", "A", "B", "G", "X", "N", "M", "P"}

    def test_none_safe(self) -> None:
        best: dict[str, Any] = {
            "receptor_info": None,
            "signaling_partners": None,
            "auxiliary_proteins": None,
            "ligands": None,
        }
        assert collect_ai_claimed_chains(best) == set()

    def test_semicolon_separated(self) -> None:
        best = {"auxiliary_proteins": [{"chain_id": "N;M"}]}
        assert collect_ai_claimed_chains(best) == {"N", "M"}

    def test_apo_and_empty_sentinels_dropped(self) -> None:
        best = {
            "receptor_info": {"chain_id": "apo"},
            "ligands": [{"chain_id": "None"}, {"chain_id": "n/a"}],
        }
        assert collect_ai_claimed_chains(best) == set()


class TestReconcileMissedPolymers:
    def test_unannotated_non_gpcr_chain_flagged(self) -> None:
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("N", None, "Nanobody 35"),
        )
        warnings = reconcile_missed_polymers(enriched, {"receptor_info": {"chain_id": "R"}})
        assert len(warnings) == 1
        assert "'N'" in warnings[0] and "Nanobody 35" in warnings[0]

    def test_chain_claimed_via_auxiliary_not_flagged(self) -> None:
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("N", None, "Nanobody 35"),
        )
        best = {"receptor_info": {"chain_id": "R"}, "auxiliary_proteins": [{"chain_id": "N"}]}
        assert reconcile_missed_polymers(enriched, best) == []

    def test_gpcr_chain_excluded(self) -> None:
        # An unannotated GPCR chain is the missed-protomer check's job, not ours.
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("N", None, "Nanobody 35"),
        )
        warnings = reconcile_missed_polymers(enriched, {})
        assert _flagged_chains(warnings) == {"N"}

    def test_peptide_ligand_claimed_not_flagged(self) -> None:
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("P", "iapp_human", "Peptide ligand"),
        )
        best = {
            "receptor_info": {"chain_id": "R"},
            "ligands": [{"chem_comp_id": "None", "chain_id": "P"}],
        }
        assert reconcile_missed_polymers(enriched, best) == []

    def test_g_protein_subunit_claimed_not_flagged(self) -> None:
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("A", "gnas2_human", "G-alpha"),
        )
        best = {
            "receptor_info": {"chain_id": "R"},
            "signaling_partners": {"g_protein": {"alpha_subunit": {"chain_id": "A"}}},
        }
        assert reconcile_missed_polymers(enriched, best) == []

    def test_no_polymer_entities(self) -> None:
        assert reconcile_missed_polymers({}, {"receptor_info": {"chain_id": "R"}}) == []

    def test_none_safe_inputs(self) -> None:
        enriched = _enriched(_poly_entity("N", None, "Nanobody 35"))
        best: dict[str, Any] = {
            "receptor_info": None,
            "signaling_partners": None,
            "auxiliary_proteins": None,
            "ligands": None,
        }
        assert _flagged_chains(reconcile_missed_polymers(enriched, best)) == {"N"}

    def test_warning_anchored_to_review_block(self) -> None:
        # A signaling-partner slug routes to signaling_partners; everything else
        # to auxiliary_proteins, so the curate UI can bucket the warning.
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("A", "gnas2_human", "G-alpha"),
            _poly_entity("N", None, "Nanobody 35"),
        )
        warnings = reconcile_missed_polymers(enriched, {"receptor_info": {"chain_id": "R"}})
        by_chain = {w.split("chain '")[1].split("'")[0]: w for w in warnings}
        assert "at 'signaling_partners'" in by_chain["A"]
        assert "at 'auxiliary_proteins'" in by_chain["N"]

    def test_overridden_chain_not_double_flagged(self) -> None:
        # A chain the model named as the receptor but later corrected by a
        # chain-id override already has a HALLUCINATION alert; not re-flagged.
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("N", None, "Nanobody 35"),
        )
        best = {
            "receptor_info": {"chain_id": "R"},
            "oligomer_analysis": {"chain_id_override": {"applied": True, "original_chain_id": "N"}},
        }
        assert reconcile_missed_polymers(enriched, best) == []


class TestReconcileRealData:
    def _enriched_9blw(self) -> dict[str, Any]:
        raw = json.loads((_FIXTURES / "enriched" / "9BLW.json").read_text())
        return (raw.get("data") or {}).get("entry") or raw

    def test_9blw_only_receptor_claimed_flags_all_partners(self) -> None:
        # 9BLW: R=GPCR, N=nanobody, E=RAMP, P=peptide, A/B/G=G-protein.
        enriched = self._enriched_9blw()
        warnings = reconcile_missed_polymers(enriched, {"receptor_info": {"chain_id": "R"}})
        assert _flagged_chains(warnings) == {"N", "E", "P", "A", "B", "G"}

    def test_9blw_fully_annotated_no_warnings(self) -> None:
        enriched = self._enriched_9blw()
        best = {
            "receptor_info": {"chain_id": "R"},
            "signaling_partners": {
                "g_protein": {
                    "alpha_subunit": {"chain_id": "A"},
                    "beta_subunit": {"chain_id": "B"},
                    "gamma_subunit": {"chain_id": "G"},
                }
            },
            "auxiliary_proteins": [{"chain_id": "N"}, {"chain_id": "E"}],
            "ligands": [{"chain_id": "P"}],
        }
        assert reconcile_missed_polymers(enriched, best) == []


class TestWarningBucketsToBlock:
    def test_unannotated_chain_surfaces_in_its_block(self) -> None:
        # The MAJOR fix: the warning must bucket to a real review block (the UI
        # matches by substring), not float in 'polymer_entities' (which matches
        # nothing and would hide the content during per-block review).
        enriched = _enriched(
            _poly_entity("R", "calcr_human", "Calcitonin receptor"),
            _poly_entity("N", None, "Nanobody 35"),
        )
        warnings = reconcile_missed_polymers(enriched, {"receptor_info": {"chain_id": "R"}})
        validation_data = {"critical_warnings": warnings, "algo_conflicts": []}
        assert get_relevant_validation_warnings("auxiliary_proteins", validation_data) == warnings
