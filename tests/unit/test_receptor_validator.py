"""Tests for receptor identity validation (Epic 3).

Covers: receptor match, UniProt clash, missing chain, None-safety,
and warning format compliance.
"""

from __future__ import annotations

import re
from typing import Any

from gpcr_tools.config import (
    VALIDATION_RECEPTOR_MATCH,
    VALIDATION_UNIPROT_CLASH,
)
from gpcr_tools.validator.receptor_validator import validate_receptor_identity

_WARNING_REGEX = re.compile(r"at ['\"]([^'\"]+)['\"]")


def _make_enriched(entities: list[dict[str, Any]]) -> dict[str, Any]:
    return {"polymer_entities": entities}


def _make_entity(
    chain_ids: list[str],
    slugs: list[str],
) -> dict[str, Any]:
    return {
        "rcsb_polymer_entity_container_identifiers": {
            "auth_asym_ids": chain_ids,
        },
        "uniprots": [{"gpcrdb_entry_name_slug": slug} for slug in slugs],
    }


class TestReceptorMatch:
    def test_match(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["drd2_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH
        assert data["receptor_info"]["api_reality"] == ["drd2_human"]


class TestUniprotClash:
    def test_clash(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["5ht2a_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert len(warnings) == 1
        assert data["receptor_info"]["validation_status"] == VALIDATION_UNIPROT_CLASH
        assert "drd2_human" in warnings[0]
        assert "5ht2a_human" in str(data["receptor_info"]["api_reality"])
        assert _WARNING_REGEX.search(warnings[0]) is not None


class TestMissingChain:
    def test_chain_not_found(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "Z",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["drd2_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert "validation_status" not in data["receptor_info"]


class TestEdgeCases:
    def test_no_receptor_info(self) -> None:
        data: dict[str, Any] = {}
        warnings = validate_receptor_identity("TEST", data, {})
        assert warnings == []

    def test_receptor_info_not_dict(self) -> None:
        data: dict[str, Any] = {"receptor_info": "string"}
        warnings = validate_receptor_identity("TEST", data, {})
        assert warnings == []

    def test_missing_uniprot(self) -> None:
        data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        warnings = validate_receptor_identity("TEST", data, {})
        assert warnings == []

    def test_missing_chain_id(self) -> None:
        data: dict[str, Any] = {"receptor_info": {"uniprot_entry_name": "drd2_human"}}
        warnings = validate_receptor_identity("TEST", data, {})
        assert warnings == []

    def test_empty_enriched(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A",
            }
        }
        warnings = validate_receptor_identity("TEST", data, {})
        assert warnings == []

    def test_null_polymer_entities(self) -> None:
        """Blood Lesson 1: explicit null polymer_entities."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A",
            }
        }
        enriched: dict[str, Any] = {"polymer_entities": None}
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []

    def test_null_uniprots(self) -> None:
        """Blood Lesson 1: explicit null uniprots list."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A",
            }
        }
        enriched = _make_enriched(
            [
                {
                    "rcsb_polymer_entity_container_identifiers": {
                        "auth_asym_ids": ["A"],
                    },
                    "uniprots": None,
                }
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        # Chain found but no slugs -> clash with empty api_reality
        assert len(warnings) == 1
        assert data["receptor_info"]["validation_status"] == VALIDATION_UNIPROT_CLASH

    def test_multiple_slugs_match(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["drd2_human", "drd3_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH
        assert len(data["receptor_info"]["api_reality"]) == 2


class TestMultiChain:
    """Multi-chain chain_id scenarios (comma-separated)."""

    def test_same_entity_match(self) -> None:
        """Both chains in same entity — homodimer, slug matches."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A, B",
            }
        }
        enriched = _make_enriched([_make_entity(["A", "B"], ["drd2_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH
        assert data["receptor_info"]["api_reality"] == ["drd2_human"]

    def test_different_entities_all_match(self) -> None:
        """Chains in different entities, both map to same UniProt."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A, C",
            }
        }
        enriched = _make_enriched(
            [
                _make_entity(["A"], ["drd2_human"]),
                _make_entity(["C"], ["drd2_human"]),
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH

    def test_different_entities_all_clash(self) -> None:
        """Chains in different entities, neither matches AI UniProt."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A, C",
            }
        }
        enriched = _make_enriched(
            [
                _make_entity(["A"], ["5ht2a_human"]),
                _make_entity(["C"], ["oprm_human"]),
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert len(warnings) == 1
        assert data["receptor_info"]["validation_status"] == VALIDATION_UNIPROT_CLASH
        assert "Chain A" in warnings[0]
        assert "Chain C" in warnings[0]

    def test_different_entities_partial_clash(self) -> None:
        """One chain matches, the other doesn't — still a clash."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "B, F",
            }
        }
        enriched = _make_enriched(
            [
                _make_entity(["B"], ["drd2_human"]),
                _make_entity(["F"], ["oprm_human"]),
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert len(warnings) == 1
        assert data["receptor_info"]["validation_status"] == VALIDATION_UNIPROT_CLASH
        # Only the clashing chain appears in the warning
        assert "Chain F" in warnings[0]
        assert "Chain B" not in warnings[0]
        # api_reality aggregates slugs from all matched entities
        assert "drd2_human" in data["receptor_info"]["api_reality"]
        assert "oprm_human" in data["receptor_info"]["api_reality"]

    def test_one_chain_found_one_missing(self) -> None:
        """One chain exists in enriched, the other doesn't — validate what we can."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A, Z",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["drd2_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH

    def test_no_spaces_in_chain_id(self) -> None:
        """chain_id without spaces (e.g. 'A,B') still works."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A,B",
            }
        }
        enriched = _make_enriched([_make_entity(["A", "B"], ["drd2_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert warnings == []
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH


class TestWarningFormat:
    def test_all_warnings_match_regex(self) -> None:
        """Blood Lesson 3: every warning must match the UI regex contract."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "wrong_slug",
                "chain_id": "A",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["real_slug"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        for warn in warnings:
            assert _WARNING_REGEX.search(warn) is not None, f"Warning fails regex: {warn}"
