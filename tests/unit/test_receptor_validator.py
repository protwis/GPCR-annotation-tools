"""Tests for receptor identity validation (Epic 3).

Covers: receptor match, UniProt clash, missing chain, None-safety,
and warning format compliance.
"""

from __future__ import annotations

import re
from typing import Any

from gpcr_tools.config import (
    VALIDATION_RECEPTOR_MATCH,
    VALIDATION_RECEPTOR_NO_API_DATA,
    VALIDATION_RECEPTOR_RCSB_UNMAPPED,
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


class TestCommaJoinedHeterodimer:
    """A Class C heterodimer where the model crammed both receptor slugs into the
    single uniprot_entry_name field ('grm2_human, grm7_human', chains 'A, B').
    Each slug genuinely sits on one reported chain, so this must MATCH -- not raise
    a false UNIPROT_CLASH (the comma-join variant the earlier single-slug fix did
    not cover)."""

    def test_comma_joined_both_present_matches(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "grm2_human, grm7_human",
                "chain_id": "A, B",
            }
        }
        enriched = _make_enriched(
            [_make_entity(["A"], ["grm2_human"]), _make_entity(["B"], ["grm7_human"])]
        )
        warnings = validate_receptor_identity("7EPD", data, enriched)
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH
        assert not any("UNIPROT_CLASH" in w for w in warnings)

    def test_comma_joined_none_present_still_clashes(self) -> None:
        # Genuine-fusion-mask protection preserved: if NEITHER joined slug is on any
        # reported chain, it is still a clash.
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "grm2_human, grm7_human",
                "chain_id": "A",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["5ht2a_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert data["receptor_info"]["validation_status"] == VALIDATION_UNIPROT_CLASH
        assert any("UNIPROT_CLASH" in w for w in warnings)

    def test_comma_joined_one_real_one_fake_matches_on_real(self) -> None:
        # Leniency by design (two-layer defence): if at least one joined slug is
        # genuinely on a reported chain, identity is MATCHED here. Catching a
        # hallucinated second slug is the integrity checker's job (per-slug UniProtKB
        # lookup), not this structural-presence validator's.
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "grm2_human, hallucinated_human",
                "chain_id": "A, B",
            }
        }
        enriched = _make_enriched([_make_entity(["A"], ["grm2_human"])])
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH
        assert not any("UNIPROT_CLASH" in w for w in warnings)


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
        # Chain found but the API has no slug -> a soft "no data" note, NOT a clash:
        # an absent slug means "cannot verify", not "identity disagrees".
        assert len(warnings) == 1
        assert "RECEPTOR_NO_API_DATA" in warnings[0]
        assert "UNIPROT_CLASH" not in warnings[0]
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_NO_API_DATA

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

    def test_partner_chain_present_is_not_a_clash(self) -> None:
        """The AI's receptor is on one reported chain; the other carries a different
        gene (a hetero-oligomer partner / fusion entity). Identity is confirmed on
        its own chain, so this is a MATCH, not a clash -- the partner is surfaced by
        the oligomer analysis, not here."""
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
        assert not any("UNIPROT_CLASH" in w for w in warnings)
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH
        # api_reality still aggregates slugs from all matched entities (partner visible).
        assert "drd2_human" in data["receptor_info"]["api_reality"]
        assert "oprm_human" in data["receptor_info"]["api_reality"]

    def test_class_c_heterodimer_no_false_clash(self) -> None:
        """Class C heterodimer (GABA-B): AI names GABBR2 with both protomer chains;
        chain A carries GABBR1 (the partner). Must NOT false-clash -- this is the
        47-structure Class C bug. Regression for 7C7Q/9NOR-style heterodimers."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "gabr2_human",
                "chain_id": "A, B",
            }
        }
        enriched = _make_enriched(
            [
                _make_entity(["A"], ["gabr1_human"]),
                _make_entity(["B"], ["gabr2_human"]),
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert not any("UNIPROT_CLASH" in w for w in warnings)
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_MATCH

    def test_named_receptor_absent_is_a_clash(self) -> None:
        """True masking/hallucination: the AI's receptor is on NONE of its reported
        chains (both carry other genes). This must still clash."""
        data: dict[str, Any] = {
            "receptor_info": {
                "uniprot_entry_name": "drd2_human",
                "chain_id": "A, B",
            }
        }
        enriched = _make_enriched(
            [
                _make_entity(["A"], ["gabr1_human"]),
                _make_entity(["B"], ["gabr2_human"]),
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert len(warnings) == 1
        assert "UNIPROT_CLASH" in warnings[0]
        assert data["receptor_info"]["validation_status"] == VALIDATION_UNIPROT_CLASH

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


def _entity_with_accession_no_slug(chain_ids: list[str], accession: str) -> dict[str, Any]:
    """An entity carrying a UniProt accession (rcsb_id) but no resolved GPCRdb slug."""
    return {
        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": chain_ids},
        "uniprots": [{"rcsb_id": accession}],
    }


class TestRcsbUnmapped:
    """A reported chain whose entity carries a UniProt accession but no resolved
    GPCRdb slug is a recoverable mapping gap (RECEPTOR_RCSB_UNMAPPED), distinct from
    a chain with no UniProt reference at all (RECEPTOR_NO_API_DATA). It must NOT pass
    silently: the receptor identity is unconfirmable, so a gating warning is raised."""

    def test_accession_present_no_slug_sets_unmapped_status(self) -> None:
        data: dict[str, Any] = {
            "receptor_info": {"uniprot_entry_name": "ccr6_human", "chain_id": "A"}
        }
        enriched = _make_enriched([_entity_with_accession_no_slug(["A"], "Q9NXX")])
        warnings = validate_receptor_identity("9D3G", data, enriched)
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_RCSB_UNMAPPED
        # A single, gating warning is raised; no duplicate soft NO_API_DATA note.
        assert len(warnings) == 1
        assert "RECEPTOR_RCSB_UNMAPPED" in warnings[0]
        assert "RECEPTOR_NO_API_DATA" not in warnings[0]
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_no_accession_no_slug_stays_no_api_data(self) -> None:
        # No UniProt reference at all (uniprots null) -> the existing NO_API_DATA
        # status / soft note, NOT the recoverable-unmapped gating warning.
        data: dict[str, Any] = {
            "receptor_info": {"uniprot_entry_name": "drd2_human", "chain_id": "A"}
        }
        enriched = _make_enriched(
            [
                {
                    "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A"]},
                    "uniprots": None,
                }
            ]
        )
        warnings = validate_receptor_identity("TEST", data, enriched)
        assert data["receptor_info"]["validation_status"] == VALIDATION_RECEPTOR_NO_API_DATA
        assert any("RECEPTOR_NO_API_DATA" in w for w in warnings)
        assert not any("RECEPTOR_RCSB_UNMAPPED" in w for w in warnings)


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
