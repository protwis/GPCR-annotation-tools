"""Tests for the integrity checker.

Covers: ghost chain detection, ghost ligand detection, fake UniProt,
fake PubChem, method consistency, multi-error scenarios, None-safety,
and warning format compliance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

from gpcr_tools.validator.cache import ValidationCache
from gpcr_tools.validator.integrity_checker import validate_all

_WARNING_REGEX = re.compile(r"at ['\"]([^'\"]+)['\"]")


def _enriched_with_chains_and_ligands(
    chains: list[str] | None = None,
    ligands: list[str] | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """Build a minimal enriched_entry with given chains, ligands, and method."""
    entry: dict[str, Any] = {}

    if method is not None:
        entry["exptl"] = [{"method": method}]

    if chains is not None:
        entry["polymer_entities"] = [
            {
                "polymer_entity_instances": [
                    {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": c}}
                    for c in chains
                ]
            }
        ]

    if ligands is not None:
        entry["nonpolymer_entities"] = [
            {
                "nonpolymer_comp": {"chem_comp": {"id": lid}},
                "nonpolymer_entity_instances": [],
            }
            for lid in ligands
        ]

    return entry


def _enriched_with_branched(
    comp_ids: list[str],
    chains: list[str] | None = None,
    nonpolymer: list[str] | None = None,
    *,
    source: str = "connect_target",
) -> dict[str, Any]:
    """Build an enriched_entry whose ligands include branched (sugar) units.

    Mirrors the real RCSB shape: branched component ids are not a flat list but
    surface through one of three sources, selected by *source*:
    ``connect_target`` / ``connect_partner`` (struct-conn records) or
    ``feature`` (instance features).
    """
    entry = _enriched_with_chains_and_ligands(chains=chains or ["A"], ligands=nonpolymer)
    if source == "feature":
        instance: dict[str, Any] = {
            "rcsb_branched_instance_feature": [
                {"feature_value": [{"comp_id": cid} for cid in comp_ids]}
            ]
        }
    else:
        instance = {
            "rcsb_branched_struct_conn": [{source: {"label_comp_id": cid}} for cid in comp_ids]
        }
    entry["branched_entities"] = [{"branched_entity_instances": [instance]}]
    return entry


class TestGhostChain:
    def test_ghost_chain_detected(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": "Z"}}
        enriched = _enriched_with_chains_and_ligands(chains=["A", "B"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert any("Ghost Chain" in w and "'Z'" in w for w in warnings)

    def test_valid_chain_no_warning(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        enriched = _enriched_with_chains_and_ligands(chains=["A", "B"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Chain" in w for w in warnings)

    def test_empty_chain_id_skipped(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": ""}}
        enriched = _enriched_with_chains_and_ligands(chains=["A"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Chain" in w for w in warnings)

    def test_none_chain_id_skipped(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": None}}
        enriched = _enriched_with_chains_and_ligands(chains=["A"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Chain" in w for w in warnings)

    def test_multi_chain_comma_separated(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": "A,Z"}}
        enriched = _enriched_with_chains_and_ligands(chains=["A", "B"])
        warnings = validate_all("TEST", ai_data, enriched)
        ghost_warnings = [w for w in warnings if "Ghost Chain" in w]
        assert len(ghost_warnings) == 1
        assert "'Z'" in ghost_warnings[0]


class TestGhostLigand:
    def test_ghost_ligand_detected(self) -> None:
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "FAKE"}]}
        enriched = _enriched_with_chains_and_ligands(ligands=["ATP", "GTP"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert any("Ghost Ligand" in w and "'FAKE'" in w for w in warnings)

    def test_valid_ligand_no_warning(self) -> None:
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "ATP"}]}
        enriched = _enriched_with_chains_and_ligands(ligands=["ATP"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_empty_ligand_skipped(self) -> None:
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": ""}]}
        enriched = _enriched_with_chains_and_ligands(ligands=["ATP"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_apo_ligand_skipped(self) -> None:
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "apo"}]}
        enriched = _enriched_with_chains_and_ligands(ligands=["ATP"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_ghost_ligand_in_ligand_free_structure_detected(self) -> None:
        # No nonpolymer entities at all: a hallucinated small molecule must not
        # slip through just because the structure has no real small molecules.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "FAKE"}]}
        enriched = _enriched_with_chains_and_ligands(chains=["A"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert any("Ghost Ligand" in w and "'FAKE'" in w for w in warnings)

    def test_branched_sugar_not_flagged(self) -> None:
        # A real glycan (NAG) lives in branched_entities, not nonpolymer; the
        # model legitimately referencing it must not be called a ghost.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "NAG"}]}
        enriched = _enriched_with_branched(["NAG", "MAN"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_ghost_ligand_detected_alongside_branched(self) -> None:
        # Branched present but the claimed code is in neither bucket -> ghost.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "FAKE"}]}
        enriched = _enriched_with_branched(["NAG"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert any("Ghost Ligand" in w and "'FAKE'" in w for w in warnings)

    def test_branched_sugar_via_connect_partner_not_flagged(self) -> None:
        # A sugar code can arrive on the connect_partner side; it must be harvested.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "NAG"}]}
        enriched = _enriched_with_branched(["NAG"], source="connect_partner")
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_branched_sugar_via_instance_feature_not_flagged(self) -> None:
        # Some depositions expose sugar codes only through instance features.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "BMA"}]}
        enriched = _enriched_with_branched(["BMA"], source="feature")
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_unresolved_branched_suppresses_ghost(self) -> None:
        # Branched entity present but no component id extractable (e.g. a free
        # glycan with no connectivity record): the inventory is not fully known,
        # so a claimed sugar must not be flagged -- avoids a false positive.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "NAG"}]}
        enriched: dict[str, Any] = _enriched_with_chains_and_ligands(chains=["A"])
        enriched["branched_entities"] = [{}]  # present but no extractable comp id
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)

    def test_protein_ligand_sentinel_skipped_without_nonpolymer(self) -> None:
        # Peptide/protein ligands carry the "None" sentinel and stay skipped
        # even when the structure has no small-molecule entities.
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "None"}]}
        enriched = _enriched_with_chains_and_ligands(chains=["A"])
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Ghost Ligand" in w for w in warnings)


class TestMethodConsistency:
    def test_method_conflict_xray(self) -> None:
        ai_data: dict[str, Any] = {"structure_info": {"method": "ELECTRON MICROSCOPY"}}
        enriched = _enriched_with_chains_and_ligands(method="X-RAY DIFFRACTION")
        warnings = validate_all("TEST", ai_data, enriched)
        assert any("Method Conflict" in w for w in warnings)

    def test_method_match_no_warning(self) -> None:
        ai_data: dict[str, Any] = {"structure_info": {"method": "X-RAY DIFFRACTION"}}
        enriched = _enriched_with_chains_and_ligands(method="X-RAY DIFFRACTION")
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Method Conflict" in w for w in warnings)

    def test_method_conflict_em(self) -> None:
        ai_data: dict[str, Any] = {"structure_info": {"method": "X-RAY DIFFRACTION"}}
        enriched = _enriched_with_chains_and_ligands(method="ELECTRON MICROSCOPY")
        warnings = validate_all("TEST", ai_data, enriched)
        assert any("Method Conflict" in w for w in warnings)

    def test_cryo_em_matches_electron(self) -> None:
        """'cryo' in AI method should match 'electron' in PDB method."""
        ai_data: dict[str, Any] = {"structure_info": {"method": "Cryo-EM"}}
        enriched = _enriched_with_chains_and_ligands(method="ELECTRON MICROSCOPY")
        warnings = validate_all("TEST", ai_data, enriched)
        assert not any("Method Conflict" in w for w in warnings)


class TestFakeUniProt:
    def test_fake_uniprot_detected(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {"receptor_info": {"uniprot_entry_name": "fake_human"}}
        enriched: dict[str, Any] = {}
        with patch(
            "gpcr_tools.validator.integrity_checker.check_uniprot_existence",
            return_value=False,
        ):
            warnings = validate_all("TEST", ai_data, enriched, cache=cache)
        assert any("Fake UniProt" in w and "'fake_human'" in w for w in warnings)

    def test_valid_uniprot_no_warning(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {"receptor_info": {"uniprot_entry_name": "drd2_human"}}
        enriched: dict[str, Any] = {}
        with patch(
            "gpcr_tools.validator.integrity_checker.check_uniprot_existence",
            return_value=True,
        ):
            warnings = validate_all("TEST", ai_data, enriched, cache=cache)
        assert not any("Fake UniProt" in w for w in warnings)

    def test_comma_joined_uniprot_checked_per_slug(self, tmp_path: Path) -> None:
        # A comma-joined heterodimer receptor name must be looked up one slug at a
        # time, never as the joined string (which is not a real UniProt ID and would
        # spuriously fail to resolve).
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {
            "receptor_info": {"uniprot_entry_name": "grm2_human, grm7_human"}
        }
        seen: list[str] = []

        def _record(uid: str, _cache: Any) -> bool:
            seen.append(uid)
            return True

        with patch(
            "gpcr_tools.validator.integrity_checker.check_uniprot_existence",
            side_effect=_record,
        ):
            warnings = validate_all("7EPD", ai_data, {}, cache=cache)
        assert seen == ["grm2_human", "grm7_human"]  # each slug separately, never joined
        assert not any("API_UNAVAILABLE" in w or "Fake UniProt" in w for w in warnings)

    def test_api_unavailable_warning(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {"receptor_info": {"uniprot_entry_name": "test_human"}}
        enriched: dict[str, Any] = {}
        with patch(
            "gpcr_tools.validator.integrity_checker.check_uniprot_existence",
            return_value=None,
        ):
            warnings = validate_all("TEST", ai_data, enriched, cache=cache)
        assert any("API_UNAVAILABLE" in w for w in warnings)

    def test_invalid_format(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"uniprot_entry_name": "nodashes"}}
        warnings = validate_all("TEST", ai_data, {})
        assert any("Invalid Format" in w for w in warnings)

    def test_no_cache_skips_api(self) -> None:
        """When cache is None, API checks are skipped."""
        ai_data: dict[str, Any] = {"receptor_info": {"uniprot_entry_name": "drd2_human"}}
        warnings = validate_all("TEST", ai_data, {}, cache=None)
        assert not any("Fake UniProt" in w for w in warnings)
        assert not any("API_UNAVAILABLE" in w for w in warnings)


class TestFakePubChem:
    def test_fake_pubchem_detected(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {"ligands": [{"pubchem_id": "999999"}]}
        enriched: dict[str, Any] = {}
        with patch(
            "gpcr_tools.validator.integrity_checker.check_pubchem_existence",
            return_value=False,
        ):
            warnings = validate_all("TEST", ai_data, enriched, cache=cache)
        assert any("PubChem CID" in w and "'999999'" in w for w in warnings)


class TestMultiError:
    def test_multiple_errors_in_one_pdb(self, tmp_path: Path) -> None:
        """Multiple distinct validation errors in a single PDB."""
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {
            "receptor_info": {
                "chain_id": "Z",
                "uniprot_entry_name": "fake_human",
            },
            "ligands": [{"chem_comp_id": "FAKE"}],
            "structure_info": {"method": "X-RAY"},
        }
        enriched = _enriched_with_chains_and_ligands(
            chains=["A"], ligands=["ATP"], method="ELECTRON MICROSCOPY"
        )
        with patch(
            "gpcr_tools.validator.integrity_checker.check_uniprot_existence",
            return_value=False,
        ):
            warnings = validate_all("TEST", ai_data, enriched, cache=cache)

        types_found = set()
        for w in warnings:
            if "Ghost Chain" in w:
                types_found.add("ghost_chain")
            elif "Fake UniProt" in w:
                types_found.add("fake_uniprot")
            elif "Ghost Ligand" in w:
                types_found.add("ghost_ligand")
            elif "Method Conflict" in w:
                types_found.add("method_conflict")

        assert "ghost_chain" in types_found
        assert "fake_uniprot" in types_found
        assert "ghost_ligand" in types_found
        assert "method_conflict" in types_found


class TestNoneSafety:
    def test_null_exptl(self) -> None:
        ai_data: dict[str, Any] = {"structure_info": {"method": "EM"}}
        enriched: dict[str, Any] = {"exptl": None}
        # Should not crash
        validate_all("TEST", ai_data, enriched)

    def test_null_polymer_entities(self) -> None:
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        enriched: dict[str, Any] = {"polymer_entities": None}
        validate_all("TEST", ai_data, enriched)

    def test_null_nonpolymer_entities(self) -> None:
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "ATP"}]}
        enriched: dict[str, Any] = {"nonpolymer_entities": None}
        validate_all("TEST", ai_data, enriched)

    def test_null_container_identifiers(self) -> None:
        """None-safe: null rcsb_polymer_entity_instance_container_identifiers must not crash."""
        ai_data: dict[str, Any] = {"receptor_info": {"chain_id": "A"}}
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "polymer_entity_instances": [
                        {"rcsb_polymer_entity_instance_container_identifiers": None}
                    ]
                }
            ]
        }
        validate_all("TEST", ai_data, enriched)

    def test_null_nonpolymer_comp(self) -> None:
        """None-safe: null nonpolymer_comp must not crash."""
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "ATP"}]}
        enriched: dict[str, Any] = {
            "nonpolymer_entities": [
                {
                    "nonpolymer_comp": None,
                    "nonpolymer_entity_instances": [],
                }
            ]
        }
        validate_all("TEST", ai_data, enriched)

    def test_null_branched_subfields(self) -> None:
        """None-safe: null branched struct-conn / feature / prd must not crash."""
        ai_data: dict[str, Any] = {"ligands": [{"chem_comp_id": "NAG"}]}
        enriched: dict[str, Any] = {
            "branched_entities": [
                {
                    "prd": None,
                    "branched_entity_instances": [
                        {"rcsb_branched_struct_conn": None, "rcsb_branched_instance_feature": None}
                    ],
                }
            ]
        }
        validate_all("TEST", ai_data, enriched)


class TestWarningFormat:
    def test_all_warnings_match_regex(self, tmp_path: Path) -> None:
        """Every warning must match the UI parsing regex contract."""
        cache = ValidationCache(tmp_path / "cache.json")
        ai_data: dict[str, Any] = {
            "receptor_info": {
                "chain_id": "Z",
                "uniprot_entry_name": "fake_human",
            },
            "ligands": [
                {"chem_comp_id": "FAKE", "pubchem_id": "999"},
            ],
            "structure_info": {"method": "X-RAY"},
        }
        enriched = _enriched_with_chains_and_ligands(
            chains=["A"], ligands=["ATP"], method="ELECTRON MICROSCOPY"
        )
        with (
            patch(
                "gpcr_tools.validator.integrity_checker.check_uniprot_existence",
                return_value=False,
            ),
            patch(
                "gpcr_tools.validator.integrity_checker.check_pubchem_existence",
                return_value=False,
            ),
        ):
            warnings = validate_all("TEST", ai_data, enriched, cache=cache)

        assert len(warnings) >= 4  # ghost chain, fake uniprot, ghost ligand, method, fake pubchem
        for warn in warnings:
            assert _WARNING_REGEX.search(warn) is not None, f"Warning fails regex: {warn}"
