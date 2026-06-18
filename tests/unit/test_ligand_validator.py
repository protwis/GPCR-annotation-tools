"""Tests for ligand cross-validation and chemical identity injection (Epic 3).

Covers: small molecule match, polymer match, ghost ligand, buffer exclusion,
APO handling, None-safety, and warning format compliance.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

from gpcr_tools.config import (
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_MATCHED_POLYMER,
    VALIDATION_MATCHED_SMALL_MOLECULE,
    VALIDATION_SKIPPED_APO,
)
from gpcr_tools.validator import api_clients
from gpcr_tools.validator.api_clients import check_pubchem_synonym_match
from gpcr_tools.validator.ligand_validator import validate_and_enrich_ligands


class _FakeSynonymCache:
    """In-memory stand-in for the enrichment JsonCache (CID -> synonym list)."""

    def __init__(self, initial: dict[str, list[str]] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


def _mock_synonym_response(status: int, synonyms: list[str] | None = None) -> MagicMock:
    """Build a mocked PubChem synonyms HTTP response."""
    resp = MagicMock()
    resp.status_code = status
    if synonyms is not None:
        resp.json.return_value = {"InformationList": {"Information": [{"Synonym": synonyms}]}}
    else:
        resp.json.return_value = {}
    return resp


# Regex contract from Blood Lesson 3
_WARNING_REGEX = re.compile(r"at ['\"]([^'\"]+)['\"]")


def _make_enriched(
    *,
    nonpolymer: list[dict[str, Any]] | None = None,
    polymer: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if nonpolymer is not None:
        entry["nonpolymer_entities"] = nonpolymer
    if polymer is not None:
        entry["polymer_entities"] = polymer
    return entry


def _np_entity(
    comp_id: str,
    name: str = "Test",
    inchikey: str = "IK123",
    pubchem_cid: str = "12345",
) -> dict[str, Any]:
    return {
        "nonpolymer_comp": {
            "chem_comp": {"id": comp_id, "name": name},
            "rcsb_chem_comp_descriptor": {
                "InChIKey": inchikey,
                "SMILES": "C=O",
                "SMILES_stereo": "C=O",
            },
            "gpcrdb_pubchem_cid": pubchem_cid,
        }
    }


def _poly_entity(
    chain_id: str,
    sequence: str = "MDEF",
    description: str = "Test protein",
    slug: str | None = None,
) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "entity_poly": {
            "pdbx_seq_one_letter_code_can": sequence,
            "type": "polypeptide(L)",
        },
        "rcsb_polymer_entity": {"pdbx_description": description},
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": chain_id}}
        ],
    }
    if slug is not None:
        entity["uniprots"] = [{"gpcrdb_entry_name_slug": slug}]
    return entity


def _lig(
    *,
    name: str = "LIG",
    comp: str = "LIG",
    type_: str = "small-molecule",
    role: str | None = None,
    site: str | None = None,
) -> dict[str, Any]:
    lig: dict[str, Any] = {"name": name, "chem_comp_id": comp, "type": type_}
    if role is not None:
        lig["role"] = {"value": role}
    if site is not None:
        lig["site_ref"] = site
    return lig


class TestSmallMoleculeMatch:
    def test_matched(self) -> None:
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "ATP", "name": "Adenosine"}]}
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        lig = data["ligands"][0]
        assert lig["validation_status"] == VALIDATION_MATCHED_SMALL_MOLECULE
        assert lig["InChIKey"] == "IK123"
        assert lig["api_pubchem_cid"] == "12345"
        assert lig["SMILES"] == "C=O"
        assert lig["SMILES_stereo"] == "C=O"


class TestPolymerMatch:
    def test_peptide_by_chain(self) -> None:
        data: dict[str, Any] = {
            "ligands": [{"chain_id": "B", "name": "Peptide X", "type": "peptide"}]
        }
        enriched = _make_enriched(polymer=[_poly_entity("B", sequence="ACDEF")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        lig = data["ligands"][0]
        assert lig["validation_status"] == VALIDATION_MATCHED_POLYMER
        assert lig["Sequence"] == "ACDEF"

    def test_protein_by_chain(self) -> None:
        data: dict[str, Any] = {
            "ligands": [{"chain_id": "C", "name": "Some protein", "type": "protein"}]
        }
        enriched = _make_enriched(polymer=[_poly_entity("C")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        assert data["ligands"][0]["validation_status"] == VALIDATION_MATCHED_POLYMER

    def test_protein_multi_chain(self) -> None:
        data: dict[str, Any] = {
            "ligands": [
                {"chain_id": "X, Y", "name": "Follicle stimulating hormone", "type": "protein"}
            ]
        }
        # Simulate a PDB where chain X and chain Y exist
        enriched = _make_enriched(polymer=[_poly_entity("X"), _poly_entity("Y")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        lig = data["ligands"][0]
        assert lig["validation_status"] == VALIDATION_MATCHED_POLYMER
        assert lig["Sequence"] == "MDEF / MDEF"


class TestGhostLigand:
    def test_ghost_ligand_with_comp_id(self) -> None:
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "XYZ", "name": "Fake Drug"}]}
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1
        assert data["ligands"][0]["validation_status"] == VALIDATION_GHOST_LIGAND
        assert "GHOST_LIGAND" in warnings[0]
        assert "XYZ" in warnings[0]
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_ghost_ligand_no_comp_id(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "Mystery"}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1
        assert "Mystery" in warnings[0]
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_ghost_ligand_is_flagged_not_removed(self) -> None:
        # The validator only flags; it never drops the entry. Deciding whether
        # an unverified ligand reaches the export is left to the curator and the
        # CSV writer, so the flagged ligand must survive validation intact.
        data: dict[str, Any] = {
            "ligands": [
                {"chem_comp_id": "ATP", "name": "Adenosine"},
                {"chem_comp_id": "XYZ", "name": "Fake Drug"},
            ]
        }
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        validate_and_enrich_ligands("TEST", data, enriched)
        assert len(data["ligands"]) == 2
        assert data["ligands"][1]["name"] == "Fake Drug"
        assert data["ligands"][1]["validation_status"] == VALIDATION_GHOST_LIGAND


class TestBufferExclusion:
    def test_excluded_buffer(self) -> None:
        """Buffer comp_ids in LIGAND_EXCLUDE_LIST are excluded at context build time,
        so they won't match as small molecules and should be ghost unless the
        buffer appears in the AI data explicitly."""
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "GOL", "name": "Glycerol"}]}
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        validate_and_enrich_ligands("TEST", data, enriched)
        assert data["ligands"][0]["validation_status"] == VALIDATION_EXCLUDED_BUFFER


class TestApoHandling:
    def test_apo_name(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "apo", "chem_comp_id": ""}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        assert data["ligands"][0]["validation_status"] == VALIDATION_SKIPPED_APO

    def test_apo_comp_id(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "No ligand", "chem_comp_id": "apo"}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        assert data["ligands"][0]["validation_status"] == VALIDATION_SKIPPED_APO

    def test_warns_when_apo_coexists_with_real_ligand(self) -> None:
        # An apo (ligand-free) placeholder alongside a real ligand is
        # contradictory; surface it for the curator, do not silently edit.
        data: dict[str, Any] = {
            "ligands": [
                {"name": "apo", "chem_comp_id": ""},
                {"name": "ATP", "chem_comp_id": "ATP"},
            ]
        }
        warnings = validate_and_enrich_ligands("TEST", data, _make_enriched())
        assert any("apo" in w.lower() and "coexist" in w.lower() for w in warnings)
        # warning-only: the apo entry must NOT be removed
        assert len(data["ligands"]) == 2

    def test_no_apo_coexistence_warning_for_pure_apo(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "apo", "chem_comp_id": ""}]}
        warnings = validate_and_enrich_ligands("TEST", data, _make_enriched())
        assert not any("coexist" in w.lower() for w in warnings)

    def test_no_apo_coexistence_warning_with_only_buffer(self) -> None:
        # Apo + an excluded buffer (glycerol) is normal, not contradictory.
        data: dict[str, Any] = {
            "ligands": [
                {"name": "apo", "chem_comp_id": ""},
                {"name": "glycerol", "chem_comp_id": "GOL"},
            ]
        }
        warnings = validate_and_enrich_ligands("TEST", data, _make_enriched())
        assert not any("coexist" in w.lower() for w in warnings)


class TestNoneSafety:
    def test_null_chem_comp_id(self) -> None:
        """Blood Lesson 1: explicit null chem_comp_id must not crash."""
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": None, "name": "Test"}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1

    def test_null_pdbx_description(self) -> None:
        """Null pdbx_description must not crash polymer context building."""
        data: dict[str, Any] = {"ligands": [{"chain_id": "A", "name": "Test", "type": "peptide"}]}
        enriched = _make_enriched(
            polymer=[
                {
                    "entity_poly": {},
                    "rcsb_polymer_entity": {"pdbx_description": None},
                    "polymer_entity_instances": [
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A"
                            }
                        }
                    ],
                }
            ]
        )
        # Should not crash
        validate_and_enrich_ligands("TEST", data, enriched)

    def test_missing_enriched_fields(self) -> None:
        """Empty enriched entry must not crash."""
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "TEST", "name": "Test"}]}
        enriched: dict[str, Any] = {}
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1

    def test_no_ligands_key(self) -> None:
        data: dict[str, Any] = {}
        warnings = validate_and_enrich_ligands("TEST", data, {})
        assert warnings == []

    def test_ligands_not_a_list(self) -> None:
        data: dict[str, Any] = {"ligands": "not a list"}
        warnings = validate_and_enrich_ligands("TEST", data, {})
        assert warnings == []


class TestWarningFormat:
    def test_all_warnings_match_regex(self) -> None:
        """Blood Lesson 3: every warning must match the UI regex contract."""
        data: dict[str, Any] = {
            "ligands": [
                {"chem_comp_id": "FAKE1", "name": "Drug1"},
                {"chem_comp_id": None, "name": "Drug2"},
                {"name": "Drug3"},
            ]
        }
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        for warn in warnings:
            assert _WARNING_REGEX.search(warn) is not None, f"Warning fails regex: {warn}"


class TestRoleSiteMismatch:
    """The role/site self-consistency net flags only unambiguous contradictions in
    the AI's own output; legitimate non-Class-A sites must never be flagged."""

    def _mismatch_warnings(self, lig: dict[str, Any]) -> list[str]:
        warnings = validate_and_enrich_ligands("TEST", {"ligands": [lig]}, _make_enriched())
        return [w for w in warnings if "ROLE_SITE_MISMATCH" in w]

    def test_lipid_at_orthosteric_warns(self) -> None:
        # A lipid with NO functional role at the orthosteric pocket is the
        # mislabelled-structural-lipid case (the original 7E2X failure).
        assert self._mismatch_warnings(_lig(type_="lipid", site="orthosteric"))

    def test_lipid_agonist_at_orthosteric_ok(self) -> None:
        # An endogenous lipid agonist (S1P / LPA / 2-AG / prostaglandin) legitimately
        # binds the orthosteric pocket as type 'lipid' -> not a contradiction.
        assert not self._mismatch_warnings(_lig(type_="lipid", role="Agonist", site="orthosteric"))

    def test_allosteric_role_at_orthosteric_warns(self) -> None:
        assert self._mismatch_warnings(_lig(role="PAM", site="orthosteric"))
        assert self._mismatch_warnings(_lig(role="Allosteric agonist", site="orthosteric"))

    def test_functional_role_at_membrane_facing_ok(self) -> None:
        # Any role can occur at any position: a functional role at the
        # membrane_facing surface (e.g. a lipid-facing positive allosteric modulator) is
        # legitimate, not a contradiction -> must not be flagged.
        assert not self._mismatch_warnings(_lig(role="Agonist", site="membrane_facing"))

    def test_agonist_at_orthosteric_ok(self) -> None:
        assert not self._mismatch_warnings(_lig(role="Agonist", site="orthosteric"))

    def test_agonist_at_extracellular_domain_ok(self) -> None:
        # Class C / B orthosteric site IS the extracellular domain -> not flagged.
        assert not self._mismatch_warnings(_lig(role="Agonist", site="extracellular_domain"))

    def test_agonist_at_intracellular_ok(self) -> None:
        # Intracellular agonist binding sites are real -> not flagged.
        assert not self._mismatch_warnings(_lig(role="Agonist", site="intracellular"))

    def test_pam_at_allosteric_ok(self) -> None:
        assert not self._mismatch_warnings(_lig(role="PAM", site="allosteric_7tm"))

    def test_lipid_at_membrane_facing_ok(self) -> None:
        assert not self._mismatch_warnings(_lig(type_="lipid", site="membrane_facing"))

    def test_warning_matches_regex(self) -> None:
        warnings = self._mismatch_warnings(_lig(role="PAM", site="orthosteric"))
        assert warnings and _WARNING_REGEX.search(warnings[0]) is not None


_G_ALPHA_DESC = "GUANINE NUCLEOTIDE-BINDING PROTEIN G(T) SUBUNIT ALPHA-3"


class TestGProteinPeptideAsLigand:
    """A transducer-derived / G-protein-mimetic peptide filed as a receptor ligand
    with a functional pocket role is a signaling partner mis-annotation. The net
    surfaces it for the curator; an honest abstention (unknown role) is never
    flagged, and genuine small-molecule agonists / auxiliary proteins are untouched.
    """

    def _gp_warnings(
        self,
        lig: dict[str, Any],
        polymer: list[dict[str, Any]],
    ) -> list[str]:
        warnings = validate_and_enrich_ligands(
            "TEST", {"ligands": [lig]}, _make_enriched(polymer=polymer)
        )
        return [w for w in warnings if "G-PROTEIN PEPTIDE AS LIGAND" in w]

    def test_g_alpha_description_peptide_agonist_warns(self) -> None:
        # Chain B's polymer description reads as a G-alpha subunit, yet the peptide
        # is filed as an Agonist -> surface for the curator.
        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        warnings = self._gp_warnings(lig, polymer)
        assert warnings
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_g_alpha_warning_reaches_accept_all_disabling_channel(self) -> None:
        # The returned warning list is extended verbatim into critical_warnings,
        # the channel that disables one-click accept-all.
        from gpcr_tools.aggregator.runner import _build_validation_report

        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        ligand_warnings = validate_and_enrich_ligands(
            "TEST", {"ligands": [lig]}, _make_enriched(polymer=polymer)
        )
        report = _build_validation_report("TEST", {}, {}, ligand_warnings, {}, None)
        assert any("G-PROTEIN PEPTIDE AS LIGAND" in w for w in report["critical_warnings"])

    def test_slug_branch_peptide_agonist_warns(self) -> None:
        # Description does NOT read as a G-alpha, but the chain's GPCRdb slug is a
        # G-protein subunit -> the slug branch (+ poly_by_chain slug extension) fire.
        lig = {"chain_id": "B", "name": "Transducin mimetic", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [
            _poly_entity(
                "B", sequence="ACDEF", description="Engineered peptide", slug="gnat1_bovin"
            )
        ]
        warnings = self._gp_warnings(lig, polymer)
        assert warnings
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_slug_branch_gna_prefix_warns(self) -> None:
        lig = {"chain_id": "C", "name": "Mini-G peptide", "type": "protein"}
        lig["role"] = {"value": "Agonist"}
        polymer = [
            _poly_entity("C", sequence="ACDEF", description="some construct", slug="gnas2_human")
        ]
        assert self._gp_warnings(lig, polymer)

    def test_slug_branch_gbb_and_gbg_prefixes_warn(self) -> None:
        # The G-beta / G-gamma subunit slug prefixes are also signaling-partner
        # markers: a beta/gamma chain filed as a functional ligand should fire.
        for slug in ("gbb1_human", "gbg2_human"):
            lig = {"chain_id": "D", "name": "Subunit peptide", "type": "peptide"}
            lig["role"] = {"value": "Agonist"}
            polymer = [_poly_entity("D", sequence="ACDEF", description="some construct", slug=slug)]
            assert self._gp_warnings(lig, polymer), slug

    def test_small_molecule_agonist_not_flagged(self) -> None:
        # The real agonist is a small molecule, not a peptide on a G-protein chain.
        data: dict[str, Any] = {
            "ligands": [
                {
                    "chem_comp_id": "RET",
                    "name": "all-trans-retinal",
                    "type": "small-molecule",
                    "role": {"value": "Agonist"},
                }
            ]
        }
        enriched = _make_enriched(nonpolymer=[_np_entity("RET")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert not any("G-PROTEIN PEPTIDE AS LIGAND" in w for w in warnings)

    def test_auxiliary_fab_not_flagged(self) -> None:
        # A stabilising Fab/nanobody is an auxiliary protein, not a peptide ligand
        # with a functional pocket role -> no agonist mis-annotation to flag.
        lig = {"chain_id": "H", "name": "Fab heavy chain", "type": "protein"}
        polymer = [_poly_entity("H", sequence="ACDEF", description="Antibody Fab fragment")]
        # No functional role: this is just a structural auxiliary protein.
        assert not self._gp_warnings(lig, polymer)

    def test_bare_sequence_alpha5_mimetic_peptide_warns(self) -> None:
        # A G-alpha alpha5 C-terminal mimetic deposited under a bare-sequence name
        # (no G-protein wording, no slug) is now recognised via its conserved motif.
        lig = {"chain_id": "B", "name": "alpha5 mimetic", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [_poly_entity("B", sequence="ILENLKDVGLF", description="ILENLKDVGLF peptide CT2")]
        warnings = self._gp_warnings(lig, polymer)
        assert warnings
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_gnb5_subunit_caught_via_slug(self) -> None:
        # G-beta-5's curated slug is gnb5_*, which the old "gbb" prefix missed; the
        # added "gnb" prefix catches it.
        lig = {"chain_id": "E", "name": "subunit peptide", "type": "protein"}
        lig["role"] = {"value": "Agonist"}
        polymer = [
            _poly_entity("E", sequence="ACDEF", description="some construct", slug="gnb5_human")
        ]
        assert self._gp_warnings(lig, polymer)

    def test_glp1_peptide_agonist_not_flagged(self) -> None:
        # A genuine peptide-hormone agonist on a non-G-protein chain is not flagged.
        lig = {"chain_id": "P", "name": "GLP-1", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [_poly_entity("P", sequence="HAEGTFTSD", description="Glucagon-like peptide-1")]
        assert not self._gp_warnings(lig, polymer)

    def test_g_alpha_peptide_with_unknown_role_not_flagged(self) -> None:
        # The model honestly abstaining (role unknown) must never be flagged.
        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        lig["role"] = {"value": "unknown"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        assert not self._gp_warnings(lig, polymer)

    def test_g_alpha_peptide_with_absent_role_not_flagged(self) -> None:
        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        assert not self._gp_warnings(lig, polymer)


class TestMultipleAgonists:
    """Two or more DISTINCT ligands annotated as plain 'Agonist' may be unrecognised
    co-agonists. The net surfaces them for the curator without asserting co-agonism;
    a single agonist split across two binding sites (same identity) is deduped to one
    and never fires, and non-agonist or mixed-role pairs are untouched.
    """

    def _agonist_warnings(self, ligands: list[dict[str, Any]]) -> list[str]:
        nonpolymer = [
            _np_entity(lig["chem_comp_id"])
            for lig in ligands
            if lig.get("chem_comp_id") and lig["chem_comp_id"] not in ("", "None")
        ]
        warnings = validate_and_enrich_ligands(
            "TEST", {"ligands": ligands}, _make_enriched(nonpolymer=nonpolymer or None)
        )
        return [w for w in warnings if "MULTIPLE AGONISTS" in w]

    def test_two_distinct_agonists_warn(self) -> None:
        # A metal-ion agonist and a small-molecule agonist co-occupying a structure:
        # the classic missed-co-agonist configuration.
        ligands = [
            _lig(name="Calcium ion", comp="CA", role="Agonist"),
            _lig(name="L-glutamate", comp="GLU", role="Agonist"),
        ]
        warnings = self._agonist_warnings(ligands)
        assert warnings
        assert _WARNING_REGEX.search(warnings[0]) is not None
        # Non-asserting: it asks the curator to verify, not declares co-agonism.
        assert "verify" in warnings[0].lower()
        assert "Calcium ion" in warnings[0]
        assert "L-glutamate" in warnings[0]

    def test_warning_reaches_accept_all_disabling_channel(self) -> None:
        # The returned warning list is extended verbatim into critical_warnings,
        # the channel that disables one-click accept-all for the PDB.
        from gpcr_tools.aggregator.runner import _build_validation_report

        ligands = [
            _lig(name="Calcium ion", comp="CA", role="Agonist"),
            _lig(name="L-glutamate", comp="GLU", role="Agonist"),
        ]
        ligand_warnings = validate_and_enrich_ligands(
            "TEST", {"ligands": ligands}, _make_enriched()
        )
        report = _build_validation_report("TEST", {}, {}, ligand_warnings, {}, None)
        assert any("MULTIPLE AGONISTS" in w for w in report["critical_warnings"])

    def test_single_agonist_two_sites_deduped_by_comp_id(self) -> None:
        # One agonist modelled at two sites -> two entries (site_ref split) with the
        # SAME chem_comp_id -> ONE distinct molecule -> no reminder.
        ligands = [
            _lig(name="L-glutamate", comp="GLU", role="Agonist", site="orthosteric"),
            _lig(name="L-glutamate", comp="GLU", role="Agonist", site="allosteric_7tm"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_single_agonist_two_sites_deduped_by_name(self) -> None:
        # Same molecule at two sites with no chem_comp_id -> deduped by
        # case-insensitive name -> one distinct molecule -> no reminder.
        ligands = [
            _lig(name="L-glutamate", comp="", role="Agonist", site="orthosteric"),
            _lig(name="l-glutamate", comp="", role="Agonist", site="allosteric_7tm"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_single_agonist_no_warning(self) -> None:
        assert not self._agonist_warnings([_lig(name="L-glutamate", comp="GLU", role="Agonist")])

    def test_agonist_plus_antagonist_no_warning(self) -> None:
        ligands = [
            _lig(name="L-glutamate", comp="GLU", role="Agonist"),
            _lig(name="Inhibitor", comp="INH", role="Antagonist"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_two_non_agonist_ligands_no_warning(self) -> None:
        ligands = [
            _lig(name="Inhibitor", comp="INH", role="Antagonist"),
            _lig(name="Modulator", comp="MOD", role="PAM"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_co_agonist_role_not_counted(self) -> None:
        # If the model already said 'Co-agonist' it recognised the relationship;
        # only the plain 'Agonist' role is in scope, so this pair does not fire.
        ligands = [
            _lig(name="Calcium ion", comp="CA", role="Co-agonist"),
            _lig(name="L-glutamate", comp="GLU", role="Co-agonist"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_distinct_agonist_mechanisms_not_counted(self) -> None:
        # 'Allosteric agonist' / 'Agonist (partial)' describe different mechanisms;
        # mixing them with a plain agonist leaves only one plain agonist -> no fire.
        ligands = [
            _lig(name="L-glutamate", comp="GLU", role="Agonist"),
            _lig(name="Allosteric drug", comp="ALL", role="Allosteric agonist"),
            _lig(name="Partial drug", comp="PAR", role="Agonist (partial)"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_agonist_plus_co_agonist_no_warning(self) -> None:
        # 'Co-agonist' means the model already recognised the relationship, so it is
        # out of scope: one plain Agonist + one Co-agonist leaves only one plain
        # agonist -> no fire.
        ligands = [
            _lig(name="L-glutamate", comp="GLU", role="Agonist"),
            _lig(name="Calcium ion", comp="CA", role="Co-agonist"),
        ]
        assert not self._agonist_warnings(ligands)

    def test_agonist_with_no_usable_identity_skipped(self) -> None:
        # An agonist entry with neither a usable chem_comp_id nor a name resolves to
        # an empty identity and is skipped, so it cannot inflate the distinct count:
        # one such entry plus one real agonist is only ONE distinct molecule -> no fire.
        ligands = [
            _lig(name="", comp="", role="Agonist"),
            _lig(name="L-glutamate", comp="GLU", role="Agonist"),
        ]
        assert not self._agonist_warnings(ligands)


@patch("gpcr_tools.validator.api_clients.time.sleep", lambda *_: None)
class TestCheckPubChemSynonymMatch:
    """The synonym gate confirms a CID actually names the reported molecule.

    A real-but-unrelated CID (one that an existence check would wave through) is
    caught because its own synonym list does not include the molecule's name.
    """

    def test_name_in_synonyms_returns_true(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(
            api_clients.requests,
            "get",
            return_value=_mock_synonym_response(200, ["Foo Acid", "ExampleDrug", "12345-67-8"]),
        ):
            assert check_pubchem_synonym_match("111", ["ExampleDrug"], cache) is True

    def test_synonym_match_via_reported_synonyms_union(self) -> None:
        # The bare name misses, but a reported synonym is in the CID's list:
        # matching the UNION rescues a correct CID whose canonical name is an
        # IUPAC string the short name never matches.
        cache = _FakeSynonymCache()
        with patch.object(
            api_clients.requests,
            "get",
            return_value=_mock_synonym_response(200, ["N-acetyl-example-amide"]),
        ):
            verdict = check_pubchem_synonym_match(
                "111", ["ShortName", "N-acetyl-example-amide"], cache
            )
        assert verdict is True

    def test_normalization_ignores_case_and_punctuation(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(
            api_clients.requests,
            "get",
            return_value=_mock_synonym_response(200, ["Example-Drug (free base)"]),
        ):
            # Candidate differs only by case/spacing/punctuation -> still matches.
            assert check_pubchem_synonym_match("111", ["example drug free base"], cache) is True

    def test_no_overlap_returns_false(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(
            api_clients.requests,
            "get",
            return_value=_mock_synonym_response(200, ["Unrelated compound", "C11H16N3O8"]),
        ):
            assert check_pubchem_synonym_match("222", ["ExampleDrug"], cache) is False

    def test_http_404_returns_false(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(api_clients.requests, "get", return_value=_mock_synonym_response(404)):
            assert check_pubchem_synonym_match("333", ["ExampleDrug"], cache) is False
        # 404 is a definitive answer and is cached as an empty list.
        assert cache.get("333") == []

    def test_network_error_returns_none_and_not_cached(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(
            api_clients.requests,
            "get",
            side_effect=api_clients.requests.RequestException("boom"),
        ):
            assert check_pubchem_synonym_match("444", ["ExampleDrug"], cache) is None
        # A network failure must never poison the cache with a false negative.
        assert not cache.has("444")

    def test_unexpected_status_returns_none_and_not_cached(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(api_clients.requests, "get", return_value=_mock_synonym_response(503)):
            assert check_pubchem_synonym_match("555", ["ExampleDrug"], cache) is None
        assert not cache.has("555")

    def test_empty_candidates_returns_none_without_call(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(api_clients.requests, "get") as mock_get:
            assert check_pubchem_synonym_match("666", [], cache) is None
            assert check_pubchem_synonym_match("666", ["", "  "], cache) is None
            mock_get.assert_not_called()

    def test_non_numeric_cid_returns_false(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(api_clients.requests, "get") as mock_get:
            assert check_pubchem_synonym_match("not-a-cid", ["ExampleDrug"], cache) is False
            mock_get.assert_not_called()

    def test_cached_list_reused_without_network(self) -> None:
        cache = _FakeSynonymCache({"777": ["ExampleDrug"]})
        with patch.object(api_clients.requests, "get") as mock_get:
            assert check_pubchem_synonym_match("777", ["ExampleDrug"], cache) is True
            assert check_pubchem_synonym_match("777", ["Other"], cache) is False
            mock_get.assert_not_called()

    def test_200_result_cached_for_reuse(self) -> None:
        cache = _FakeSynonymCache()
        with patch.object(
            api_clients.requests,
            "get",
            return_value=_mock_synonym_response(200, ["ExampleDrug"]),
        ) as mock_get:
            check_pubchem_synonym_match("888", ["ExampleDrug"], cache)
            check_pubchem_synonym_match("888", ["ExampleDrug"], cache)
            assert mock_get.call_count == 1
        assert cache.get("888") == ["ExampleDrug"]


@patch("gpcr_tools.validator.api_clients.time.sleep", lambda *_: None)
class TestKeylessPubChemGate:
    """The validator gates a model-supplied CID only for keyless ligands (no matched
    chemical component), blanking a wrong CID and abstaining on network failure; a
    matched small molecule's authoritative CID is never touched.
    """

    def _gate_warnings(self, ligands: list[dict[str, Any]], cache: _FakeSynonymCache) -> list[str]:
        return validate_and_enrich_ligands(
            "TEST", {"ligands": ligands}, _make_enriched(), synonym_cache=cache
        )

    def test_correct_cid_kept(self) -> None:
        # A keyless peptide with a CID whose synonyms include its name -> kept.
        cache = _FakeSynonymCache({"111": ["ExampleDrug", "some-iupac-name"]})
        lig = {
            "name": "ExampleDrug",
            "chem_comp_id": "None",
            "type": "peptide",
            "pubchem_id": "111",
        }
        warnings = self._gate_warnings([lig], cache)
        assert lig["pubchem_id"] == "111"
        assert not any("Mismatch" in w for w in warnings)

    def test_keyed_excluded_buffer_not_gated(self) -> None:
        # An excluded buffer (e.g. PLM) is a keyed component: it has a chem_comp_id
        # and its CID is echoed from metadata, not guessed -> never synonym-gated,
        # and no network call is made.
        from unittest.mock import patch

        cache = _FakeSynonymCache({})
        lig = {
            "name": "Palmitic acid",
            "chem_comp_id": "PLM",
            "type": "lipid",
            "pubchem_id": "985",
        }
        with patch(
            "gpcr_tools.validator.ligand_validator.check_pubchem_synonym_match"
        ) as mock_check:
            warnings = self._gate_warnings([lig], cache)
        mock_check.assert_not_called()
        assert lig["pubchem_id"] == "985"
        assert not any("Mismatch" in w or "API_UNAVAILABLE" in w for w in warnings)

    def test_correct_cid_kept_via_reported_synonym(self) -> None:
        cache = _FakeSynonymCache({"111": ["n-acetyl-example-amide"]})
        lig = {
            "name": "example peptide",
            "chem_comp_id": "None",
            "type": "peptide",
            "pubchem_id": "111",
            "synonyms": ["n-acetyl-example-amide"],
        }
        warnings = self._gate_warnings([lig], cache)
        assert lig["pubchem_id"] == "111"
        assert not any("Mismatch" in w for w in warnings)

    def test_wrong_cid_blanked_and_warned(self) -> None:
        # A real-but-unrelated CID: synonyms do not include the molecule's name.
        cache = _FakeSynonymCache({"222": ["Unrelated compound"]})
        lig = {
            "name": "ExampleDrug",
            "chem_comp_id": "None",
            "type": "peptide",
            "pubchem_id": "222",
        }
        warnings = self._gate_warnings([lig], cache)
        assert lig["pubchem_id"] is None
        assert any("Mismatch" in w and "'222'" in w and "ExampleDrug" in w for w in warnings)
        assert any(_WARNING_REGEX.search(w) for w in warnings if "Mismatch" in w)

    def test_network_abstention_leaves_value_unchanged(self) -> None:
        cache = _FakeSynonymCache()
        lig = {
            "name": "ExampleDrug",
            "chem_comp_id": "None",
            "type": "peptide",
            "pubchem_id": "111",
        }
        with patch.object(
            api_clients.requests,
            "get",
            side_effect=api_clients.requests.RequestException("offline"),
        ):
            warnings = self._gate_warnings([lig], cache)
        assert lig["pubchem_id"] == "111"  # never blanked on a network error
        assert any("API_UNAVAILABLE" in w and "'111'" in w for w in warnings)

    def test_matched_small_molecule_cid_never_gated(self) -> None:
        # A small molecule that matched enriched metadata carries api_pubchem_cid;
        # its authoritative CID must never be synonym-checked or blanked.
        data: dict[str, Any] = {
            "ligands": [{"chem_comp_id": "ATP", "name": "Adenosine", "pubchem_id": "999"}]
        }
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP", pubchem_cid="12345")])
        cache = _FakeSynonymCache()
        with patch.object(api_clients.requests, "get") as mock_get:
            warnings = validate_and_enrich_ligands("TEST", data, enriched, synonym_cache=cache)
            mock_get.assert_not_called()
        lig = data["ligands"][0]
        assert "api_pubchem_cid" in lig
        assert not any("Mismatch" in w for w in warnings)

    def test_no_cache_keeps_offline_no_gate(self) -> None:
        # Without a cache the gate is skipped entirely (no network), preserving the
        # offline contract for callers that do not opt in.
        lig = {
            "name": "ExampleDrug",
            "chem_comp_id": "None",
            "type": "peptide",
            "pubchem_id": "222",
        }
        with patch.object(api_clients.requests, "get") as mock_get:
            warnings = validate_and_enrich_ligands("TEST", {"ligands": [lig]}, _make_enriched())
            mock_get.assert_not_called()
        assert lig["pubchem_id"] == "222"  # untouched
        assert not any("Mismatch" in w for w in warnings)

    def test_none_pubchem_id_skipped(self) -> None:
        # A keyless ligand whose model pubchem_id is the 'None' sentinel has nothing
        # to gate and must not trigger a network call.
        cache = _FakeSynonymCache()
        lig = {
            "name": "example peptide",
            "chem_comp_id": "None",
            "type": "peptide",
            "pubchem_id": "None",
        }
        with patch.object(api_clients.requests, "get") as mock_get:
            warnings = self._gate_warnings([lig], cache)
            mock_get.assert_not_called()
        assert not any("Mismatch" in w or "API_UNAVAILABLE" in w for w in warnings)

    def test_no_candidate_names_makes_no_call_and_no_warning(self) -> None:
        # A keyless ligand with a CID but no name and no usable synonyms cannot be
        # verified, makes no network call, and must NOT emit an [API_UNAVAILABLE]
        # note (which would wrongly imply a network failure that never happened).
        cache = _FakeSynonymCache()
        lig = {"name": "", "chem_comp_id": "None", "type": "peptide", "pubchem_id": "222"}
        with patch.object(api_clients.requests, "get") as mock_get:
            warnings = self._gate_warnings([lig], cache)
            mock_get.assert_not_called()
        assert lig["pubchem_id"] == "222"  # untouched
        assert not any("Mismatch" in w or "API_UNAVAILABLE" in w for w in warnings)
