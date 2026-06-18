"""Contract tests binding the Gemini tool schema to the config constants.

The schema's enums are hand-written literals; if one drifts from the config
constant the rest of the pipeline validates against, the model and the
validators silently disagree. These tests fail loudly on any such drift.
"""

from __future__ import annotations

import importlib.resources

from google.genai import types

from gpcr_tools.annotator.schema import ANNOTATION_TOOL
from gpcr_tools.config import AI_OLIGOMER_TO_RECEPTOR_LEVEL, AI_OLIGOMER_UNKNOWN, SITE_REF_VALUES


def _ligand_item_properties() -> dict:
    """The per-ligand item schema's properties, from the live tool object."""
    params = ANNOTATION_TOOL.function_declarations[0].parameters
    return params.properties["ligands"].items.properties


def _g_protein_properties() -> dict:
    """The g_protein object schema's properties, from the live tool object."""
    params = ANNOTATION_TOOL.function_declarations[0].parameters
    return params.properties["signaling_partners"].properties["g_protein"].properties


def _receptor_info():
    """The receptor_info object schema, from the live tool object."""
    params = ANNOTATION_TOOL.function_declarations[0].parameters
    return params.properties["receptor_info"]


def _v5_prompt_text() -> str:
    """The bundled annotation prompt, read from package data."""
    src = importlib.resources.files("gpcr_tools") / "data" / "prompts" / "v5.md"
    return src.read_text(encoding="utf-8")


def test_site_ref_enum_matches_config() -> None:
    # The schema's site_ref enum is the model-facing list; SITE_REF_VALUES is
    # what every downstream validator / grouping step uses. They must agree, or
    # the model can emit a value the pipeline does not recognise (and vice
    # versa). Compared as sets: an added or removed value is the real drift.
    schema_enum = _ligand_item_properties()["site_ref"].enum
    assert schema_enum is not None
    assert len(schema_enum) == len(set(schema_enum))  # no duplicate members
    assert set(schema_enum) == set(SITE_REF_VALUES)

    # The structure-state enum (incl. its 'unknown' escape) is pinned to the
    # downstream-accepted CSV tokens by test_state_vocab_contract.


def test_site_ref_justification_is_optional_string() -> None:
    # The justification is a curator-facing free-text field on each ligand: it must
    # exist with type string and must NOT be required (the model may omit it).
    props = _ligand_item_properties()
    assert "site_ref_justification" in props
    assert props["site_ref_justification"].type == types.Type.STRING
    required = (
        ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items.required
    )
    assert "site_ref_justification" not in (required or [])


def test_role_site_rule_sets_match_schema_role_enum() -> None:
    # The ligand_validator role/site safety net keys on exact role.value strings;
    # a misspelling would silently never fire. Bind its role sets to the schema enum.
    from gpcr_tools.validator.ligand_validator import (
        _ALLOSTERIC_ROLES,
        _FUNCTIONAL_POCKET_ROLES,
    )

    role_enum = set(_ligand_item_properties()["role"].properties["value"].enum)
    assert _ALLOSTERIC_ROLES.issubset(role_enum)
    assert _FUNCTIONAL_POCKET_ROLES.issubset(role_enum)
    assert "Cofactor" in role_enum


def test_g_protein_note_carries_sourcing_constraint() -> None:
    # The g_protein.note description must require that the specific composition
    # details be stated in the source -- the constraint that stops the model
    # from inventing an unsourced subtype/species in the free-text note.
    note_desc = (_g_protein_properties()["note"].description or "").lower()
    assert "sourcing requirement" in note_desc
    assert "specific composition details" in note_desc
    assert "paper or pdb metadata" in note_desc


def test_receptor_oligomeric_state_enum_matches_config() -> None:
    # The receptor oligomeric-state enum is the model-facing list; the
    # AI-vs-classifier cross-check keys on these exact strings. The non-unknown
    # members must equal the cross-check's mapping keys, plus 'unknown' (which the
    # mapping intentionally omits because it asserts no count). A drift would make
    # a model value the cross-check silently never recognises.
    props = _receptor_info().properties
    assert "oligomeric_state" in props
    enum = props["oligomeric_state"].properties["value"].enum
    assert enum is not None
    assert len(enum) == len(set(enum))  # no duplicate members
    assert set(enum) == set(AI_OLIGOMER_TO_RECEPTOR_LEVEL) | {AI_OLIGOMER_UNKNOWN}


def test_receptor_oligomeric_state_mirrors_state_evidence_shape() -> None:
    # The new field must mirror the structure_info.state shape: a value/confidence/
    # evidence object, all three required, so it votes and renders like state.
    oligo = _receptor_info().properties["oligomeric_state"]
    assert oligo.type == types.Type.OBJECT
    assert set(oligo.required or []) == {"value", "confidence", "evidence"}
    assert oligo.properties["evidence"].type == types.Type.OBJECT
    assert set(oligo.properties["evidence"].required or []) == {
        "source",
        "quote_or_path",
        "reasoning",
    }


def test_receptor_oligomeric_state_description_counts_only_receptors() -> None:
    # The field description must hardcode "count only the receptor(s)" with the
    # baked-in examples, so the model never folds partners into the count.
    desc = _receptor_info().properties["oligomeric_state"].properties["value"].description or ""
    lower = desc.lower()
    assert "do not count" in lower
    assert "g-protein" in lower
    # The three baked-in examples: receptor+G-protein -> monomer; GABA-B -> hetero;
    # mGlu2/CaSR -> homo.
    assert "monomer" in lower
    assert "gaba-b" in lower
    assert "hetero-dimer" in lower
    assert ("mglu2" in lower) or ("casr" in lower)
    assert "homo-dimer" in lower


def test_v5_prompt_asks_for_receptor_oligomeric_state() -> None:
    # The prompt must instruct the model to output the receptor's own oligomeric
    # state counting only receptors, and surface the author assembly as reference.
    lower = _v5_prompt_text().lower()
    assert "oligomeric_state" in lower or "oligomeric state" in lower
    assert "do not count" in lower
    assert "author-deposited biological assembly" in lower
    assert "reference" in lower


def test_v5_note_carries_sourcing_constraint() -> None:
    # The prompt's chimera-note guidance must carry a sourcing constraint: the
    # composition details must come from the paper or PDB metadata. (The example
    # wording and the is_chimeric definition are owner-controlled and not
    # asserted here.)
    lower = _v5_prompt_text().lower()
    assert "must come from the paper or pdb metadata" in lower
