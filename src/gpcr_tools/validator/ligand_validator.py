"""Ligand cross-validation and chemical identity injection.

Validates each AI-reported ligand against the enriched PDB metadata and
injects pre-fetched chemical identifiers (InChIKey, SMILES, etc.) in-place.

Purely offline: reads only from the pre-loaded enriched entry dict.
No network calls.
"""

from __future__ import annotations

import logging
from typing import Any

from gpcr_tools.config import (
    APO_SENTINEL,
    EMPTY_VALUES,
    LIGAND_EXCLUDE_LIST,
    LIGAND_TYPE_PEPTIDE,
    LIGAND_TYPE_PROTEIN,
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_MATCHED_POLYMER,
    VALIDATION_MATCHED_SMALL_MOLECULE,
    VALIDATION_SKIPPED_APO,
)

logger = logging.getLogger(__name__)


def _build_ligand_api_context(
    enriched_entry: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build lookup indexes from enriched entry.

    Returns ``{"np_by_comp": {...}, "poly_by_chain": {...}}``.
    All `.get()` calls use the None-safe ``or {}`` / ``or ""`` pattern
    (Blood Lesson 1).
    """
    np_by_comp: dict[str, dict[str, Any]] = {}
    for np_ent in enriched_entry.get("nonpolymer_entities") or []:
        comp = np_ent.get("nonpolymer_comp") or {}
        cc = comp.get("chem_comp") or {}
        descriptor = comp.get("rcsb_chem_comp_descriptor") or {}
        comp_id = cc.get("id") or ""
        if not comp_id or comp_id in LIGAND_EXCLUDE_LIST:
            continue
        np_by_comp[comp_id] = {
            "name": cc.get("name"),
            "InChIKey": descriptor.get("InChIKey"),
            "SMILES": descriptor.get("SMILES"),
            "SMILES_stereo": descriptor.get("SMILES_stereo"),
            "pubchem_cid": comp.get("gpcrdb_pubchem_cid"),
        }

    poly_by_chain: dict[str, dict[str, Any]] = {}
    for p_ent in enriched_entry.get("polymer_entities") or []:
        ep = p_ent.get("entity_poly") or {}
        desc = (p_ent.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        seq = ep.get("pdbx_seq_one_letter_code_can") or ""
        for inst in p_ent.get("polymer_entity_instances") or []:
            chain = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if chain:
                poly_by_chain[chain] = {
                    "description": desc,
                    "type": (ep.get("type") or ""),
                    "sequence": seq,
                }

    return {"np_by_comp": np_by_comp, "poly_by_chain": poly_by_chain}


def validate_and_enrich_ligands(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
) -> list[str]:
    """Validate AI-reported ligands and inject chemical identifiers.

    Mutates *best_run_data* ligand dicts in-place.
    Returns a list of warning strings for ``GHOST_LIGAND`` detections.

    Blood Lesson 3 — Warning format:
        ``f"GHOST_LIGAND at 'ligands[{label}]': '{name}' ({cid}) not found in API entities."``
    """
    warnings: list[str] = []
    ligands = best_run_data.get("ligands")
    if not isinstance(ligands, list) or not ligands:
        return warnings

    api = _build_ligand_api_context(enriched_entry)

    for lig in ligands:
        if not isinstance(lig, dict):
            continue

        comp_id = (lig.get("chem_comp_id") or "").strip()
        chain_id = (lig.get("chain_id") or "").strip()
        ai_name = (lig.get("name") or "").strip()
        ai_type = (lig.get("type") or "").strip()

        comp_id_valid = bool(comp_id) and comp_id.lower() not in EMPTY_VALUES
        chain_id_valid = bool(chain_id) and chain_id.lower() not in EMPTY_VALUES

        # 1. Explicit Apo
        if ai_name.lower() == APO_SENTINEL or comp_id.lower() == APO_SENTINEL:
            lig["validation_status"] = VALIDATION_SKIPPED_APO
            continue

        # 2. Polymer path: peptides/proteins validated by chain_id
        if ai_type.lower() in (LIGAND_TYPE_PEPTIDE, LIGAND_TYPE_PROTEIN) and chain_id_valid:
            chains = [c.strip() for c in chain_id.split(",")]
            matched_sequences = []
            found_any_chain = False

            for c in chains:
                poly_match = api["poly_by_chain"].get(c)
                if poly_match:
                    found_any_chain = True
                    seq = poly_match.get("sequence")
                    if seq:
                        matched_sequences.append(seq)

            if found_any_chain:
                lig["validation_status"] = VALIDATION_MATCHED_POLYMER
                lig["Sequence"] = " / ".join(matched_sequences)
                continue
        # 3. Small-molecule path
        if comp_id_valid:
            np_match = api["np_by_comp"].get(comp_id)
            if np_match:
                lig["validation_status"] = VALIDATION_MATCHED_SMALL_MOLECULE
                lig["InChIKey"] = np_match.get("InChIKey")
                lig["api_pubchem_cid"] = np_match.get("pubchem_cid")
                lig["SMILES_stereo"] = np_match.get("SMILES_stereo")
                lig["SMILES"] = np_match.get("SMILES")
                continue

            if comp_id in LIGAND_EXCLUDE_LIST:
                lig["validation_status"] = VALIDATION_EXCLUDED_BUFFER
                continue

        # 4. Ghost fallback
        lig["validation_status"] = VALIDATION_GHOST_LIGAND
        label = comp_id if comp_id_valid else ai_name
        cid_display = comp_id or "no comp_id"
        warnings.append(
            f"GHOST_LIGAND at 'ligands[{label}]': "
            f"'{ai_name}' ({cid_display}) not found in API entities."
        )

    return warnings
