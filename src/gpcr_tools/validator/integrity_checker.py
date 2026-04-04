"""Generic integrity checks (ghost chain, fake UniProt/PubChem, ghost ligand, method).

Port of the legacy ``validate_all()`` function.  Five checks, all operating
on the AI data cross-referenced against enriched PDB metadata.

Blood Lesson 3 — Warning format:
    Every warning follows ``f"TYPE at '{path}': description"``.

Blood Lesson 4 — Magic strings:
    ``EMPTY_VALUES`` imported from ``config.py``, not redefined locally.
"""

from __future__ import annotations

import logging
from typing import Any

from gpcr_tools.config import APO_SENTINEL, EMPTY_VALUES
from gpcr_tools.validator.api_clients import check_pubchem_existence, check_uniprot_existence
from gpcr_tools.validator.cache import ValidationCache

logger = logging.getLogger(__name__)


def _build_pdb_context(
    enriched_entry: dict[str, Any],
) -> dict[str, Any]:
    """Extract valid chains, ligand IDs, and method from enriched entry.

    Blood Lesson 1 — None-safety:
        Every ``.get()`` on enriched data uses ``or {}`` / ``or []`` / ``or ""``.
    """
    chains: set[str] = set()
    ligands: set[str] = set()
    method: str | None = None

    # Method
    exptl = enriched_entry.get("exptl") or []
    if exptl and isinstance(exptl, list) and isinstance(exptl[0], dict):
        method = (exptl[0].get("method") or "").lower() or None

    # Polymer chains
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        for instance in entity.get("polymer_entity_instances") or []:
            if not isinstance(instance, dict):
                continue
            cid = (instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if cid:
                chains.add(cid)

    # Non-polymer chains & ligand IDs
    for np_ent in enriched_entry.get("nonpolymer_entities") or []:
        if not isinstance(np_ent, dict):
            continue
        # Ligand ID: triple chain with or {} at each level (BL1)
        comp_id = ((np_ent.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id")
        if comp_id:
            ligands.add(comp_id)
        # Non-polymer chain IDs
        for instance in np_ent.get("nonpolymer_entity_instances") or []:
            if not isinstance(instance, dict):
                continue
            cid = (instance.get("rcsb_nonpolymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if cid:
                chains.add(cid)

    return {"chains": chains, "ligands": ligands, "method": method}


def _is_empty_value(value: Any) -> bool:
    """Check if *value* is semantically empty (using ``EMPTY_VALUES`` from config)."""
    if not value:
        return True
    return str(value).lower() in EMPTY_VALUES


def validate_all(
    pdb_id: str,
    ai_data: dict[str, Any],
    enriched_entry: dict[str, Any],
    cache: ValidationCache | None = None,
) -> list[str]:
    """Run five integrity checks on *ai_data* against *enriched_entry*.

    Checks:
      1. Ghost Chain — AI chain_id not in PDB source
      2. Fake UniProt — entry_name doesn't exist (API)
      3. Fake PubChem — CID doesn't exist (API)
      4. Ghost Ligand — chem_comp_id not in PDB metadata
      5. Method Consistency — AI method vs PDB method

    Returns list of warning strings.  All warnings follow the
    ``f"TYPE at '{path}': description"`` format (Blood Lesson 3).
    """
    warnings: list[str] = []
    ctx = _build_pdb_context(enriched_entry)
    valid_chains = ctx["chains"]
    valid_ligands = ctx["ligands"]
    real_method: str | None = ctx["method"]

    def _check_node(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            # Check 1: Ghost Chain
            if "chain_id" in node:
                val = node["chain_id"]
                if val and not _is_empty_value(val) and str(val).lower() != APO_SENTINEL:
                    current_chains = [c.strip() for c in str(val).replace(";", ",").split(",")]
                    if valid_chains:
                        for c in current_chains:
                            if c and c not in valid_chains:
                                warnings.append(
                                    f"Ghost Chain at '{path}': '{c}' not in PDB Source."
                                )

            # Check 2: Fake UniProt
            if "uniprot_entry_name" in node:
                uid = node["uniprot_entry_name"]
                if uid and isinstance(uid, str) and uid.lower() not in EMPTY_VALUES:
                    if "_" not in uid:
                        warnings.append(
                            f"Invalid Format at '{path}': '{uid}' (Expected: name_species)"
                        )
                    elif cache is not None:
                        result = check_uniprot_existence(uid, cache)
                        if result is False:
                            warnings.append(
                                f"Fake UniProt ID at '{path}': '{uid}' does not exist in UniProtKB."
                            )
                        elif result is None:
                            warnings.append(
                                f"[API_UNAVAILABLE] at '{path}': "
                                f"Could not verify UniProt ID '{uid}'."
                            )

            # Check 3: Fake PubChem
            if "pubchem_id" in node:
                cid = node["pubchem_id"]
                if cid and not _is_empty_value(cid) and cache is not None:
                    result = check_pubchem_existence(str(cid), cache)
                    if result is False:
                        warnings.append(
                            f"Invalid PubChem CID at '{path}': '{cid}' does not exist in PubChem."
                        )
                    elif result is None:
                        warnings.append(
                            f"[API_UNAVAILABLE] at '{path}': Could not verify PubChem CID '{cid}'."
                        )

            # Check 4: Ghost Ligand
            if "chem_comp_id" in node:
                lid = node["chem_comp_id"]
                if (
                    lid
                    and not _is_empty_value(lid)
                    and str(lid).lower() != APO_SENTINEL
                    and valid_ligands
                    and lid not in valid_ligands
                ):
                    warnings.append(
                        f"Ghost Ligand ID at '{path}': "
                        f"'{lid}' not found in PDB Metadata "
                        f"(valid: {sorted(valid_ligands)})."
                    )

            # Check 5: Method consistency (top level only)
            if path == ".structure_info" and "method" in node and real_method:
                ai_method = (str(node["method"]) or "").lower()
                is_conflict = ("x-ray" in real_method and "x-ray" not in ai_method) or (
                    "electron" in real_method
                    and "electron" not in ai_method
                    and "cryo" not in ai_method
                )
                if is_conflict:
                    warnings.append(
                        f"Method Conflict at 'structure_info': "
                        f"PDB says '{real_method}', AI says '{ai_method}'."
                    )

            # Recurse
            for k, v in node.items():
                _check_node(v, f"{path}.{k}")

        elif isinstance(node, list):
            for i, item in enumerate(node):
                _check_node(item, f"{path}[{i}]")

    _check_node(ai_data)
    return warnings
