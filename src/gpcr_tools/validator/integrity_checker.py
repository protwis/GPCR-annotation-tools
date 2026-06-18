"""Generic integrity checks (ghost chain, fake UniProt/PubChem, ghost ligand, method).

Five checks, all operating on the AI data cross-referenced against enriched
PDB metadata.

Conventions:
    - Every warning follows ``f"TYPE at '{path}': description"``.
    - ``EMPTY_VALUES`` is imported from ``config.py``, not redefined locally.
"""

from __future__ import annotations

import logging
from typing import Any

from gpcr_tools.config import ALERT_PREFIX_API_UNAVAILABLE, APO_SENTINEL, EMPTY_VALUES
from gpcr_tools.validator.api_clients import check_pubchem_existence, check_uniprot_existence
from gpcr_tools.validator.cache import ValidationCache

logger = logging.getLogger(__name__)


def _build_pdb_context(
    enriched_entry: dict[str, Any],
) -> dict[str, Any]:
    """Extract valid chains, ligand IDs, and method from enriched entry.

    None-safe: every ``.get()`` on enriched data uses ``or {}`` / ``or []``.
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
        # Ligand ID: defensively unwrap each nested level.
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

    # Branched (oligosaccharide) component ids. Sugars are real ligands but
    # live in a bucket separate from nonpolymer_entities. No single field is
    # populated for every deposition, so harvest from each source the fetcher
    # provides: struct-conn records, instance features, and any PRD reference.
    # A struct-conn record also carries the bonded attachment residue (e.g. the
    # asparagine an N-glycan hangs off), which only widens the whitelist and
    # never causes a false positive.
    #
    # branched_unresolved guards the ghost-ligand check: if a branched entity
    # is present but yields no component id (e.g. a free glycan with no
    # connectivity record), the chemical inventory is not fully known, so a
    # claimed sugar must not be flagged as a ghost.
    #
    # Modified polymer residues (chem_comp_nstd_monomers, e.g. SEP/TPO) are
    # intentionally not harvested: the model is not shown them as ligands and
    # admitting them would whitelist hundreds of residues -- accepted gap.
    branched_unresolved = False
    for br_ent in enriched_entry.get("branched_entities") or []:
        if not isinstance(br_ent, dict):
            continue
        entity_comps: set[str] = set()
        prd_id = ((br_ent.get("prd") or {}).get("pdbx_reference_molecule") or {}).get(
            "chem_comp_id"
        )
        if prd_id:
            entity_comps.add(prd_id)
        for instance in br_ent.get("branched_entity_instances") or []:
            if not isinstance(instance, dict):
                continue
            for conn in instance.get("rcsb_branched_struct_conn") or []:
                if not isinstance(conn, dict):
                    continue
                for side in ("connect_partner", "connect_target"):
                    comp_id = (conn.get(side) or {}).get("label_comp_id")
                    if comp_id:
                        entity_comps.add(comp_id)
            for feat in instance.get("rcsb_branched_instance_feature") or []:
                if not isinstance(feat, dict):
                    continue
                for value in feat.get("feature_value") or []:
                    if isinstance(value, dict) and value.get("comp_id"):
                        entity_comps.add(value["comp_id"])
        if entity_comps:
            ligands |= entity_comps
        else:
            branched_unresolved = True

    return {
        "chains": chains,
        "ligands": ligands,
        "method": method,
        "inventory_known": not branched_unresolved,
    }


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
    ``f"TYPE at '{path}': description"`` format.
    """
    warnings: list[str] = []
    ctx = _build_pdb_context(enriched_entry)
    valid_chains = ctx["chains"]
    valid_ligands = ctx["ligands"]
    inventory_known: bool = ctx["inventory_known"]
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

            # Check 2: Fake UniProt. The receptor field may comma-join an obligate
            # heterodimer's two slugs ("grm2_human, grm7_human"); check each slug
            # separately so a joined value is not looked up as one (non-existent) ID.
            if "uniprot_entry_name" in node:
                uid = node["uniprot_entry_name"]
                if uid and isinstance(uid, str) and uid.lower() not in EMPTY_VALUES:
                    for one_uid in (u.strip() for u in uid.split(",") if u.strip()):
                        if "_" not in one_uid:
                            warnings.append(
                                f"Invalid Format at '{path}': '{one_uid}' (Expected: name_species)"
                            )
                        elif cache is not None:
                            result = check_uniprot_existence(one_uid, cache)
                            if result is False:
                                warnings.append(
                                    f"Fake UniProt ID at '{path}': '{one_uid}' does not exist in UniProtKB."
                                )
                            elif result is None:
                                warnings.append(
                                    f"{ALERT_PREFIX_API_UNAVAILABLE} at '{path}': "
                                    f"Could not verify UniProt ID '{one_uid}'."
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
                            f"{ALERT_PREFIX_API_UNAVAILABLE} at '{path}': Could not verify PubChem CID '{cid}'."
                        )

            # Check 4: Ghost Ligand
            # valid_ligands covers every real chemical component (nonpolymer
            # plus branched sugars); protein/Apo ligands carry an empty/apo
            # sentinel and are skipped. When the inventory is fully known an
            # empty set genuinely means the structure has no chemical component,
            # so any real id the model claims is a ghost -- no short-circuit on
            # emptiness. inventory_known is False only when a branched entity is
            # present but its components could not be enumerated, in which case
            # we stay silent rather than risk flagging a real glycan.
            if "chem_comp_id" in node and inventory_known:
                lid = node["chem_comp_id"]
                if (
                    lid
                    and not _is_empty_value(lid)
                    and str(lid).lower() != APO_SENTINEL
                    and lid not in valid_ligands
                ):
                    valid_desc = (
                        f"valid: {sorted(valid_ligands)}"
                        if valid_ligands
                        else "no chemical components in PDB metadata"
                    )
                    warnings.append(
                        f"Ghost Ligand ID at '{path}': "
                        f"'{lid}' not found in PDB Metadata ({valid_desc})."
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
