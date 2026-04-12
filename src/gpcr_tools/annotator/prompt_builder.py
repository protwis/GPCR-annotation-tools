from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from gpcr_tools.config import LIGAND_EXCLUDE_LIST


def _get_entry(enriched_data: dict) -> dict:
    """Dereference the ``data.entry`` envelope from enriched JSON."""
    return (enriched_data.get("data") or {}).get("entry") or {}


def generate_chain_inventory_reminder(pdb_id: str, enriched_data: dict) -> str:
    """Generates a human-readable summary of the polymer chains in the PDB."""
    entry = _get_entry(enriched_data)
    polymers = entry.get("polymer_entities") or []
    if not polymers:
        return f"### CHAIN INVENTORY REMINDER\nThis structure ({pdb_id}) contains 0 polymer chains."

    # Group chains by description
    desc_to_chains: dict[str, list[str]] = defaultdict(list)
    unique_chains: set[str] = set()

    for poly in polymers:
        desc = (poly.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "Unknown polymer"
        # We handle both single strings and lists for descriptions
        if isinstance(desc, list):
            desc = desc[0] if desc else "Unknown polymer"

        chains = (poly.get("rcsb_polymer_entity_container_identifiers") or {}).get(
            "auth_asym_ids"
        ) or []
        if chains:
            desc_to_chains[desc].extend(chains)
            unique_chains.update(chains)

    total_chains = len(unique_chains)

    lines = [
        "### CHAIN INVENTORY REMINDER",
        f"This structure ({pdb_id}) contains EXACTLY {total_chains} unique polymer chain(s):",
    ]

    for desc, chains in desc_to_chains.items():
        chain_str = ", ".join(sorted(chains))
        lines.append(f"- Chain(s) {chain_str}: {desc}")

    return "\n".join(lines)


def enhanced_simplify_pdb_json(enriched_data: dict) -> dict:
    """Simplifies the enriched PDB JSON into a minimal dictionary for Gemini."""
    entry = _get_entry(enriched_data)
    pdb_id = entry.get("id") or "UNKNOWN"

    # Safely get structural details
    method = "Unknown"
    exptl = entry.get("exptl") or []
    if exptl:
        method = exptl[0].get("method") or "Unknown"

    resolution = None
    em_3d = entry.get("em_3d_reconstruction") or []
    if em_3d:
        resolution = em_3d[0].get("resolution")
    if resolution is None:
        refine = entry.get("refine") or []
        if refine:
            resolution = refine[0].get("ls_d_res_high")

    raw_date = (entry.get("rcsb_accession_info") or {}).get("initial_release_date") or ""
    release_date = raw_date.split("T")[0] if raw_date else None

    simplified: dict[str, Any] = {
        "structure_details": {
            "pdb_id": pdb_id,
            "title": (entry.get("struct") or {}).get("title"),
            "method": method,
            "resolution": resolution,
            "release_date": release_date,
        },
        "polymer_components": [],
        "non_polymer_components": [],
    }

    # Extract polymers
    polymers = entry.get("polymer_entities") or []
    for poly in polymers:
        chains = (poly.get("rcsb_polymer_entity_container_identifiers") or {}).get(
            "auth_asym_ids"
        ) or []
        desc = (poly.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "Unknown"
        poly_type = (poly.get("entity_poly") or {}).get("rcsb_entity_polymer_type") or "Unknown"

        uniprots = poly.get("uniprots") or []
        uniprot_accessions = [u.get("rcsb_id") for u in uniprots if u.get("rcsb_id")]
        entry_names = [
            u.get("gpcrdb_entry_name_slug") for u in uniprots if u.get("gpcrdb_entry_name_slug")
        ]

        source = poly.get("rcsb_entity_source_organism") or []
        organism = "Unknown"
        if source:
            organism = source[0].get("scientific_name") or "Unknown"

        simplified["polymer_components"].append(
            {
                "chain_ids": chains,
                "description": desc,
                "type": poly_type,
                "organism": organism,
                "uniprot_accessions": uniprot_accessions,
                "entry_names": entry_names,
            }
        )

    # Extract nonpolymers
    nonpolymers = entry.get("nonpolymer_entities") or []
    for np_entity in nonpolymers:
        comp = np_entity.get("nonpolymer_comp") or {}
        chem_comp = comp.get("chem_comp") or {}
        chem_comp_id = chem_comp.get("id")

        if not chem_comp_id:
            continue

        # Exclude common buffers and ions
        if chem_comp_id in LIGAND_EXCLUDE_LIST:
            continue

        name = chem_comp.get("name") or chem_comp_id
        description = (np_entity.get("rcsb_nonpolymer_entity") or {}).get("pdbx_description")
        chains = (np_entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get(
            "auth_asym_ids"
        ) or []

        determined_type = comp.get("gpcrdb_determined_type") or "unknown"
        pubchem_cid = comp.get("gpcrdb_pubchem_cid")
        synonyms = comp.get("gpcrdb_pubchem_synonyms") or []

        simplified["non_polymer_components"].append(
            {
                "chem_comp_id": chem_comp_id,
                "name": name,
                "description": description,
                "chain_ids": chains,
                "determined_type": determined_type,
                "pubchem_cid": pubchem_cid,
                "synonyms": synonyms,
            }
        )

    return simplified


def build_prompt_parts(
    pdb_id: str,
    enriched_data: dict,
    prompt_template: str,
) -> list[str]:
    """Assembles the prompt string parts sent to Gemini."""
    parts = []

    # 1. System prompt template
    parts.append(prompt_template)
    parts.append("\n\n")

    # 2. Chain inventory reminder
    parts.append(generate_chain_inventory_reminder(pdb_id, enriched_data))
    parts.append("\n\n")

    # 3. Sibling structures warning
    entry = _get_entry(enriched_data)
    siblings = entry.get("sibling_pdbs") or []
    if siblings:
        sib_str = ", ".join(siblings)
        parts.append(
            f"### IMPORTANT: SIBLING STRUCTURES WARNING\n"
            f"This paper also reports structures for the following PDB IDs: {sib_str}.\n"
            f"Ensure you are extracting data ONLY for {pdb_id}. Do NOT mix up ligands "
            f"or active states with the sibling structures reported in the same paper."
        )
        parts.append("\n\n")

    # 4. PDB Metadata header
    parts.append(f"--- PDB METADATA FOR {pdb_id} ---\n")

    # 5. Simplified enriched JSON
    simplified = enhanced_simplify_pdb_json(enriched_data)
    parts.append(json.dumps(simplified, indent=2))
    parts.append("\n\n")

    # 6. Full paper header
    parts.append("--- FULL PAPER ---\n")

    return parts
