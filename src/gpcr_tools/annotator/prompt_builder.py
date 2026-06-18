from __future__ import annotations

import json
import logging
from collections import defaultdict
from types import MappingProxyType
from typing import Any

from gpcr_tools.annotator.detect_orchestrator import assemble_detect_block
from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    LIGAND_EXCLUDE_LIST,
    POLYMER_FEATURES_CACHE_NAME,
    get_config,
)
from gpcr_tools.detector.signals import DetectSignal
from gpcr_tools.validator.cache import PolymerFeaturesCache
from gpcr_tools.validator.oligomer import (
    format_7tm_status,
    get_sequence_length,
    scan_all_chains_7tm,
)

logger = logging.getLogger(__name__)

# Sentinel meaning "the whole-PDB transmembrane fetch failed, so no chain has a
# known 7TM status". It is distinct from an empty map (the fetch succeeded but a
# chain genuinely carries no TM annotation, e.g. a peptide). When the table is
# built with this sentinel, the ``7tm_status`` column is OMITTED rather than
# asserting a false ``UNKNOWN 0/0`` for a real receptor (the "don't fail open on
# a fetch failure" rule applied to the prompt side).
TM_FETCH_FAILED = object()


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


# Human-readable origin labels for an assembly's ``pdbx_struct_assembly.rcsb_details``.
# Anything not listed (or absent) renders with the raw value so an unexpected
# origin is still shown rather than silently dropped.
_ASSEMBLY_ORIGIN_LABELS: MappingProxyType[str, str] = MappingProxyType(
    {
        "author_defined_assembly": "author-defined",
        "author_and_software_defined_assembly": "author- and software-defined",
        "software_defined_assembly": "software-defined",
    }
)


def _assembly_origin_label(pdbx_struct_assembly: dict) -> str:
    """Render an assembly's origin (author vs software) for the reference line.

    Appends the software method (e.g. ``PISA``) when present so a conflicting
    software-predicted assembly is visibly distinguished from the author's.
    """
    details = pdbx_struct_assembly.get("rcsb_details")
    label = _ASSEMBLY_ORIGIN_LABELS.get(details) if isinstance(details, str) else None
    if not label:
        label = str(details) if details else "unspecified origin"
    method = pdbx_struct_assembly.get("method_details")
    if method:
        return f"{label} ({method})"
    return label


def generate_author_assembly_reference(pdb_id: str, enriched_data: dict) -> str:
    """Render the author-deposited biological assembly as a reference line.

    One structure-level block (not per chain) listing every deposited biological
    assembly with its oligomeric state, stoichiometry, and origin label
    (author-defined vs software, e.g. PISA). ALL assemblies are listed -- when
    they genuinely conflict (an author-defined monomer alongside a software
    homo-dimer) the model sees the conflict and judges for itself, rather than
    being handed a single pre-chosen answer.

    Framed explicitly as reference-only and NOT authoritative: the assembly
    counts every chain (so a receptor + G-protein complex reads as a higher-order
    "Hetero N-mer" even though the receptor itself is a monomer). All data comes
    from the enriched JSON -- no network call. Returns ``""`` when no assembly
    with a symmetry block is present, so an ordinary structure's prompt only
    grows when there is something to show.
    """
    entry = _get_entry(enriched_data)
    assemblies = [a for a in (entry.get("assemblies") or []) if isinstance(a, dict)]

    rows: list[str] = []
    for asm in assemblies:
        symmetry_blocks = [
            s for s in (asm.get("rcsb_struct_symmetry") or []) if isinstance(s, dict)
        ]
        if not symmetry_blocks:
            continue
        origin = _assembly_origin_label(asm.get("pdbx_struct_assembly") or {})
        for sym in symmetry_blocks:
            state = sym.get("oligomeric_state")
            if not state:
                continue
            stoich = sym.get("stoichiometry")
            kind = sym.get("kind")
            parts = [f"{state}"]
            if stoich:
                # ``stoichiometry`` is a list[str] (e.g. ['A1','B1']) -- render it
                # as ``[A1, B1]`` rather than the raw Python list repr.
                if isinstance(stoich, list):
                    parts.append("[" + ", ".join(str(s) for s in stoich) + "]")
                else:
                    parts.append(f"{stoich}")
            if kind:
                parts.append(f"{kind}")
            rows.append(f"- {', '.join(parts)} [{origin}]")

    if not rows:
        return ""

    header = (
        "### AUTHOR-DEPOSITED BIOLOGICAL ASSEMBLY (reference only, NOT authoritative)\n"
        f"From the structure authors' biological assembly deposited in the PDB for {pdb_id}. "
        "This is reference information to inform your own judgment, not the answer. It counts "
        "ALL chains (so a receptor + G-protein complex is reported as a higher-order complex "
        "even though the receptor itself is a monomer), and it can be wrong in either "
        "direction. A 'Homo N-mer' or software-predicted (e.g. PISA) assembly often "
        "reflects crystallographic packing rather than a true biological oligomer, so do "
        "NOT treat it as evidence that the receptor is an oligomer — defer to the paper. "
        "When more than one assembly is listed and they conflict, weigh them "
        "against the polymer table and the paper and decide for yourself:"
    )
    return header + "\n" + "\n".join(rows)


def enhanced_simplify_pdb_json(
    enriched_data: dict,
    tm_by_chain: dict[str, dict[str, Any]] | object | None = None,
) -> dict:
    """Simplifies the enriched PDB JSON into a minimal dictionary for Gemini.

    *tm_by_chain* maps an ``auth_asym_id`` to its 7TM analysis dict; when given,
    every polymer chain gains two factual columns (its 7TM status and residue
    length) so the model can tell a 7TM receptor (``COMPLETE 7/7``) from a
    peptide ligand or partner (``UNKNOWN 0/0``) without the code pre-judging
    which chains are receptors. A chain absent from the map reads ``UNKNOWN 0/0``
    -- a real fact (the fetch succeeded; this chain has no TM annotation).

    When the whole-PDB transmembrane fetch failed entirely, pass the
    ``TM_FETCH_FAILED`` sentinel: the ``7tm_status`` column is then OMITTED for
    every chain rather than asserting a false ``UNKNOWN 0/0`` that would make a
    genuine receptor look identical to a peptide. The ``residue_length`` column
    does not depend on the fetch and is always emitted.
    """
    tm_fetch_failed = tm_by_chain is TM_FETCH_FAILED
    tm_map: dict[str, dict[str, Any]] = (
        {} if (tm_fetch_failed or not isinstance(tm_by_chain, dict)) else tm_by_chain
    )
    entry = _get_entry(enriched_data)
    pdb_id = entry.get("rcsb_id") or "UNKNOWN"

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

        # Two factual columns per chain, neutral for every polymer (receptor, G
        # protein, peptide, antibody): the 7TM status (resolved/total) and the
        # residue length. A 7TM receptor reads COMPLETE 6-7/7; a peptide ligand or
        # partner reads UNKNOWN 0/0. Keyed per chain so the model is told the facts
        # rather than which chains the code thinks are receptors. When the whole
        # fetch failed (TM_FETCH_FAILED) the 7TM column is omitted entirely so a
        # real receptor is never asserted as a false UNKNOWN 0/0; residue_length
        # is independent of the fetch and is always emitted.
        component: dict[str, Any] = {
            "chain_ids": chains,
            "description": desc,
            "type": poly_type,
            "organism": organism,
            "uniprot_accessions": uniprot_accessions,
            "entry_names": entry_names,
            "residue_length": get_sequence_length(poly),
        }
        if not tm_fetch_failed:
            component["7tm_status"] = {c: format_7tm_status(tm_map.get(c)) for c in chains}

        simplified["polymer_components"].append(component)

    # Extract nonpolymers
    nonpolymers = entry.get("nonpolymer_entities") or []
    for np_entity in nonpolymers:
        comp = np_entity.get("nonpolymer_comp") or {}
        chem_comp = comp.get("chem_comp") or {}
        chem_comp_id = chem_comp.get("id")

        if not chem_comp_id:
            continue

        # Exclude common buffers and ions -- but never strip an incidental-candidate molecule
        # (e.g. palmitate): the model must see it to judge its role (accommodate
        # and guide, rather than conceal).
        if chem_comp_id in LIGAND_EXCLUDE_LIST and chem_comp_id not in INCIDENTAL_CANDIDATES:
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


def _collect_polymer_chain_ids(enriched_data: dict) -> set[str]:
    """Collect every polymer chain's ``auth_asym_id`` from enriched JSON."""
    entry = _get_entry(enriched_data)
    chain_ids: set[str] = set()
    for poly in entry.get("polymer_entities") or []:
        chains = (poly.get("rcsb_polymer_entity_container_identifiers") or {}).get(
            "auth_asym_ids"
        ) or []
        chain_ids.update(c for c in chains if c)
    return chain_ids


def _polymer_tm_by_chain(pdb_id: str, enriched_data: dict) -> dict[str, dict[str, Any]] | object:
    """Fetch per-chain 7TM analysis for every polymer chain (best-effort).

    Reuses the oligomer classifier's cached TM-data path so the prompt table and
    the classifier warm one another off the same cache file. Returns a per-chain
    map on success (a chain genuinely lacking a TM annotation is simply absent ->
    renders ``UNKNOWN 0/0``). Returns the ``TM_FETCH_FAILED`` sentinel when the
    whole-PDB fetch failed entirely (or no cache directory exists), so the caller
    OMITS the 7TM column rather than asserting a false ``UNKNOWN 0/0`` for a real
    receptor.
    """
    chain_ids = _collect_polymer_chain_ids(enriched_data)
    if not chain_ids:
        return {}
    try:
        cfg = get_config()
        cache = PolymerFeaturesCache(cfg.cache_dir / POLYMER_FEATURES_CACHE_NAME)
        results, graphql_entry = scan_all_chains_7tm(pdb_id, chain_ids, cache=cache)
        if graphql_entry is None:
            # The whole-PDB feature fetch returned nothing: omit the column rather
            # than assert a false UNKNOWN 0/0 for every chain (a real receptor must
            # never be made to look like a peptide).
            return TM_FETCH_FAILED
        # Persist the successful fetch so subsequent runs / the aggregate stage hit
        # the cache instead of re-issuing 10 live RCSB requests per PDB.
        try:
            cache.save()
        except OSError as exc:
            logger.warning("[%s] Failed to save polymer-features cache: %s", pdb_id, exc)
        return results
    except Exception as exc:
        logger.warning("[%s] 7TM scan for polymer table unavailable: %s", pdb_id, exc)
        return TM_FETCH_FAILED


def build_prompt_parts(
    pdb_id: str,
    enriched_data: dict,
    prompt_template: str,
    detect_signals: list[DetectSignal] | None = None,
) -> list[str]:
    """Assembles the prompt string parts sent to Gemini.

    *detect_signals* (advisory ones) add an evidence block between the metadata
    and the paper; with none, the parts are byte-for-byte unchanged.
    """
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

    # 5. Simplified enriched JSON, with each polymer chain's 7TM status + length
    tm_by_chain = _polymer_tm_by_chain(pdb_id, enriched_data)
    simplified = enhanced_simplify_pdb_json(enriched_data, tm_by_chain=tm_by_chain)
    parts.append(json.dumps(simplified, indent=2))
    parts.append("\n\n")

    # 5a. Author-deposited biological assembly, as a structure-level reference line
    # for the receptor oligomeric-state call (reference only, not authoritative).
    # Nothing is appended when no assembly carries a symmetry block, so an ordinary
    # structure's prompt only grows when there is something to show.
    assembly_reference = generate_author_assembly_reference(pdb_id, enriched_data)
    if assembly_reference:
        parts.append(assembly_reference)
        parts.append("\n\n")

    # 5b. Detector evidence (advisory detect signals only). No advisory signals
    # -> nothing appended, so an ordinary structure's prompt is byte-identical.
    detect_block = assemble_detect_block(detect_signals or [])
    if detect_block:
        parts.append(detect_block)
        parts.append("\n\n")

    # 6. Full paper header
    parts.append("--- FULL PAPER ---\n")

    return parts
