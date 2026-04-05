"""Receptor identity validation against enriched PDB data.

Validates the AI-extracted receptor UniProt entry name against the
enriched polymer entity data.  Injects ``validation_status`` and
``api_reality`` into the receptor_info dict in-place.

Purely offline: reads only from the pre-loaded enriched entry dict.
"""

from __future__ import annotations

import logging
from typing import Any

from gpcr_tools.config import (
    VALIDATION_RECEPTOR_MATCH,
    VALIDATION_UNIPROT_CLASH,
)

logger = logging.getLogger(__name__)


def validate_receptor_identity(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
) -> list[str]:
    """Validate receptor identity and inject validation status.

    Mutates *best_run_data["receptor_info"]* in-place.
    Returns a list of warning strings for ``UNIPROT_CLASH`` detections.

    Blood Lesson 3 — Warning format:
        ``f"UNIPROT_CLASH at 'receptor_info': '{ai_uid}' on Chain {chain} vs API reality [...]"``
    """
    warnings: list[str] = []
    receptor_info = best_run_data.get("receptor_info")

    if not isinstance(receptor_info, dict):
        return warnings

    ai_uniprot = receptor_info.get("uniprot_entry_name")
    ai_chain = receptor_info.get("chain_id")

    if not ai_uniprot or not ai_chain:
        return warnings

    # chain_id can be comma-separated (e.g. "B, F" for homodimers)
    ai_chains = [c.strip() for c in ai_chain.split(",") if c.strip()]

    # Traverse polymer entities to collect slugs for every reported chain
    polymer_entities = enriched_entry.get("polymer_entities") or []

    # Per-chain results: chain -> list of slugs from its entity
    chain_slugs: dict[str, list[str]] = {}

    for chain in ai_chains:
        for entity in polymer_entities:
            if not isinstance(entity, dict):
                continue
            identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            auth_asym_ids = identifiers.get("auth_asym_ids") or []

            if chain in auth_asym_ids:
                slugs: list[str] = []
                for u in entity.get("uniprots") or []:
                    if not isinstance(u, dict):
                        continue
                    slug = u.get("gpcrdb_entry_name_slug")
                    if slug:
                        slugs.append(slug)
                chain_slugs[chain] = slugs
                break  # found entity for this chain, move to next chain

    if not chain_slugs:
        return warnings

    # Determine match / clash per chain
    clashed_chains = [c for c, s in chain_slugs.items() if ai_uniprot not in s]

    # Aggregate all unique slugs across matched entities for api_reality
    all_slugs: list[str] = []
    seen: set[str] = set()
    for s_list in chain_slugs.values():
        for s in s_list:
            if s not in seen:
                seen.add(s)
                all_slugs.append(s)

    if not clashed_chains:
        receptor_info["validation_status"] = VALIDATION_RECEPTOR_MATCH
        receptor_info["api_reality"] = all_slugs
    else:
        receptor_info["validation_status"] = VALIDATION_UNIPROT_CLASH
        receptor_info["api_reality"] = all_slugs
        clash_detail = ", ".join(f"Chain {c} -> {chain_slugs[c]}" for c in clashed_chains)
        warnings.append(
            f"UNIPROT_CLASH at 'receptor_info': '{ai_uniprot}' clashes on {clash_detail}."
        )

    return warnings
