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

    # Traverse polymer entities to find the chain
    polymer_entities = enriched_entry.get("polymer_entities") or []

    api_slugs: list[str] = []
    found_chain = False

    for entity in polymer_entities:
        if not isinstance(entity, dict):
            continue
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        auth_asym_ids = identifiers.get("auth_asym_ids") or []

        if ai_chain in auth_asym_ids:
            found_chain = True
            uniprots = entity.get("uniprots") or []
            for u in uniprots:
                if not isinstance(u, dict):
                    continue
                slug = u.get("gpcrdb_entry_name_slug")
                if slug:
                    api_slugs.append(slug)
            break

    if not found_chain:
        return warnings

    match = ai_uniprot in api_slugs

    if match:
        receptor_info["validation_status"] = VALIDATION_RECEPTOR_MATCH
        receptor_info["api_reality"] = api_slugs
    else:
        receptor_info["validation_status"] = VALIDATION_UNIPROT_CLASH
        receptor_info["api_reality"] = api_slugs
        warnings.append(
            f"UNIPROT_CLASH at 'receptor_info': "
            f"'{ai_uniprot}' on Chain {ai_chain} vs API reality {api_slugs}."
        )

    return warnings
