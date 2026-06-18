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
    VALIDATION_RECEPTOR_NO_API_DATA,
    VALIDATION_RECEPTOR_RCSB_UNMAPPED,
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

    Warning format:
        ``f"UNIPROT_CLASH at 'receptor_info': '{ai_uid}' is not present on any reported
        chain; API reality: Chain {c} -> [slugs]."``
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

    # uniprot_entry_name can likewise be comma-joined when the model crams an
    # obligate heterodimer's two receptor slugs into the single field (e.g.
    # "grm2_human, grm7_human"). Split it the same way as chain_id so a real
    # heterodimer is matched per-slug rather than mistaken for one absent name.
    ai_uniprots = [u.strip() for u in ai_uniprot.split(",") if u.strip()]

    # Traverse polymer entities to collect slugs for every reported chain
    polymer_entities = enriched_entry.get("polymer_entities") or []

    # Per-chain results: chain -> list of slugs from its entity
    chain_slugs: dict[str, list[str]] = {}
    # Chains whose entity carries a UniProt accession (rcsb_id) but no resolved
    # GPCRdb slug -- a recoverable upstream mapping gap, distinct from a chain the
    # source has no UniProt reference for at all.
    unmapped_accession_chains: set[str] = set()

    for chain in ai_chains:
        for entity in polymer_entities:
            if not isinstance(entity, dict):
                continue
            identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            auth_asym_ids = identifiers.get("auth_asym_ids") or []

            if chain in auth_asym_ids:
                slugs: list[str] = []
                has_accession = False
                for u in entity.get("uniprots") or []:
                    if not isinstance(u, dict):
                        continue
                    if u.get("rcsb_id"):
                        has_accession = True
                    slug = u.get("gpcrdb_entry_name_slug")
                    if slug:
                        slugs.append(slug)
                chain_slugs[chain] = slugs
                if not slugs and has_accession:
                    unmapped_accession_chains.add(chain)
                break  # found entity for this chain, move to next chain

    if not chain_slugs:
        return warnings

    # Aggregate all unique slugs the API exposes (across chains that carry any).
    all_slugs: list[str] = []
    seen: set[str] = set()
    for s_list in chain_slugs.values():
        for s in s_list:
            if s not in seen:
                seen.add(s)
                all_slugs.append(s)

    # A chain the API has no UniProt slug for cannot be checked -- that is "no
    # data", not a clash. Validate only against chains that actually carry a slug;
    # surface the unverifiable chains as a soft note for the curator. A chain whose
    # accession is present but unmapped gets a stronger gating warning below instead,
    # so it is excluded here to avoid a duplicate soft note.
    chains_with_slugs = {c: s for c, s in chain_slugs.items() if s}
    # In the pure no-slug case the unmapped chains get the gating warning below, so
    # they are excluded here; in the mixed case (identity confirmable elsewhere) the
    # soft note is kept for every unverifiable chain.
    suppress_unmapped_note = not chains_with_slugs
    no_data_chains = sorted(
        c
        for c, s in chain_slugs.items()
        if not s and not (suppress_unmapped_note and c in unmapped_accession_chains)
    )
    if no_data_chains:
        warnings.append(
            f"RECEPTOR_NO_API_DATA at 'receptor_info': the API exposes no UniProt slug for "
            f"chain(s) {', '.join(no_data_chains)}; receptor identity is not verifiable there."
        )

    # The AI names ONE receptor, but a Class C dimer -- and any hetero-oligomer --
    # legitimately carries a DIFFERENT receptor on its partner chain (e.g. GABA-B:
    # GABBR1 on one chain, GABBR2 on the other). So a clash is the AI naming a
    # receptor that is present on NONE of the chains it reports (a hallucinated or
    # fusion-masked identity) -- NOT merely some reported chain carrying a different
    # gene. When the AI's receptor IS on one of its chains the identity is confirmed.
    #
    # A differing partner chain is then treated as a co-protomer. Its identity is
    # RECORDED in the oligomer analysis (all_gpcr_chains) for the curator, but is not
    # necessarily alerted: a partner the AI did NOT report raises MISSED_PROTOMER,
    # while one the AI folded under its own (multi-chain) chain_id does not. So if the
    # AI over-claims its chain_id across a genuinely wrong partner, that partner is
    # accepted here unflagged -- a deliberate trade-off, far better than the old
    # all-or-nothing clash that destructively blocked legitimate heterodimers. (A
    # low-severity "heteromer carries an unconfirmed second receptor" note is a
    # follow-up for the AI-gated phase, once real model chain_id behaviour is known.)
    matched_chains = [c for c, s in chains_with_slugs.items() if any(u in s for u in ai_uniprots)]

    if not chains_with_slugs:
        # No chain carried any slug at all -> nothing to validate against. Two
        # distinct causes: a chain carrying a UniProt accession that simply has no
        # resolved GPCRdb slug is a recoverable mapping gap (RECEPTOR_RCSB_UNMAPPED),
        # whereas a chain the source has no UniProt reference for at all stays
        # RECEPTOR_NO_API_DATA. The unmapped case must NOT silently assert a
        # confident absence, so it raises a warning that gates one-click accept.
        if unmapped_accession_chains:
            receptor_info["validation_status"] = VALIDATION_RECEPTOR_RCSB_UNMAPPED
            receptor_info["api_reality"] = all_slugs
            warnings.append(
                f"RECEPTOR_RCSB_UNMAPPED at 'receptor_info': chain(s) "
                f"{', '.join(sorted(unmapped_accession_chains))} carry a UniProt "
                f"accession with no resolved GPCRdb slug; the receptor identity "
                f"cannot be confirmed (upstream mapping gap). Confirm manually."
            )
        else:
            receptor_info["validation_status"] = VALIDATION_RECEPTOR_NO_API_DATA
            receptor_info["api_reality"] = all_slugs
    elif matched_chains:
        receptor_info["validation_status"] = VALIDATION_RECEPTOR_MATCH
        receptor_info["api_reality"] = all_slugs
    else:
        receptor_info["validation_status"] = VALIDATION_UNIPROT_CLASH
        receptor_info["api_reality"] = all_slugs
        clash_detail = ", ".join(
            f"Chain {c} -> {chains_with_slugs[c]}" for c in sorted(chains_with_slugs)
        )
        warnings.append(
            f"UNIPROT_CLASH at 'receptor_info': '{ai_uniprot}' is not present on any "
            f"reported chain; API reality: {clash_detail}."
        )

    return warnings
