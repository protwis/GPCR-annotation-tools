"""Pre-annotation G-protein coupling-protomer detector (structure geometry, no AI).

A Class C receptor is an obligate dimer, and only ONE protomer engages the G
protein. In a heterodimer that coupling protomer is often NOT the agonist-binding
one (GABA-B: GABBR1 binds the agonist in its VFT, GABBR2 is the only G-protein
coupler), so "which protomer is primary/active" cannot be read off the ligand. It
can be read off the coordinates: the G-alpha contacts exactly one receptor 7TM, so
the protomer with the G-alpha interface is the coupling (active) one.

This is geometry-only and advisory: it surfaces which protomer the G protein
actually engages in THIS structure (an objective fact, not a canonical assumption),
as evidence for the model and for primary-protomer selection. It is also
upstream-independent -- it never consults GPCRdb's own answer.

It reads the coordinate file, so it runs only when API checks are enabled (it needs
the network to fetch the structure). It stays silent unless the structure has a
G-alpha and at least two receptor protomer chains to tell apart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    GEOMETRY_COUPLING_DECISIVE_RATIO,
    GEOMETRY_COUPLING_MIN_CONTACTS,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_COUPLING_PROTOMER,
    DetectSignal,
)
from gpcr_tools.validator.geometry import load_structure, receptor_gprotein_contacts
from gpcr_tools.validator.oligomer import is_gpcr_slug

logger = logging.getLogger(__name__)

_COUPLING_LOCUS = "receptor_info"


def _is_role_slug(slug: str) -> bool:
    """A receptor (GPCR) slug or a G-alpha (``gna*``) slug -- the role-relevant ones."""
    return is_gpcr_slug(slug) or slug.startswith("gna")


def _chain_slugs(enriched_entry: dict[str, Any]) -> dict[str, str]:
    """auth_chain -> its GPCRdb slug, from polymer_entity_instances (None-safe).

    A fusion construct lists several slugs for one entity; prefer the role-relevant
    one (the receptor or the G-alpha) over a fusion partner that happens to be listed
    first, so a chain is labelled by what it actually is.
    """
    out: dict[str, str] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        slugs: list[str] = []
        for u in entity.get("uniprots") or []:
            if not isinstance(u, dict):
                continue
            s = u.get("gpcrdb_entry_name_slug")
            if isinstance(s, str) and s:
                slugs.append(s)
        slug = next((s for s in slugs if _is_role_slug(s)), slugs[0] if slugs else None)
        if not slug:
            continue
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            cid = inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}
            auth = cid.get("auth_asym_id")
            if auth:
                out[auth] = slug
    return out


def galpha_auth_chains(enriched_entry: dict[str, Any]) -> set[str]:
    """Author chain ids carrying a G-alpha (``gna*``) slug (None-safe, may be empty).

    Shared definition of "which chains are the G-alpha" so the coupling detector,
    membrane-side orientation, and any future transducer check all agree on one
    rule instead of each re-deriving the ``gna*`` prefix test.
    """
    return {c for c, s in _chain_slugs(enriched_entry).items() if s.startswith("gna")}


def detect_coupling_protomer(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache_dir: Path,
) -> list[DetectSignal]:
    """One advisory signal naming the G-protein-coupling receptor protomer.

    Fires only when the structure has a G-alpha chain and >= 2 receptor protomer
    chains (a dimer to disambiguate) and the geometry resolves a single coupling
    protomer decisively -- one chain carries a real G-alpha interface and the
    runner-up carries far fewer contacts. An apo / inactive structure (no G-alpha),
    a monomer, or an ambiguous interface yields no signal.
    """
    chain_slug = _chain_slugs(enriched_entry)
    galpha_chains = galpha_auth_chains(enriched_entry)
    receptor_chains = {c: s for c, s in chain_slug.items() if is_gpcr_slug(s)}
    # Need a G protein plus more than one protomer; with a single receptor chain
    # there is nothing to tell apart.
    if not galpha_chains or len(receptor_chains) < 2:
        return []

    structure = load_structure(pdb_id, cache_dir)
    if structure is None:
        return []

    counts = receptor_gprotein_contacts(structure, set(receptor_chains), galpha_chains)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_chain, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0

    # No real interface resolved, or two protomers both heavily contact the G-alpha
    # (cannot claim one) -> stay silent rather than guess.
    if top < GEOMETRY_COUPLING_MIN_CONTACTS:
        return []
    if second > GEOMETRY_COUPLING_DECISIVE_RATIO * top:
        return []

    slug = receptor_chains[top_chain]
    return [
        DetectSignal(
            kind=SIGNAL_COUPLING_PROTOMER,
            target_ref=_COUPLING_LOCUS,
            summary=(
                f"The G protein engages receptor chain {top_chain} ({slug}); that "
                f"protomer is the G-protein-coupling (active) protomer of the dimer. "
                f"In a heterodimer this need not be the agonist-binding protomer."
            ),
            payload={
                "coupling_chain": top_chain,
                "coupling_slug": slug,
                "galpha_chains": sorted(galpha_chains),
                "contacts": dict(counts),
            },
            severity=SEVERITY_ADVISORY,
        )
    ]
