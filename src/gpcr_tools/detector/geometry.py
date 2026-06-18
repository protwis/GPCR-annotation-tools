"""Pre-annotation dual-role ligand detector (structure geometry, no AI).

A ligand modelled in two distinct functional pockets on one receptor chain plays
more than one role (e.g. an orthosteric agonist that also sits in an allosteric
pocket). The annotation schema records one entry per ligand component, so without
a nudge the model can collapse both copies into a single role. This detector
finds such cases from the coordinates and routes per-copy evidence into the
prompt so the model emits a separate entry per site.

It is geometry-only and advisory: it surfaces a genuine two-pocket case as
evidence (accommodate + guide); it never assigns the per-site pharmacology
itself. A ligand modelled in two distinct, deeply buried pockets on one receptor
chain -- with the copy count capped so detergent floods do not qualify -- is the
validated discriminator (no reliance on the unreliable subject-of-investigation
flag; the geometry does the work).

This detector reads the coordinate file, so it runs only when API checks are
enabled (it needs the network to fetch the structure).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    GEOMETRY_BURIAL_MIN,
    GEOMETRY_DUAL_ROLE_MAX_COPIES,
    GEOMETRY_DUAL_ROLE_POCKET_JACCARD_MAX,
    GEOMETRY_MIN_POCKET_RESIDUES,
    INCIDENTAL_CANDIDATES,
    LIGAND_EXCLUDE_LIST,
    LOCUS_LIGANDS,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_DUAL_ROLE_LIGAND,
    DetectSignal,
)
from gpcr_tools.validator.geometry import (
    LigandCopyGeometry,
    analyze_ligand_copies,
    load_structure,
)
from gpcr_tools.validator.oligomer import is_gpcr_slug

logger = logging.getLogger(__name__)


def _gpcr_auth_chains(enriched_entry: dict[str, Any]) -> set[str]:
    """Author chain ids of every GPCR receptor polymer instance (None-safe)."""
    chains: set[str] = set()
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        is_gpcr = any(
            isinstance(u, dict) and is_gpcr_slug(u.get("gpcrdb_entry_name_slug") or "")
            for u in entity.get("uniprots") or []
        )
        if not is_gpcr:
            continue
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            cid = inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}
            auth = cid.get("auth_asym_id")
            if auth:
                chains.add(auth)
    return chains


def _candidate_comp_ids(enriched_entry: dict[str, Any]) -> set[str]:
    """Non-polymer component ids worth checking for a dual-role pocket (None-safe).

    Every present non-polymer minus the stripped buffers (incidental-candidate molecules are
    kept). Identity is NOT filtered by RCSB's subject-of-investigation flag --
    that flag is unreliable (it misses many real ligands), so the geometry gates
    (deep burial + a small copy count + distinct pockets) do the discrimination
    instead: a structural lipid scattered across shallow surface grooves fails the
    burial gate, and a detergent flood fails the copy-count cap.
    """
    comp_ids: set[str] = set()
    for entity in enriched_entry.get("nonpolymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        comp_id = (entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get(
            "nonpolymer_comp_id"
        )
        if comp_id:
            comp_ids.add(comp_id)
    return comp_ids - (LIGAND_EXCLUDE_LIST - INCIDENTAL_CANDIDATES)


def _cluster_pockets(group: list[LigandCopyGeometry], chain: str) -> list[list[LigandCopyGeometry]]:
    """Cluster buried copies on *chain* into distinct pockets by residue overlap.

    Two copies belong to the same pocket when their pocket-residue sets overlap
    (Jaccard >= the cap); copies below the cap occupy distinct pockets. So three
    copies where two share a site collapse to two pockets, not three -- the pocket
    count, not the copy count, is what the dual-role evidence reports.
    """
    clusters: list[dict[str, Any]] = []
    for copy in group:
        residues = copy.residue_numbers_on(chain)
        placed = False
        for cluster in clusters:
            union = cluster["residues"] | residues
            overlap = len(cluster["residues"] & residues) / len(union) if union else 0.0
            if overlap >= GEOMETRY_DUAL_ROLE_POCKET_JACCARD_MAX:
                cluster["residues"] |= residues
                cluster["copies"].append(copy)
                placed = True
                break
        if not placed:
            clusters.append({"residues": set(residues), "copies": [copy]})
    return [cluster["copies"] for cluster in clusters]


def _dual_role_pocket(
    copies: list[LigandCopyGeometry],
) -> tuple[str, list[LigandCopyGeometry]] | None:
    """Return the receptor chain and one representative copy per distinct pocket.

    Requires copies that are each deeply buried in a real pocket, on the same
    receptor chain, occupying at least two genuinely different pockets. The most
    buried copy represents each pocket, so the evidence reflects distinct sites.
    """
    buried = [
        c
        for c in copies
        if c.burial >= GEOMETRY_BURIAL_MIN and c.n_pocket_residues >= GEOMETRY_MIN_POCKET_RESIDUES
    ]
    if len(buried) < 2:
        return None
    by_chain: dict[str, list[LigandCopyGeometry]] = {}
    for copy in buried:
        chain = copy.primary_gpcr_chain()
        if chain is not None:
            by_chain.setdefault(chain, []).append(copy)
    for chain, group in by_chain.items():
        if len(group) < 2:
            continue
        pockets = [p for p in _cluster_pockets(group, chain) if p]
        if len(pockets) < 2:
            continue
        representatives = [max(pocket, key=lambda c: c.burial) for pocket in pockets]
        return chain, representatives
    return None


def _copy_payload(copy: LigandCopyGeometry, chain: str) -> dict[str, Any]:
    # Count and list are both the residues on this receptor chain, so the evidence
    # is internally consistent even for a copy that also grazes another chain.
    residues = sorted(copy.residue_numbers_on(chain))
    return {
        "chain": copy.auth_chain,
        "seq_id": copy.seq_id,
        "burial": round(copy.burial, 2),
        "n_pocket_residues": len(residues),
        "pocket_residues": residues,
        "contacts_partner": copy.contacts_partner,
    }


def _build_signal(
    comp_id: str, chain: str, representatives: list[LigandCopyGeometry]
) -> DetectSignal:
    copies_payload = [
        _copy_payload(copy, chain)
        for copy in sorted(representatives, key=lambda c: (c.auth_chain, c.seq_id))
    ]
    return DetectSignal(
        kind=SIGNAL_DUAL_ROLE_LIGAND,
        target_ref=LOCUS_LIGANDS,
        summary=(
            f"{comp_id} is modelled in {len(representatives)} distinct buried pockets on "
            f"receptor chain {chain}; it may play more than one role. Record one ligand "
            f"entry per site."
        ),
        payload={"comp_id": comp_id, "gpcr_chain": chain, "copies": copies_payload},
        severity=SEVERITY_ADVISORY,
    )


def detect_dual_role_ligands(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache_dir: Path,
) -> list[DetectSignal]:
    """One advisory signal per ligand bound at two distinct receptor pockets.

    A ligand qualifies on geometry alone -- a small number of copies, each deeply
    buried in a distinct pocket -- so detergent floods (too many copies) and
    surface lipids (too shallow) never qualify, with no reliance on RCSB's
    subject-of-investigation flag. The coordinate file is fetched (cached) on
    first use; a missing structure or no GPCR / candidate ligand yields no signal.
    """
    comp_ids = _candidate_comp_ids(enriched_entry)
    if not comp_ids:
        return []
    gpcr_chains = _gpcr_auth_chains(enriched_entry)
    if not gpcr_chains:
        return []
    structure = load_structure(pdb_id, cache_dir)
    if structure is None:
        return []

    signals: list[DetectSignal] = []
    for comp_id in sorted(comp_ids):
        copies = analyze_ligand_copies(structure, comp_id, gpcr_chains)
        # 2-3 copies: a dual-role drug appears 2-3x; a structural lipid appears 5-34x.
        if not 2 <= len(copies) <= GEOMETRY_DUAL_ROLE_MAX_COPIES:
            continue
        pocket = _dual_role_pocket(copies)
        if pocket is None:
            continue
        chain, representatives = pocket
        signals.append(_build_signal(comp_id, chain, representatives))
    return signals
