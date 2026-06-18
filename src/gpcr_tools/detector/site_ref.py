"""Pre-annotation ligand binding-site EVIDENCE (site_ref), from structure geometry.

For every ligand the model will annotate, this computes the objective geometric
FACTS about where each modelled copy sits -- contact residues (gemmi) -> UniProt
positions (RCSB alignment) -> GPCRdb generic numbers + segments (shipped table),
plus enclosure, lipid-vs-pocket facing, and membrane depth -- and routes them into
the prompt as evidence. It deliberately does NOT compute a site verdict: the model
infers ``site_ref`` from these facts plus the paper. The old deterministic
classifier was retired -- a confidently-wrong label dragged the model off the
right answer; a missing fact is better than a wrong conclusion.

Needs the coordinate file and the RCSB alignment, so it runs only when API checks
are enabled. Sparse or unmatched contacts yield no facts for that copy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gemmi

from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    LIGAND_EXCLUDE_LIST,
    LOCUS_LIGANDS,
    MEMBRANE_INTRACELLULAR_ANCHOR_GENERIC,
    MEMBRANE_INTRACELLULAR_SEGMENTS,
    MEMBRANE_MIN_ORIENT_LANDMARKS,
    ORTHOSTERIC_CORE_GENERIC,
    SITE_REF_MIN_MAPPED_CONTACTS,
)
from gpcr_tools.detector.coupling import galpha_auth_chains
from gpcr_tools.detector.signals import SEVERITY_ADVISORY, SIGNAL_SITE_REF, DetectSignal
from gpcr_tools.validator.api_clients import fetch_polymer_alignment
from gpcr_tools.validator.generic_numbering import (
    load_numbering_table,
    map_contacts,
    map_uniprot_position,
)
from gpcr_tools.validator.geometry import (
    centroid,
    is_protein_atom,
    ligand_contact_residues,
    load_structure,
)
from gpcr_tools.validator.membrane import (
    MembraneFrame,
    intracellular_side_sign,
    ligand_facing_fractions,
    ligand_membrane_depth,
    membrane_frame,
)
from gpcr_tools.validator.oligomer import build_nonpolymer_instance_index, is_gpcr_slug

logger = logging.getLogger(__name__)


def _gpcr_chain_accessions(enriched_entry: dict[str, Any]) -> dict[str, str]:
    """Map each GPCR receptor author chain to its UniProt accession (None-safe).

    ``is_gpcr_slug`` is a permissive denylist, so a crystallization fusion partner
    (rubredoxin, flavodoxin, ...) on the same entity can also pass it. Among the
    candidates, prefer the accession that is actually in the numbering table (the
    real receptor) over a fusion partner that merely survives the denylist.
    """
    table = load_numbering_table()
    chains: dict[str, str] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        candidates: list[str] = [
            u["rcsb_id"]
            for u in entity.get("uniprots") or []
            if isinstance(u, dict)
            and is_gpcr_slug(u.get("gpcrdb_entry_name_slug") or "")
            and u.get("rcsb_id")
        ]
        if not candidates:
            continue
        accession = next((acc for acc in candidates if acc in table), candidates[0])
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth:
                chains[auth] = accession
    return chains


def _annotated_ligands(enriched_entry: dict[str, Any]) -> set[str]:
    """Component ids the model annotates: present non-polymers minus the stripped
    buffers (incidental-candidate molecules are un-stripped, so kept). Broad on purpose -- a
    site fact helps every real ligand, not only the studied one."""
    present = set(build_nonpolymer_instance_index(enriched_entry).keys())
    return present - (LIGAND_EXCLUDE_LIST - INCIDENTAL_CANDIDATES)


def _copy_evidence(
    contacts: list[tuple[str, int, str]],
    chain_accessions: dict[str, str],
    alignment: dict[str, dict[str, list[tuple[int, int, int]]]],
) -> dict[str, Any] | None:
    """Objective contact facts for one ligand copy, or ``None`` if too sparse.

    Maps the copy's receptor contacts to GPCRdb generic numbers + segments and
    counts how many fall in the canonical orthosteric core (a FACT, not a verdict).
    """
    by_chain: dict[str, list[tuple[int, str]]] = {}
    for auth_chain, label_seq, amino_acid in contacts:
        by_chain.setdefault(auth_chain, []).append((label_seq, amino_acid))
    if not by_chain:
        return None
    generic_numbers: set[str] = set()
    segments: set[str] = set()
    mapped = 0
    for auth_chain, chain_contacts in by_chain.items():
        accession = chain_accessions.get(auth_chain)
        regions = (alignment.get(auth_chain) or {}).get(accession) if accession else None
        if not accession or not regions:
            continue
        g, s, m, _ = map_contacts(chain_contacts, regions, accession)
        generic_numbers |= g
        segments |= s
        mapped += m
    if mapped < SITE_REF_MIN_MAPPED_CONTACTS:
        return None
    return {
        "generic_numbers": sorted(generic_numbers),
        "segments": sorted(segments),
        "core_hits": len(generic_numbers & ORTHOSTERIC_CORE_GENERIC),
        "mapped": mapped,
    }


def _build_signal(comp_id: str, copies: list[dict[str, Any]]) -> DetectSignal:
    summary = (
        f"{comp_id}: geometry facts for {len(copies)} modelled "
        f"cop{'y' if len(copies) == 1 else 'ies'} (infer the binding site)."
    )
    return DetectSignal(
        kind=SIGNAL_SITE_REF,
        target_ref=LOCUS_LIGANDS,
        summary=summary,
        payload={"comp_id": comp_id, "copies": copies},
        severity=SEVERITY_ADVISORY,
    )


def _intracellular_landmark_centroid(
    structure: gemmi.Structure,
    chain_accessions: dict[str, str],
    alignment: dict[str, dict[str, list[tuple[int, int, int]]]],
) -> gemmi.Position | None:
    """Centroid of the receptor cytoplasmic-face landmark Cα, or ``None``.

    Uses the receptor's own 7TM backbone via the shipped generic numbering: a
    residue is a landmark if its segment is on the cytoplasmic face (H8 / ICL1-3)
    or it carries a canonical intracellular-anchor generic number (DRY 3x50,
    NPxxY 7x49-7x53). Each residue's entity SEQRES index maps to a UniProt
    position (RCSB alignment) -> numbering entry, with the same amino-acid
    identity gate as the contact mapping so a mutated / mis-aligned residue is
    skipped. Returns ``None`` if fewer than ``MEMBRANE_MIN_ORIENT_LANDMARKS``
    landmark Cα are located -- an honest abstain rather than a noisy orientation.
    This is universal: it needs only the receptor, so it works for apo /
    no-G-protein structures where a G-alpha reference is unavailable.
    """
    table = load_numbering_table()
    model = structure[0]
    landmark_atoms: list[gemmi.Atom] = []
    for chain in model:
        accession = chain_accessions.get(chain.name)
        if accession is None:
            continue
        regions = (alignment.get(chain.name) or {}).get(accession)
        residues = (table.get(accession) or {}).get("r") if isinstance(table, dict) else None
        if not regions or not isinstance(residues, dict):
            continue
        for residue in chain:
            if not is_protein_atom(residue):
                continue
            label_seq = residue.label_seq
            if label_seq is None:
                continue
            uniprot_pos = map_uniprot_position(label_seq, regions)
            if uniprot_pos is None:
                continue
            entry = residues.get(str(uniprot_pos))
            if not entry:
                continue
            generic_number, segment, reference_aa = entry[0], entry[1], entry[2]
            info = gemmi.find_tabulated_residue(residue.name)
            amino_acid = info.one_letter_code.upper() if info else "X"
            if amino_acid != reference_aa:
                continue  # amino-acid identity gate (mutation / mis-alignment)
            is_landmark = (
                segment in MEMBRANE_INTRACELLULAR_SEGMENTS
                or generic_number in MEMBRANE_INTRACELLULAR_ANCHOR_GENERIC
            )
            if not is_landmark:
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is not None:
                landmark_atoms.append(ca)
    if len(landmark_atoms) < MEMBRANE_MIN_ORIENT_LANDMARKS:
        return None
    return centroid(landmark_atoms)


def _galpha_centroid(structure: gemmi.Structure, galpha_chains: set[str]) -> gemmi.Position | None:
    """Centroid of the G-alpha chains' protein Cα (always on the cytoplasmic side),
    or ``None`` if no G-alpha Cα are present. A confirming reference only."""
    if not galpha_chains:
        return None
    model = structure[0]
    cas: list[gemmi.Atom] = []
    for chain in model:
        if chain.name not in galpha_chains:
            continue
        for residue in chain:
            if not is_protein_atom(residue):
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is not None:
                cas.append(ca)
    return centroid(cas) if cas else None


def _resolve_orientation(
    structure: gemmi.Structure,
    frame: MembraneFrame,
    chain_accessions: dict[str, str],
    alignment: dict[str, dict[str, list[tuple[int, int, int]]]],
    galpha_chains: set[str],
) -> tuple[int | None, str | None]:
    """Determine which side of the bilayer is intracellular.

    Returns ``(ic_sign, note)`` where ``ic_sign`` is +1/-1 so a copy's signed
    depth times ``ic_sign`` is positive on the intracellular side (``None`` when
    the structure cannot be oriented -> honest abstain). The PRIMARY reference is
    the receptor's own cytoplasmic-face landmarks (works without a G protein); a
    present G-alpha is only a confirming cross-check, and a disagreement is
    recorded as the soft *note* (not used to override the landmark call).
    """
    landmark_centroid = _intracellular_landmark_centroid(structure, chain_accessions, alignment)
    galpha_centroid = _galpha_centroid(structure, galpha_chains)

    landmark_sign = (
        intracellular_side_sign(
            frame, (landmark_centroid.x, landmark_centroid.y, landmark_centroid.z)
        )
        if landmark_centroid is not None
        else None
    )
    galpha_sign = (
        intracellular_side_sign(frame, (galpha_centroid.x, galpha_centroid.y, galpha_centroid.z))
        if galpha_centroid is not None
        else None
    )

    if landmark_sign is not None:
        note = None
        if galpha_sign is not None and galpha_sign != landmark_sign:
            note = (
                "the G-protein position disagrees with the receptor intracellular "
                "landmarks about which side is cytoplasmic"
            )
        return landmark_sign, note
    # Degrade: no usable landmarks but a G-alpha is present -> orient by it.
    if galpha_sign is not None:
        return galpha_sign, None
    return None, None  # neither reference -> stay unoriented


def _membrane_side(depth: float, in_band: bool, ic_sign: int) -> str:
    """Qualitative side fact for a copy from its signed depth + band + orientation.

    A copy inside the bilayer band is reported as ``mid-membrane``; otherwise the
    oriented depth places it on the intracellular or extracellular side.
    """
    if in_band:
        return "mid-membrane"
    return "on the intracellular side" if depth * ic_sign > 0 else "on the extracellular side"


def detect_site_refs(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache_dir: Path,
) -> list[DetectSignal]:
    """One advisory signal per annotated ligand carrying per-copy geometry facts.

    Each copy contributes its contact generic numbers + segments + canonical-core
    count + enclosure + lipid-vs-pocket facing + membrane depth. The model reads
    these facts (plus the paper) to assign ``site_ref`` itself. Missing coordinates
    / alignment, or only sparse/unmatched contacts, yield no facts rather than a
    guess. A ligand modelled at distinct sites simply shows distinct per-copy facts;
    the model decides whether to emit one entry per site.
    """
    chain_accessions = _gpcr_chain_accessions(enriched_entry)
    if not chain_accessions:
        return []
    comp_ids = _annotated_ligands(enriched_entry)
    if not comp_ids:
        return []
    structure = load_structure(pdb_id, cache_dir)
    if structure is None:
        return []
    alignment = fetch_polymer_alignment(pdb_id)
    if not alignment:
        return []

    receptor_chains = set(chain_accessions)
    frame = membrane_frame(structure)
    # model + per-copy atom lists are only needed for the membrane-depth fact, so
    # they are resolved only when a frame was fitted.
    model = structure[0] if frame is not None else None
    # Orient the (sign-arbitrary) membrane normal so a copy's signed depth gains a
    # physical "which side" meaning -- primary reference is the receptor's own
    # cytoplasmic-face landmarks (works for apo / no-G-protein structures), with a
    # present G-alpha only as a confirming cross-check. None -> stay unoriented.
    ic_sign: int | None = None
    orientation_note: str | None = None
    if frame is not None:
        ic_sign, orientation_note = _resolve_orientation(
            structure,
            frame,
            chain_accessions,
            alignment,
            galpha_auth_chains(enriched_entry),
        )
        if orientation_note is not None:
            logger.debug("%s membrane orientation: %s", pdb_id, orientation_note)
    signals: list[DetectSignal] = []
    for comp_id in sorted(comp_ids):
        contact_copies = ligand_contact_residues(structure, comp_id, receptor_chains)
        if not contact_copies:
            continue
        # facing + per-copy atom lists are in model order, the same order as
        # ligand_contact_residues, so they align by index.
        facings = (
            ligand_facing_fractions(structure, comp_id, frame)
            if frame is not None
            else [None] * len(contact_copies)
        )
        atom_lists = (
            [list(res) for chain in model for res in chain if res.name == comp_id]
            if model is not None
            else []
        )
        copies: list[dict[str, Any]] = []
        for i, (burial, contacts) in enumerate(contact_copies):
            evidence = _copy_evidence(contacts, chain_accessions, alignment)
            if evidence is None:
                continue
            evidence["enclosure"] = round(burial, 2)
            evidence["facing"] = facings[i] if i < len(facings) else None
            if frame is not None and i < len(atom_lists):
                depth_band = ligand_membrane_depth(frame, atom_lists[i])
                if depth_band is not None:
                    depth, in_band = depth_band
                    evidence["depth"], evidence["in_band"] = depth, in_band
                    # Keep the signed depth number unchanged; add the oriented side
                    # only when the structure could be oriented (else honest abstain).
                    if ic_sign is not None:
                        evidence["side"] = _membrane_side(depth, in_band, ic_sign)
                        if orientation_note is not None:
                            evidence["orientation_note"] = orientation_note
            copies.append(evidence)
        if copies:
            signals.append(_build_signal(comp_id, copies))
    return signals
