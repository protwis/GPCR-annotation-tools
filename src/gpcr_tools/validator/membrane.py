"""Membrane spatial frame from coordinates alone (gemmi, no AI, no GPCRdb).

Fits an approximate lipid-bilayer frame for a membrane protein by an ANVIL-style
hydrophobic-slab search: over many candidate membrane normals (golden-spiral
sphere points) and slab thicknesses, find the slab that best encloses exposed
HYDROPHOBIC residues while excluding hydrophilic ones. From that frame we derive,
per ligand copy, its signed depth relative to the bilayer mid-plane and whether it
sits within the membrane band.

Pure geometry, offline, no GPCRdb dependency. The exposure proxy + objective are
deliberately simple and want calibration against OPM/PPM on a labelled GPCR set
before the absolute numbers are trusted; the *direction* (membrane normal) is the
robust output. The fitted half-thickness is only approximate -- the simplified
objective plateaus once the slab spans the hydrophobic belt -- so
``ligand_membrane_depth`` reports the signed depth as the primary value and treats
the membrane band as a soft margin. Note: the normal's sign is arbitrary (the slab
is symmetric), so this
frame gives depth + in/out-of-band, NOT which side is extracellular vs
intracellular — orienting the sides needs a separate reference and is left to the
caller. Failures (too few residues, degenerate geometry) return ``None`` so a
detector never breaks on an odd structure.

The membrane normal is computed from the whole protein; for a receptor-plus-
transducer complex the soluble partner's surface is mostly hydrophilic and does
not pull the slab off the bilayer belt (a calibration note, not a correctness
claim).
"""

from __future__ import annotations

import bisect
import logging
import math
from collections import Counter
from dataclasses import dataclass

import gemmi

from gpcr_tools.config import (
    GEOMETRY_CONTACT_RADIUS,
    GEOMETRY_NEIGHBOR_SEARCH_RADIUS,
    MEMBRANE_BAND_MARGIN,
    MEMBRANE_EXPOSURE_CA_RADIUS,
    MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA,
    MEMBRANE_FACING_DEADZONE_COS,
    MEMBRANE_HYDROPHOBIC_RESIDUES,
    MEMBRANE_MIN_TM_CA,
    MEMBRANE_NORMAL_CANDIDATES,
    MEMBRANE_OFFSET_STEP,
    MEMBRANE_THICKNESS_MAX,
    MEMBRANE_THICKNESS_MIN,
    MEMBRANE_THICKNESS_STEP,
)

# Reused geometry helpers (golden-spiral directions; polymer-residue test).
from gpcr_tools.validator.geometry import fibonacci_directions, is_protein_atom

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MembraneFrame:
    """A fitted bilayer frame. ``normal`` is a unit vector (arbitrary sign);
    ``center`` is the bilayer mid-plane's projection along ``normal``;
    ``half_thickness`` is half the hydrophobic slab thickness (Å)."""

    normal: tuple[float, float, float]
    center: float
    half_thickness: float


def _exposed_residue_projections(
    model: gemmi.Model, cell: gemmi.UnitCell
) -> list[tuple[gemmi.Position, bool]]:
    """Surface-exposed protein Cα as (position, is_hydrophobic).

    Exposure proxy (no SASA dependency): a Cα with fewer than
    ``MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA`` other Cα within
    ``MEMBRANE_EXPOSURE_CA_RADIUS`` is treated as surface-facing.
    """
    cas: list[tuple[gemmi.Atom, bool]] = []
    for chain in model:
        for residue in chain:
            if not is_protein_atom(residue):
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is not None:
                cas.append((ca, residue.name in MEMBRANE_HYDROPHOBIC_RESIDUES))
    if len(cas) < MEMBRANE_MIN_TM_CA:
        return []
    ns = gemmi.NeighborSearch(model, cell, MEMBRANE_EXPOSURE_CA_RADIUS).populate()
    exposed: list[tuple[gemmi.Position, bool]] = []
    for ca, hydrophobic in cas:
        neighbors = 0
        for mark in ns.find_atoms(ca.pos, "\0", radius=MEMBRANE_EXPOSURE_CA_RADIUS):
            cra = mark.to_cra(model)
            # Protein-gate: a calcium ion is also named "CA"; only protein
            # Cα count toward residue density.
            if (
                is_protein_atom(cra.residue)
                and cra.atom.name == "CA"
                and ca.pos.dist(cra.atom.pos) <= MEMBRANE_EXPOSURE_CA_RADIUS
            ):
                neighbors += 1
        if neighbors - 1 < MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA:  # minus self
            exposed.append((ca.pos, hydrophobic))
    return exposed


def membrane_frame(structure: gemmi.Structure) -> MembraneFrame | None:
    """Fit a bilayer frame to *structure*, or ``None`` if it cannot be resolved."""
    if len(structure) == 0:
        return None
    model = structure[0]
    exposed = _exposed_residue_projections(model, structure.cell)
    if len(exposed) < MEMBRANE_MIN_TM_CA:
        return None

    best: tuple[float, tuple[float, float, float], float, float] | None = None
    for normal in fibonacci_directions(MEMBRANE_NORMAL_CANDIDATES):
        projected = sorted(
            (p.x * normal.x + p.y * normal.y + p.z * normal.z, hydrophobic)
            for p, hydrophobic in exposed
        )
        coords = [c for c, _ in projected]
        # Prefix sum of hydrophobic count over the sorted projections, so the
        # hydrophobic/hydrophilic counts inside any [a, b] window are O(log n).
        prefix_hydrophobic = [0]
        for _, hydrophobic in projected:
            prefix_hydrophobic.append(prefix_hydrophobic[-1] + (1 if hydrophobic else 0))
        lo, hi = coords[0], coords[-1]
        thickness = MEMBRANE_THICKNESS_MIN
        while thickness <= MEMBRANE_THICKNESS_MAX:
            half = thickness / 2.0
            center = lo + half
            while center <= hi - half + 1e-9:
                left = bisect.bisect_left(coords, center - half)
                right = bisect.bisect_right(coords, center + half)
                hydrophobic_in = prefix_hydrophobic[right] - prefix_hydrophobic[left]
                hydrophilic_in = (right - left) - hydrophobic_in
                # ANVIL principle: the membrane slab maximises enclosed exposed
                # hydrophobics minus enclosed hydrophilics (a thickness-bounded,
                # simplified form of the published Q score).
                score = float(hydrophobic_in - hydrophilic_in)
                if best is None or score > best[0]:
                    best = (score, (normal.x, normal.y, normal.z), center, half)
                center += MEMBRANE_OFFSET_STEP
            thickness += MEMBRANE_THICKNESS_STEP

    # A non-positive best score means the exposed set has no hydrophobic belt
    # to enclose (e.g. a soluble fragment): return None rather than a noise fit.
    if best is None or best[0] <= 0:
        return None
    return MembraneFrame(normal=best[1], center=best[2], half_thickness=best[3])


def ligand_membrane_depth(
    frame: MembraneFrame, ligand_atoms: list[gemmi.Atom]
) -> tuple[float, bool] | None:
    """Signed depth of a ligand copy's centroid vs the bilayer mid-plane.

    Returns ``(signed_depth, in_band)`` (depth in Å along the membrane normal,
    sign arbitrary; ``in_band`` true within the hydrophobic slab plus
    ``MEMBRANE_BAND_MARGIN``), or ``None`` for an empty atom list.
    """
    if not ligand_atoms:
        return None
    n = len(ligand_atoms)
    cx = sum(a.pos.x for a in ligand_atoms) / n
    cy = sum(a.pos.y for a in ligand_atoms) / n
    cz = sum(a.pos.z for a in ligand_atoms) / n
    nx, ny, nz = frame.normal
    signed_depth = (cx * nx + cy * ny + cz * nz) - frame.center
    in_band = abs(signed_depth) <= frame.half_thickness + MEMBRANE_BAND_MARGIN
    return round(signed_depth, 1), in_band


def intracellular_side_sign(
    frame: MembraneFrame, reference_point: tuple[float, float, float]
) -> int | None:
    """Sign that makes a signed membrane depth POSITIVE on the intracellular side.

    The fitted frame leaves the normal's sign arbitrary. Given an intracellular
    reference point (e.g. the centroid of the receptor's cytoplasmic-face
    landmark residues, or the G-alpha centroid), this returns ``+1`` or ``-1``
    such that ``signed_depth * sign > 0`` means a copy sits on the intracellular
    side and ``< 0`` the extracellular side. Returns ``None`` when the reference
    lies in the bilayer mid-plane (no resolvable side). The slug / numbering logic
    that locates the reference stays in the caller; this is pure geometry.
    """
    rx, ry, rz = reference_point
    nx, ny, nz = frame.normal
    reference_depth = (rx * nx + ry * ny + rz * nz) - frame.center
    if abs(reference_depth) < 1e-6:
        return None
    return 1 if reference_depth > 0 else -1


def _in_plane(
    vx: float, vy: float, vz: float, n: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Component of a vector in the membrane plane (its normal component removed)."""
    d = vx * n[0] + vy * n[1] + vz * n[2]
    return (vx - d * n[0], vy - d * n[1], vz - d * n[2])


def _bundle_center(
    model: gemmi.Model, frame: MembraneFrame, chain_name: str
) -> tuple[float, float, float] | None:
    """Centroid of *chain_name*'s protein Cα within the bilayer band ≈ that chain's
    TM-bundle axis point, the radial origin for lipid-vs-pocket facing.

    Restricted to a single chain on purpose: averaging across all chains would put
    the origin between the two bundles of a dimer / receptor-plus-transducer
    complex and flip the pocket/lipid sign for off-centre residues.
    """
    nx, ny, nz = frame.normal
    sx = sy = sz = 0.0
    k = 0
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if not is_protein_atom(residue):
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is None:
                continue
            depth = (ca.pos.x * nx + ca.pos.y * ny + ca.pos.z * nz) - frame.center
            if abs(depth) <= frame.half_thickness:
                sx += ca.pos.x
                sy += ca.pos.y
                sz += ca.pos.z
                k += 1
    if k == 0:
        return None
    return (sx / k, sy / k, sz / k)


def ligand_facing_fractions(
    structure: gemmi.Structure, comp_id: str, frame: MembraneFrame
) -> list[float | None]:
    """Per modelled copy of *comp_id*: the fraction of the contacts on its primary
    (most-contacted) protein chain that are POCKET-facing rather than lipid-facing.

    Breaks the generic-number "face-blindness": a residue's number is the same
    whether the ligand touches its pocket-facing (inward) or lipid-facing
    (outward) side. Radial test in the membrane plane: relative to the TM-bundle
    axis, a ligand sitting further OUT than the contacted residue is lipid-facing,
    further IN is pocket-facing; near-tangential contacts (within the cosine
    dead-zone) are ambiguous and excluded. ``1.0`` = all clear contacts
    pocket-facing, ``0.0`` = all lipid-facing; ``None`` for a copy with no clear
    contacts, no resolvable bundle axis, or one sitting outside the membrane band
    (where a facing fraction would be meaningless). One value per copy, in model order.
    """
    if len(structure) == 0:
        return []
    model = structure[0]
    ns = gemmi.NeighborSearch(model, structure.cell, GEOMETRY_NEIGHBOR_SEARCH_RADIUS).populate()
    normal = frame.normal
    results: list[float | None] = []
    for chain in model:
        for residue in chain:
            if residue.name != comp_id:
                continue
            ligand_atoms = list(residue)
            if not ligand_atoms:
                results.append(None)
                continue
            n_at = len(ligand_atoms)
            lcx = sum(a.pos.x for a in ligand_atoms) / n_at
            lcy = sum(a.pos.y for a in ligand_atoms) / n_at
            lcz = sum(a.pos.z for a in ligand_atoms) / n_at
            # Facing is only meaningful inside the bilayer band; an off-bundle copy
            # (e.g. a nucleotide bound to a soluble partner, far from any TM bundle)
            # has no membrane face, so abstain rather than report a misleading
            # fraction from whichever residues it happens to brush.
            depth = (lcx * normal[0] + lcy * normal[1] + lcz * normal[2]) - frame.center
            if abs(depth) > frame.half_thickness + MEMBRANE_BAND_MARGIN:
                results.append(None)
                continue
            # Contacted protein residues -> (chain, Cα), one per residue (the
            # insertion code is kept so 100 and 100A stay distinct).
            contact_ca: dict[tuple[str, int, str], tuple[str, gemmi.Position]] = {}
            for atom in ligand_atoms:
                for mark in ns.find_atoms(atom.pos, "\0", radius=GEOMETRY_CONTACT_RADIUS):
                    cra = mark.to_cra(model)
                    if not is_protein_atom(cra.residue):
                        continue
                    res_num = cra.residue.seqid.num
                    if res_num is None or atom.pos.dist(cra.atom.pos) > GEOMETRY_CONTACT_RADIUS:
                        continue
                    key = (cra.chain.name, res_num, cra.residue.seqid.icode)
                    if key not in contact_ca:
                        ca = next((a for a in cra.residue if a.name == "CA"), None)
                        if ca is not None:
                            contact_ca[key] = (cra.chain.name, ca.pos)
            if not contact_ca:
                results.append(None)
                continue
            # Bundle axis = the ligand's primary (most-contacted) chain; judge only
            # that chain's contacts, so a dimer's other protomer (its own axis)
            # cannot flip the sign.
            primary = Counter(cn for cn, _ in contact_ca.values()).most_common(1)[0][0]
            center = _bundle_center(model, frame, primary)
            if center is None:
                results.append(None)
                continue
            pocket = lipid = 0
            for cn, ca_pos in contact_ca.values():
                if cn != primary:
                    continue
                vr = _in_plane(
                    ca_pos.x - center[0], ca_pos.y - center[1], ca_pos.z - center[2], normal
                )
                vl = _in_plane(lcx - ca_pos.x, lcy - ca_pos.y, lcz - ca_pos.z, normal)
                mr = math.sqrt(vr[0] ** 2 + vr[1] ** 2 + vr[2] ** 2)
                ml = math.sqrt(vl[0] ** 2 + vl[1] ** 2 + vl[2] ** 2)
                if mr < 1e-6 or ml < 1e-6:
                    continue
                cosang = (vr[0] * vl[0] + vr[1] * vl[1] + vr[2] * vl[2]) / (mr * ml)
                if cosang <= -MEMBRANE_FACING_DEADZONE_COS:
                    pocket += 1
                elif cosang >= MEMBRANE_FACING_DEADZONE_COS:
                    lipid += 1
            total = pocket + lipid
            results.append(round(pocket / total, 2) if total else None)
    return results
