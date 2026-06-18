"""Structure-geometry analysis for the detect stage (gemmi-based, no AI).

Downloads a PDB's coordinate file (mmCIF) once, caches it, and computes per-copy
geometry for small-molecule ligands:

* **burial** -- the fraction of evenly spread directions around a copy's centroid
  that have a protein atom within a narrow cone. A copy enclosed in a pocket is
  shielded from most directions; a copy on the membrane-facing surface (a
  scattered structural lipid) is shielded from a narrow cone only. This is the
  load-bearing separator between a functional pocket and a surface lipid.
* **pocket residues** -- the receptor residues lining the copy, used to tell two
  copies in different pockets apart from two copies in the same pocket.
* **partner contact** -- whether the copy also touches a non-receptor protein
  chain (a G-protein / peptide), a hint that it sits in the active-state pocket.

Network and parsing failures degrade to ``None`` / an empty result so the detect
stage never breaks on a missing or unreadable structure.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gemmi
import requests

from gpcr_tools.config import (
    API_MAX_RETRIES,
    GEOMETRY_BURIAL_CONE_DEG,
    GEOMETRY_BURIAL_SPHERE_DIRS,
    GEOMETRY_CONTACT_RADIUS,
    GEOMETRY_ENV_RADIUS,
    GEOMETRY_HBOND_HEAVY_DIST,
    GEOMETRY_HYDROPHOBIC_DIST,
    GEOMETRY_METAL_COORD_DIST,
    GEOMETRY_NEIGHBOR_SEARCH_RADIUS,
    RCSB_STRUCTURE_DOWNLOAD_URL,
    SLEEP_VALIDATION_RETRY,
    STRUCTURE_CACHE_SUBDIR,
    TIMEOUT_RCSB_STRUCTURE,
    VALIDATION_RETRY_BACKOFF_FACTOR,
)

logger = logging.getLogger(__name__)

_HETATM_FLAG = "H"  # gemmi residue.het_flag for HETATM (ligands, waters); ATOM is "A"


def is_protein_atom(residue: gemmi.Residue) -> bool:
    """True for a polymer (protein) residue atom; False for waters / ligands / HET.

    A modified standard residue (e.g. selenomethionine) is recorded as HETATM but
    is still part of the polymer, so the polymer entity type -- not het_flag alone
    -- decides. ``setup_entities`` assigns the entity type before this is called.
    """
    if residue.het_flag != _HETATM_FLAG:
        return True
    return residue.entity_type == gemmi.EntityType.Polymer


@dataclass(frozen=True)
class LigandCopyGeometry:
    """Geometry of one modelled copy of a small-molecule ligand."""

    auth_chain: str  # the copy's own author chain (gemmi chain name)
    seq_id: int
    burial: float  # angular coverage in [0, 1]; higher = more enclosed
    pocket_residues: frozenset[tuple[str, int]]  # (gpcr_chain, residue_number)
    contacts_partner: bool  # touches a non-GPCR protein chain (e.g. a G-protein)

    @property
    def n_pocket_residues(self) -> int:
        return len(self.pocket_residues)

    def primary_gpcr_chain(self) -> str | None:
        """The GPCR chain contributing the most pocket residues, or None."""
        if not self.pocket_residues:
            return None
        counts = Counter(chain for chain, _ in self.pocket_residues)
        return counts.most_common(1)[0][0]

    def residue_numbers_on(self, chain: str) -> frozenset[int]:
        """Residue numbers of this copy's pocket on *chain*."""
        return frozenset(num for c, num in self.pocket_residues if c == chain)


def fibonacci_directions(n: int) -> list[gemmi.Vec3]:
    """Return *n* roughly evenly spread unit directions on the sphere."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    directions: list[gemmi.Vec3] = []
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        directions.append(gemmi.Vec3(radius * math.cos(theta), y, radius * math.sin(theta)))
    return directions


# Computed once: the directions are fixed for a given configured count.
_SPHERE_DIRECTIONS = fibonacci_directions(GEOMETRY_BURIAL_SPHERE_DIRS)
_CONE_COS = math.cos(math.radians(GEOMETRY_BURIAL_CONE_DEG))


def fetch_structure(pdb_id: str, cache_dir: Path) -> Path | None:
    """Return the cached mmCIF path for *pdb_id*, downloading it once if absent.

    A released PDB's coordinates are immutable, so a cached file is never
    refetched. A transient failure (timeout / connection error / HTTP 5xx / 429)
    is retried, then logs a warning and returns ``None``; a definitive HTTP 404
    (structure genuinely absent) returns ``None`` immediately without retrying.
    """
    subdir = cache_dir / STRUCTURE_CACHE_SUBDIR
    path = subdir / f"{pdb_id.lower()}.cif.gz"
    if path.is_file():
        return path
    url = f"{RCSB_STRUCTURE_DOWNLOAD_URL}/{pdb_id.upper()}.cif.gz"
    resp = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT_RCSB_STRUCTURE)
            if resp.status_code == 200:
                break
            if resp.status_code == 404:
                # The structure is genuinely absent; re-fetching will not help. An
                # apo / missing structure is indistinguishable downstream, where
                # ``load_structure`` treats ``None`` uniformly -- the intended design.
                logger.warning("[geometry] %s: coordinates not found (HTTP 404)", pdb_id)
                return None
            # 5xx / 429 / other: service unavailable, not a verdict. Retry, then abstain.
            if attempt == API_MAX_RETRIES - 1:
                logger.warning(
                    "[geometry] %s: coordinates unavailable (HTTP %s)", pdb_id, resp.status_code
                )
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("[geometry] %s: could not download coordinates: %s", pdb_id, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY * VALIDATION_RETRY_BACKOFF_FACTOR**attempt)
    if resp is None or resp.status_code != 200:
        return None
    subdir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(subdir), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_structure(pdb_id: str, cache_dir: Path) -> gemmi.Structure | None:
    """Fetch (cached) and parse *pdb_id*'s coordinates, or ``None`` on failure."""
    path = fetch_structure(pdb_id, cache_dir)
    if path is None:
        return None
    try:
        structure = gemmi.read_structure(str(path))
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("[geometry] %s: could not parse coordinates: %s", pdb_id, exc)
        return None
    if len(structure) == 0:
        return None
    structure.setup_entities()
    return structure


def centroid(atoms: list[gemmi.Atom]) -> gemmi.Position:
    """Return the mean position of the given atoms (caller ensures non-empty)."""
    n = len(atoms)
    return gemmi.Position(
        sum(a.pos.x for a in atoms) / n,
        sum(a.pos.y for a in atoms) / n,
        sum(a.pos.z for a in atoms) / n,
    )


def _burial(centroid: gemmi.Position, env_positions: list[gemmi.Position]) -> float:
    """Fraction of sphere directions shielded by an environment atom within a cone."""
    units: list[gemmi.Vec3] = []
    for p in env_positions:
        vec = gemmi.Vec3(p.x - centroid.x, p.y - centroid.y, p.z - centroid.z)
        length = vec.length()
        if length < 1e-6:
            continue
        units.append(gemmi.Vec3(vec.x / length, vec.y / length, vec.z / length))
    if not units:
        return 0.0
    covered = 0
    for direction in _SPHERE_DIRECTIONS:
        if any(unit.dot(direction) >= _CONE_COS for unit in units):
            covered += 1
    return covered / len(_SPHERE_DIRECTIONS)


def _analyze_copy(
    model: gemmi.Model,
    neighbor_search: gemmi.NeighborSearch,
    chain_name: str,
    residue: gemmi.Residue,
    gpcr_chains: set[str],
) -> LigandCopyGeometry:
    # Burial measures enclosure by ANY protein atom (receptor or partner), as the
    # threshold was calibrated; a copy being a genuine RECEPTOR pocket is enforced
    # separately by the minimum GPCR pocket-residue count, so interface copies are
    # not mistaken for receptor pockets. Pocket residues are keyed by seqid.num;
    # insertion-code residues sharing a number are rare in receptor pockets and
    # would only undercount (a miss, never a false positive).
    ligand_atoms = list(residue)
    env: dict[tuple[str, int, str], gemmi.Position] = {}
    pocket: set[tuple[str, int]] = set()
    contacts_partner = False
    for atom in ligand_atoms:
        for mark in neighbor_search.find_atoms(
            atom.pos, "\0", radius=GEOMETRY_NEIGHBOR_SEARCH_RADIUS
        ):
            cra = mark.to_cra(model)
            if not is_protein_atom(cra.residue):  # skip ligands, waters, other HET
                continue
            res_num = cra.residue.seqid.num
            if res_num is None:
                continue
            distance = cra.atom.pos.dist(atom.pos)
            if distance <= GEOMETRY_ENV_RADIUS:
                env[(cra.chain.name, res_num, cra.atom.name)] = cra.atom.pos
            if distance <= GEOMETRY_CONTACT_RADIUS:
                if cra.chain.name in gpcr_chains:
                    pocket.add((cra.chain.name, res_num))
                else:
                    contacts_partner = True
    burial = _burial(centroid(ligand_atoms), list(env.values())) if ligand_atoms else 0.0
    copy_seq = residue.seqid.num
    return LigandCopyGeometry(
        auth_chain=chain_name,
        seq_id=copy_seq if copy_seq is not None else 0,
        burial=burial,
        pocket_residues=frozenset(pocket),
        contacts_partner=contacts_partner,
    )


def analyze_ligand_copies(
    structure: gemmi.Structure,
    comp_id: str,
    gpcr_chains: set[str],
) -> list[LigandCopyGeometry]:
    """Per-copy geometry for every modelled copy of *comp_id* in the structure.

    Copies are read directly from the coordinates (one residue named *comp_id* is
    one copy), which is robust to label-vs-author chain id mismatches.
    """
    model = structure[0]
    neighbor_search = gemmi.NeighborSearch(
        model, structure.cell, GEOMETRY_NEIGHBOR_SEARCH_RADIUS
    ).populate()
    copies: list[LigandCopyGeometry] = []
    for chain in model:
        for residue in chain:
            if residue.name == comp_id:
                copies.append(
                    _analyze_copy(model, neighbor_search, chain.name, residue, gpcr_chains)
                )
    return copies


def ligand_contact_residues(
    structure: gemmi.Structure,
    comp_id: str,
    receptor_chains: set[str],
) -> list[tuple[float, list[tuple[str, int, str]]]]:
    """Per-copy burial + receptor contacts of *comp_id*, for site classification.

    Returns one ``(burial, contacts)`` per modelled copy. *burial* is the angular
    coverage in [0, 1] (how enclosed the copy is, by any protein atom) -- it lets
    the caller require that a multi-site split come from deeply-buried copies (real
    pockets) rather than a structural lipid scattered across surface grooves.
    *contacts* are ``(receptor_auth_chain, label_seq, amino_acid_one_letter)`` for
    each receptor residue the copy touches; ``label_seq`` (the entity SEQRES index)
    is what the RCSB alignment maps to a UniProt position, and the amino acid lets
    the caller gate the mapping against the reference sequence.
    """
    model = structure[0]
    neighbor_search = gemmi.NeighborSearch(
        model, structure.cell, GEOMETRY_NEIGHBOR_SEARCH_RADIUS
    ).populate()
    copies: list[tuple[float, list[tuple[str, int, str]]]] = []
    for chain in model:
        for residue in chain:
            if residue.name != comp_id:
                continue
            ligand_atoms = list(residue)
            env: dict[tuple[str, int, str], gemmi.Position] = {}
            contacts: dict[tuple[str, int], str] = {}
            for atom in ligand_atoms:
                for mark in neighbor_search.find_atoms(
                    atom.pos, "\0", radius=GEOMETRY_NEIGHBOR_SEARCH_RADIUS
                ):
                    cra = mark.to_cra(model)
                    if not is_protein_atom(cra.residue):
                        continue
                    distance = cra.atom.pos.dist(atom.pos)
                    res_num = cra.residue.seqid.num
                    if distance <= GEOMETRY_ENV_RADIUS and res_num is not None:
                        env[(cra.chain.name, res_num, cra.atom.name)] = cra.atom.pos
                    if distance <= GEOMETRY_CONTACT_RADIUS and cra.chain.name in receptor_chains:
                        label_seq = cra.residue.label_seq
                        if label_seq is not None:
                            info = gemmi.find_tabulated_residue(cra.residue.name)
                            contacts[(cra.chain.name, label_seq)] = (
                                info.one_letter_code.upper() if info else "X"
                            )
            burial = _burial(centroid(ligand_atoms), list(env.values())) if ligand_atoms else 0.0
            copies.append(
                (burial, [(chain_name, ls, aa) for (chain_name, ls), aa in contacts.items()])
            )
    return copies


def receptor_gprotein_contacts(
    structure: gemmi.Structure,
    receptor_chains: set[str],
    galpha_chains: set[str],
) -> dict[str, int]:
    """Per receptor chain, the number of its residues that touch a G-alpha chain.

    A residue counts once if any of its atoms is within ``GEOMETRY_CONTACT_RADIUS``
    of any atom of a G-alpha chain. The G protein engages exactly one protomer of an
    obligate dimer, so the chain with the contacts is the coupling protomer; the
    partner protomer has none. Returns a count per requested receptor chain (0 for
    a chain that never touches the G-alpha).
    """
    model = structure[0]
    neighbor_search = gemmi.NeighborSearch(
        model, structure.cell, GEOMETRY_NEIGHBOR_SEARCH_RADIUS
    ).populate()
    counts: dict[str, int] = {c: 0 for c in receptor_chains}
    for chain in model:
        if chain.name not in receptor_chains:
            continue
        for residue in chain:
            if not is_protein_atom(residue):  # count receptor protein residues only
                continue
            touched = False
            for atom in residue:
                if touched:
                    break
                for mark in neighbor_search.find_atoms(
                    atom.pos, "\0", radius=GEOMETRY_NEIGHBOR_SEARCH_RADIUS
                ):
                    cra = mark.to_cra(model)
                    # Only a protein atom of a G-alpha chain counts -- not a bound
                    # nucleotide / ion that happens to sit on the G-alpha chain.
                    if (
                        cra.chain.name in galpha_chains
                        and is_protein_atom(cra.residue)
                        and atom.pos.dist(cra.atom.pos) <= GEOMETRY_CONTACT_RADIUS
                    ):
                        touched = True
                        break
            if touched:
                counts[chain.name] += 1
    return counts


_POLAR_ELEMENTS = frozenset({"N", "O", "S"})


def ligand_interaction_counts(structure: gemmi.Structure, comp_id: str) -> list[dict[str, int]]:
    """Per modelled copy of *comp_id*: a coarse, element-level count of the protein
    residues it contacts by interaction chemistry.

    Heavy-atom only (no hydrogens, as deposited): ``polar`` = residues with an
    N/O/S within ``GEOMETRY_HBOND_HEAVY_DIST`` of a ligand N/O/S (a hydrogen-bond
    proxy — distance only, no donor/acceptor or angle); ``metal`` = residues in a
    metal coordination (a metal on either side within ``GEOMETRY_METAL_COORD_DIST``
    of a coordinating N/O/S or metal on the other); ``hydrophobic`` = residues with
    a carbon within ``GEOMETRY_HYDROPHOBIC_DIST`` of a ligand carbon. Each count is
    DISTINCT protein residues; a residue may contribute to more than one type. This
    is a proxy, not full interaction typing (no aromatic/pi, salt-bridge, or
    directional hydrogen bonds — those need ligand chemistry/connectivity). A
    metal ligand atom counts only toward ``metal`` (never ``polar``), and a
    metal-near-carbon contact falls in no bucket. One dict per modelled copy, in
    model order.
    """
    if len(structure) == 0:
        return []
    model = structure[0]
    neighbor_search = gemmi.NeighborSearch(
        model, structure.cell, GEOMETRY_NEIGHBOR_SEARCH_RADIUS
    ).populate()
    counts: list[dict[str, int]] = []
    for chain in model:
        for residue in chain:
            if residue.name != comp_id:
                continue
            polar: set[tuple[str, int, str]] = set()
            metal: set[tuple[str, int, str]] = set()
            hydrophobic: set[tuple[str, int, str]] = set()
            for atom in residue:
                lig_el = atom.element.name
                lig_metal = atom.element.is_metal
                lig_polar = lig_el in _POLAR_ELEMENTS
                for mark in neighbor_search.find_atoms(
                    atom.pos, "\0", radius=GEOMETRY_HYDROPHOBIC_DIST
                ):
                    cra = mark.to_cra(model)
                    if not is_protein_atom(cra.residue):
                        continue
                    res_num = cra.residue.seqid.num
                    if res_num is None:
                        continue
                    dist = atom.pos.dist(cra.atom.pos)
                    if dist > GEOMETRY_HYDROPHOBIC_DIST:
                        continue
                    p_el = cra.atom.element.name
                    p_metal = cra.atom.element.is_metal
                    key = (cra.chain.name, res_num, cra.residue.seqid.icode)
                    if lig_polar and p_el in _POLAR_ELEMENTS and dist <= GEOMETRY_HBOND_HEAVY_DIST:
                        polar.add(key)
                    # A metal on either side coordinating an N/O/S (or another
                    # metal). Protein-side metals are rare (HETATM, usually already
                    # excluded), so in practice this fires for a metal-ion ligand
                    # near a protein N/O/S.
                    if dist <= GEOMETRY_METAL_COORD_DIST and (
                        (lig_metal and (p_el in _POLAR_ELEMENTS or p_metal))
                        or (p_metal and lig_el in _POLAR_ELEMENTS)
                    ):
                        metal.add(key)
                    # Hydrophobic carbon-carbon; the outer query radius IS the
                    # hydrophobic cutoff, so no extra distance check is needed.
                    if lig_el == "C" and p_el == "C":
                        hydrophobic.add(key)
            counts.append(
                {"polar": len(polar), "metal": len(metal), "hydrophobic": len(hydrophobic)}
            )
    return counts
