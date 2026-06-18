"""Tests for the membrane-frame geometry foundation (gemmi, synthetic structures).

No network, no fixtures: a synthetic "membrane belt" (a hydrophobic slab of Cα
perpendicular to z, with hydrophilic caps above/below) is built in-memory so the
ANVIL-style fit can be checked deterministically.
"""

import math

import gemmi

from gpcr_tools.validator.geometry import is_protein_atom, ligand_interaction_counts
from gpcr_tools.validator.membrane import (
    MembraneFrame,
    intracellular_side_sign,
    ligand_facing_fractions,
    ligand_membrane_depth,
    membrane_frame,
)


def _atom(
    name: str, x: float, y: float, z: float, b_iso: float = 20.0, element: str = "C"
) -> gemmi.Atom:
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.b_iso = b_iso
    a.element = gemmi.Element(element)
    return a


def _residue(name: str, seqid: int, atoms: list[gemmi.Atom], het: str = "A") -> gemmi.Residue:
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(seqid, " ")
    res.het_flag = het
    for a in atoms:
        res.add_atom(a)
    return res


def _structure(residues: list[gemmi.Residue], chain_name: str = "A") -> gemmi.Structure:
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_name)
    for res in residues:
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def _multichain_structure(chains: dict[str, list[gemmi.Residue]]) -> gemmi.Structure:
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(400, 400, 400, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    for chain_name, residues in chains.items():
        chain = gemmi.Chain(chain_name)
        for res in residues:
            chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model)
    return st


def _membrane_belt_structure() -> gemmi.Structure:
    """Hydrophobic Cα slab in z in [-14, 14]; hydrophilic caps at z = +/-28."""
    residues: list[gemmi.Residue] = []
    for i in range(80):  # belt: hydrophobic (LEU), spread around a cylinder
        ang = i * 0.7
        z = -14.0 + 28.0 * (i / 79.0)
        residues.append(
            _residue("LEU", i + 1, [_atom("CA", 18 * math.cos(ang), 18 * math.sin(ang), z)])
        )
    for j in range(30):  # caps: hydrophilic (ARG)
        ang = j * 0.9
        z = 28.0 if j % 2 == 0 else -28.0
        residues.append(
            _residue("ARG", 200 + j, [_atom("CA", 6 * math.cos(ang), 6 * math.sin(ang), z)])
        )
    return _structure(residues)


class TestMembraneFrame:
    def test_recovers_belt_normal_and_band(self):
        frame = membrane_frame(_membrane_belt_structure())
        assert frame is not None
        # The membrane belt is perpendicular to z, so the normal aligns with z
        # (either sign), centred near the slab mid-plane (z=0), ~28 Å thick.
        assert abs(frame.normal[2]) > 0.8
        assert abs(frame.center) < 6.0
        # Half-thickness is approximate (objective plateaus once the slab spans the
        # belt); 20.0 is the theoretical max given MEMBRANE_THICKNESS_MAX=40.
        assert 10.0 < frame.half_thickness <= 20.0

    def test_too_few_residues_returns_none(self):
        residues = [_residue("LEU", i + 1, [_atom("CA", float(i), 0.0, 0.0)]) for i in range(10)]
        assert membrane_frame(_structure(residues)) is None

    def test_empty_structure_returns_none(self):
        assert membrane_frame(gemmi.Structure()) is None

    def test_all_hydrophilic_returns_none(self):
        # Enough residues to pass the count gate, but all hydrophilic (ARG): there is
        # no hydrophobic belt to enclose, so the best slab score is <= 0 -> None.
        residues = [
            _residue(
                "ARG",
                i + 1,
                [
                    _atom(
                        "CA",
                        18 * math.cos(i * 0.7),
                        18 * math.sin(i * 0.7),
                        -14.0 + 28.0 * (i / 79.0),
                    )
                ],
            )
            for i in range(80)
        ]
        assert membrane_frame(_structure(residues)) is None


class TestLigandMembraneDepth:
    _FRAME = MembraneFrame(normal=(0.0, 0.0, 1.0), center=0.0, half_thickness=14.0)

    def test_centre_in_band(self):
        depth, in_band = ligand_membrane_depth(self._FRAME, [_atom("C1", 0.0, 0.0, 0.0)])
        assert depth == 0.0
        assert in_band

    def test_far_out_of_band(self):
        depth, in_band = ligand_membrane_depth(self._FRAME, [_atom("C1", 0.0, 0.0, 30.0)])
        assert depth == 30.0
        assert not in_band

    def test_edge_within_margin(self):
        # 15 Å is outside the 14 Å half-slab but within the +/-3 Å band margin.
        _, in_band = ligand_membrane_depth(self._FRAME, [_atom("C1", 0.0, 0.0, 15.0)])
        assert in_band

    def test_empty_atoms_returns_none(self):
        assert ligand_membrane_depth(self._FRAME, []) is None


class TestIntracellularSideSign:
    _FRAME = MembraneFrame(normal=(0.0, 0.0, 1.0), center=0.0, half_thickness=14.0)

    def test_reference_below_midplane_gives_negative_sign(self):
        # An intracellular reference at z=-30 (below the mid-plane) makes the
        # negative-depth side intracellular -> sign -1.
        assert intracellular_side_sign(self._FRAME, (0.0, 0.0, -30.0)) == -1

    def test_reference_above_midplane_gives_positive_sign(self):
        assert intracellular_side_sign(self._FRAME, (0.0, 0.0, 30.0)) == 1

    def test_reference_on_midplane_is_unresolvable(self):
        # A reference exactly in the mid-plane has no side -> None (abstain).
        assert intracellular_side_sign(self._FRAME, (0.0, 0.0, 0.0)) is None

    def test_sign_follows_the_normal_direction(self):
        # With the normal flipped (sign-arbitrary frame), the same reference point
        # yields the opposite sign -- the helper resolves the side, not the depth.
        flipped = MembraneFrame(normal=(0.0, 0.0, -1.0), center=0.0, half_thickness=14.0)
        assert intracellular_side_sign(flipped, (0.0, 0.0, -30.0)) == 1


class TestLigandFacing:
    _FRAME = MembraneFrame(normal=(0.0, 0.0, 1.0), center=0.0, half_thickness=14.0)

    def _ring_with_ligand(self, lig_x: float) -> gemmi.Structure:
        # 12 protein Cα on a radius-10 ring in the bilayer band (centroid = origin);
        # one Cα sits at (10, 0, 0). A one-atom ligand on the x-axis contacts only
        # that residue (the others are > 4.5 Å away).
        residues: list[gemmi.Residue] = []
        for i in range(12):
            ang = i * (math.pi / 6.0)
            residues.append(
                _residue("LEU", i + 1, [_atom("CA", 10 * math.cos(ang), 10 * math.sin(ang), 0.0)])
            )
        residues.append(_residue("LIG", 100, [_atom("C1", lig_x, 0.0, 0.0)], het="H"))
        return _structure(residues)

    def test_pocket_facing_when_inward(self):
        # Ligand at x=7 is inward of the residue at x=10 -> pocket-facing.
        assert ligand_facing_fractions(self._ring_with_ligand(7.0), "LIG", self._FRAME) == [1.0]

    def test_lipid_facing_when_outward(self):
        # Ligand at x=13 is outward of the residue at x=10 -> lipid-facing.
        assert ligand_facing_fractions(self._ring_with_ligand(13.0), "LIG", self._FRAME) == [0.0]

    def test_none_without_contacts(self):
        # Ligand far from the ring -> no clear contacts.
        assert ligand_facing_fractions(self._ring_with_ligand(40.0), "LIG", self._FRAME) == [None]

    @staticmethod
    def _ring(center_x: float) -> list[gemmi.Residue]:
        return [
            _residue(
                "LEU",
                i + 1,
                [
                    _atom(
                        "CA",
                        center_x + 10 * math.cos(i * math.pi / 6.0),
                        10 * math.sin(i * math.pi / 6.0),
                        0.0,
                    )
                ],
            )
            for i in range(12)
        ]

    def test_primary_chain_isolates_bundle_in_dimer(self):
        # Two bundles: chain A around origin, chain B 60 Å away. The ligand sits in
        # A's pocket (inward of A's residue at x=10). Using A's own axis -> pocket;
        # a composite A+B centroid (~x=30) would flip it to lipid.
        lig = _residue("LIG", 100, [_atom("C1", 7.0, 0.0, 0.0)], het="H")
        st = _multichain_structure({"A": [*self._ring(0.0), lig], "B": self._ring(60.0)})
        assert ligand_facing_fractions(st, "LIG", self._FRAME) == [1.0]

    def test_mixed_fraction(self):
        # Ligand at x=7 contacts the ring residue at x=10 (inward -> pocket) and an
        # extra inner residue at x=5 (outward -> lipid): 1 of 2 pocket-facing.
        residues = [*self._ring(0.0), _residue("LEU", 50, [_atom("CA", 5.0, 0.0, 0.0)])]
        residues.append(_residue("LIG", 100, [_atom("C1", 7.0, 0.0, 0.0)], het="H"))
        assert ligand_facing_fractions(_structure(residues), "LIG", self._FRAME) == [0.5]

    def test_none_when_primary_chain_not_in_band(self):
        # Contacted residue (and ligand) lie outside the bilayer band -> the chain
        # has no in-band Cα, so there is no bundle axis -> None.
        res = _residue("LEU", 1, [_atom("CA", 10.0, 0.0, 50.0)])
        lig = _residue("LIG", 2, [_atom("C1", 7.0, 0.0, 50.0)], het="H")
        assert ligand_facing_fractions(_structure([res, lig]), "LIG", self._FRAME) == [None]

    def test_none_when_ligand_out_of_band(self):
        # A copy whose centroid is far outside the bilayer band (e.g. a nucleotide on
        # a soluble partner) gets no facing fraction, even though it brushes a residue
        # whose chain has an in-band bundle axis -> abstain, not a misleading value.
        near = _residue("LEU", 50, [_atom("CA", 10.0, 0.0, 30.0)])
        lig = _residue("LIG", 100, [_atom("C1", 7.0, 0.0, 30.0)], het="H")
        st = _structure([*self._ring(0.0), near, lig])
        assert ligand_facing_fractions(st, "LIG", self._FRAME) == [None]


class TestLigandInteractionCounts:
    def test_polar_contact(self):
        # Ligand O 3.0 Å from a protein N -> one polar (H-bond proxy) residue.
        prot = _residue("ASN", 1, [_atom("ND2", 0.0, 0.0, 0.0, element="N")])
        lig = _residue("LIG", 2, [_atom("O1", 3.0, 0.0, 0.0, element="O")], het="H")
        assert ligand_interaction_counts(_structure([prot, lig]), "LIG") == [
            {"polar": 1, "metal": 0, "hydrophobic": 0}
        ]

    def test_hydrophobic_contact(self):
        # Ligand C 3.8 Å from a protein C -> one hydrophobic residue.
        prot = _residue("LEU", 1, [_atom("CD1", 0.0, 0.0, 0.0, element="C")])
        lig = _residue("LIG", 2, [_atom("C1", 3.8, 0.0, 0.0, element="C")], het="H")
        assert ligand_interaction_counts(_structure([prot, lig]), "LIG") == [
            {"polar": 0, "metal": 0, "hydrophobic": 1}
        ]

    def test_metal_coordination(self):
        # A metal-ion ligand 2.5 Å from a protein O -> one metal-coordination residue.
        prot = _residue("ASP", 1, [_atom("OD1", 0.0, 0.0, 0.0, element="O")])
        lig = _residue("ZN", 2, [_atom("ZN", 2.5, 0.0, 0.0, element="Zn")], het="H")
        assert ligand_interaction_counts(_structure([prot, lig]), "ZN") == [
            {"polar": 0, "metal": 1, "hydrophobic": 0}
        ]

    def test_two_polar_residues(self):
        # Ligand O within 3.5 Å of a polar atom (N or O) on each of two residues.
        r1 = _residue("ASN", 1, [_atom("ND2", 0.0, 0.0, 0.0, element="N")])
        r2 = _residue("SER", 2, [_atom("OG", 6.0, 0.0, 0.0, element="O")])
        lig = _residue("LIG", 3, [_atom("O1", 3.0, 0.0, 0.0, element="O")], het="H")
        assert ligand_interaction_counts(_structure([r1, r2, lig]), "LIG") == [
            {"polar": 2, "metal": 0, "hydrophobic": 0}
        ]

    def test_residue_counts_in_two_types(self):
        # One residue with both a polar (N) and a carbon atom near the ligand's
        # O and C respectively contributes to BOTH polar and hydrophobic.
        prot = _residue(
            "ASN",
            1,
            [_atom("ND2", 0.0, 0.0, 0.0, element="N"), _atom("CB", 0.0, 3.5, 0.0, element="C")],
        )
        lig = _residue(
            "LIG",
            2,
            [
                _atom("O1", 3.0, 0.0, 0.0, element="O"),
                _atom("C1", 0.0, 3.5 + 3.8, 0.0, element="C"),
            ],
            het="H",
        )
        assert ligand_interaction_counts(_structure([prot, lig]), "LIG") == [
            {"polar": 1, "metal": 0, "hydrophobic": 1}
        ]

    def test_nothing_in_range(self):
        prot = _residue("ASN", 1, [_atom("ND2", 0.0, 0.0, 0.0, element="N")])
        lig = _residue("LIG", 2, [_atom("O1", 5.0, 0.0, 0.0, element="O")], het="H")
        assert ligand_interaction_counts(_structure([prot, lig]), "LIG") == [
            {"polar": 0, "metal": 0, "hydrophobic": 0}
        ]

    def test_modified_residue_is_protein_only_after_setup_entities(self):
        # The counter (and every geometry function) gates protein-vs-HET on
        # is_protein_atom(), which for a HETATM-recorded residue depends on the
        # PRODUCTION entity typing: load_structure() calls setup_entities(), which
        # types a modified standard residue (e.g. selenomethionine, MSE) as Polymer
        # so it counts as protein. Before that call its entity is Unknown and it is
        # wrongly excluded. The other synthetic fixtures here skip setup_entities,
        # leaving residues Unknown; this pins the branch that call controls.
        def _backbone(i: int) -> list[gemmi.Atom]:
            x = i * 3.8
            return [
                _atom("N", x, 0.0, 0.0, element="N"),
                _atom("CA", x + 1.0, 0.0, 0.0),
                _atom("C", x + 2.0, 0.0, 0.0),
                _atom("O", x + 2.5, 0.0, 0.0, element="O"),
            ]

        chain = [
            _residue("MSE" if i == 1 else "ALA", i + 1, _backbone(i), het="H" if i == 1 else "A")
            for i in range(6)
        ]
        st = _structure(chain)
        assert not is_protein_atom(st[0][0][1])  # MSE Unknown before setup_entities -> excluded
        st.setup_entities()
        assert is_protein_atom(st[0][0][1])  # MSE typed Polymer after -> counted as protein
