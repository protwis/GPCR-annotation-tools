"""Tests for the generic-numbering table loader + contact mapping.

These exercise the real shipped artifact (the lookup must match GPCRdb), plus
the entity->UniProt->generic mapping and the amino-acid identity gate.
"""

from __future__ import annotations

from gpcr_tools.validator.generic_numbering import (
    load_numbering_table,
    map_contacts,
    map_uniprot_position,
    receptor_class,
)

ADRB2 = "P07550"  # class A; 3x32 = D113, 6x48 = W286


class TestLoadTable:
    def test_known_receptors_present(self) -> None:
        table = load_numbering_table()
        assert ADRB2 in table
        assert table[ADRB2]["c"] == "001"
        # 3x32 maps to D113 in adrb2.
        assert table[ADRB2]["r"]["113"] == ["3x32", "TM3", "D"]

    def test_receptor_class(self) -> None:
        assert receptor_class(ADRB2) == "001"
        assert receptor_class("Q9NYV8") == "009"  # TAS2R14, taste type 2
        assert receptor_class("NOT_A_REAL_ACCESSION") is None


class TestMapUniprotPosition:
    def test_identity_region(self) -> None:
        assert map_uniprot_position(113, [(1, 1, 500)]) == 113

    def test_offset_region(self) -> None:
        # entity index 30 with entity_beg=10 -> ref_beg=2 -> 2 + (30-10) = 22.
        assert map_uniprot_position(30, [(10, 2, 100)]) == 22

    def test_multi_region_fusion(self) -> None:
        # Two receptor regions split by a fusion insert: the gap maps to nothing.
        regions = [(12, 2, 200), (325, 219, 99)]
        assert map_uniprot_position(12, regions) == 2
        assert map_uniprot_position(325, regions) == 219
        assert map_uniprot_position(250, regions) is None  # inside the fusion insert


class TestMapContacts:
    def test_maps_and_gates(self) -> None:
        # Identity alignment; adrb2 113=D (3x32), 286=W (6x48). A wrong residue
        # at 113 must fail the amino-acid gate, not silently mis-map.
        contacts = [(113, "D"), (286, "W"), (113, "A")]
        gnums, segs, mapped, gate_fails = map_contacts(contacts, [(1, 1, 500)], ADRB2)
        assert gnums == {"3x32", "6x48"}
        assert segs == {"TM3", "TM6"}
        assert mapped == 2
        assert gate_fails == 1

    def test_unknown_accession_is_empty(self) -> None:
        gnums, segs, mapped, gate_fails = map_contacts([(1, "M")], [(1, 1, 9)], "NOPE")
        assert gnums == set() and segs == set() and mapped == 0 and gate_fails == 0
