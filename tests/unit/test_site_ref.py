"""Tests for the site_ref geometry-evidence detector (detect_site_refs).

The deterministic ``classify_site`` rule was retired (the architecture flip): the
detector now emits per-copy geometric FACTS and the model infers the site. These
tests cover the enriched parsing and the per-copy evidence orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gemmi
import pytest

from gpcr_tools.config import INCIDENTAL_CANDIDATES
from gpcr_tools.detector import site_ref as sr
from gpcr_tools.validator.membrane import MembraneFrame


def _entry(comp: str = "LIG", slug: str = "t2r14_human", acc: str = "Q9NYV8") -> dict:
    return {
        "polymer_entities": [
            {
                "uniprots": [{"gpcrdb_entry_name_slug": slug, "rcsb_id": acc}],
                "polymer_entity_instances": [
                    {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "R"}}
                ],
            }
        ],
        "nonpolymer_entities": [
            {
                "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": comp},
                "nonpolymer_entity_instances": [
                    {"rcsb_nonpolymer_entity_instance_container_identifiers": {"asym_id": "B"}}
                ],
            }
        ],
    }


class TestEnrichedParsing:
    def test_gpcr_chain_accessions(self) -> None:
        assert sr._gpcr_chain_accessions(_entry()) == {"R": "Q9NYV8"}

    def test_non_gpcr_yields_no_chains(self) -> None:
        assert sr._gpcr_chain_accessions(_entry(slug="gnas2_human")) == {}

    def test_annotated_keeps_real_ligand(self) -> None:
        # Every real (non-buffer) ligand is annotated and gets facts (recall);
        # whether it is one or several sites is left to the model.
        assert sr._annotated_ligands(_entry("LIG")) == {"LIG"}

    def test_incidental_candidate_molecule_annotated(self) -> None:
        incidental_candidate = sorted(INCIDENTAL_CANDIDATES)[0]
        assert incidental_candidate in sr._annotated_ligands(_entry(incidental_candidate))

    def test_prefers_table_accession_over_fusion_partner(self) -> None:
        # Both UniProts pass the permissive slug denylist; the real receptor
        # (in the shipped table) must win over a crystallization fusion partner.
        entry = {
            "polymer_entities": [
                {
                    "uniprots": [
                        {"gpcrdb_entry_name_slug": "rubr_clopa", "rcsb_id": "P00268"},
                        {"gpcrdb_entry_name_slug": "adrb2_human", "rcsb_id": "P07550"},
                    ],
                    "polymer_entity_instances": [
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A"
                            }
                        }
                    ],
                }
            ]
        }
        assert sr._gpcr_chain_accessions(entry) == {"A": "P07550"}


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Stub the I/O so detect_site_refs exercises only its orchestration.

    Set the fixture list to ``(enclosure, evidence_or_None)`` per copy: the stub
    feeds each copy's enclosure through ligand_contact_residues and its evidence
    dict (or None for a too-sparse copy) through _copy_evidence. membrane_frame is
    stubbed to None so the facing/depth facts are skipped in these tests.
    """
    copies: list[tuple[float, dict | None]] = []
    monkeypatch.setattr(sr, "load_structure", lambda *a, **k: object())
    monkeypatch.setattr(
        sr, "fetch_polymer_alignment", lambda *a, **k: {"R": {"Q9NYV8": [(1, 1, 400)]}}
    )
    monkeypatch.setattr(sr, "membrane_frame", lambda *a, **k: None)
    monkeypatch.setattr(sr, "ligand_contact_residues", lambda *a, **k: list(copies))
    monkeypatch.setattr(sr, "_copy_evidence", lambda contacts, *a, **k: contacts)
    return copies


_ORTH = {
    "generic_numbers": ["3x33", "6x51"],
    "segments": ["TM3", "TM6"],
    "core_hits": 2,
    "mapped": 10,
}
_VEST = {"generic_numbers": ["45x52"], "segments": ["ECL2"], "core_hits": 0, "mapped": 8}


class TestDetectSiteRefs:
    def test_single_copy_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(0.92, dict(_ORTH))]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals) == 1
        copies = signals[0].payload["copies"]
        assert len(copies) == 1
        assert copies[0]["generic_numbers"] == ["3x33", "6x51"]
        assert copies[0]["core_hits"] == 2
        assert copies[0]["enclosure"] == 0.92  # the burial is recorded as an enclosure fact

    def test_multi_copy_facts_not_collapsed(self, stub_pipeline, tmp_path: Path) -> None:
        # Both copies' facts are emitted (distinct sites); the model decides whether
        # to emit one entry per site -- the detector no longer makes that call.
        stub_pipeline[:] = [(0.95, dict(_ORTH)), (0.9, dict(_VEST))]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals) == 1
        copies = signals[0].payload["copies"]
        assert len(copies) == 2
        assert {c["segments"][0] for c in copies} == {"TM3", "ECL2"}

    def test_shallow_copy_still_emitted_as_fact(self, stub_pipeline, tmp_path: Path) -> None:
        # A low-enclosure copy is no longer gated out; its enclosure is just a fact
        # (the model reads low enclosure + lipid-facing as a structural-lipid hint).
        stub_pipeline[:] = [(0.40, dict(_VEST))]
        signals = sr.detect_site_refs("X", _entry("CLR"), tmp_path)
        assert signals[0].payload["copies"][0]["enclosure"] == 0.40

    def test_sparse_copy_skipped(self, stub_pipeline, tmp_path: Path) -> None:
        # A copy with too few mapped contacts (_copy_evidence -> None) is dropped.
        stub_pipeline[:] = [(0.9, dict(_ORTH)), (0.5, None)]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals[0].payload["copies"]) == 1

    def test_all_sparse_emits_no_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(0.9, None), (0.8, None)]
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []

    def test_no_gpcr_chain_short_circuits(self, tmp_path: Path) -> None:
        assert sr.detect_site_refs("X", _entry(slug="gnas2_human"), tmp_path) == []

    def test_missing_structure_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sr, "load_structure", lambda *a, **k: None)
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []


_FRAME = MembraneFrame(normal=(0.0, 0.0, 1.0), center=0.0, half_thickness=14.0)


def _ca_chain(name: str, positions: list[tuple[float, float, float]]) -> gemmi.Chain:
    chain = gemmi.Chain(name)
    for i, (x, y, z) in enumerate(positions):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(i + 1, " ")
        res.het_flag = "A"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.pos = gemmi.Position(x, y, z)
        atom.element = gemmi.Element("C")
        res.add_atom(atom)
        chain.add_residue(res)
    return chain


def _structure(chains: list[gemmi.Chain]) -> gemmi.Structure:
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    for chain in chains:
        model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


class TestMembraneSide:
    """The qualitative side fact from a copy's oriented depth + band."""

    def test_in_band_is_mid_membrane(self) -> None:
        assert sr._membrane_side(1.0, True, -1) == "mid-membrane"

    def test_intracellular_side(self) -> None:
        # ic_sign=-1: a copy at depth -24 (below the mid-plane) is intracellular.
        assert sr._membrane_side(-24.0, False, -1) == "on the intracellular side"

    def test_extracellular_side(self) -> None:
        # ic_sign=-1: a copy at depth +24 (above) is extracellular.
        assert sr._membrane_side(24.0, False, -1) == "on the extracellular side"

    def test_sign_flips_the_side(self) -> None:
        # With ic_sign=+1, the same +24 depth becomes the intracellular side.
        assert sr._membrane_side(24.0, False, 1) == "on the intracellular side"


class TestGalphaCentroid:
    def test_averages_galpha_ca_only(self) -> None:
        # G-alpha chain G sits at z=-40 (cytoplasmic); receptor chain R is ignored.
        st = _structure(
            [
                _ca_chain("R", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
                _ca_chain("G", [(0.0, 0.0, -40.0), (2.0, 0.0, -40.0)]),
            ]
        )
        centroid = sr._galpha_centroid(st, {"G"})
        assert centroid is not None
        assert round(centroid.z, 1) == -40.0
        assert round(centroid.x, 1) == 1.0

    def test_no_galpha_chains_returns_none(self) -> None:
        st = _structure([_ca_chain("R", [(0.0, 0.0, 0.0)])])
        assert sr._galpha_centroid(st, set()) is None
        assert sr._galpha_centroid(st, {"G"}) is None  # named chain absent


class TestIntracellularLandmarkCentroid:
    def test_no_receptor_chains_returns_none(self) -> None:
        st = _structure([_ca_chain("R", [(0.0, 0.0, 0.0)])])
        # No chain_accessions -> no landmark residues located -> abstain.
        assert sr._intracellular_landmark_centroid(st, {}, {}) is None


class TestResolveOrientation:
    """Orientation degrade chain: receptor landmarks (primary) -> G-alpha
    (confirming) -> neither (abstain). G-alpha never overrides the landmarks."""

    def _patch_references(
        self,
        monkeypatch: pytest.MonkeyPatch,
        landmark: gemmi.Position | None,
        galpha: gemmi.Position | None,
    ) -> None:
        monkeypatch.setattr(sr, "_intracellular_landmark_centroid", lambda *a, **k: landmark)
        monkeypatch.setattr(sr, "_galpha_centroid", lambda *a, **k: galpha)

    def test_landmarks_orient_without_g_protein(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Apo / no-G-protein: the receptor's own landmarks at z=-30 orient it.
        self._patch_references(monkeypatch, gemmi.Position(0.0, 0.0, -30.0), None)
        sign, note = sr._resolve_orientation(object(), _FRAME, {}, {}, set())
        assert sign == -1
        assert note is None

    def test_g_protein_only_confirms_when_no_landmarks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No usable landmarks but a G-alpha present -> orient by the G-alpha.
        self._patch_references(monkeypatch, None, gemmi.Position(0.0, 0.0, 30.0))
        sign, note = sr._resolve_orientation(object(), _FRAME, {}, {}, {"G"})
        assert sign == 1
        assert note is None

    def test_g_protein_agreement_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Landmarks and G-alpha agree (both below) -> no note.
        self._patch_references(
            monkeypatch, gemmi.Position(0.0, 0.0, -30.0), gemmi.Position(0.0, 0.0, -25.0)
        )
        sign, note = sr._resolve_orientation(object(), _FRAME, {}, {}, {"G"})
        assert sign == -1
        assert note is None

    def test_g_protein_disagreement_records_soft_note_without_overriding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Landmarks say intracellular = below (sign -1); G-alpha is above (sign +1).
        # The landmark call wins; the disagreement is only a soft note.
        self._patch_references(
            monkeypatch, gemmi.Position(0.0, 0.0, -30.0), gemmi.Position(0.0, 0.0, 30.0)
        )
        sign, note = sr._resolve_orientation(object(), _FRAME, {}, {}, {"G"})
        assert sign == -1  # landmark primary, not overridden by G-alpha
        assert note is not None and "disagree" in note

    def test_no_reference_stays_unoriented(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_references(monkeypatch, None, None)
        assert sr._resolve_orientation(object(), _FRAME, {}, {}, set()) == (None, None)


class TestSidePropagation:
    """The oriented side fact reaches each copy's payload (or is absent when the
    structure cannot be oriented). The signed depth number is always preserved."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        ic_sign: int | None,
        note: str | None = None,
    ) -> list:
        # A structure carrying one LIG copy so the per-copy atom list is non-empty
        # and the depth fact is computed; orientation + depth are stubbed so the
        # test controls the side outcome deterministically (no network / no AI).
        st = _structure([_ca_chain("R", [(0.0, 0.0, 0.0)])])
        lig_chain = gemmi.Chain("B")
        lig = gemmi.Residue()
        lig.name = "LIG"
        lig.seqid = gemmi.SeqId(1, " ")
        lig.het_flag = "H"
        atom = gemmi.Atom()
        atom.name = "C1"
        atom.pos = gemmi.Position(0.0, 0.0, -24.0)
        atom.element = gemmi.Element("C")
        lig.add_atom(atom)
        lig_chain.add_residue(lig)
        st[0].add_chain(lig_chain)

        monkeypatch.setattr(sr, "load_structure", lambda *a, **k: st)
        monkeypatch.setattr(
            sr, "fetch_polymer_alignment", lambda *a, **k: {"R": {"Q9NYV8": [(1, 1, 400)]}}
        )
        monkeypatch.setattr(sr, "membrane_frame", lambda *a, **k: _FRAME)
        monkeypatch.setattr(sr, "ligand_facing_fractions", lambda *a, **k: [0.9])
        monkeypatch.setattr(sr, "ligand_contact_residues", lambda *a, **k: [(0.85, [])])
        monkeypatch.setattr(sr, "_copy_evidence", lambda *a, **k: dict(_ORTH))
        monkeypatch.setattr(sr, "galpha_auth_chains", lambda *a, **k: set())
        monkeypatch.setattr(sr, "ligand_membrane_depth", lambda *a, **k: (-24.0, False))
        monkeypatch.setattr(sr, "_resolve_orientation", lambda *a, **k: (ic_sign, note))
        return sr.detect_site_refs("X", _entry("LIG"), tmp_path)

    def test_intracellular_side_propagates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # ic_sign=-1 + a copy at depth -24 (below the mid-plane) -> intracellular.
        copy = self._run(monkeypatch, tmp_path, ic_sign=-1)[0].payload["copies"][0]
        assert copy["side"] == "on the intracellular side"
        assert copy["depth"] == -24.0  # the signed depth number is preserved

    def test_extracellular_side_propagates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # ic_sign=+1 flips the same -24 copy to the extracellular side.
        copy = self._run(monkeypatch, tmp_path, ic_sign=1)[0].payload["copies"][0]
        assert copy["side"] == "on the extracellular side"
        assert copy["depth"] == -24.0

    def test_unoriented_copy_has_no_side(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No orientation -> no side fact, but the depth is still reported.
        copy = self._run(monkeypatch, tmp_path, ic_sign=None)[0].payload["copies"][0]
        assert "side" not in copy
        assert copy["depth"] == -24.0

    def test_orientation_disagreement_note_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A soft orientation note (e.g. G-alpha vs landmark disagreement) surfaces
        # in the debug log so the cross-check caveat is observable.
        note = "the G-protein position disagrees with the receptor intracellular landmarks"
        with caplog.at_level(logging.DEBUG, logger="gpcr_tools.detector.site_ref"):
            self._run(monkeypatch, tmp_path, ic_sign=-1, note=note)
        assert note in caplog.text
