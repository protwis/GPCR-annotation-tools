"""Tests for the G-protein coupling-protomer detector (the rule + enriched parsing).

The gemmi compute (receptor<->G-alpha contacts) and the coordinate fetch are stubbed,
so these exercise the decision logic only: when a structure with a G-alpha and >=2
receptor chains resolves a single coupling protomer decisively.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpcr_tools.detector import coupling as coupling_mod
from gpcr_tools.detector.coupling import _chain_slugs, detect_coupling_protomer
from gpcr_tools.detector.signals import SIGNAL_COUPLING_PROTOMER


def _entity(auth: str, slug: str) -> dict:
    return {
        "uniprots": [{"gpcrdb_entry_name_slug": slug}],
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": auth}}
        ],
    }


def _entry(chains: dict[str, str]) -> dict:
    """chains: auth_chain -> slug (one entity per chain)."""
    return {"polymer_entities": [_entity(auth, slug) for auth, slug in chains.items()]}


# A GABA-B-shaped heterodimer + Gi: chain A = GABBR1 (binds), chain B = GABBR2
# (couples), chain G = G-alpha.
_GABAB = {"A": "gabr1_human", "B": "gabr2_human", "G": "gnai1_human"}


@pytest.fixture
def stub_geometry(monkeypatch: pytest.MonkeyPatch):
    """Stub load_structure (non-None) and let a test set the contact counts."""
    monkeypatch.setattr(coupling_mod, "load_structure", lambda *a, **k: object())
    counts: dict[str, int] = {}
    monkeypatch.setattr(coupling_mod, "receptor_gprotein_contacts", lambda *a, **k: dict(counts))
    return counts


class TestEnrichedParsing:
    def test_chain_slugs(self) -> None:
        assert _chain_slugs(_entry(_GABAB)) == _GABAB


class TestCouplingRule:
    def test_decisive_coupler_flagged(self, stub_geometry, tmp_path: Path) -> None:
        # Only GABBR2 (chain B) contacts the G-alpha -> it is the coupling protomer.
        stub_geometry.update({"A": 0, "B": 12})
        signals = detect_coupling_protomer("7C7Q", _entry(_GABAB), tmp_path)
        assert len(signals) == 1
        assert signals[0].kind == SIGNAL_COUPLING_PROTOMER
        assert signals[0].payload["coupling_chain"] == "B"
        assert signals[0].payload["coupling_slug"] == "gabr2_human"

    def test_homodimer_picks_the_contacting_chain(self, stub_geometry, tmp_path: Path) -> None:
        # Same receptor on both chains; the G-alpha resolves which protomer couples.
        chains = {"R": "grm2_human", "S": "grm2_human", "A": "gnai1_human"}
        stub_geometry.update({"R": 17, "S": 0})
        signals = detect_coupling_protomer("7E9G", _entry(chains), tmp_path)
        assert len(signals) == 1
        assert signals[0].payload["coupling_chain"] == "R"

    def test_ambiguous_interface_no_signal(self, stub_geometry, tmp_path: Path) -> None:
        # Both protomers heavily contact the G-alpha -> cannot claim one.
        stub_geometry.update({"A": 20, "B": 18})
        assert detect_coupling_protomer("X", _entry(_GABAB), tmp_path) == []

    def test_below_min_contacts_no_signal(self, stub_geometry, tmp_path: Path) -> None:
        stub_geometry.update({"A": 0, "B": 2})
        assert detect_coupling_protomer("X", _entry(_GABAB), tmp_path) == []

    def test_min_contacts_boundary_emits(self, stub_geometry, tmp_path: Path) -> None:
        # Exactly at the floor (top == MIN_CONTACTS=4, runner-up 0) -> emits.
        stub_geometry.update({"A": 0, "B": 4})
        assert len(detect_coupling_protomer("X", _entry(_GABAB), tmp_path)) == 1

    def test_just_below_boundary_silent(self, stub_geometry, tmp_path: Path) -> None:
        stub_geometry.update({"A": 0, "B": 3})
        assert detect_coupling_protomer("X", _entry(_GABAB), tmp_path) == []

    def test_decisive_ratio_boundary(self, stub_geometry, tmp_path: Path) -> None:
        # top=8, runner-up=2 (== 0.25*8) is decisive; =3 (> 0.25*8) is ambiguous.
        stub_geometry.update({"A": 2, "B": 8})
        assert len(detect_coupling_protomer("X", _entry(_GABAB), tmp_path)) == 1
        stub_geometry.update({"A": 3, "B": 8})
        assert detect_coupling_protomer("X", _entry(_GABAB), tmp_path) == []


class TestShortCircuits:
    def test_no_galpha_skips(self, stub_geometry, tmp_path: Path) -> None:
        chains = {"A": "gabr1_human", "B": "gabr2_human"}  # no G-alpha chain
        stub_geometry.update({"A": 0, "B": 12})
        assert detect_coupling_protomer("X", _entry(chains), tmp_path) == []

    def test_single_receptor_chain_skips(self, stub_geometry, tmp_path: Path) -> None:
        chains = {"A": "casr_human", "G": "gnai1_human"}  # one protomer -> nothing to tell apart
        stub_geometry.update({"A": 25})
        assert detect_coupling_protomer("X", _entry(chains), tmp_path) == []

    def test_missing_structure_skips(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(coupling_mod, "load_structure", lambda *a, **k: None)
        assert detect_coupling_protomer("X", _entry(_GABAB), tmp_path) == []
