"""Live calibration test for the dual-role detector against real coordinates.

Fetches a known two-pocket structure (9IIX: the studied ligand A1AEI modelled in
two distinct buried pockets on the TAS2R14 chain, one of which contacts the
G-protein) and confirms the production geometry flags it. Gated on the network so
the unit suite stays offline and fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpcr_tools.detector.geometry import detect_dual_role_ligands
from gpcr_tools.detector.signals import SIGNAL_DUAL_ROLE_LIGAND

pytestmark = pytest.mark.skipif(
    not os.environ.get("GPCR_RUN_LIVE_TESTS"),
    reason="Live network tests disabled; set GPCR_RUN_LIVE_TESTS=1 to enable",
)

# Minimal enriched fields the detector reads: the studied ligand and the GPCR chain.
_ENTRY_9IIX = {
    "polymer_entities": [
        {
            "uniprots": [{"gpcrdb_entry_name_slug": "t2r14_human"}],
            "polymer_entity_instances": [
                {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "R"}}
            ],
        }
    ],
    "nonpolymer_entities": [
        {
            "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": "A1AEI"},
        }
    ],
}


def test_9iix_dual_role_flagged(tmp_path: Path) -> None:
    signals = detect_dual_role_ligands("9IIX", _ENTRY_9IIX, tmp_path)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.kind == SIGNAL_DUAL_ROLE_LIGAND
    assert signal.payload["comp_id"] == "A1AEI"
    assert signal.payload["gpcr_chain"] == "R"
    copies = signal.payload["copies"]
    assert len(copies) == 2
    assert all(c["burial"] >= 0.80 for c in copies)
    # The orthosteric copy contacts the G-protein (active-state hint).
    assert any(c["contacts_partner"] for c in copies)
