"""Tests for receptor-side crystallization-fusion detection (BRIL / T4 lysozyme).

A BRIL or T4-lysozyme fused into a receptor is a crystallization aid, not part
of the biological receptor; it is surfaced as an advisory (non-blocking) note.
A standalone lysozyme that is not fused into a receptor must not be flagged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpcr_tools.aggregator.runner import _build_validation_report
from gpcr_tools.config import CHIMERA_STATUS_SKIPPED
from gpcr_tools.validator.oligomer import detect_crystallization_fusions

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "real_pdbs"


def _entity(auth: str, slug: str | None, description: str) -> dict[str, Any]:
    return {
        "uniprots": [{"gpcrdb_entry_name_slug": slug}] if slug else [],
        "rcsb_polymer_entity": {"pdbx_description": description},
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": auth}}
        ],
    }


def _enriched(*entities: Any) -> dict[str, Any]:
    return {"polymer_entities": list(entities)}


class TestCrystallizationFusions:
    def test_bril_in_description_flagged(self) -> None:
        enriched = _enriched(
            _entity(
                "A", "acm4_human", "Muscarinic acetylcholine receptor M4,Soluble cytochrome b562"
            )
        )
        notes = detect_crystallization_fusions(enriched)
        assert len(notes) == 1
        assert "chain(s) A" in notes[0] and "receptor_info" in notes[0]

    def test_t4_lysozyme_flagged(self) -> None:
        enriched = _enriched(
            _entity("A", "aa2ar_human", "Adenosine receptor A2A-T4 lysozyme chimera")
        )
        assert len(detect_crystallization_fusions(enriched)) == 1

    def test_fusion_slug_flagged(self) -> None:
        # BRIL carried as a separate uniprot slug (c562) on the receptor entity.
        entity = _entity("A", "aa2ar_human", "Adenosine receptor A2A")
        entity["uniprots"].append({"gpcrdb_entry_name_slug": "c562_ecolx"})
        assert len(detect_crystallization_fusions(_enriched(entity))) == 1

    def test_plain_receptor_not_flagged(self) -> None:
        enriched = _enriched(_entity("A", "aa2ar_human", "Adenosine receptor A2A"))
        assert detect_crystallization_fusions(enriched) == []

    def test_standalone_lysozyme_not_flagged(self) -> None:
        # A lysozyme not fused into a receptor (no GPCR slug) must be ignored.
        enriched = _enriched(_entity("A", None, "T4 lysozyme"))
        assert detect_crystallization_fusions(enriched) == []

    def test_none_safe(self) -> None:
        assert detect_crystallization_fusions({}) == []
        assert detect_crystallization_fusions({"polymer_entities": [None, {}]}) == []


class TestRealData:
    def _enriched(self, pdb: str) -> dict[str, Any]:
        raw = json.loads((_FIXTURES / "enriched" / f"{pdb}.json").read_text())
        return (raw.get("data") or {}).get("entry") or raw

    def test_9iqs_bril_fusion_flagged(self) -> None:
        notes = detect_crystallization_fusions(self._enriched("9IQS"))
        assert len(notes) == 1
        assert "chain(s) B" in notes[0] and "b562" in notes[0]

    def test_9o38_gprotein_fusion_not_flagged(self) -> None:
        # 9O38 carries a receptor-Galpha fusion, not BRIL/T4L -> no fusion note.
        assert detect_crystallization_fusions(self._enriched("9O38")) == []


class TestRoutingInvariant:
    def test_fusion_note_is_advisory_not_blocking(self) -> None:
        # The fusion note must land in detector_notes (non-blocking), never in
        # critical_warnings (which would gate accept-all).
        enriched = _enriched(
            _entity(
                "A", "acm4_human", "Muscarinic acetylcholine receptor M4,Soluble cytochrome b562"
            )
        )
        best = {"receptor_info": {"chain_id": "A"}}
        report = _build_validation_report(
            "TEST", best, enriched, [], {"status": CHIMERA_STATUS_SKIPPED, "score": 0}, None
        )
        assert any("CRYSTALLIZATION FUSION" in n for n in report["detector_notes"])
        assert not any("CRYSTALLIZATION FUSION" in w for w in report["critical_warnings"])
