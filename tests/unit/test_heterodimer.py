"""Tests for the Class C multi-protomer detector (metadata-only, no fetch).

The detector fires only when both gates hold: a GPCR protomer is GPCRdb Class C
AND the GPCR roster has more than one protomer. Real GPCRdb accessions are used so
the shipped numbering table answers the class lookup directly.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.detector.heterodimer import detect_class_c_multi_protomer
from gpcr_tools.detector.signals import SEVERITY_ADVISORY, SIGNAL_CLASS_C_MULTI_PROTOMER

# Real GPCRdb accessions: GABA-B is an obligate Class C heterodimer (GABBR1 +
# GABBR2); ADRB2 / DRD2 are Class A monomers used as negatives.
_GABBR1 = "Q9UBS5"
_GABBR2 = "O75899"
_ADRB2 = "P07550"
_DRD2 = "P14416"


def _entity(chain: str, slug: str, accession: str) -> dict[str, Any]:
    return {
        "uniprots": [{"gpcrdb_entry_name_slug": slug, "rcsb_id": accession}],
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": chain}}
        ],
    }


def _enriched(*entities: dict[str, Any]) -> dict[str, Any]:
    return {"polymer_entities": list(entities)}


class TestClassCMultiProtomerDetector:
    def test_class_c_heterodimer_fires(self) -> None:
        entry = _enriched(
            _entity("A", "gabr1_human", _GABBR1),
            _entity("B", "gabr2_human", _GABBR2),
        )
        signals = detect_class_c_multi_protomer("TEST", entry)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.kind == SIGNAL_CLASS_C_MULTI_PROTOMER
        assert sig.severity == SEVERITY_ADVISORY  # must reach the prompt
        assert sig.target_ref == "receptor_info"

    def test_class_c_monomer_silent(self) -> None:
        # A single Class C protomer is not a multi-protomer structure.
        entry = _enriched(_entity("A", "gabr2_human", _GABBR2))
        assert detect_class_c_multi_protomer("TEST", entry) == []

    def test_class_a_heterodimer_silent(self) -> None:
        # Two GPCR protomers but neither is Class C -> no advisory.
        entry = _enriched(
            _entity("A", "adrb2_human", _ADRB2),
            _entity("B", "drd2_human", _DRD2),
        )
        assert detect_class_c_multi_protomer("TEST", entry) == []

    def test_no_gpcr_chains_silent(self) -> None:
        assert detect_class_c_multi_protomer("TEST", {"polymer_entities": []}) == []

    def test_mixed_class_a_and_class_c_fires(self) -> None:
        # If any protomer is Class C and there is more than one protomer, fire.
        entry = _enriched(
            _entity("A", "gabr2_human", _GABBR2),
            _entity("B", "adrb2_human", _ADRB2),
        )
        signals = detect_class_c_multi_protomer("TEST", entry)
        assert len(signals) == 1
        assert signals[0].kind == SIGNAL_CLASS_C_MULTI_PROTOMER
