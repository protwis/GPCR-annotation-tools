"""Pre-annotation Class C multi-protomer advisory (metadata-only, no AI, no fetch).

A Class C GPCR (e.g. metabotropic glutamate, GABA-B, calcium-sensing) is an
obligate dimer: a structure routinely carries more than one GPCR protomer, and
every protomer is itself a receptor. When the model annotates such a structure it
tends to file the partner protomer under ``auxiliary_proteins`` instead of
treating it as a receptor. This advisory simply states the structural fact -- a
Class C receptor with more than one GPCR protomer -- so the model treats both
protomers as receptors.

The fact is read from already-enriched metadata only (the receptor's GPCRdb class
plus the GPCR roster), so this detector needs no network fetch and runs in the
always-on metadata block. It is advisory: it surfaces a fact for the prompt, never
a verdict, and never computes which protomer couples or is primary.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_CLASS_C_MULTI_PROTOMER,
    DetectSignal,
)
from gpcr_tools.detector.site_ref import _gpcr_chain_accessions
from gpcr_tools.validator.generic_numbering import receptor_class
from gpcr_tools.validator.oligomer import _build_gpcr_roster

# GPCRdb class slug for Class C receptors (metabotropic glutamate, GABA-B,
# calcium-sensing, taste 1, ...). Its protomers are obligate dimers.
_GPCRDB_CLASS_C: str = "004"

_HETERODIMER_LOCUS = "receptor_info"


def detect_class_c_multi_protomer(
    pdb_id: str,
    enriched_entry: dict[str, Any],
) -> list[DetectSignal]:
    """One advisory when a Class C receptor structure has >1 GPCR protomer.

    Double-gated, both from already-enriched metadata: (1) a GPCR protomer's
    UniProt accession maps to GPCRdb Class C, AND (2) the full GPCR roster has
    more than one protomer. Returns no signal otherwise (a Class A/B/F structure,
    a single-protomer Class C structure, or one with no GPCR chains).
    """
    roster = _build_gpcr_roster(enriched_entry)
    if len(roster) <= 1:
        return []

    accessions = _gpcr_chain_accessions(enriched_entry)
    is_class_c = any(receptor_class(acc) == _GPCRDB_CLASS_C for acc in set(accessions.values()))
    if not is_class_c:
        return []

    return [
        DetectSignal(
            kind=SIGNAL_CLASS_C_MULTI_PROTOMER,
            target_ref=_HETERODIMER_LOCUS,
            summary="Class C receptor structure with more than one GPCR protomer.",
            payload={
                "gpcr_chains": sorted(roster.keys()),
                "accessions": sorted(set(accessions.values())),
            },
            severity=SEVERITY_ADVISORY,
        )
    ]
