"""Pre-annotation detector for incidental-candidate molecules.

A molecule on the curated ``INCIDENTAL_CANDIDATES`` set (cholesterol, palmitate)
can be EITHER a functional ligand OR an incidental structural additive. When one
is present this surfaces it as advisory evidence so the model judges the role
itself -- and the exclude-list strip is bypassed so the model can see it.

This is metadata-only (no sequence/UniProt fetch), so it always runs.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    LOCUS_LIGANDS,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_INCIDENTAL_CANDIDATE,
    DetectSignal,
)


def _nonpolymer_comp_ids(enriched_entry: dict[str, Any]) -> list[str]:
    """Collect the chem_comp ids of every non-polymer entity (None-safe)."""
    entities = enriched_entry.get("nonpolymer_entities")
    if not isinstance(entities, list):
        return []
    ids: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        comp_id = ((entity.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id")
        if comp_id:
            ids.append(comp_id)
    return ids


def detect_incidental_candidates(
    pdb_id: str,
    enriched_entry: dict[str, Any],
) -> list[DetectSignal]:
    """One advisory signal per incidental-candidate molecule (cholesterol / palmitate) present.

    An incidental-candidate molecule can be EITHER a functional ligand OR an incidental
    structural lipid. The signal is advisory: it routes evidence into the prompt
    so the model judges the role itself (and any incidental-candidate member stripped by the
    exclude list is un-stripped so the model can see it) -- it does not silently
    send the case to human review.
    """
    present = sorted(set(_nonpolymer_comp_ids(enriched_entry)) & INCIDENTAL_CANDIDATES)
    return [
        DetectSignal(
            kind=SIGNAL_INCIDENTAL_CANDIDATE,
            target_ref=LOCUS_LIGANDS,
            summary=(
                f"{code} is present; it can be a functional ligand or an incidental "
                f"structural component. Judge its role from the paper and record a "
                f"pharmacological_role_check."
            ),
            payload={"comp_id": code},
            severity=SEVERITY_ADVISORY,
        )
        for code in present
    ]
