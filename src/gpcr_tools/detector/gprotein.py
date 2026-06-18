"""Pre-annotation G-protein identity detector.

Wraps the alpha5 sequence analysis (``validator.chimera``) into a
``DetectSignal``. When the coupling family is found but the subtype cannot be
told apart by the alpha5 (or no single family fits), the signal is marked for
review so a human confirms the subtype rather than the pipeline forcing one.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.config import CHIMERA_STATUS_SUCCESS, CHIMERA_SUBTYPE_LOW_CONFIDENCE
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    DetectSignal,
)
from gpcr_tools.validator.cache import SequenceCache
from gpcr_tools.validator.chimera import get_chimera_analysis

G_PROTEIN_LOCUS = "signaling_partners.g_protein.alpha_subunit"


def detect_g_protein_identity(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache: SequenceCache,
) -> tuple[list[DetectSignal], bool]:
    """Emit at most one G-protein identity signal for *enriched_entry*.

    A cleanly resolved subtype yields an advisory signal (evidence for the
    prompt). A family-only / indistinguishable / low-confidence result yields a
    review signal (routes the subtype to a human).

    Returns ``(signals, degraded)`` where *degraded* is True when one or more
    reference sequences transiently failed to fetch (timeout/5xx, not a 404), so
    the result is weaker than the input warrants and a later run should recompute
    it once UniProt recovers. A genuinely G-alpha-free structure is not degraded.
    """
    result = get_chimera_analysis(pdb_id, enriched_entry, cache)
    degraded = bool(result.get("transient_abstained"))
    if result.get("status") != CHIMERA_STATUS_SUCCESS:
        return [], degraded

    family = result.get("family")
    subtype = result.get("subtype")
    candidates = result.get("candidate_set") or []
    a5_tail = result.get("a5_tail")
    payload = {
        "family": family,
        "subtype": subtype,
        "subtype_resolution": result.get("subtype_resolution"),
        "candidate_set": candidates,
        "a5_tail": a5_tail,
        "score": result.get("score"),
    }

    if subtype is not None:
        return [
            DetectSignal(
                kind=SIGNAL_CHIMERIC_GPROTEIN,
                target_ref=G_PROTEIN_LOCUS,
                summary=f"alpha5 '{a5_tail}' resolves the G-alpha to {subtype}.",
                payload=payload,
                severity=SEVERITY_ADVISORY,
            )
        ], degraded

    members = ", ".join(candidates) or "no recognised subtype"
    if result.get("subtype_resolution") == CHIMERA_SUBTYPE_LOW_CONFIDENCE:
        summary = (
            f"alpha5 window match too weak to identify the G-alpha "
            f"(best window '{a5_tail}'); confirm manually."
        )
    elif family:
        summary = (
            f"alpha5 '{a5_tail}' indicates the {family} family but cannot "
            f"distinguish the subtype ({members}); confirm manually."
        )
    else:
        summary = (
            f"alpha5 '{a5_tail}' does not map to a single coupling family "
            f"({members}); confirm manually."
        )
    return [
        DetectSignal(
            kind=SIGNAL_CHIMERIC_GPROTEIN,
            target_ref=G_PROTEIN_LOCUS,
            summary=summary,
            payload=payload,
            severity=SEVERITY_REVIEW,
        )
    ], degraded
