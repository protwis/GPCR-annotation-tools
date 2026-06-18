"""The detect-stage signal contract.

A ``DetectSignal`` is one structural / bioinformatic observation about a PDB,
made before annotation. ``severity`` decides routing: ``advisory`` signals are
recorded for later prompt branching; ``review`` signals are also surfaced to the
curator as critical warnings (which disables one-click accept-all).

Severity is fail-safe (see ``DetectSignal.__post_init__``): only an explicit,
recognised ``advisory`` feeds the model prompt. Anything else -- a missing,
empty, or unrecognised severity -- is normalised to ``review`` so an
unclassified signal is surfaced to a human, never silently injected as prompt
evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Signal kinds (domain vocabulary; more are added as detectors land).
SIGNAL_CHIMERIC_GPROTEIN: str = "chimeric_g_protein"
SIGNAL_INCIDENTAL_CANDIDATE: str = "incidental_candidate"
SIGNAL_DUAL_ROLE_LIGAND: str = "dual_role_ligand"
SIGNAL_SITE_REF: str = "site_ref"
SIGNAL_COUPLING_PROTOMER: str = "coupling_protomer"
SIGNAL_CLASS_C_MULTI_PROTOMER: str = "class_c_multi_protomer"

SEVERITY_ADVISORY: str = "advisory"
SEVERITY_REVIEW: str = "review"


@dataclass(frozen=True)
class DetectSignal:
    """One pre-annotation observation, anchored to a JSON-path locus."""

    kind: str
    target_ref: str  # stable anchor, e.g. a JSON path / chain id / component id
    summary: str  # one-line, human-readable
    payload: dict[str, Any] = field(default_factory=dict)
    severity: str = SEVERITY_REVIEW  # fail-safe default: opt in to advisory explicitly

    def __post_init__(self) -> None:
        # Fail safe: only an explicit, recognised ``advisory`` reaches the model
        # prompt. Empty / malformed / future severities become ``review`` so an
        # unclassified signal is surfaced to a human, not silently injected as
        # prompt evidence. (frozen dataclass -> object.__setattr__ to normalise.)
        if self.severity not in (SEVERITY_ADVISORY, SEVERITY_REVIEW):
            object.__setattr__(self, "severity", SEVERITY_REVIEW)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectSignal:
        # A missing/empty severity is left blank here and normalised to
        # ``review`` by __post_init__ -- deserialising a malformed or
        # future-version signal must not silently default to advisory.
        return cls(
            kind=data["kind"],
            target_ref=data.get("target_ref") or "",
            summary=data.get("summary") or "",
            payload=data.get("payload") or {},
            severity=data.get("severity") or "",
        )


def to_critical_warnings(signals: list[DetectSignal]) -> list[str]:
    """Format the ``review`` signals as critical-warning strings.

    Uses the established locus convention ``"<kind> at '<target_ref>': <summary>"``
    so the existing curate gating (disable accept-all on any critical warning,
    route by the ``at '<path>'`` locus) consumes them unchanged.
    """
    return [
        f"{s.kind} at '{s.target_ref}': {s.summary}"
        for s in signals
        if s.severity == SEVERITY_REVIEW
    ]
