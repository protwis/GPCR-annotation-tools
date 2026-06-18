"""The pre-annotation detect stage.

Reads a PDB's enriched metadata, runs the detectors, and persists the resulting
signals to ``detect/{pdb}.json`` for the annotate and curate stages to consume.
No AI and no paper are used here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from gpcr_tools.config import DETECT_INCOMPLETE_MARKER_KEY, get_config
from gpcr_tools.detector.coupling import detect_coupling_protomer
from gpcr_tools.detector.geometry import detect_dual_role_ligands
from gpcr_tools.detector.gprotein import detect_g_protein_identity
from gpcr_tools.detector.heterodimer import detect_class_c_multi_protomer
from gpcr_tools.detector.ligands import detect_incidental_candidates
from gpcr_tools.detector.signals import DetectSignal
from gpcr_tools.detector.site_ref import detect_site_refs
from gpcr_tools.fetcher.enricher import INCOMPLETE_MARKER_KEY
from gpcr_tools.validator.cache import SequenceCache

logger = logging.getLogger(__name__)

_SEQUENCE_CACHE_NAME = "uniprot_sequence_cache.json"


def _enriched_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the ``data.entry`` envelope if present, else use the object."""
    return (raw.get("data") or {}).get("entry") or raw


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path* atomically (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _detect_is_incomplete(path: Path) -> bool:
    """Whether the detect output at *path* must be recomputed on a resume.

    True when the file is unreadable/corrupt (redo it) or carries the incomplete
    marker (a transient reference-fetch failure degraded it). A clean, readable
    record -- including one that legitimately produced no signal -- is complete
    and skipped, so a healthy run is never needlessly recomputed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get(DETECT_INCOMPLETE_MARKER_KEY))


def run_detect(
    pdb_id: str,
    *,
    skip_api_checks: bool = False,
    force: bool = False,
    cache: SequenceCache | None = None,
) -> list[DetectSignal]:
    """Run the detectors on one PDB, persist ``detect/{pdb}.json``, return signals.

    Sequence-based detectors are skipped when *skip_api_checks* is set (they need
    UniProt reference sequences). The detect file is written regardless, so the
    stage's output is always present and inspectable.

    Top-up resume: a PDB whose detect output already exists and is complete is
    skipped (its persisted signals are returned). One that is missing, unreadable,
    or marked incomplete -- a transient reference-fetch failure degraded it -- is
    recomputed. *force* recomputes regardless.

    A *cache* may be supplied by a caller running many PDBs so fetched sequences
    persist across them; the caller then owns saving it. When none is given (a
    single-PDB run) one is created and saved here.
    """
    cfg = get_config()
    out_path = cfg.detect_dir / f"{pdb_id}.json"
    prior_incomplete = out_path.is_file() and _detect_is_incomplete(out_path)
    if not force and out_path.is_file() and not prior_incomplete:
        logger.debug("[detect] %s: complete output exists; skipping.", pdb_id)
        return load_detect_signals(pdb_id)

    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"
    if not enriched_path.is_file():
        logger.warning("[detect] %s: no enriched data found; skipping.", pdb_id)
        return []

    try:
        raw = json.loads(enriched_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[detect] %s: could not read enriched data: %s", pdb_id, exc)
        return []

    # Enrichment written during a transient API outage carries this top-level
    # marker. Detecting on it would derive signals from a recoverable gap; skip
    # and let a re-run of fetch heal the record first.
    if isinstance(raw, dict) and raw.get(INCOMPLETE_MARKER_KEY):
        logger.warning(
            "[detect] %s: enrichment incomplete (transient API gap); skipping — "
            "re-run fetch to complete enrichment first.",
            pdb_id,
        )
        return []

    entry = _enriched_entry(raw)
    signals: list[DetectSignal] = []

    # Metadata-only detectors (no external fetch) always run.
    signals.extend(detect_incidental_candidates(pdb_id, entry))
    signals.extend(detect_class_c_multi_protomer(pdb_id, entry))

    # Detectors needing a network fetch (UniProt references, coordinate files).
    detect_incomplete = False
    if not skip_api_checks:
        owns_cache = cache is None
        if cache is None:
            cache = SequenceCache(cfg.cache_dir / _SEQUENCE_CACHE_NAME)
        gp_signals, gp_degraded = detect_g_protein_identity(pdb_id, entry, cache)
        signals.extend(gp_signals)
        detect_incomplete = gp_degraded
        # The coordinate-file detectors below deliberately do NOT contribute to
        # detect_incomplete: load_structure returns None indistinguishably for a
        # transient fetch failure, a parse error, and a genuinely apo/absent
        # structure, so marking on it would re-run apo structures forever. Only
        # the UniProt-reference path above has a transient-vs-definitive signal.
        signals.extend(detect_dual_role_ligands(pdb_id, entry, cfg.cache_dir))
        signals.extend(detect_site_refs(pdb_id, entry, cfg.cache_dir))
        signals.extend(detect_coupling_protomer(pdb_id, entry, cfg.cache_dir))
        if owns_cache:
            cache.save()
    # Future detectors (entity reconciliation, ...) append here.

    payload: dict[str, Any] = {"pdb_id": pdb_id, "signals": [s.to_dict() for s in signals]}
    # Stamp the marker on a fresh transient failure, and preserve an existing one
    # when API checks were skipped this pass -- a metadata-only run cannot resolve
    # the sequence-fetch gap that set it, so it must not silently clear it. A clean
    # full run omits the marker, which clears a stale one on rewrite.
    if detect_incomplete or (skip_api_checks and prior_incomplete):
        payload[DETECT_INCOMPLETE_MARKER_KEY] = True
    _atomic_write_json(out_path, payload)
    logger.info("[detect] %s: %d signal(s) -> %s", pdb_id, len(signals), out_path)
    return signals


def run_detect_stage(
    pdb_id: str | None = None, *, skip_api_checks: bool = False, force: bool = False
) -> dict[str, int]:
    """Run detect on one PDB or every enriched PDB. Returns {pdb_id: signal count}.

    By default this tops up: PDBs with a complete detect output are skipped and
    only missing or transiently-degraded ones are (re)computed. *force* recomputes
    every PDB.
    """
    cfg = get_config()

    # Fail fast on a stale / missing storage contract before any detection work,
    # so a layout mismatch is caught before the expensive annotate stage.
    from gpcr_tools.workspace import validate_contract

    validate_contract(cfg)

    if pdb_id:
        targets = [pdb_id]
    elif cfg.enriched_dir.exists():
        targets = sorted(p.stem for p in cfg.enriched_dir.glob("*.json"))
    else:
        targets = []

    # One shared cache across all PDBs so UniProt references fetched for an early
    # PDB are reused (and persisted) rather than refetched for each one.
    cache = None if skip_api_checks else SequenceCache(cfg.cache_dir / _SEQUENCE_CACHE_NAME)
    summary = {
        pid: len(run_detect(pid, skip_api_checks=skip_api_checks, force=force, cache=cache))
        for pid in targets
    }
    if cache is not None:
        cache.save()
    logger.info("[detect] processed %d PDB(s).", len(targets))
    return summary


def load_detect_signals(pdb_id: str) -> list[DetectSignal]:
    """Read back the persisted detect signals for *pdb_id* (empty if none)."""
    cfg = get_config()
    path = cfg.detect_dir / f"{pdb_id}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [DetectSignal.from_dict(s) for s in (data.get("signals") or [])]
