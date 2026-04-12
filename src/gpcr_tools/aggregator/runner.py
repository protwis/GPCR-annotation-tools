"""Orchestration layer — wire all aggregation + validation components.

``aggregate_pdb()`` runs the full pipeline for a single PDB ID:
    AI runs → voting → best run → deepcopy → ground truth → validators →
    discrepancies → integrity → chimera → validation report → atomic writes.

``aggregate_all()`` iterates pending PDBs with per-PDB error isolation.

Blood Lesson 2 — Atomic writes:
    ALL output files are written to temp files first, then ``os.replace``-d
    together after all writes succeed.  ``try...finally`` guarantees cleanup.

Blood Lesson 5 — Truthiness:
    ``if enriched is None:`` — NOT ``if not enriched:``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpcr_tools.aggregator.ai_results_loader import get_pending_pdb_ids, load_ai_runs
from gpcr_tools.aggregator.enriched_loader import load_enriched_data
from gpcr_tools.aggregator.ground_truth import inject_ground_truth
from gpcr_tools.aggregator.voting import (
    extract_ai_g_protein,
    find_discrepancies,
    get_majority_votes,
    select_best_run,
)
from gpcr_tools.config import (
    AGG_STATUS_COMPLETED,
    AGG_STATUS_FAILED,
    ALERT_PREFIX_ALGO_WARNING,
    ALERT_PREFIX_HALLUCINATION,
    ALERT_PREFIX_TIE_BREAKER_ALIGNED,
    ALERT_PREFIX_TIE_BREAKER_OVERRIDE,
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_SKIPPED,
    CHIMERA_STATUS_SUCCESS,
    EMPTY_VALUES,
    get_config,
)
from gpcr_tools.validator.cache import SequenceCache, ValidationCache
from gpcr_tools.validator.chimera import get_chimera_analysis
from gpcr_tools.validator.integrity_checker import validate_all
from gpcr_tools.validator.ligand_validator import validate_and_enrich_ligands
from gpcr_tools.validator.oligomer import analyze_oligomer
from gpcr_tools.validator.receptor_validator import validate_receptor_identity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AggregateResult:
    """Container for a single PDB aggregation result."""

    pdb_id: str
    success: bool
    aggregated_path: Path | None = None
    voting_log_path: Path | None = None
    validation_path: Path | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation report assembly
# ---------------------------------------------------------------------------


def _build_validation_report(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
    all_warnings: list[str],
    chimera_result: dict[str, Any],
    validation_cache: ValidationCache | None,
) -> dict[str, Any]:
    """Assemble the validation report from all warning sources.

    Blood Lesson 1: ``chimera_result.get("score") or 0`` — NOT
    ``chimera_result.get("score", 0)``.

    Blood Lesson 4: all status comparisons use constants.

    Review 4 A-2, Review 7: full chimera vs AI conflict classification.
    """
    report: dict[str, Any] = {
        "critical_warnings": list(all_warnings),
        "algo_conflicts": [],
        "algo_notes": [],
        "chimera_score": chimera_result.get("score") or 0,
        "chimera_status": chimera_result.get("status") or CHIMERA_STATUS_SKIPPED,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    # Integrity checks (ghost chain, fake UniProt/PubChem, ghost ligand, method)
    integrity_warnings = validate_all(pdb_id, best_run_data, enriched_entry, cache=validation_cache)
    report["critical_warnings"].extend(integrity_warnings)

    # Chimera vs AI comparison (Review 4 A-2, Review 7)
    status = chimera_result.get("status") or CHIMERA_STATUS_SKIPPED
    can_best = chimera_result.get("can_best")
    ai_uniprot = extract_ai_g_protein(best_run_data)

    if status == CHIMERA_STATUS_SUCCESS:
        max_matches = chimera_result.get("max_score_matches") or ([can_best] if can_best else [])
        if ai_uniprot:
            tail_seq = chimera_result.get("tail_seq") or "N/A"
            if ai_uniprot in max_matches:
                report["algo_notes"].append(
                    f"{ALERT_PREFIX_TIE_BREAKER_ALIGNED} at 'chimera_analysis': "
                    f"Tail '{tail_seq}' matched {len(max_matches)} slugs. "
                    f"AI choice '{ai_uniprot}' retained."
                )
            else:
                report["algo_conflicts"].append(
                    f"{ALERT_PREFIX_TIE_BREAKER_OVERRIDE} at 'chimera_analysis': "
                    f"Tail '{tail_seq}' matched {len(max_matches)} slugs. "
                    f"AI '{ai_uniprot}' failed. "
                    f"Defaulted to canonical '{can_best}'."
                )
        # Review 4 L-4: guard None interpolation
        if can_best:
            report["algo_notes"].append(f"Matched G-alpha tail to '{can_best}'.")
    elif status == CHIMERA_STATUS_NO_G_PROTEIN:
        if ai_uniprot and str(ai_uniprot).lower() not in EMPTY_VALUES:
            report["algo_conflicts"].append(
                f"{ALERT_PREFIX_HALLUCINATION} at 'chimera_analysis': "
                f"AI found '{ai_uniprot}' but algorithm found NO G-protein "
                f"in source PDB."
            )
    elif status != CHIMERA_STATUS_SKIPPED:
        error_msg = chimera_result.get("error")
        report["algo_conflicts"].append(
            f"{ALERT_PREFIX_ALGO_WARNING} at 'chimera_analysis': "
            f"Verification could not run. Status: '{status}'. "
            f"Details: {error_msg}"
        )

    return report


# ---------------------------------------------------------------------------
# Atomic write block
# ---------------------------------------------------------------------------


def _write_outputs(
    pdb_id: str,
    best_run_data: dict[str, Any],
    discrepancies: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> AggregateResult:
    """Write aggregated JSON, voting log, and validation report atomically.

    Blood Lesson 2: all temp files written first, then ``os.replace``-d.
    ``try...finally`` guarantees cleanup on failure.
    """
    cfg = get_config()
    aggregated_path = cfg.aggregated_dir / f"{pdb_id}.json"
    voting_log_dir = cfg.aggregated_dir / "logs"
    validation_dir = cfg.aggregated_dir / "validation_logs"

    aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    voting_log_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    voting_log_path = voting_log_dir / f"{pdb_id}_voting_log.json" if discrepancies else None
    validation_path = validation_dir / f"{pdb_id}_validation.json"

    tmp_paths: list[str] = []
    try:
        # Write all temp files
        tmp_agg = _write_temp_json(aggregated_path.parent, best_run_data)
        tmp_paths.append(tmp_agg)

        tmp_val = _write_temp_json(validation_path.parent, validation_report)
        tmp_paths.append(tmp_val)

        tmp_log: str | None = None
        if voting_log_path is not None:
            tmp_log = _write_temp_json(voting_log_path.parent, discrepancies)
            tmp_paths.append(tmp_log)

        # Commit all at once
        os.replace(tmp_agg, str(aggregated_path))
        os.replace(tmp_val, str(validation_path))
        if voting_log_path is not None and tmp_log is not None:
            os.replace(tmp_log, str(voting_log_path))

        # Clear committed paths from cleanup list
        tmp_paths.clear()

        return AggregateResult(
            pdb_id=pdb_id,
            success=True,
            aggregated_path=aggregated_path,
            voting_log_path=voting_log_path,
            validation_path=validation_path,
        )
    finally:
        for tmp in tmp_paths:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _write_temp_json(directory: Path, data: Any) -> str:
    """Write *data* to a temp file in *directory* and return the temp path."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(directory),
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as fd:
        json.dump(data, fd, indent=4)
        return fd.name


# ---------------------------------------------------------------------------
# Aggregate log
# ---------------------------------------------------------------------------


def _update_aggregate_log(
    pdb_id: str,
    status: str,
) -> None:
    """Record *pdb_id* processing status in ``aggregate_log.json``.

    Blood Lesson 2: uses atomic write.
    Never swallows exceptions silently — logs warnings.
    """
    cfg = get_config()
    log_path = cfg.state_dir / "aggregate_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_data: dict[str, Any] = {}
    if log_path.is_file():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                log_data = raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read aggregate log: %s", exc)

    log_data[pdb_id] = {
        "status": status,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(log_path.parent),
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fd:
            tmp_path = fd.name
            json.dump(log_data, fd, indent=2)
        os.replace(tmp_path, str(log_path))
        tmp_path = None
    except OSError as exc:
        logger.warning("Failed to update aggregate log for %s: %s", pdb_id, exc)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def aggregate_pdb(
    pdb_id: str,
    *,
    skip_api_checks: bool = False,
    validation_cache: ValidationCache | None = None,
    sequence_cache: SequenceCache | None = None,
) -> AggregateResult:
    """Run the full aggregation + validation pipeline for a single PDB.

    Steps:
        1. Load AI runs
        2. Majority voting
        3. Select best run + deepcopy (mutation boundary)
        4. Load enriched data
        5. Inject ground truth
        6. Ligand validation
        7. Receptor validation
        8. Oligomer analysis
        9. Compute discrepancies
        10. Chimera analysis
        11. Assemble validation report
        12. Atomic write block

    Blood Lesson 5: ``if enriched is None:`` — NOT ``if not enriched:``.
    """
    # 1. Load AI runs
    runs = load_ai_runs(pdb_id)
    if not runs:
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="No valid AI runs found",
        )

    # 2. Majority voting
    majority_votes, all_votes = get_majority_votes(runs)

    # 3. Select best run + deepcopy
    _best_idx, best_run_original = select_best_run(runs, majority_votes)
    best_run_data = copy.deepcopy(best_run_original)

    # 4. Load enriched data
    enriched = load_enriched_data(pdb_id)
    # BL5: if enriched is None — empty dict {} is valid
    if enriched is None:
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="Enriched data not available",
        )

    try:
        # 5. Inject ground truth (mutates best_run_data)
        inject_ground_truth(pdb_id, best_run_data, enriched)

        # 6. Ligand validation (mutates best_run_data, returns warnings)
        all_warnings: list[str] = []
        ligand_warnings = validate_and_enrich_ligands(pdb_id, best_run_data, enriched)
        all_warnings.extend(ligand_warnings)

        # 7. Receptor validation (mutates best_run_data, returns warnings)
        receptor_warnings = validate_receptor_identity(pdb_id, best_run_data, enriched)
        all_warnings.extend(receptor_warnings)

        # 8. Oligomer analysis (mutates best_run_data — may override chain_id)
        analyze_oligomer(pdb_id, best_run_data, enriched)

        # 9. Compute discrepancies
        discrepancies = find_discrepancies(best_run_data, majority_votes, all_votes)

        # 10. Chimera analysis
        chimera_result: dict[str, Any] = {
            "status": CHIMERA_STATUS_SKIPPED,
            "score": 0,
        }
        if not skip_api_checks and sequence_cache is not None:
            chimera_result = get_chimera_analysis(pdb_id, enriched, sequence_cache)

        # 11. Assemble validation report
        v_cache = validation_cache if not skip_api_checks else None
        report = _build_validation_report(
            pdb_id,
            best_run_data,
            enriched,
            all_warnings,
            chimera_result,
            v_cache,
        )

        # 12. Atomic write block
        result = _write_outputs(pdb_id, best_run_data, discrepancies, report)
        result.warnings = report["critical_warnings"]
        return result
    except Exception as exc:
        logger.error("[%s] Pipeline failure: %s", pdb_id, exc)
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error=str(exc),
        )


def aggregate_all(
    *,
    skip_api_checks: bool = False,
    force: bool = False,
) -> list[AggregateResult]:
    """Aggregate all pending PDBs with per-PDB error isolation.

    Args:
        skip_api_checks: Skip UniProt/PubChem/chimera API calls.
        force: Re-process PDBs already in the aggregate log.

    Returns list of :class:`AggregateResult` for each processed PDB.
    """
    try:
        cfg = get_config()
    except Exception as exc:
        logger.error("Failed to initialize workspace config: %s", exc)
        return []

    # Cache initialization (Review 5 H4)
    try:
        validation_cache = ValidationCache(cfg.cache_dir / "id_validation_cache.json")
        sequence_cache = SequenceCache(cfg.cache_dir / "uniprot_sequence_cache.json")
    except Exception as exc:
        logger.error("Failed to initialize caches: %s", exc)
        return []

    if force:
        # Get ALL PDB IDs with AI results (bypass aggregate log)
        ai_dir = cfg.ai_results_dir
        if not ai_dir.is_dir():
            return []
        pending = sorted(
            d.name for d in ai_dir.iterdir() if d.is_dir() and list(d.glob("run_*.json"))
        )
    else:
        pending = get_pending_pdb_ids()

    from tqdm import tqdm

    results: list[AggregateResult] = []
    for pdb_id in tqdm(pending, desc="Progress"):
        try:
            result = aggregate_pdb(
                pdb_id,
                skip_api_checks=skip_api_checks,
                validation_cache=validation_cache,
                sequence_cache=sequence_cache,
            )
            results.append(result)
            status = AGG_STATUS_COMPLETED if result.success else AGG_STATUS_FAILED
            _update_aggregate_log(pdb_id, status)
        except Exception as exc:
            logger.error("[%s] Critical failure: %s", pdb_id, exc)
            results.append(AggregateResult(pdb_id=pdb_id, success=False, error=str(exc)))
            _update_aggregate_log(pdb_id, AGG_STATUS_FAILED)

    # Save caches (best-effort, after output commit)
    try:
        validation_cache.save()
    except OSError as exc:
        logger.warning("Failed to save validation cache: %s", exc)
    try:
        sequence_cache.save()
    except OSError as exc:
        logger.warning("Failed to save sequence cache: %s", exc)

    return results
