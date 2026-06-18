"""Orchestration layer — wire all aggregation + validation components.

``aggregate_pdb()`` runs the full pipeline for a single PDB ID:
    AI runs → voting → best run → deepcopy → ground truth → validators →
    discrepancies → integrity → chimera → validation report → atomic writes.

``aggregate_all()`` iterates pending PDBs with per-PDB error isolation.

Conventions:
    - Atomic writes: all output files are written to temp files first, then
      ``os.replace``-d together after every write succeeds (``try...finally`` cleanup).
    - Truthiness: ``if enriched is None:`` — NOT ``if not enriched:``.
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

from gpcr_tools.aggregator.ai_results_loader import (
    get_pending_pdb_ids,
    load_ai_runs,
    pdb_has_runs,
)
from gpcr_tools.aggregator.enriched_loader import enriched_is_incomplete, load_enriched_data
from gpcr_tools.aggregator.ground_truth import inject_ground_truth
from gpcr_tools.aggregator.voting import (
    extract_ai_g_protein,
    find_discrepancies,
    flag_low_confidence_consensus,
    get_majority_votes,
    select_best_run,
)
from gpcr_tools.config import (
    A5_SUBTYPE_FAMILY,
    AGG_STATUS_COMPLETED,
    AGG_STATUS_FAILED,
    ALERT_PREFIX_ALGO_WARNING,
    ALERT_PREFIX_ALPHA5_GRAFT,
    ALERT_PREFIX_API_UNAVAILABLE,
    ALERT_PREFIX_CHIMERIC_REVIEW,
    ALERT_PREFIX_HALLUCINATION,
    ALERT_PREFIX_TIE_BREAKER_ALIGNED,
    ALERT_PREFIX_TIE_BREAKER_OVERRIDE,
    ALERT_PREFIX_UNRECOGNISED_G_ALPHA,
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_SKIPPED,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_SUBTYPE_LOW_CONFIDENCE,
    EMPTY_VALUES,
    FULL_G_ALPHA_CANDIDATES,
    LOW_CONFIDENCE_LEVELS,
    POLYMER_FEATURES_CACHE_NAME,
    get_config,
)
from gpcr_tools.detector.signals import (
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_COUPLING_PROTOMER,
    to_critical_warnings,
)
from gpcr_tools.detector.stage import load_detect_signals
from gpcr_tools.fetcher.cache import JsonCache
from gpcr_tools.validator.api_clients import SynonymCache
from gpcr_tools.validator.cache import (
    PolymerFeaturesCache,
    SequenceCache,
    ValidationCache,
)
from gpcr_tools.validator.chimera import get_chimera_analysis
from gpcr_tools.validator.integrity_checker import validate_all
from gpcr_tools.validator.ligand_validator import validate_and_enrich_ligands
from gpcr_tools.validator.oligomer import (
    analyze_oligomer,
    detect_crystallization_fusions,
    reconcile_missed_polymers,
)
from gpcr_tools.validator.receptor_validator import validate_receptor_identity

logger = logging.getLogger(__name__)

# Detect REVIEW signals of these kinds are NOT re-surfaced as critical warnings
# here: the aggregator re-derives the G-protein review from its own alpha5
# analysis below, with finer severity tuning (low-confidence -> note, not a
# blocker). Routing the detect copy too would both duplicate the warning and
# override that tuning. (The deferred chimera-logic consolidation will collapse
# the two into a single source.)
_AGGREGATOR_OWNED_REVIEW_KINDS = frozenset({SIGNAL_CHIMERIC_GPROTEIN})


def _coupling_protomer(pdb_id: str) -> str | None:
    """The geometric G-protein-coupling protomer chain from the detect sidecar, if any.

    Returns ``None`` when the detect stage did not run, found no G protein, or could
    not resolve a single protomer -- in which case primary selection falls back to the
    lower ranks.
    """
    for signal in load_detect_signals(pdb_id):
        if signal.kind == SIGNAL_COUPLING_PROTOMER:
            chain = signal.payload.get("coupling_chain")
            return chain if isinstance(chain, str) else None
    return None


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

# The recognised G-alpha subunit slugs (the values of the curated human G-alpha
# roster). A specific alpha-subunit slug outside this set is not a known G-alpha
# candidate and must reach a human. Built once at import.
_RECOGNISED_G_ALPHA_SLUGS = frozenset(FULL_G_ALPHA_CANDIDATES.values())


def _warn_on_unrecognised_g_alpha(best_run_data: dict[str, Any]) -> list[str]:
    """Flag (for the curator) a G-protein alpha subunit named with a specific slug
    that is NOT in the curated G-alpha candidate set.

    The candidate roster is alpha-specific, so this checks the alpha subunit only;
    beta/gamma carry their own slugs and are out of scope. An honest abstention
    (missing / empty / 'unknown' name) is never flagged -- only a specific,
    off-roster slug, which is the signature of an invented subtype/species. The
    warning is a critical warning so it disables one-click accept-all for the PDB.
    """
    slug = extract_ai_g_protein(best_run_data)
    if not isinstance(slug, str):
        return []
    normalised = slug.strip().lower()
    if normalised in EMPTY_VALUES or normalised == "unknown":
        return []
    if normalised in _RECOGNISED_G_ALPHA_SLUGS:
        return []
    return [
        f"{ALERT_PREFIX_UNRECOGNISED_G_ALPHA} at "
        f"'signaling_partners.g_protein.alpha_subunit': G-protein alpha subunit "
        f"'{slug}' is not a recognised G-alpha candidate (off the curated human "
        f"G-alpha set); verify the subtype/species against the paper."
    ]


def _build_validation_report(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
    all_warnings: list[str],
    chimera_result: dict[str, Any],
    validation_cache: ValidationCache | None,
) -> dict[str, Any]:
    """Assemble the validation report from all warning sources.

    None-safe reads throughout (``chimera_result.get("score") or 0``) and all
    status comparisons go through the shared constants. The G-alpha sequence
    finding is classified against the model's claim: family agreement, subtype
    resolution, and routing of an indistinguishable subtype to human review.
    """
    report: dict[str, Any] = {
        "critical_warnings": list(all_warnings),
        "algo_conflicts": [],
        "detector_notes": [],
        "chimera_score": chimera_result.get("score") or 0,
        "chimera_status": chimera_result.get("status") or CHIMERA_STATUS_SKIPPED,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    # Integrity checks (ghost chain, fake UniProt/PubChem, ghost ligand, method)
    integrity_warnings = validate_all(pdb_id, best_run_data, enriched_entry, cache=validation_cache)
    report["critical_warnings"].extend(integrity_warnings)

    # Candidate-membership backstop: a specific G-alpha slug outside the curated
    # roster reaches a human. Deterministic, independent of the alpha5 API check,
    # so it fires even under --skip-api-checks.
    report["critical_warnings"].extend(_warn_on_unrecognised_g_alpha(best_run_data))

    # Non-GPCR polymer chains present in the structure but never annotated by the
    # model (the oligomer missed-protomer check covers GPCR chains only).
    report["critical_warnings"].extend(reconcile_missed_polymers(enriched_entry, best_run_data))

    # Detect-stage REVIEW signals -> curator critical warnings. This is the
    # production consumer of the detect review route (advisory signals already
    # went into the annotation prompt upstream); without it, a detector's review
    # signal -- and any signal whose severity failed safe to review -- never
    # reaches a human. Kinds the aggregator re-derives itself are excluded above.
    detect_reviews = [
        s for s in load_detect_signals(pdb_id) if s.kind not in _AGGREGATOR_OWNED_REVIEW_KINDS
    ]
    report["critical_warnings"].extend(to_critical_warnings(detect_reviews))

    # Receptor-side crystallization fusions (BRIL / T4 lysozyme) -- advisory,
    # non-blocking: recorded for the curator, does not gate accept-all.
    report["detector_notes"].extend(detect_crystallization_fusions(enriched_entry))

    # Chimeric G-protein review is driven by the deterministic alpha5 analysis,
    # NOT the model's optional is_chimeric flag (which the model can silently
    # omit -> a false negative that skips review). The model flag is kept only as
    # a fallback for when the alpha5 was INCONCLUSIVE -- it never ran
    # (--skip-api-checks), or ran but could not decide (too short / no reference
    # comparisons / error). In those cases there is no deterministic ruling, so a
    # self-declared chimera must still reach a human. When the alpha5 reached a
    # conclusion the fallback is suppressed: SUCCESS -> the alpha5 routing below
    # owns the review (it forces review only when it genuinely cannot resolve the
    # subtype, and stays silent once the identity is settled); NO_G_PROTEIN -> the
    # hallucination branch below owns it (the algorithm positively found none).
    status = chimera_result.get("status") or CHIMERA_STATUS_SKIPPED
    ai_uniprot = extract_ai_g_protein(best_run_data)
    g_protein = (best_run_data.get("signaling_partners") or {}).get("g_protein") or {}
    alpha5_inconclusive = status not in (CHIMERA_STATUS_SUCCESS, CHIMERA_STATUS_NO_G_PROTEIN)
    if alpha5_inconclusive and g_protein.get("is_chimeric") is True:
        report["critical_warnings"].append(
            f"{ALERT_PREFIX_CHIMERIC_REVIEW} at "
            f"'signaling_partners.g_protein.alpha_subunit': chimeric G-protein — "
            f"confirm the alpha-subunit identity manually."
        )

    # Compare the alpha5 sequence finding against the model's G-alpha claim.

    if status == CHIMERA_STATUS_SUCCESS:
        family = chimera_result.get("family")
        subtype = chimera_result.get("subtype")
        resolution = chimera_result.get("subtype_resolution")
        candidate_set = chimera_result.get("candidate_set") or []
        a5_tail = chimera_result.get("a5_tail") or "N/A"
        ai_family = A5_SUBTYPE_FAMILY.get(ai_uniprot) if ai_uniprot else None

        if subtype is not None:
            # The alpha5 resolves to a single subtype.
            if ai_uniprot and ai_uniprot != subtype:
                report["algo_conflicts"].append(
                    f"{ALERT_PREFIX_TIE_BREAKER_OVERRIDE} at 'chimera_analysis': "
                    f"alpha5 '{a5_tail}' resolves G-alpha to '{subtype}', but the "
                    f"model chose '{ai_uniprot}'. Confirm the identity."
                )
            else:
                report["detector_notes"].append(
                    f"{ALERT_PREFIX_TIE_BREAKER_ALIGNED} at 'chimera_analysis': "
                    f"alpha5 '{a5_tail}' resolves G-alpha to '{subtype}'."
                )
        elif resolution == CHIMERA_SUBTYPE_LOW_CONFIDENCE:
            report["detector_notes"].append(
                f"{ALERT_PREFIX_ALGO_WARNING} at 'chimera_analysis': "
                f"alpha5 match is weak (best window score "
                f"{chimera_result.get('score') or 0}); G-alpha identity unverified."
            )
        elif ai_family and family and ai_family != family:
            # The model's family disagrees with the alpha5 coupling family.
            report["algo_conflicts"].append(
                f"{ALERT_PREFIX_TIE_BREAKER_OVERRIDE} at 'chimera_analysis': "
                f"alpha5 '{a5_tail}' indicates the {family} family, but the model "
                f"chose '{ai_uniprot}' ({ai_family}). Confirm the G-alpha identity."
            )
        elif family:
            # Family is confident but the subtype cannot be told apart by the
            # alpha5; route the subtype to a human rather than forcing a member.
            members = ", ".join(candidate_set) or "indistinguishable subtypes"
            report["critical_warnings"].append(
                f"{ALERT_PREFIX_CHIMERIC_REVIEW} at "
                f"'signaling_partners.g_protein.alpha_subunit': alpha5 confirms the "
                f"{family} family but cannot distinguish the subtype ({members}); "
                f"confirm manually."
            )
        else:
            # The best match spans more than one coupling family or an
            # unrecognised slug, so even the family is undetermined. Never leave
            # this silent: surface it as a conflict for manual resolution.
            members = ", ".join(candidate_set) or "no recognised subtype"
            report["algo_conflicts"].append(
                f"{ALERT_PREFIX_ALGO_WARNING} at 'chimera_analysis': "
                f"alpha5 '{a5_tail}' does not map to a single coupling family "
                f"({members}); G-alpha identity cannot be determined automatically."
            )

        # alpha5-graft: the engineered scaffold differs from the functional
        # alpha5 (~6% of G-alpha structures). Record the backbone for provenance
        # (export still collapses to the functional identity) and note it --
        # informational, not a conflict: the alpha5 helix is the principal
        # receptor-coupling determinant, so it defines the G-alpha identity.
        if chimera_result.get("is_alpha5_graft"):
            backbone_slug = chimera_result.get("backbone_slug")
            backbone_family = chimera_result.get("backbone_family")
            g_block = (best_run_data.get("signaling_partners") or {}).get("g_protein")
            if isinstance(g_block, dict):
                g_block["chimera_backbone"] = f"{backbone_slug} ({backbone_family} scaffold)"
            report["detector_notes"].append(
                f"{ALERT_PREFIX_ALPHA5_GRAFT} at "
                f"'signaling_partners.g_protein.alpha_subunit': alpha5-graft chimera "
                f"-- backbone {backbone_slug} ({backbone_family}), functional alpha5 "
                f"= {family}; identity follows the alpha5 per convention."
            )
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

    All temp files are written first, then ``os.replace``-d (atomic write).
    ``try...finally`` guarantees cleanup on failure.
    """
    cfg = get_config()
    aggregated_path = cfg.aggregated_dir / f"{pdb_id}.json"
    voting_log_dir = cfg.aggregated_dir / "logs"
    validation_dir = cfg.aggregated_dir / "validation_logs"

    aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    voting_log_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # The voting log is written for every PDB, not only when discrepancies exist.
    # A clean PDB's log is an empty list -- an explicit, audit-friendly record that
    # aggregation ran and found no disagreement, distinct from a missing file. Each
    # discrepancy record already carries its per-field vote tallies (all_votes), so
    # the log preserves the vote shape that drove every flagged field. The payload
    # stays a list of records to honour the curate loader's contract; an empty list
    # yields an empty controversy map downstream, so a clean PDB is not gated.
    voting_log_path = voting_log_dir / f"{pdb_id}_voting_log.json"
    validation_path = validation_dir / f"{pdb_id}_validation.json"

    tmp_paths: list[str] = []
    try:
        # Write all temp files
        tmp_agg = _write_temp_json(aggregated_path.parent, best_run_data)
        tmp_paths.append(tmp_agg)

        tmp_val = _write_temp_json(validation_path.parent, validation_report)
        tmp_paths.append(tmp_val)

        tmp_log = _write_temp_json(voting_log_path.parent, discrepancies)
        tmp_paths.append(tmp_log)

        # Commit all at once
        os.replace(tmp_agg, str(aggregated_path))
        os.replace(tmp_val, str(validation_path))
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

    Uses atomic write.
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
    synonym_cache: SynonymCache | None = None,
    polymer_features_cache: PolymerFeaturesCache | None = None,
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

    The ``validation_cache`` / ``sequence_cache`` / ``synonym_cache`` are supplied by
    the batch ``aggregate_all`` path. Single-PDB CLI invocations pass none, so the
    cache-gated API steps (chimera analysis and the PubChem synonym gate) are skipped
    there, consistent with the existing offline behaviour of the single-PDB path.

    Use ``if enriched is None:`` — NOT ``if not enriched:`` (an empty dict is valid).
    """
    # Fail fast on a stale / missing storage contract before doing real work.
    from gpcr_tools.workspace import validate_contract

    cfg = get_config()
    validate_contract(cfg)

    # The single-PDB paths (CLI `aggregate <PDB>` and the pipeline) pass no
    # polymer-features cache. Lazily construct one from the same cache file the
    # batch path uses, so single-PDB runs get the same fail-open protection (a
    # genuine TM-fetch failure routes to a curator instead of silently counting a
    # peptide as a receptor) and warm the shared cache for later runs. A
    # locally-built cache is saved at the end of this call; a caller-supplied one
    # is owned and saved by that caller (aggregate_all).
    owns_polymer_cache = polymer_features_cache is None
    if owns_polymer_cache:
        polymer_features_cache = PolymerFeaturesCache(cfg.cache_dir / POLYMER_FEATURES_CACHE_NAME)

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
    # if enriched is None — empty dict {} is valid
    if enriched is None:
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="Enriched data not available",
        )

    # An enrichment written during a transient API outage must NOT be aggregated:
    # consuming it would turn an unresolved (but recoverable) field into an
    # affirmative answer (e.g. a missing receptor slug → "no GPCR"). Refuse and
    # signal re-enrichment; a plain re-run of fetch self-heals the record.
    if enriched_is_incomplete(pdb_id):
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="Enrichment incomplete (transient API gap) — re-run fetch to complete it before aggregating",
        )

    try:
        # 5. Inject ground truth (mutates best_run_data)
        inject_ground_truth(pdb_id, best_run_data, enriched)

        # 6. Ligand validation (mutates best_run_data, returns warnings)
        all_warnings: list[str] = []
        ligand_warnings = validate_and_enrich_ligands(
            pdb_id,
            best_run_data,
            enriched,
            synonym_cache=synonym_cache if not skip_api_checks else None,
        )
        all_warnings.extend(ligand_warnings)

        # 7. Receptor validation (mutates best_run_data, returns warnings)
        receptor_warnings = validate_receptor_identity(pdb_id, best_run_data, enriched)
        all_warnings.extend(receptor_warnings)

        # 8. Oligomer analysis (mutates best_run_data — may override chain_id). The
        # detect stage's geometric coupling-protomer signal, when present, selects the
        # primary protomer (the G-protein coupler) over the AI's chain guess.
        analyze_oligomer(
            pdb_id,
            best_run_data,
            enriched,
            coupling_chain=_coupling_protomer(pdb_id),
            polymer_features_cache=polymer_features_cache,
        )

        # Persist a locally-built cache (single-PDB path) so the warmed TM fetch
        # survives to the next run; a caller-supplied cache is saved by the caller.
        if owns_polymer_cache and polymer_features_cache is not None:
            try:
                polymer_features_cache.save()
            except OSError as exc:
                logger.warning("[%s] Failed to save polymer-features cache: %s", pdb_id, exc)

        # 9. Compute discrepancies
        discrepancies = find_discrepancies(best_run_data, majority_votes, all_votes)
        # Also surface unanimous-but-low-confidence decision units for review
        # (consensus is not correctness); dedupe by path so a field already
        # flagged as a real disagreement or near-tie is not duplicated.
        low_conf = flag_low_confidence_consensus(best_run_data, LOW_CONFIDENCE_LEVELS)
        seen_paths = {d["path"] for d in discrepancies}
        discrepancies.extend(d for d in low_conf if d["path"] not in seen_paths)

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


def _pdbs_with_api_unavailable(cfg: Any) -> list[str]:
    """PDB ids whose last validation report recorded a transient API abstention.

    A report carrying an ``[API_UNAVAILABLE]`` warning means a UniProt/PubChem
    lookup abstained (the result was not cached), so re-aggregating that PDB
    retries only those lookups -- the definitive results stay cached.
    """
    validation_dir = cfg.aggregated_dir / "validation_logs"
    if not validation_dir.is_dir():
        return []
    suffix = "_validation.json"
    pdb_ids: list[str] = []
    for path in sorted(validation_dir.glob(f"*{suffix}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        warnings = data.get("critical_warnings") or []
        if any(ALERT_PREFIX_API_UNAVAILABLE in str(w) for w in warnings):
            pdb_ids.append(path.name[: -len(suffix)])
    return pdb_ids


def aggregate_all(
    *,
    skip_api_checks: bool = False,
    force: bool = False,
    retry_unavailable: bool = False,
) -> list[AggregateResult]:
    """Aggregate all pending PDBs with per-PDB error isolation.

    Args:
        skip_api_checks: Skip UniProt/PubChem/chimera API calls.
        force: Re-process PDBs already in the aggregate log.
        retry_unavailable: Re-process only PDBs whose last validation report
            recorded a transient API abstention (``[API_UNAVAILABLE]``).
            Definitive results are cached, so only the failed lookups are retried.
            Cannot be combined with ``skip_api_checks`` (which would skip the very
            checks the retry exists to re-run). It also cannot recover a cache that
            an older build poisoned with a transient failure stored as a definitive
            result -- those read back as a verdict, not an abstention, and must be
            evicted manually.

    Returns list of :class:`AggregateResult` for each processed PDB.
    """
    if retry_unavailable and skip_api_checks:
        raise ValueError(
            "retry_unavailable cannot be combined with skip_api_checks: the retry "
            "exists to re-run the API checks that skip_api_checks disables."
        )

    try:
        cfg = get_config()
    except Exception as exc:
        logger.error("Failed to initialize workspace config: %s", exc)
        return []

    # Fail fast on a stale / missing storage contract before the batch begins.
    from gpcr_tools.workspace import validate_contract

    validate_contract(cfg)

    # Cache initialization
    try:
        validation_cache = ValidationCache(cfg.cache_dir / "id_validation_cache.json")
        sequence_cache = SequenceCache(cfg.cache_dir / "uniprot_sequence_cache.json")
        synonym_cache = JsonCache(cfg.cache_dir / "pubchem_synonym_cache.json")
        polymer_features_cache = PolymerFeaturesCache(cfg.cache_dir / POLYMER_FEATURES_CACHE_NAME)
    except Exception as exc:
        logger.error("Failed to initialize caches: %s", exc)
        return []

    if retry_unavailable:
        # Only re-run PDBs whose AI results are still present, so a stale
        # validation report never downgrades a PDB that can no longer be rebuilt.
        pending = [
            pdb_id
            for pdb_id in _pdbs_with_api_unavailable(cfg)
            if pdb_has_runs(cfg.ai_results_dir / pdb_id)
        ]
        if not pending:
            logger.info(
                "[aggregate] --retry-unavailable: no PDB with AI results recorded a "
                "transient API failure; nothing to retry."
            )
            return []
    elif force:
        # Get ALL PDB IDs with AI results (bypass aggregate log)
        ai_dir = cfg.ai_results_dir
        if not ai_dir.is_dir():
            return []
        pending = sorted(d.name for d in ai_dir.iterdir() if pdb_has_runs(d))
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
                synonym_cache=synonym_cache,
                polymer_features_cache=polymer_features_cache,
            )
            results.append(result)
            status = AGG_STATUS_COMPLETED if result.success else AGG_STATUS_FAILED
            _update_aggregate_log(pdb_id, status)
        except Exception as exc:
            logger.error("[%s] Critical failure: %s", pdb_id, exc)
            results.append(AggregateResult(pdb_id=pdb_id, success=False, error=str(exc)))
            _update_aggregate_log(pdb_id, AGG_STATUS_FAILED)

    # Surface transient API abstentions so they can be retried in isolation.
    unavailable = [
        r.pdb_id for r in results if any(ALERT_PREFIX_API_UNAVAILABLE in str(w) for w in r.warnings)
    ]
    if unavailable:
        logger.warning(
            "[aggregate] %d PDB(s) hit a transient API failure (not cached): %s. "
            "Re-run 'gpcr-tools aggregate --retry-unavailable' to retry only these.",
            len(unavailable),
            ", ".join(unavailable),
        )

    # Save caches (best-effort, after output commit)
    try:
        validation_cache.save()
    except OSError as exc:
        logger.warning("Failed to save validation cache: %s", exc)
    try:
        sequence_cache.save()
    except OSError as exc:
        logger.warning("Failed to save sequence cache: %s", exc)
    try:
        synonym_cache.save()
    except OSError as exc:
        logger.warning("Failed to save synonym cache: %s", exc)
    try:
        polymer_features_cache.save()
    except OSError as exc:
        logger.warning("Failed to save polymer-features cache: %s", exc)

    return results
