"""End-to-end pipeline orchestration.

Runs the stages in dependency order -- fetch -> fetch-papers -> detect ->
annotate -> aggregate -- with prerequisite checks between them.  Stops before the
interactive ``curate`` step.  Each stage logs what it does; a dry run prints the
planned sequence without executing anything.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_files(directory: Path, pattern: str) -> bool:
    return directory.exists() and any(directory.glob(pattern))


def run_pipeline(
    pdb_id: str | None = None,
    *,
    dry_run: bool = False,
    batch: bool = False,
    num_runs: int | None = None,
    skip_fetch_papers: bool = False,
    skip_api_checks: bool = False,
) -> None:
    """Run fetch -> fetch-papers -> annotate -> aggregate in order.

    Each stage is gated on the prior stage's output. In ``batch`` mode the
    pipeline stops after submitting the batch (results arrive asynchronously);
    the curator runs ``annotate --check-batch`` then ``aggregate`` later.
    """
    from gpcr_tools.config import GEMINI_DEFAULT_RUNS, get_config

    cfg = get_config()
    runs = num_runs if num_runs is not None else GEMINI_DEFAULT_RUNS

    # Fail fast on a stale / missing storage contract BEFORE any expensive
    # stage runs (the AI annotate stage in particular). Previously only the
    # interactive curate step validated the contract, so a mismatch surfaced
    # only after the costly work had already completed.
    if not dry_run:
        from gpcr_tools.workspace import validate_contract

        validate_contract(cfg)

    stages = ["fetch"]
    if not skip_fetch_papers:
        stages.append("fetch-papers")
    stages += ["detect", "annotate", "aggregate"]

    if dry_run:
        target = pdb_id or "all targets (targets.txt / auto-discovery)"
        # In batch mode the pipeline submits the batch and stops; aggregate is a
        # later, separate step — so don't list it in the planned sequence.
        planned = stages[:-1] if batch else stages
        suffix = (
            "  (batch: detect runs; stops after annotate submission; aggregate runs later)"
            if batch
            else ""
        )
        logger.info("[pipeline] dry run -- would run: %s%s", " -> ".join(planned), suffix)
        logger.info(
            "[pipeline] target=%s, annotate mode=%s, runs=%d",
            target,
            "batch" if batch else "single",
            runs,
        )
        return

    # 1. fetch -------------------------------------------------------------
    logger.info("[pipeline] stage: fetch")
    from gpcr_tools.fetcher.runner import run_fetch

    run_fetch(pdb_id=pdb_id, targets_file=None, force=False)

    if not _has_files(cfg.enriched_dir, "*.json"):
        logger.warning("[pipeline] no enriched data after fetch -- stopping.")
        return

    # 2. fetch-papers ------------------------------------------------------
    if not skip_fetch_papers:
        logger.info("[pipeline] stage: fetch-papers")
        from gpcr_tools.papers.runner import run_fetch_papers

        run_fetch_papers(pdb_id=pdb_id, targets_file=None, auto_only=True, force=False)

    # 3. detect ------------------------------------------------------------
    logger.info("[pipeline] stage: detect")
    from gpcr_tools.detector.stage import run_detect_stage

    run_detect_stage(pdb_id=pdb_id, skip_api_checks=skip_api_checks)

    # 4. annotate ----------------------------------------------------------
    logger.info("[pipeline] stage: annotate (%s)", "batch" if batch else "single")
    from gpcr_tools.annotator.runner import run_annotation_stage

    try:
        run_annotation_stage(pdb_id=pdb_id, num_runs=runs, batch=batch)
    except FileNotFoundError as exc:
        logger.error("[pipeline] annotate aborted: %s", exc)
        return

    if batch:
        logger.info(
            "[pipeline] batch submitted -- run 'gpcr-tools annotate --check-batch' to retrieve "
            "results, then 'gpcr-tools aggregate'. Stopping here."
        )
        return

    # 5. aggregate ---------------------------------------------------------
    logger.info("[pipeline] stage: aggregate")
    # Aggregate only the requested PDB when one was given — mirroring the
    # standalone `aggregate <PDB>` command — instead of sweeping every pending
    # PDB in the workspace (which would also mark unrelated ones processed).
    if pdb_id:
        from gpcr_tools.aggregator.runner import aggregate_pdb

        result = aggregate_pdb(pdb_id, skip_api_checks=skip_api_checks)
        ok, fail = (1, 0) if result.success else (0, 1)
    else:
        from gpcr_tools.aggregator.runner import aggregate_all

        results = aggregate_all(skip_api_checks=skip_api_checks, force=False)
        ok = sum(1 for r in results if r.success)
        fail = sum(1 for r in results if not r.success)
    logger.info("[pipeline] aggregate complete: %d succeeded, %d failed.", ok, fail)
    logger.info("[pipeline] done -- next: 'gpcr-tools curate' to review.")
