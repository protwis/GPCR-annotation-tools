"""Entry point for ``python -m gpcr_tools`` and the ``gpcr-tools`` console script."""

from __future__ import annotations

import argparse
import logging
import sys


def cli() -> None:
    # Surface our own INFO progress (pipeline stages, fetch/annotate/aggregate)
    # while keeping third-party libraries quiet. Without this, logging defaults
    # to WARNING and all progress messages are swallowed.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("gpcr_tools").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        prog="gpcr-tools",
        description="GPCR Annotation Tools — Human-in-the-loop curation suite.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-workspace ---------------------------------------------------
    subparsers.add_parser(
        "init-workspace",
        help="Initialize a workspace with the v3.1 directory contract.",
    )

    # curate (alias for the current csv-generator workflow) ------------
    curate_parser = subparsers.add_parser(
        "curate",
        help="Interactive CSV generator for expert review of AI annotations.",
    )
    curate_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: target a specific PDB ID instead of processing all pending.",
    )
    curate_parser.add_argument(
        "--auto-accept",
        action="store_true",
        default=False,
        help="Run non-interactively with accept-all behavior (for CI smoke tests).",
    )

    # fetch ---------------------------------------------------------------
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download PDB metadata from RCSB and enrich with UniProt/PubChem data.",
    )
    fetch_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: fetch a specific PDB ID.",
    )
    fetch_parser.add_argument(
        "--targets",
        default=None,
        metavar="FILE",
        help="Override: read PDB IDs from this file instead of targets.txt.",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-fetch even if output files already exist.",
    )

    # fetch-papers -----------------------------------------------------
    fp_parser = subparsers.add_parser(
        "fetch-papers",
        help="Download open-access papers for enriched PDB entries.",
    )
    fp_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: fetch paper for a specific PDB ID.",
    )
    fp_parser.add_argument(
        "--targets",
        default=None,
        metavar="FILE",
        help="Override: read PDB IDs from this file.",
    )
    fp_parser.add_argument(
        "--auto-only",
        action="store_true",
        default=False,
        help="Skip watch mode for paywalled papers (for CI/scripting).",
    )
    fp_parser.add_argument(
        "--watch-only",
        action="store_true",
        default=False,
        help=(
            "Skip the auto-download retry and go straight to watch mode for the "
            "papers already marked paywalled in the download log. Watches ALL "
            "paywalled entries; any pdb_id / --targets argument is ignored."
        ),
    )
    fp_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if PDF already exists.",
    )

    # annotate ---------------------------------------------------------
    ann_parser = subparsers.add_parser(
        "annotate",
        help="Run Gemini AI annotation (single + batch modes).",
    )
    ann_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: annotate a specific PDB ID.",
    )
    ann_parser.add_argument(
        "--targets",
        default=None,
        metavar="FILE",
        help="Override: read PDB IDs from this file.",
    )
    ann_parser.add_argument(
        "--prompt",
        default=None,
        metavar="FILE",
        help="Path to system prompt file.",
    )

    from gpcr_tools.config import GEMINI_DEFAULT_RUNS, GEMINI_MODEL_NAME

    ann_parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help=(
            f"Gemini model name (default: {GEMINI_MODEL_NAME}). "
            "Can also be set via GPCR_GEMINI_MODEL env var. "
            "CLI flag takes highest priority."
        ),
    )

    def _positive_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"--runs must be >= 1, got {ivalue}")
        return ivalue

    ann_parser.add_argument(
        "--runs",
        type=_positive_int,
        default=GEMINI_DEFAULT_RUNS,
        help=f"Number of annotation runs per PDB (default: {GEMINI_DEFAULT_RUNS}).",
    )
    ann_parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Use Gemini Batch API instead of single calls.",
    )
    ann_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Sampling temperature for generation. Omit to use the model's "
            "default (no override sent, in either single or batch mode)."
        ),
    )
    ann_parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help=(
            "Reasoning depth for generation. Omit to use the model's default "
            "(high; no override sent, in either single or batch mode)."
        ),
    )
    ann_parser.add_argument(
        "--check-batch",
        action="store_true",
        default=False,
        help="Poll current batch status.",
    )
    ann_parser.add_argument(
        "--recover",
        action="store_true",
        default=False,
        help="Re-process raw JSONL output files.",
    )

    # detect -----------------------------------------------------------
    detect_parser = subparsers.add_parser(
        "detect",
        help="Pre-annotation structural detection: flag hard cases before annotate.",
    )
    detect_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: detect on a specific PDB ID instead of all enriched.",
    )
    detect_parser.add_argument(
        "--skip-api-checks",
        action="store_true",
        default=False,
        help="Skip sequence-based detectors that need UniProt reference fetches.",
    )
    detect_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Recompute every detect output, including ones already complete "
        "(default tops up: skip complete outputs, redo only missing or degraded ones).",
    )

    # aggregate --------------------------------------------------------
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate multi-run AI results and validate against PDB metadata.",
    )
    agg_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: aggregate a specific PDB ID instead of all pending.",
    )
    agg_parser.add_argument(
        "--skip-api-checks",
        action="store_true",
        default=False,
        help="Skip UniProt/PubChem/chimera API validation calls.",
    )
    # --force (reprocess everything) and --retry-unavailable (reprocess only the
    # API-abstention subset) are mutually exclusive scopes.
    agg_scope = agg_parser.add_mutually_exclusive_group()
    agg_scope.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process PDBs already in the aggregate log.",
    )
    agg_scope.add_argument(
        "--retry-unavailable",
        action="store_true",
        default=False,
        help=(
            "Re-aggregate only PDBs whose last run recorded a transient API "
            "failure ([API_UNAVAILABLE]); cached results are reused, so only the "
            "failed lookups are retried. Incompatible with --skip-api-checks."
        ),
    )

    # csv-generator (kept temporarily for backward compat) -------------
    csv_parser = subparsers.add_parser(
        "csv-generator",
        help="(deprecated) Use 'curate' instead.",
    )
    csv_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: target a specific PDB ID instead of processing all pending.",
    )

    # pipeline ---------------------------------------------------------
    pipe_parser = subparsers.add_parser(
        "pipeline",
        help="Run fetch -> fetch-papers -> detect -> annotate -> aggregate in dependency order.",
    )
    pipe_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: run the pipeline for a specific PDB ID.",
    )
    pipe_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the planned stage sequence without running anything.",
    )
    pipe_parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Annotate via the Batch API (the pipeline stops after submission).",
    )
    pipe_parser.add_argument(
        "--runs",
        type=_positive_int,
        default=GEMINI_DEFAULT_RUNS,
        help=f"Number of annotation runs per PDB (default: {GEMINI_DEFAULT_RUNS}).",
    )
    pipe_parser.add_argument(
        "--skip-fetch-papers",
        action="store_true",
        default=False,
        help="Skip the paper-download stage.",
    )
    pipe_parser.add_argument(
        "--skip-api-checks",
        action="store_true",
        default=False,
        help="Skip UniProt/PubChem/chimera validation in the aggregate stage.",
    )

    # migrate-papers ---------------------------------------------------
    subparsers.add_parser(
        "migrate-papers",
        help=(
            "One-time, idempotent consolidation of per-PDB paper PDFs into "
            "DOI-named canonical files (safe to re-run; never deletes a source "
            "until its canonical exists and validates)."
        ),
    )

    # report -----------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="Print an operational report over pipeline outputs.",
    )
    report_parser.add_argument(
        "kind",
        choices=["pdf-coverage", "full-audit", "tail-analysis", "run-manifest"],
        help=(
            "pdf-coverage: paper-PDF outcomes; "
            "full-audit: validation warnings + chimera conflicts across PDBs; "
            "tail-analysis: G-protein chimera score distribution; "
            "run-manifest: write a full run record (output/run_manifest.{json,md})."
        ),
    )

    args = parser.parse_args()

    if args.command == "init-workspace":
        from gpcr_tools.workspace import init_workspace

        init_workspace()

    elif args.command == "fetch":
        from gpcr_tools.fetcher.runner import run_fetch

        run_fetch(
            pdb_id=args.pdb_id,
            targets_file=args.targets,
            force=args.force,
        )

    elif args.command == "fetch-papers":
        if args.auto_only and args.watch_only:
            print(
                "Error: --auto-only and --watch-only are mutually exclusive.",
                file=sys.stderr,
            )
            sys.exit(2)
        from gpcr_tools.papers.runner import run_fetch_papers

        run_fetch_papers(
            pdb_id=args.pdb_id,
            targets_file=args.targets,
            auto_only=args.auto_only,
            watch_only=args.watch_only,
            force=args.force,
        )

    elif args.command == "annotate":
        if args.check_batch:
            from gpcr_tools.annotator.runner import check_batch_status

            check_batch_status()
        elif args.recover:
            from gpcr_tools.annotator.runner import recover_batch

            recover_batch()
        else:
            from gpcr_tools.annotator.runner import run_annotation_stage

            try:
                run_annotation_stage(
                    pdb_id=args.pdb_id,
                    targets_file=args.targets,
                    prompt_file=args.prompt,
                    model=args.model,
                    num_runs=args.runs,
                    batch=args.batch,
                    temperature=args.temperature,
                    thinking_level=args.thinking_level,
                )
            except FileNotFoundError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

    elif args.command == "csv-generator":
        import warnings

        warnings.warn(
            "'gpcr-tools csv-generator' is deprecated. Use 'gpcr-tools curate' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        print(
            "WARNING: 'gpcr-tools csv-generator' is deprecated. Use 'gpcr-tools curate' instead.",
            file=sys.stderr,
        )
        from gpcr_tools.csv_generator.app import main

        main(target_pdb=args.pdb_id, auto_accept=False)

    elif args.command == "detect":
        from gpcr_tools.detector.stage import run_detect_stage

        summary = run_detect_stage(
            args.pdb_id, skip_api_checks=args.skip_api_checks, force=args.force
        )
        total = sum(summary.values())
        print(f"Detect complete: {len(summary)} PDB(s), {total} signal(s).")

    elif args.command == "aggregate":
        from gpcr_tools.aggregator.runner import aggregate_all, aggregate_pdb

        if args.retry_unavailable and args.skip_api_checks:
            print(
                "Error: --retry-unavailable cannot be combined with --skip-api-checks "
                "(the retry re-runs the API checks that --skip-api-checks disables).",
                file=sys.stderr,
            )
            sys.exit(2)

        if args.pdb_id:
            result = aggregate_pdb(
                args.pdb_id,
                skip_api_checks=args.skip_api_checks,
            )
            if result.success:
                print(f"Aggregated {args.pdb_id} -> {result.aggregated_path}")
            else:
                print(f"Failed {args.pdb_id}: {result.error}", file=sys.stderr)
                sys.exit(1)
        else:
            results = aggregate_all(
                skip_api_checks=args.skip_api_checks,
                force=args.force,
                retry_unavailable=args.retry_unavailable,
            )
            ok = sum(1 for r in results if r.success)
            fail = sum(1 for r in results if not r.success)
            print(f"Aggregation complete: {ok} succeeded, {fail} failed.")
            if fail > 0:
                sys.exit(1)

    elif args.command == "curate":
        from gpcr_tools.csv_generator.app import main

        auto_accept = getattr(args, "auto_accept", False)
        main(target_pdb=args.pdb_id, auto_accept=auto_accept)

    elif args.command == "pipeline":
        from gpcr_tools.pipeline import run_pipeline

        run_pipeline(
            pdb_id=args.pdb_id,
            dry_run=args.dry_run,
            batch=args.batch,
            num_runs=args.runs,
            skip_fetch_papers=args.skip_fetch_papers,
            skip_api_checks=args.skip_api_checks,
        )

    elif args.command == "migrate-papers":
        from gpcr_tools.papers.migrate import migrate_papers_to_doi_storage

        migration = migrate_papers_to_doi_storage()
        print(
            f"Paper migration complete: {migration.consolidated} consolidated, "
            f"{migration.redundant_removed} redundant removed, "
            f"{migration.no_doi_kept} no-DOI kept, {migration.skipped_invalid} invalid skipped."
        )

    elif args.command == "report":
        from gpcr_tools import reports

        report_funcs = {
            "pdf-coverage": reports.report_pdf_coverage,
            "full-audit": reports.report_full_audit,
            "tail-analysis": reports.report_tail_analysis,
            "run-manifest": reports.report_run_manifest,
        }
        print(report_funcs[args.kind]())

    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
