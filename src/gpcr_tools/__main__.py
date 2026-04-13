"""Entry point for ``python -m gpcr_tools`` and the ``gpcr-tools`` console script."""

from __future__ import annotations

import argparse
import sys


def cli() -> None:
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
    agg_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process PDBs already in the aggregate log.",
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
        from gpcr_tools.papers.runner import run_fetch_papers

        run_fetch_papers(
            pdb_id=args.pdb_id,
            targets_file=args.targets,
            auto_only=args.auto_only,
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
            import json
            from pathlib import Path

            from gpcr_tools.config import get_config

            cfg = get_config()

            # Resolve target PDB IDs
            if args.pdb_id:
                pdb_ids = [args.pdb_id.upper()]
            elif args.targets:
                from gpcr_tools.fetcher.targets import read_targets

                pdb_ids = read_targets(Path(args.targets))
            else:
                # Auto-discover: enriched PDBs missing complete ai_results
                enriched_pdbs = {p.stem.upper() for p in cfg.enriched_dir.glob("*.json")}
                done_pdbs: set[str] = set()
                if cfg.ai_results_dir.exists():
                    for d in cfg.ai_results_dir.iterdir():
                        if d.is_dir():
                            completed = sum(
                                1 for n in range(1, args.runs + 1) if (d / f"run_{n}.json").exists()
                            )
                            if completed >= args.runs:
                                done_pdbs.add(d.name.upper())
                pdb_ids = sorted(enriched_pdbs - done_pdbs)

            # Resolve prompt text
            if args.prompt:
                prompt_text = Path(args.prompt).read_text(encoding="utf-8")
            else:
                if not cfg.default_prompt_file.exists():
                    print(
                        f"Error: default prompt file not found at {cfg.default_prompt_file}\n"
                        "Please create it or use --prompt to specify one.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                prompt_text = cfg.default_prompt_file.read_text(encoding="utf-8")

            # Resolve model name: --model flag > GPCR_GEMINI_MODEL env > config default
            from gpcr_tools.config import get_gemini_model_name

            model_name = args.model or get_gemini_model_name()

            if args.batch:
                from gpcr_tools.annotator.runner import build_and_submit_batch

                build_and_submit_batch(
                    pdb_ids, prompt_text, num_runs=args.runs, model_name=model_name
                )
            else:
                from gpcr_tools.annotator.runner import run_single_pdb

                for pdb_id in pdb_ids:
                    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"
                    if not enriched_path.exists():
                        print(
                            f"Skipping {pdb_id}: no enriched data at {enriched_path}",
                            file=sys.stderr,
                        )
                        continue

                    with open(enriched_path, encoding="utf-8") as fh:
                        enriched_data = json.load(fh)

                    # Find PDF (required for annotation)
                    pdf_path = cfg.papers_dir / f"{pdb_id}.pdf"
                    if not pdf_path.exists():
                        print(
                            f"Skipping {pdb_id}: no PDF at {pdf_path}",
                            file=sys.stderr,
                        )
                        continue

                    run_single_pdb(
                        pdb_id=pdb_id,
                        enriched_data=enriched_data,
                        prompt_text=prompt_text,
                        pdf_path=pdf_path,
                        num_runs=args.runs,
                        model_name=model_name,
                    )

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

    elif args.command == "aggregate":
        from gpcr_tools.aggregator.runner import aggregate_all, aggregate_pdb

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

    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
