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
