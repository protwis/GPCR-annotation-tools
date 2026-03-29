"""Entry point for `python -m gpcr_tools`."""

import argparse
import sys


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="gpcr-tools",
        description="GPCR Annotation Tools — Human-in-the-loop curation suite.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: csv-generator
    csv_parser = subparsers.add_parser(
        "csv-generator",
        help="Interactive CSV generator for expert review of AI annotations.",
    )
    csv_parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: target a specific PDB ID instead of processing all pending.",
    )

    args = parser.parse_args()

    if args.command == "csv-generator":
        from gpcr_tools.csv_generator.app import main

        main(target_pdb=args.pdb_id)
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
