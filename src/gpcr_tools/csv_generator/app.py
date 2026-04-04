"""Main application orchestration for the Interactive CSV Generator.

This is the entry point that ties together data loading, the review engine,
CSV writing, and the Rich UI dashboard.
"""

import copy
import sys

from rich import box
from rich.panel import Panel
from rich.pretty import Pretty
from rich.prompt import Confirm, Prompt
from tqdm import tqdm

from gpcr_tools.csv_generator.audit import log_audit_trail
from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv
from gpcr_tools.csv_generator.data_loader import (
    get_pending_pdbs,
    load_pdb_data,
    update_processed_log,
)
from gpcr_tools.csv_generator.exceptions import CsvSchemaMismatchError
from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks
from gpcr_tools.csv_generator.ui import (
    console,
    create_display_copy,
    display_dashboard_header,
    display_oligomer_analysis_panel,
)
from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts


def main(target_pdb: str | None = None, auto_accept: bool = False) -> None:
    """Run the interactive CSV generator dashboard.

    Args:
        target_pdb: If provided, only process this specific PDB ID.
        auto_accept: If True, run non-interactively with accept-all behavior.
            Intended for CI smoke tests — fully deterministic, no prompts.
    """
    from gpcr_tools.workspace import startup_checks

    startup_checks()

    if auto_accept:
        _run_auto_accept(target_pdb)
        return

    if target_pdb:
        target_pdb = target_pdb.upper()
        pending_pdbs = [target_pdb]
        total_count = 1
    else:
        pending_pdbs, skipped_pdbs, total_count = get_pending_pdbs()

        if skipped_pdbs:
            console.print(
                Panel(
                    f"Found [bold yellow]{len(skipped_pdbs)}[/] previously skipped "
                    f"PDB(s): {', '.join(skipped_pdbs[:10])}"
                    + (f" ... and {len(skipped_pdbs) - 10} more" if len(skipped_pdbs) > 10 else ""),
                    title="[bold yellow]Skipped PDBs Detected[/]",
                    border_style="yellow",
                )
            )
            if Confirm.ask(
                f"Re-review these {len(skipped_pdbs)} skipped PDB(s) this session?",
                default=False,
            ):
                pending_pdbs.extend(skipped_pdbs)
                pending_pdbs.sort()
                console.print(
                    f"[green]Merged {len(skipped_pdbs)} skipped PDB(s) into the queue.[/green]"
                )

    console.clear()
    display_dashboard_header(total_count, len(pending_pdbs))

    if not pending_pdbs:
        console.print(Panel("All PDBs processed!", style="success"))
        return

    try:
        for pdb_id in tqdm(pending_pdbs, desc="Progress"):
            console.clear()
            display_dashboard_header(total_count, len(pending_pdbs))
            console.print(Panel(f"Current PDB: [bold magenta]{pdb_id}[/]", box=box.HEAVY))

            main_data, controversies, validation_data = load_pdb_data(pdb_id)
            if not main_data:
                update_processed_log(pdb_id, "failed")
                continue

            console.print(
                Panel(
                    Pretty(create_display_copy(main_data)),
                    title="Initial Summary",
                    border_style="blue",
                )
            )

            # Unified Oligomer Analysis panel (replaces legacy heteromer + 7TM panels)
            display_oligomer_analysis_panel(main_data)

            oligo = main_data.get("oligomer_analysis") or {}
            inject_oligomer_alerts(oligo, validation_data)

            has_crit_issues = validation_data.get("critical_warnings") or validation_data.get(
                "algo_conflicts"
            )
            if has_crit_issues:
                console.print(
                    Panel(
                        "This PDB has VALIDATION ERRORS or ALGORITHM CONFLICTS.\n"
                        "Global 'Accept All' is DISABLED. Check RED sections carefully.",
                        style="bold white on red",
                        box=box.DOUBLE,
                    )
                )

            choices = ["r", "s", "f"]
            prompt_txt = "Select mode ([bold]r[/]eview, [bold]s[/]kip, [bold]f[/]ix issues only"

            if not has_crit_issues and not controversies:
                choices.insert(0, "a")
                prompt_txt += ", [bold]a[/]ccept all"

            prompt_txt += "):"

            mode = Prompt.ask(prompt_txt, choices=choices, default="r").lower()

            if mode == "s":
                update_processed_log(pdb_id, "skipped")
                log_audit_trail(pdb_id, "*", "skip_pdb", "N/A", "SKIPPED")
                continue

            final_data = None
            if mode == "a":
                final_data = main_data
                log_audit_trail(pdb_id, "*", "accept_all_pdb", "N/A", "ACCEPTED")

            if mode == "r":
                final_data = review_toplevel_blocks(
                    pdb_id, copy.deepcopy(main_data), controversies, validation_data
                )
                if final_data is None:
                    raise KeyboardInterrupt

            if mode == "f":
                final_data = review_toplevel_blocks(
                    pdb_id,
                    copy.deepcopy(main_data),
                    controversies,
                    validation_data,
                    fix_mode=True,
                )
                if final_data is None:
                    raise KeyboardInterrupt

            if final_data is not None:
                console.print(
                    Panel(
                        Pretty(create_display_copy(final_data)),
                        title="Final Data",
                        border_style="green",
                    )
                )
                if Confirm.ask("Write to CSV?"):
                    try:
                        append_to_csvs(transform_for_csv(pdb_id, final_data))
                        update_processed_log(pdb_id, "completed")
                        console.print("[green]Saved![/green]")
                    except CsvSchemaMismatchError as e:
                        console.print(
                            Panel(
                                f"[bold red]SCHEMA MISMATCH:[/] {e.message}",
                                border_style="red",
                                box=box.DOUBLE,
                            )
                        )
                        update_processed_log(pdb_id, "failed")
                else:
                    console.print(
                        f"[yellow]PDB {pdb_id} NOT saved. "
                        f"It will reappear as pending in the next session.[/yellow]"
                    )
                    log_audit_trail(pdb_id, "*", "csv_write_declined", "N/A", "DEFERRED")

    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")


def _run_auto_accept(target_pdb: str | None) -> None:
    """Non-interactive accept-all pass for CI smoke tests.

    Processes all pending PDBs (or a single targeted PDB) without any
    interactive prompts.  Previously-skipped PDBs are NOT re-included
    unless explicitly targeted via *target_pdb*.
    """
    if target_pdb:
        target_pdb = target_pdb.upper()
        pending_pdbs = [target_pdb]
    else:
        pending_pdbs, _skipped, _total = get_pending_pdbs()

    if not pending_pdbs:
        print("auto-accept: nothing to process", file=sys.stderr)
        return

    for pdb_id in pending_pdbs:
        main_data, _controversies, _validation = load_pdb_data(pdb_id)
        if not main_data:
            update_processed_log(pdb_id, "failed")
            continue

        log_audit_trail(pdb_id, "*", "auto_accept", "N/A", "ACCEPTED")
        try:
            append_to_csvs(transform_for_csv(pdb_id, main_data))
            update_processed_log(pdb_id, "completed")
            print(f"auto-accept: processed {pdb_id}", file=sys.stderr)
        except CsvSchemaMismatchError as e:
            print(f"ERROR: {e.message}", file=sys.stderr)
            print(f"auto-accept: FAILED {pdb_id} (schema mismatch)", file=sys.stderr)
            update_processed_log(pdb_id, "failed")


def cli_entry() -> None:
    """Deprecated CLI entry point kept for backward compatibility.

    Emits a deprecation warning and delegates to the canonical ``gpcr-tools curate`` CLI.
    """
    import warnings

    warnings.warn(
        "gpcr-csv-generator is deprecated. Use 'gpcr-tools curate' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print(
        "WARNING: gpcr-csv-generator is deprecated. Use 'gpcr-tools curate' instead.",
        file=sys.stderr,
    )

    import argparse

    parser = argparse.ArgumentParser(
        prog="gpcr-csv-generator",
        description="(Deprecated) Use 'gpcr-tools curate' instead.",
    )
    parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: target a specific PDB ID instead of processing all pending.",
    )
    args = parser.parse_args()
    main(target_pdb=args.pdb_id)
