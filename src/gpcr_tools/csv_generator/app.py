"""Main application orchestration for the Interactive CSV Generator.

This is the entry point that ties together data loading, the review engine,
CSV writing, and the Rich UI dashboard.
"""

import copy

from rich import box
from rich.panel import Panel
from rich.pretty import Pretty
from rich.prompt import Confirm, Prompt
from rich.text import Text
from tqdm import tqdm

from gpcr_tools.csv_generator.audit import log_audit_trail
from gpcr_tools.csv_generator.csv_writer import append_to_csvs, transform_for_csv
from gpcr_tools.csv_generator.data_loader import (
    get_pending_pdbs,
    load_pdb_data,
    update_processed_log,
)
from gpcr_tools.csv_generator.review_engine import review_toplevel_blocks
from gpcr_tools.csv_generator.ui import console, create_display_copy, display_dashboard_header


def main(target_pdb: str | None = None) -> None:
    """Run the interactive CSV generator dashboard.

    Args:
        target_pdb: If provided, only process this specific PDB ID.
    """
    from gpcr_tools.config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if target_pdb:
        target_pdb = target_pdb.upper()
        pending_pdbs = [target_pdb]
        total_count = 1
    else:
        pending_pdbs, skipped_pdbs, total_count = get_pending_pdbs()

        # Prompt the user to re-review previously skipped PDBs
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

            # Summary
            console.print(
                Panel(
                    Pretty(create_display_copy(main_data)),
                    title="Initial Summary",
                    border_style="blue",
                )
            )

            # ── Epic 3: Heteromer Resolution Panel ──────────────────
            hetero_res = main_data.get("heteromer_resolution", {})
            if hetero_res.get("is_heteromer"):
                primary = hetero_res.get("primary_chain", "?")
                r_slug = main_data.get("receptor_info", {}).get("uniprot_entry_name", "?")
                reason = hetero_res.get("reason", "?")
                ignored = ", ".join(
                    [
                        f"Chain {ign.get('chain_id')} ({ign.get('slug')})"
                        for ign in hetero_res.get("ignored_chains", [])
                    ]
                )

                hetero_text = Text()
                hetero_text.append(
                    f"PRIMARY RECEPTOR: Chain {primary} ({r_slug})\n",
                    style="bold white",
                )
                hetero_text.append(f"REASON: {reason}\n", style="white")
                hetero_text.append(f"IGNORED SECONDARY CHAINS: {ignored}", style="dim white")

                console.print(
                    Panel(
                        hetero_text,
                        title="[bold cyan]HETEROMER AUTOMATICALLY RESOLVED[/]",
                        border_style="cyan",
                        box=box.DOUBLE,
                    )
                )

            # ── Epic 4: 7TM Completeness Warning ───────────────────
            tm_comp = main_data.get("tm_completeness", {})
            if tm_comp.get("status") == "INCOMPLETE_7TM":
                val_res = tm_comp.get("resolved_tms", "?")
                val_tot = tm_comp.get("total_tms", "?")

                tm_text = Text()
                tm_text.append(
                    "WARNING: SEVERELY TRUNCATED RECEPTOR DETECTED\n",
                    style="bold white",
                )
                tm_text.append(
                    f"Only {val_res} out of {val_tot} Transmembrane (TM) domains "
                    f"are structurally resolved.\n",
                    style="white",
                )
                tm_text.append(
                    "This structure lacks a complete 7TM barrel and may be biologically inactive.",
                    style="dim white",
                )

                console.print(
                    Panel(
                        tm_text,
                        title="[bold red blink]⚠️ STRUCTURAL QUALITY ALERT[/]",
                        border_style="red",
                        box=box.DOUBLE,
                    )
                )

            # ── Global Alert Check ─────────────────────────────────
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

            # ── Mode Selection ─────────────────────────────────────
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

            if final_data:
                console.print(
                    Panel(
                        Pretty(create_display_copy(final_data)),
                        title="Final Data",
                        border_style="green",
                    )
                )
                if Confirm.ask("Write to CSV?"):
                    append_to_csvs(transform_for_csv(pdb_id, final_data))
                    update_processed_log(pdb_id, "completed")
                    console.print("[green]Saved![/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")


def cli_entry() -> None:
    """CLI entry point for the `gpcr-csv-generator` console script.

    Parses sys.argv via argparse and delegates to main().
    This is necessary because setuptools console_scripts entry points
    do not forward sys.argv to the target function.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="gpcr-csv-generator",
        description="Interactive CSV generator for expert review of AI-generated GPCR annotations.",
    )
    parser.add_argument(
        "pdb_id",
        nargs="?",
        default=None,
        help="Optional: target a specific PDB ID instead of processing all pending.",
    )
    args = parser.parse_args()
    main(target_pdb=args.pdb_id)
