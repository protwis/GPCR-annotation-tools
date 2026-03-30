"""Rich console UI helpers for the CSV generator dashboard.

All Rich rendering — themes, panels, tables, display functions — lives here.
Review logic is in review_engine.py; this module only handles presentation.
"""

from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Console Setup ───────────────────────────────────────────────────────

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "highlight": "magenta",
        "key": "bold blue",
        "value": "white",
        "panel.border": "blue",
        "header": "bold white on blue",
    }
)

console = Console(theme=custom_theme)


# ── Dashboard ───────────────────────────────────────────────────────────


def display_dashboard_header(pdb_count: int, pending_count: int) -> None:
    """Render the top-of-screen dashboard banner."""
    from datetime import datetime

    title = Text(
        "PDB Annotation Review Dashboard",
        style="bold white on blue",
        justify="center",
    )
    stats = Text.assemble(
        ("Total PDBs: ", "bold"),
        (str(pdb_count), "cyan"),
        " | ",
        ("Pending: ", "bold"),
        (str(pending_count), "yellow"),
        " | ",
        ("Date: ", "bold"),
        (datetime.now().strftime("%Y-%m-%d"), "green"),
    )
    panel = Panel(
        Align.center(Group(title, Text(""), stats)),
        box=box.ROUNDED,
        border_style="blue",
        title="[bold]Interactive Review System[/bold]",
        subtitle="Docker-Ready · Full Audit Trail",
    )
    console.print(panel)


# ── Display Helpers ─────────────────────────────────────────────────────


def create_display_copy(data: Any) -> Any:
    """Create a display-friendly copy of data, truncating long synonym lists."""
    if isinstance(data, dict):
        display_dict = {}
        for key, value in data.items():
            if key == "synonyms" and isinstance(value, list):
                if len(value) > 3:
                    display_dict[key] = [*value[:3], f"... ({len(value) - 3} more)"]
                else:
                    display_dict[key] = value
            else:
                display_dict[key] = create_display_copy(value)
        return display_dict
    elif isinstance(data, list):
        return [create_display_copy(item) for item in data]
    else:
        return data


def display_ligand_validation_panel(ligands_data: list) -> None:
    """Render a status-aware summary panel for ligands before the main review.

    GHOST_LIGAND = stark red, EXCLUDED_BUFFER = dim grey, MATCHED = green.
    """
    if not ligands_data or not isinstance(ligands_data, list):
        return

    has_any_status = any(
        isinstance(lig, dict) and lig.get("validation_status") for lig in ligands_data
    )
    if not has_any_status:
        return

    table = Table(
        box=box.SIMPLE_HEAVY,
        expand=True,
        show_header=True,
        title="Ligand Validation Summary",
    )
    table.add_column("Name", style="cyan", width=20)
    table.add_column("chem_comp_id", width=12)
    table.add_column("Status", width=28)
    table.add_column("Details", ratio=1)

    for lig in ligands_data:
        if not isinstance(lig, dict):
            continue
        name = lig.get("name", "?")
        comp_id = lig.get("chem_comp_id", "?")
        status = lig.get("validation_status", "")

        if status == "GHOST_LIGAND":
            status_text = Text("GHOST_LIGAND", style="bold red")
            detail = Text("AI hallucination. Recommend DELETE.", style="red")
        elif status == "EXCLUDED_BUFFER":
            status_text = Text("EXCLUDED_BUFFER", style="dim")
            detail = Text("Crystallization artifact / buffer. Safe to ignore.", style="dim")
        elif status == "MATCHED_SMALL_MOLECULE":
            smiles = lig.get("SMILES_stereo") or lig.get("SMILES", "")
            inchikey = lig.get("InChIKey", "")
            detail_str = f"InChIKey: {inchikey[:20]}..." if inchikey else ""
            if smiles:
                detail_str += (
                    f"  SMILES: {smiles[:30]}..." if len(smiles) > 30 else f"  SMILES: {smiles}"
                )
            status_text = Text("MATCHED (small-molecule)", style="green")
            detail = Text(detail_str, style="green")
        elif status == "MATCHED_POLYMER":
            seq = lig.get("Sequence", "")
            seq_display = f"Seq: {seq[:40]}..." if len(seq) > 40 else f"Seq: {seq}"
            status_text = Text("MATCHED (polymer)", style="green")
            detail = Text(seq_display, style="green")
        elif status == "SKIPPED_APO":
            status_text = Text("SKIPPED (apo)", style="dim")
            detail = Text("", style="dim")
        else:
            status_text = Text(status or "N/A", style="dim")
            detail = Text("")

        table.add_row(name, comp_id, status_text, detail)

    ghost_count = sum(
        1
        for lig in ligands_data
        if isinstance(lig, dict) and lig.get("validation_status") == "GHOST_LIGAND"
    )

    border_style = "red" if ghost_count > 0 else "green"
    panel_title = (
        f"[bold red]GHOST LIGAND(S) DETECTED: {ghost_count}[/]"
        if ghost_count > 0
        else "[bold green]All Ligands Validated[/]"
    )

    console.print(Panel(table, title=panel_title, border_style=border_style, box=box.DOUBLE))
