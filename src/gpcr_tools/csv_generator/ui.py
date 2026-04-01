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


# ── Oligomer Analysis Panel ────────────────────────────────────────────


def _should_highlight_oligomer(oligo: dict, receptor_chain: str) -> bool:
    """Determine if the Oligomer Analysis panel needs visual highlighting."""
    if not oligo:
        return False
    if oligo.get("chain_id_override", {}).get("applied"):
        return True
    alert_types = {a.get("type") for a in oligo.get("alerts", [])}
    if alert_types & {"HALLUCINATION", "MISSED_PROTOMER", "CHAIN_ID_OVERRIDDEN"}:
        return True
    if oligo.get("classification") in ("HOMOMER", "HETEROMER"):
        return True
    if receptor_chain and "," in receptor_chain:
        return True
    return any(c.get("7tm_status") == "INCOMPLETE_7TM" for c in oligo.get("all_gpcr_chains", []))


def display_oligomer_analysis_panel(main_data: dict) -> None:
    """Render the unified Oligomer Analysis panel.

    Replaces the legacy heteromer_resolution + tm_completeness panels.
    Shows classification, GPCR chain table, primary protomer suggestion,
    alerts, and assembly cross-check information.
    """
    oligo = main_data.get("oligomer_analysis")
    if not oligo:
        return

    receptor_chain = main_data.get("receptor_info", {}).get("chain_id", "")
    highlight = _should_highlight_oligomer(oligo, receptor_chain)
    override = oligo.get("chain_id_override", {})
    classification = oligo.get("classification", "UNKNOWN")
    alerts = oligo.get("alerts", [])

    elements: list = []

    # ── Override banner (highest priority) ──
    if override.get("applied"):
        override_text = Text()
        override_text.append("CHAIN_ID CORRECTED  ", style="bold white on red")
        override_text.append(
            f"  {override.get('original_chain_id')} -> {override.get('corrected_chain_id')}"
            f"  |  UniProt: {override.get('original_uniprot')} -> "
            f"{override.get('corrected_uniprot')}"
            f"\n  Trigger: {override.get('trigger')}  |  {override.get('reason', '')}",
            style="red",
        )
        elements.append(
            Panel(override_text, border_style="bold red", box=box.HEAVY, padding=(0, 1))
        )
        elements.append(Text())

    # ── Classification line ──
    cls_style = {
        "HOMOMER": "bold cyan",
        "HETEROMER": "bold magenta",
        "MONOMER": "bold green",
        "NO_GPCR": "dim",
    }.get(classification, "white")
    cls_text = Text()
    cls_text.append("Classification: ", style="bold")
    cls_text.append(classification, style=cls_style)
    elements.append(cls_text)

    # ── GPCR chains table ──
    gpcr_chains = oligo.get("all_gpcr_chains", [])
    if gpcr_chains:
        chain_table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True, padding=(0, 1))
        chain_table.add_column("Chain", style="bold", width=6)
        chain_table.add_column("Slug", width=24)
        chain_table.add_column("7TM Status", width=16)
        chain_table.add_column("TMs", width=8, justify="center")
        for chain in gpcr_chains:
            tm_status = chain.get("7tm_status", "UNKNOWN")
            tm_style = {
                "COMPLETE": "green",
                "INCOMPLETE_7TM": "bold red",
                "UNKNOWN": "dim",
                "NOT_GPCR": "dim",
            }.get(tm_status, "white")
            chain_table.add_row(
                chain.get("chain_id", "?"),
                chain.get("slug", "?"),
                Text(tm_status, style=tm_style),
                f"{chain.get('resolved_tms', '?')}/{chain.get('total_tms', '?')}",
            )
        elements.append(Text())
        elements.append(chain_table)

    # ── Primary protomer suggestion ──
    suggestion = oligo.get("primary_protomer_suggestion", {})
    if suggestion.get("chain_id"):
        sug_text = Text()
        sug_text.append("Primary Protomer: ", style="bold")
        sug_text.append(f"Chain {suggestion['chain_id']}", style="bold cyan")
        sug_text.append(f"  (Rank {suggestion.get('rank_used', '?')})", style="dim")
        sug_text.append(f"\n  {suggestion.get('reason', '')}", style="white")
        elements.append(Text())
        elements.append(sug_text)

    # ── Alerts ──
    if alerts:
        alert_text = Text()
        for alert in alerts:
            atype = alert.get("type", "")
            style = {
                "HALLUCINATION": "bold red",
                "CHAIN_ID_OVERRIDDEN": "bold red",
                "MISSED_PROTOMER": "bold yellow",
                "CONFIRMED_OLIGOMER": "green",
            }.get(atype, "white")
            alert_text.append(f"  [{atype}] ", style=style)
            alert_text.append(f"{alert.get('message', '')}\n", style="white")
        elements.append(Text())
        elements.append(Text("Alerts:", style="bold underline"))
        elements.append(alert_text)

    # ── Assembly cross-check (informational) ──
    asm = oligo.get("assembly_cross_check", {})
    if asm.get("oligomeric_state"):
        asm_text = Text()
        asm_text.append("Assembly: ", style="dim bold")
        asm_text.append(
            f"{asm.get('oligomeric_state', '')}  "
            f"Stoich: {asm.get('stoichiometry', '')}  "
            f"Symmetry: {asm.get('type', '')}",
            style="dim",
        )
        elements.append(Text())
        elements.append(asm_text)

    # ── Panel styling ──
    if override.get("applied"):
        border_style = "bold red"
        title = "[bold white on red] OLIGOMER ANALYSIS — CHAIN CORRECTED [/]"
    elif highlight:
        border_style = "yellow"
        title = "[bold yellow]OLIGOMER ANALYSIS — REVIEW RECOMMENDED[/]"
    else:
        border_style = "green"
        title = "[bold green]Oligomer Analysis[/]"

    console.print(
        Panel(
            Group(*elements),
            title=title,
            border_style=border_style,
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )
