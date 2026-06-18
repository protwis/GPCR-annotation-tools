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

from gpcr_tools.config import (
    ALERT_ASSEMBLY_MISMATCH,
    ALERT_CHAIN_ID_OVERRIDDEN,
    ALERT_CONFIRMED_OLIGOMER,
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    ALERT_MULTI_COPY_LIGAND,
    ALERT_OLIGOMER_DISAGREEMENT,
    ALERT_PROTOMER_IN_AUXILIARY,
    ALERT_SUSPICIOUS_7TM,
    ALERT_TM_DATA_UNAVAILABLE,
    OLIGOMER_HETEROMER,
    OLIGOMER_HOMOMER,
    OLIGOMER_MONOMER,
    OLIGOMER_NO_GPCR,
    TM_STATUS_COMPLETE,
    TM_STATUS_INCOMPLETE,
    TM_STATUS_UNKNOWN,
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_MATCHED_POLYMER,
    VALIDATION_MATCHED_SMALL_MOLECULE,
    VALIDATION_SKIPPED_APO,
    ensure_alert_prefix,
)

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


def display_pdb_footer(pdb_id: str) -> None:
    """Re-show the PDB being curated so the reviewer keeps context after the
    screen scrolls.  Intentionally faint — a quiet marker, not a banner.
    """
    console.print(f"[dim]── PDB {pdb_id} ──[/dim]")


def display_critical_warnings_summary(validation_data: dict) -> bool:
    """Render the validation critical warnings + algorithm conflicts up front.

    These are the same findings that gate the PDB (disable global accept-all)
    and surface as RED sections once review mode is entered.  Showing them on
    the Initial Summary lets the curator see *why* a PDB is gated immediately,
    without first stepping into review.  Returns True if anything was rendered.
    """
    if not validation_data:
        return False
    critical = validation_data.get("critical_warnings") or []
    conflicts = validation_data.get("algo_conflicts") or []
    if not critical and not conflicts:
        return False

    warn_text = Text()
    for w in critical:
        warn_text.append(f"• {w}\n", style="bold red")
    for c in conflicts:
        warn_text.append(f"• {c}\n", style="bold yellow")

    count = len(critical) + len(conflicts)
    console.print(
        Panel(
            warn_text,
            title=f"[bold red]CRITICAL VALIDATION FINDINGS ({count})[/]",
            border_style="red",
            box=box.DOUBLE,
        )
    )
    return True


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
                    display_dict[key] = list(value)
            else:
                display_dict[key] = create_display_copy(value)
        return display_dict
    elif isinstance(data, list):
        return [create_display_copy(item) for item in data]
    else:
        return data


def ligand_detector_notes(lig: dict) -> list[str]:
    """Advisory one-liners for a ligand's detector findings (None-safe).

    The geometry-derived binding site (``site_ref``) and the model's judgment on
    an incidental-candidate molecule (``pharmacological_role_check``). Returned as plain strings so
    the formatting logic is testable independently of the Rich panel.
    """
    notes: list[str] = []
    site = lig.get("site_ref")
    if site:
        notes.append(f"Site: {site}")
    justification = lig.get("site_ref_justification")
    if justification:
        notes.append(f"Site justification: {justification}")
    assessment = lig.get("pharmacological_role_check")
    if isinstance(assessment, dict):
        verdict = (
            "functional ligand"
            if assessment.get("is_functional_ligand")
            else "incidental / structural"
        )
        conf = assessment.get("confidence") or "?"
        evidence = assessment.get("evidence") or ""
        notes.append(f"Pharmacological role: {verdict} (confidence {conf}) — {evidence}")
    return notes


def display_ligand_validation_panel(ligands_data: list) -> None:
    """Render a status-aware summary panel for ligands before the main review.

    GHOST_LIGAND = stark red, EXCLUDED_BUFFER = dim grey, MATCHED = green.
    """
    if not ligands_data or not isinstance(ligands_data, list):
        return

    has_any_status = any(
        isinstance(lig, dict)
        and (
            lig.get("validation_status")
            or lig.get("pharmacological_role_check")
            or lig.get("site_ref")
        )
        for lig in ligands_data
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
        name = lig.get("name") or "?"
        comp_id = lig.get("chem_comp_id") or "?"
        status = lig.get("validation_status") or ""

        if status == VALIDATION_GHOST_LIGAND:
            status_text = Text(VALIDATION_GHOST_LIGAND, style="bold red")
            detail = Text("AI hallucination. Recommend DELETE.", style="red")
        elif status == VALIDATION_EXCLUDED_BUFFER:
            status_text = Text(VALIDATION_EXCLUDED_BUFFER, style="dim")
            detail = Text("Crystallization artifact / buffer. Safe to ignore.", style="dim")
        elif status == VALIDATION_MATCHED_SMALL_MOLECULE:
            smiles = lig.get("SMILES_stereo") or lig.get("SMILES") or ""
            inchikey = lig.get("InChIKey") or ""
            detail_str = f"InChIKey: {inchikey[:20]}..." if inchikey else ""
            if smiles:
                detail_str += (
                    f"  SMILES: {smiles[:30]}..." if len(smiles) > 30 else f"  SMILES: {smiles}"
                )
            status_text = Text("MATCHED (small-molecule)", style="green")
            detail = Text(detail_str, style="green")
        elif status == VALIDATION_MATCHED_POLYMER:
            seq = lig.get("Sequence") or ""
            seq_display = f"Seq: {seq[:40]}..." if len(seq) > 40 else f"Seq: {seq}"
            status_text = Text("MATCHED (polymer)", style="green")
            detail = Text(seq_display, style="green")
        elif status == VALIDATION_SKIPPED_APO:
            status_text = Text("SKIPPED (apo)", style="dim")
            detail = Text("", style="dim")
        else:
            status_text = Text(status or "N/A", style="dim")
            detail = Text("")

        # Detector advisories (non-gating): the geometry-derived binding site and
        # the model's judgment on an incidental-candidate molecule, so the curator can act on
        # them. They never block the review -- they only inform it.
        for note in ligand_detector_notes(lig):
            detail.append(f"\n{note}", style="magenta")

        table.add_row(name, comp_id, status_text, detail)

    ghost_count = sum(
        1
        for lig in ligands_data
        if isinstance(lig, dict) and lig.get("validation_status") == VALIDATION_GHOST_LIGAND
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
    if (oligo.get("chain_id_override") or {}).get("applied"):
        return True
    alert_types = {a.get("type") for a in oligo.get("alerts") or []}
    if alert_types & {
        ALERT_HALLUCINATION,
        ALERT_MISSED_PROTOMER,
        ALERT_CHAIN_ID_OVERRIDDEN,
        ALERT_SUSPICIOUS_7TM,
        ALERT_OLIGOMER_DISAGREEMENT,
        ALERT_TM_DATA_UNAVAILABLE,
    }:
        return True
    if oligo.get("classification") in (OLIGOMER_HOMOMER, OLIGOMER_HETEROMER):
        return True
    if receptor_chain and "," in receptor_chain:
        return True
    return any(
        c.get("7tm_status") == TM_STATUS_INCOMPLETE for c in oligo.get("all_gpcr_chains") or []
    )


def display_oligomer_analysis_panel(main_data: dict) -> None:
    """Render the unified Oligomer Analysis panel.

    Replaces the legacy heteromer_resolution + tm_completeness panels.
    Shows classification, GPCR chain table, primary protomer suggestion,
    alerts, and assembly cross-check information.
    """
    oligo = main_data.get("oligomer_analysis")
    if not oligo:
        return

    receptor_chain = (main_data.get("receptor_info") or {}).get("chain_id") or ""
    highlight = _should_highlight_oligomer(oligo, receptor_chain)
    override = oligo.get("chain_id_override") or {}
    classification = oligo.get("classification") or "UNKNOWN"
    alerts = oligo.get("alerts") or []

    elements: list = []

    # ── Override banner (highest priority) ──
    if override.get("applied"):
        override_text = Text()
        override_text.append("CHAIN_ID CORRECTED  ", style="bold white on red")
        override_text.append(
            f"  {override.get('original_chain_id')} -> {override.get('corrected_chain_id')}"
            f"  |  UniProt: {override.get('original_uniprot')} -> "
            f"{override.get('corrected_uniprot')}"
            f"\n  Trigger: {override.get('trigger')}  |  {override.get('reason') or ''}",
            style="red",
        )
        elements.append(
            Panel(override_text, border_style="bold red", box=box.HEAVY, padding=(0, 1))
        )
        elements.append(Text())

    # ── Classification line ──
    cls_style = {
        OLIGOMER_HOMOMER: "bold cyan",
        OLIGOMER_HETEROMER: "bold magenta",
        OLIGOMER_MONOMER: "bold green",
        OLIGOMER_NO_GPCR: "dim",
    }.get(classification) or "white"
    cls_text = Text()
    cls_text.append("Classification: ", style="bold")
    cls_text.append(classification, style=cls_style)
    elements.append(cls_text)

    # ── GPCR chains table ──
    gpcr_chains = oligo.get("all_gpcr_chains") or []
    if gpcr_chains:
        chain_table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True, padding=(0, 1))
        chain_table.add_column("Chain", style="bold", width=6)
        chain_table.add_column("Slug", width=24)
        chain_table.add_column("7TM Status", width=16)
        chain_table.add_column("TMs", width=8, justify="center")
        for chain in gpcr_chains:
            tm_status = chain.get("7tm_status") or "UNKNOWN"
            tm_style = {
                TM_STATUS_COMPLETE: "green",
                TM_STATUS_INCOMPLETE: "bold red",
                TM_STATUS_UNKNOWN: "dim",
                "NOT_GPCR": "dim",
            }.get(tm_status) or "white"
            chain_table.add_row(
                chain.get("chain_id") or "?",
                chain.get("slug") or "?",
                Text(tm_status, style=tm_style),
                f"{chain.get('resolved_tms') or '?'}/{chain.get('total_tms') or '?'}",
            )
        elements.append(Text())
        elements.append(chain_table)

    # ── Primary protomer suggestion ──
    suggestion = oligo.get("primary_protomer_suggestion") or {}
    if suggestion.get("chain_id"):
        sug_text = Text()
        sug_text.append("Primary Protomer: ", style="bold")
        sug_text.append(f"Chain {suggestion['chain_id']}", style="bold cyan")
        sug_text.append(f"  (Rank {suggestion.get('rank_used') or '?'})", style="dim")
        sug_text.append(f"\n  {suggestion.get('reason') or ''}", style="white")
        elements.append(Text())
        elements.append(sug_text)

    # ── Alerts ──
    # ASSEMBLY_MISMATCH is a chain-count vs functional-oligomer note that fires on
    # most receptor + transducer complexes; it is informational, not actionable, so
    # it is surfaced below with the assembly cross-check rather than as an alert.
    actionable_alerts = [a for a in alerts if (a.get("type") or "") != ALERT_ASSEMBLY_MISMATCH]
    if actionable_alerts:
        alert_text = Text()
        for alert in actionable_alerts:
            atype = alert.get("type") or ""
            style = {
                ALERT_HALLUCINATION: "bold red",
                ALERT_CHAIN_ID_OVERRIDDEN: "bold red",
                ALERT_MISSED_PROTOMER: "bold yellow",
                ALERT_CONFIRMED_OLIGOMER: "green",
                ALERT_SUSPICIOUS_7TM: "bold yellow on red",
                ALERT_MULTI_COPY_LIGAND: "bold yellow",
                ALERT_PROTOMER_IN_AUXILIARY: "bold yellow",
                ALERT_OLIGOMER_DISAGREEMENT: "bold red",
                ALERT_TM_DATA_UNAVAILABLE: "bold red",
            }.get(atype) or "white"
            # Keep the "[TYPE]" label present exactly once and use the type
            # only to pick a style. Current validator messages already carry the
            # prefix (prepending it again would duplicate it, e.g.
            # "[MULTI_COPY_LIGAND] [MULTI_COPY_LIGAND] ..."), while older recorded
            # data needs it added.
            message = ensure_alert_prefix(atype, alert.get("message"))
            alert_text.append(f"  {message}\n", style=style)
        elements.append(Text())
        elements.append(Text("Alerts:", style="bold underline"))
        elements.append(alert_text)

    # ── Assembly cross-check (informational) ──
    asm = oligo.get("assembly_cross_check") or {}
    if asm.get("oligomeric_state"):
        asm_text = Text()
        asm_text.append("Assembly: ", style="dim bold")
        asm_text.append(
            f"{asm.get('oligomeric_state') or ''}  "
            f"Stoich: {asm.get('stoichiometry') or ''}  "
            f"Symmetry: {asm.get('type') or ''}",
            style="dim",
        )
        elements.append(Text())
        elements.append(asm_text)

    # Assembly-vs-GPCR-centric mismatch: informational, shown dim alongside the
    # cross-check above (the difference between the chain-count assembly and the
    # functional GPCR-centric oligomer label), not an actionable alert.
    for alert in alerts:
        if (alert.get("type") or "") == ALERT_ASSEMBLY_MISMATCH:
            note = Text()
            note.append("  ", style="dim")
            note.append(
                ensure_alert_prefix(ALERT_ASSEMBLY_MISMATCH, alert.get("message")),
                style="dim",
            )
            elements.append(note)

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
