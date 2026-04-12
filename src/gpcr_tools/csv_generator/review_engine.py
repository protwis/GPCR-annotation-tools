"""Core review logic for the interactive CSV generator.

Handles the recursive review tree: decision units, leaf nodes, controversy
resolution, auto-resolve for trivial keys, and top-level block orchestration.
"""

import copy
import json
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from gpcr_tools.config import AUTO_RESOLVE_KEYS, BLACKLISTED_KEYS, TOPLEVEL_BLOCK_KEYS
from gpcr_tools.csv_generator.audit import log_audit_trail
from gpcr_tools.csv_generator.ui import (
    console,
    create_display_copy,
    display_ligand_validation_panel,
)
from gpcr_tools.csv_generator.validation_display import (
    analyze_validation_impact,
    display_validation_alert,
    get_relevant_validation_warnings,
)

# ── Type Coercion ──────────────────────────────────────────────────────


def coerce_type(original: Any, new_str: str) -> Any:
    """Preserve the original value's type after a Prompt.ask edit.

    Attempts to parse *new_str* back to the type of *original*.
    Falls back to returning *new_str* as-is if parsing fails.
    """
    if new_str == str(original):
        return original
    if isinstance(original, bool):
        if new_str.lower() in ("true", "1", "yes"):
            return True
        if new_str.lower() in ("false", "0", "no"):
            return False
        return new_str
    if isinstance(original, int):
        try:
            return int(new_str)
        except ValueError:
            pass
    if isinstance(original, float):
        try:
            return float(new_str)
        except ValueError:
            pass
    if isinstance(original, list | dict):
        try:
            parsed = json.loads(new_str)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, type(original)):
            return parsed
    return new_str


# ── Controversy Detection ───────────────────────────────────────────────


def has_downstream_controversy(path_prefix: str, controversies: dict) -> bool:
    """Check whether any controversy exists at or below the given path."""
    if not path_prefix:
        return bool(controversies)
    if path_prefix in controversies:
        return True
    for p in controversies:
        if p.startswith(path_prefix + ".") or p.startswith(path_prefix + "["):
            return True
    return False


def is_controversy_significant(
    path_prefix: str, controversies: dict, validation_data: dict
) -> bool:
    """Returns True if there are significant (non-trivial) controversies or validation warnings.

    Returns False if all controversies are trivial (in AUTO_RESOLVE_KEYS)
    and there are no validation warnings.
    """
    if get_relevant_validation_warnings(path_prefix, validation_data):
        return True

    if not path_prefix:
        controversy_paths = list(controversies.keys())
    else:
        controversy_paths = [path_prefix] if path_prefix in controversies else []
        for p in controversies:
            if p.startswith(path_prefix + ".") or p.startswith(path_prefix + "["):
                controversy_paths.append(p)

    for controversy_path in controversy_paths:
        terminal_key = ""
        if controversy_path:
            terminal_key = controversy_path.split(".")[-1]
            if "[" in terminal_key:
                terminal_key = terminal_key.split("[")[0]
        if terminal_key and terminal_key not in AUTO_RESOLVE_KEYS:
            return True

    return False


def get_verified_paths(main_data: dict) -> set:
    """Extract paths that have been verified by algorithmic validators."""
    verified: set = set()
    for block_name, block_data in main_data.items():
        if isinstance(block_data, dict):
            vf = block_data.get("_verified_fields") or []
            if isinstance(vf, list):
                for field in vf:
                    verified.add(f"{block_name}.{field}")
    return verified


# ── Review Functions ────────────────────────────────────────────────────


def review_decision_unit(
    pdb_id: str,
    d_node: dict,
    controversies: dict,
    path: str,
    validation_data: dict,
    fix_mode: bool = False,
    verified_paths: set | None = None,
) -> dict | None:
    """Review a decision unit (dict with value/confidence/evidence)."""
    display_validation_alert(path, validation_data)

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(style="bold cyan", width=12)
    grid.add_column(style="value")
    grid.add_row("Value:", str(d_node.get("value", "N/A")))
    conf = d_node.get("confidence", 0)
    conf_style = "success" if isinstance(conf, int | float) and conf >= 0.8 else "warning"
    grid.add_row("Confidence:", f"[{conf_style}]{conf}[/{conf_style}]")

    content = Group(
        grid,
        Text("\nEvidence:", style="bold underline"),
        Pretty(d_node.get("evidence") or {}),
    )
    console.print(
        Panel(
            content,
            title=f"[bold]Reviewing Decision: [cyan]{path}[/]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )

    if has_downstream_controversy(path, controversies):
        console.print(Panel("[bold yellow]Controversy detected downstream.[/]", style="yellow"))
        return review_node(
            pdb_id,
            d_node,
            controversies,
            path,
            True,
            validation_data,
            fix_mode,
            verified_paths,
        )

    action = Prompt.ask(
        "\n[prompt]Accept block? ([bold]Y[/]es, [bold]e[/]dit, "
        "[bold]d[/]eep-dive, [bold]s[/]kip, [bold]q[/]uit):[/]",
        choices=["y", "e", "d", "s", "q"],
        default="y",
    ).lower()
    if action == "y":
        log_audit_trail(pdb_id, path, "accept_block", d_node.get("value"), d_node.get("value"))
        return d_node
    if action == "s":
        log_audit_trail(pdb_id, path, "skip_field", d_node.get("value"), d_node.get("value"))
        return d_node
    if action == "q":
        return None
    if action == "d":
        return review_node(
            pdb_id,
            d_node,
            controversies,
            path,
            True,
            validation_data,
            fix_mode,
            verified_paths,
        )
    if action == "e":
        orig_val = d_node["value"]
        raw = Prompt.ask(f"[prompt]New value for [cyan]{path}.value[/]:[/] ", default=str(orig_val))
        new_val = coerce_type(orig_val, raw)
        new_d_node = dict(d_node)
        new_d_node["value"] = new_val
        log_audit_trail(pdb_id, path + ".value", "edit_in_block", orig_val, new_val)
        return new_d_node
    return d_node  # fallback


def review_leaf(
    pdb_id: str,
    leaf_val: Any,
    controversies: dict,
    path: str,
    validation_data: dict,
    verified_paths: set | None = None,
) -> Any | None:
    """Review a leaf value (scalar or non-decision-unit)."""
    display_validation_alert(path, validation_data)

    if path in controversies:
        controversy = controversies[path]
        best_run_value = controversy.get("best_run_value")
        majority_value = controversy.get("majority_vote_value")
        all_votes = controversy.get("all_votes") or {}

        def parse_vote_key(raw_key):
            if isinstance(raw_key, str):
                try:
                    return json.loads(raw_key)
                except json.JSONDecodeError:
                    return raw_key
            return raw_key

        def canonicalize(value: Any) -> str:
            try:
                return json.dumps(value, sort_keys=True)
            except (TypeError, ValueError):
                return str(value)

        def format_value(value: Any) -> str:
            if isinstance(value, dict | list):
                try:
                    return json.dumps(value, sort_keys=True)
                except (TypeError, ValueError):
                    return str(value)
            return str(value)

        candidates_map: dict[str, dict[str, Any]] = {}
        for raw_key, raw_count in all_votes.items():
            parsed_val = parse_vote_key(raw_key)
            canon_key = canonicalize(parsed_val)
            candidate = candidates_map.setdefault(canon_key, {"value": parsed_val, "count": 0})
            try:
                candidate["count"] += int(raw_count)
            except (TypeError, ValueError):
                continue

        for anchor_value in (best_run_value, majority_value):
            if anchor_value is None:
                continue
            canon_key = canonicalize(anchor_value)
            candidates_map.setdefault(canon_key, {"value": anchor_value, "count": 0})

        if not candidates_map:
            fallback_key = canonicalize(leaf_val)
            candidates_map[fallback_key] = {"value": leaf_val, "count": 0}

        candidates: list[dict[str, Any]] = []
        for candidate in candidates_map.values():
            value_obj = candidate["value"]
            candidates.append(
                {
                    "value": value_obj,
                    "count": candidate.get("count", 0),
                    "is_best": value_obj == best_run_value,
                    "is_majority": value_obj == majority_value,
                    "display": format_value(value_obj),
                    "canonical": canonicalize(value_obj),
                }
            )

        candidates.sort(key=lambda c: (-c["count"], c["canonical"]))

        table = Table(box=box.SIMPLE, expand=True, show_header=True)
        table.add_column("Opt", style="bold magenta", width=4, justify="center")
        table.add_column("Tags/Votes", style="cyan")
        table.add_column("Value", style="green")

        for idx, candidate in enumerate(candidates, start=1):
            vote_count = candidate["count"]
            vote_label = f"{vote_count} vote{'s' if vote_count != 1 else ''}"
            tags = [vote_label]

            is_verified = False
            if verified_paths and path in verified_paths and candidate["is_best"]:
                tags.append("[Verified]")
                is_verified = True
            elif candidate["is_best"]:
                tags.append("[Best Run]")

            if candidate["is_majority"]:
                tags.append("[Majority]")

            candidate["is_verified"] = is_verified
            table.add_row(str(idx), " ".join(tags).strip(), candidate["display"])

        console.print(
            Panel(
                Group(Text(f"Path: {path}", style="bold cyan"), table),
                title="[bold red]CONTROVERSY[/]",
                border_style="red",
                box=box.HEAVY,
            )
        )

        option_choices = [str(i) for i in range(1, len(candidates) + 1)]
        prompt_choices = [*option_choices, "s", "e", "q"]

        target_default_idx = None
        for idx, cand in enumerate(candidates):
            if cand.get("is_verified"):
                target_default_idx = idx
                break
        if target_default_idx is None:
            for idx, cand in enumerate(candidates):
                if cand["is_majority"]:
                    target_default_idx = idx
                    break
        if target_default_idx is None:
            for idx, cand in enumerate(candidates):
                if cand["is_best"]:
                    target_default_idx = idx
                    break
        if target_default_idx is None:
            target_default_idx = 0

        default_choice = option_choices[target_default_idx] if option_choices else "e"

        choice = Prompt.ask(
            "\n[prompt]Select option, [bold]s[/]kip, [bold]e[/]dit, or [bold]q[/]uit:[/]",
            choices=prompt_choices,
            default=default_choice,
        ).lower()

        if choice == "q":
            return None
        if choice == "s":
            log_audit_trail(pdb_id, path, "skip_field", leaf_val, leaf_val)
            return leaf_val
        if choice == "e":
            raw = Prompt.ask("[prompt]Enter value:[/] ", default=str(leaf_val))
            new_val = coerce_type(leaf_val, raw)
            log_audit_trail(pdb_id, path, "edit_controversy", leaf_val, new_val)
            return new_val

        selected_idx = int(choice) - 1
        selected_candidate = candidates[selected_idx]
        selected_value = copy.deepcopy(selected_candidate["value"])
        if selected_candidate["is_best"]:
            action_str = "select_best"
        elif selected_candidate["is_majority"]:
            action_str = "select_majority"
        else:
            action_str = "select_alternative"

        log_audit_trail(pdb_id, path, action_str, leaf_val, selected_value)
        return selected_value
    else:
        console.print(Panel(f"Path: {path}\nValue: {leaf_val}", border_style="blue"))
        action = Prompt.ask(
            "[prompt]Action ([bold]Y[/]es, [bold]e[/]dit, [bold]s[/]kip, [bold]q[/]uit):[/]",
            choices=["y", "e", "s", "q"],
            default="y",
        ).lower()
        if action == "q":
            return None
        if action in ("y", "s"):
            audit_action = "accept" if action == "y" else "skip_field"
            log_audit_trail(pdb_id, path, audit_action, leaf_val, leaf_val)
            return leaf_val
        if action == "e":
            raw = Prompt.ask("[prompt]New value:[/] ", default=str(leaf_val))
            new_val = coerce_type(leaf_val, raw)
            log_audit_trail(pdb_id, path, "edit", leaf_val, new_val)
            return new_val
    return leaf_val  # fallback


def review_node(
    pdb_id: str,
    d_node: Any,
    controversies: dict,
    path: str = "",
    force_deep: bool = False,
    validation_data: dict | None = None,
    fix_mode: bool = False,
    verified_paths: set | None = None,
) -> Any | None:
    """Recursively review a JSON node (dict, list, or leaf)."""
    if validation_data is None:
        validation_data = {}

    terminal_key = ""
    if path:
        terminal_key = path.split(".")[-1]
        if "[" in terminal_key:
            terminal_key = terminal_key.split("[")[0]

    # Auto-resolve trivial keys
    if (
        terminal_key in AUTO_RESOLVE_KEYS
        and path in controversies
        and not get_relevant_validation_warnings(path, validation_data)
    ):
        best_value = copy.deepcopy(controversies[path].get("best_run_value"))
        log_audit_trail(pdb_id, path, "auto_resolve_trivial", d_node, best_value)
        return best_value

    # Clean branch check in fix mode
    if (
        fix_mode
        and not has_downstream_controversy(path, controversies)
        and not get_relevant_validation_warnings(path, validation_data)
    ):
        return d_node

    # Decision unit shortcut
    if (
        not force_deep
        and isinstance(d_node, dict)
        and all(k in d_node for k in ["value", "confidence", "evidence"])
    ):
        return review_decision_unit(
            pdb_id,
            d_node,
            controversies,
            path,
            validation_data,
            fix_mode,
            verified_paths,
        )

    if isinstance(d_node, dict):
        new_dict = {}
        for key, val in d_node.items():
            curr_path = f"{path}.{key}" if path else key
            is_clean = not has_downstream_controversy(curr_path, controversies)
            has_warning = bool(get_relevant_validation_warnings(curr_path, validation_data))

            if is_clean and not has_warning and key in BLACKLISTED_KEYS:
                new_dict[key] = val
                continue

            res = review_node(
                pdb_id,
                val,
                controversies,
                curr_path,
                force_deep,
                validation_data,
                fix_mode,
                verified_paths,
            )
            if res is None:
                return None
            new_dict[key] = res
        return new_dict

    elif isinstance(d_node, list):
        if (
            len(d_node) > 5
            and "synonyms" in path
            and not has_downstream_controversy(path, controversies)
        ) and Confirm.ask(f"Accept list at {path} ({len(d_node)} items)?"):
            log_audit_trail(pdb_id, path, "accept_list_batch", "N/A", "N/A")
            return d_node

        new_list = []
        key_field = (
            "chem_comp_id"
            if "ligands" in path
            else ("name" if "auxiliary_proteins" in path else None)
        )
        for idx, item in enumerate(d_node):
            curr_path = (
                f"{path}[{item.get(key_field, idx)}]"
                if key_field and isinstance(item, dict)
                else f"{path}[{idx}]"
            )
            res = review_node(
                pdb_id,
                item,
                controversies,
                curr_path,
                force_deep,
                validation_data,
                fix_mode,
                verified_paths,
            )
            if res is None:
                return None
            new_list.append(res)
        return new_list
    else:
        return review_leaf(pdb_id, d_node, controversies, path, validation_data, verified_paths)


# ── Top-Level Block Review ──────────────────────────────────────────────


def review_toplevel_blocks(
    pdb_id: str,
    main_data: dict,
    controversies: dict,
    validation_data: dict,
    fix_mode: bool = False,
) -> dict | None:
    """Review each top-level block with appropriate context and UI."""
    verified_paths = get_verified_paths(main_data)
    final_data: dict = {}

    for key in TOPLEVEL_BLOCK_KEYS:
        if key not in main_data:
            continue
        block_data = main_data[key]

        has_alert = bool(get_relevant_validation_warnings(key, validation_data))
        has_contra = has_downstream_controversy(key, controversies)

        is_effectively_clean = fix_mode and not is_controversy_significant(
            key, controversies, validation_data
        )

        if is_effectively_clean:
            if has_contra:
                resolved_block = review_node(
                    pdb_id,
                    block_data,
                    controversies,
                    key,
                    False,
                    validation_data,
                    fix_mode,
                    verified_paths,
                )
                if resolved_block is None:
                    return None
                final_data[key] = resolved_block
                log_audit_trail(pdb_id, key, "auto_accept_trivial_block", "N/A", "ACCEPTED")
            else:
                final_data[key] = block_data
                log_audit_trail(pdb_id, key, "auto_accept_clean_block", "N/A", "ACCEPTED")
            continue

        # Ligand validation panel
        if key == "ligands" and isinstance(block_data, list):
            display_ligand_validation_panel(block_data)

        # Receptor identity clash
        if key == "receptor_info" and isinstance(block_data, dict):
            status = block_data.get("validation_status")
            if status == "UNIPROT_CLASH":
                chain_id = block_data.get("chain_id") or "?"
                api_reality = block_data.get("api_reality") or []
                ai_uniprot = block_data.get("uniprot_entry_name") or "?"
                console.print(
                    Panel(
                        f"⚠️ IDENTITY CLASH: API identifies chain {chain_id} as {api_reality}, "
                        f"but AI identified it as {ai_uniprot}. "
                        f"Potential Fusion Protein Masking! Expert verification required.",
                        title="[bold red blink]UNIPROT CLASH DETECTED[/]",
                        style="bold white on red",
                        box=box.DOUBLE,
                    )
                )

        title = f"[bold]Top-Level Block: [cyan]{key}[/]"
        style = "green"
        if has_contra:
            title += " [bold yellow](CONTROVERSY)[/]"
            style = "yellow"
        if has_alert:
            title += " [bold red blink](VALIDATION ERROR)[/]"
            style = "red"

        if has_contra or has_alert:
            current_block = copy.deepcopy(block_data)
            while True:
                display_validation_alert(key, validation_data)
                console.print(
                    Panel(
                        Pretty(create_display_copy(current_block)),
                        title=title,
                        border_style=style,
                        box=box.ROUNDED,
                    )
                )

                suggestion = analyze_validation_impact(key, current_block, validation_data)
                if suggestion:
                    suggestion_text = suggestion.get("reason") or ""
                    if suggestion.get("invalid_indices") is not None:
                        suggestion_text += f"\nIndices: {suggestion['invalid_indices']}"
                    console.print(
                        Panel(
                            f"💡 {suggestion_text}",
                            title="[bold green]Validation Suggestion[/]",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                    )

                choices = ["r", "a", "s", "q"]
                choice_labels = [
                    "[bold]r[/]eview",
                    "[bold]a[/]ccept",
                    "[bold]s[/]kip/delete",
                    "[bold]q[/]uit",
                ]

                if suggestion and suggestion.get("action") == "DELETE_BLOCK":
                    choices.append("d")
                    choice_labels.append("[bold]d[/]elete block")
                if (
                    suggestion
                    and suggestion.get("action") == "CLEAN_ENTRIES"
                    and isinstance(current_block, list)
                ):
                    choices.append("c")
                    choice_labels.append("[bold]c[/]lean invalid")

                action = Prompt.ask(
                    f"[prompt]Action for '{key}' ({', '.join(choice_labels)}):[/]",
                    choices=choices,
                    default="r",
                )

                if action == "q":
                    return None
                if action == "s":
                    core_blocks = {"receptor_info", "ligands", "signaling_partners"}
                    if key in core_blocks and not Confirm.ask(
                        f"[bold red]'{key}' is a core block. "
                        f"Skipping will leave its CSV fields empty. Continue?[/]",
                        default=False,
                    ):
                        continue
                    console.print(f"[yellow]Skipping/Deleting block '{key}'[/yellow]")
                    log_audit_trail(
                        pdb_id,
                        key,
                        "skip_block",
                        create_display_copy(current_block),
                        "SKIPPED",
                    )
                    break
                if action == "d" and suggestion and suggestion.get("action") == "DELETE_BLOCK":
                    console.print(
                        f"[red]Deleting block '{key}' per validation recommendation.[/red]"
                    )
                    log_audit_trail(
                        pdb_id,
                        key,
                        "delete_block_validation",
                        create_display_copy(current_block),
                        "REMOVED",
                    )
                    current_block = None
                    break
                if action == "c":
                    if (
                        suggestion
                        and suggestion.get("action") == "CLEAN_ENTRIES"
                        and isinstance(current_block, list)
                    ):
                        invalid_indices = suggestion.get("invalid_indices") or []
                        invalid_set = set(invalid_indices)
                        before_snapshot = {
                            "removed_indices": invalid_indices,
                            "before": create_display_copy(current_block),
                        }
                        cleaned_block = [
                            item for idx, item in enumerate(current_block) if idx not in invalid_set
                        ]
                        removed_count = len(current_block) - len(cleaned_block)
                        current_block = cleaned_block
                        log_audit_trail(
                            pdb_id,
                            key,
                            "clean_list_entries",
                            before_snapshot,
                            {"after": create_display_copy(current_block)},
                        )
                        console.print(
                            f"[green]Removed {removed_count} invalid "
                            f"entr{'y' if removed_count == 1 else 'ies'}. "
                            f"Re-evaluating block.[/green]"
                        )
                        continue
                    console.print("[yellow]No cleanable entries detected.[/yellow]")
                    continue
                if action == "a":
                    final_data[key] = current_block
                    log_audit_trail(pdb_id, key, "accept_block_forced", "N/A", "ACCEPTED")
                    break
                if action == "r":
                    # P0 fix: user explicitly requested review — override fix_mode
                    # so child nodes actually stop for input instead of auto-accepting.
                    res = review_node(
                        pdb_id,
                        current_block,
                        controversies,
                        key,
                        False,
                        validation_data,
                        False,
                        verified_paths,
                    )
                    if res is None:
                        return None
                    final_data[key] = res
                    break
            continue
        else:
            console.print(
                Panel(
                    Pretty(create_display_copy(block_data)),
                    title=title,
                    border_style=style,
                    box=box.ROUNDED,
                )
            )
            if Confirm.ask(f"Accept '{key}'?", default=True):
                final_data[key] = block_data
                log_audit_trail(pdb_id, key, "accept_block", "N/A", "ACCEPTED")
            else:
                res = review_node(
                    pdb_id,
                    block_data,
                    controversies,
                    key,
                    True,
                    validation_data,
                    fix_mode,
                    verified_paths,
                )
                if res is None:
                    return None
                final_data[key] = res

    # Preserve any non-reviewed keys
    for k, v in main_data.items():
        if k not in final_data and k not in TOPLEVEL_BLOCK_KEYS:
            final_data[k] = v

    return final_data
