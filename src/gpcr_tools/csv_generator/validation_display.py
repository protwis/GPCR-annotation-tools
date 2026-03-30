"""Validation display and analysis for the CSV generator.

Handles rendering validation alerts, extracting warning entries,
and analyzing whether validation findings warrant block deletion or cleanup.
"""

import re
from typing import Any

from rich import box
from rich.panel import Panel
from rich.text import Text

from gpcr_tools.config import VALIDATION_FATAL_KEYWORDS
from gpcr_tools.csv_generator.ui import console

# ── Warning Helpers ─────────────────────────────────────────────────────


def get_relevant_validation_warnings(path: str, validation_data: dict) -> list[str]:
    """Return validation warnings relevant to the given JSON path."""
    relevant: list[str] = []
    if "signaling_partners" in path and validation_data.get("algo_conflicts"):
        relevant.extend(validation_data["algo_conflicts"])
    if validation_data.get("critical_warnings"):
        for w in validation_data["critical_warnings"]:
            if path in w or (path == "signaling_partners" and "g_protein" in w):
                relevant.append(w)
    # Deduplicate while preserving insertion order for deterministic display.
    return list(dict.fromkeys(relevant))


def display_validation_alert(path: str, validation_data: dict) -> bool:
    """Render a validation alert panel if there are relevant warnings."""
    warnings = get_relevant_validation_warnings(path, validation_data)
    if warnings:
        warn_text = Text()
        for w in warnings:
            if "CONFLICT" in w or "HALLUCINATION" in w:
                warn_text.append(f"⚠ {w}\n", style="bold red")
            else:
                warn_text.append(f"• {w}\n", style="yellow")

        console.print(
            Panel(
                warn_text,
                title="[bold red blink]ALGORITHM / VALIDATION ALERT[/]",
                border_style="red",
                box=box.DOUBLE,
            )
        )
        return True
    return False


# ── Validation Entry Extraction ─────────────────────────────────────────


def canonicalize_path(raw_path: str | None) -> str:
    """Normalize a validation path by stripping leading dots."""
    if not raw_path:
        return ""
    return raw_path.lstrip(".")


def extract_validation_entries(validation_data: dict | None) -> list[dict]:
    """Parse raw validation data into structured entry dicts."""
    if not validation_data:
        return []
    entries: list[dict] = []
    for bucket in ("critical_warnings", "algo_conflicts"):
        for warning in validation_data.get(bucket, []):
            warn_str = str(warning)
            path_match = re.search(r"at ['\"]([^'\"]+)['\"]", warn_str)
            entries.append(
                {
                    "text": warn_str,
                    "path": path_match.group(1) if path_match else None,
                    "bucket": bucket,
                    "is_hallucination": "HALLUCINATION ALERT" in warn_str.upper(),
                }
            )
    return entries


def warning_matches_block(entry: dict, block_path: str) -> bool:
    """Check whether a validation warning is relevant to a given block path."""
    normalized_block = canonicalize_path(block_path)
    normalized_path = canonicalize_path(entry.get("path"))
    if normalized_path:
        if normalized_path == normalized_block:
            return True
        if normalized_path.startswith(f"{normalized_block}."):
            return True
        if normalized_path.startswith(f"{normalized_block}["):
            return True
    if entry.get("is_hallucination"):
        warn_text = entry.get("text", "").lower()
        if normalized_block in warn_text:
            return True
        if normalized_block == "signaling_partners" and "g-protein" in warn_text:
            return True
    return False


# ── Validation Impact Analysis ──────────────────────────────────────────


def analyze_validation_impact(
    block_path: str, block_data: Any, validation_data: dict
) -> dict | None:
    """Analyze whether validation findings warrant deletion or cleanup of a block.

    Returns:
        A suggestion dict with 'action' and 'reason' keys, or None.
    """
    entries = [
        entry
        for entry in extract_validation_entries(validation_data)
        if warning_matches_block(entry, block_path)
    ]
    if not entries or block_data is None:
        return None

    fatal_entries = [
        entry
        for entry in entries
        if entry.get("is_hallucination")
        or any(keyword in entry.get("text", "").lower() for keyword in VALIDATION_FATAL_KEYWORDS)
    ]

    if isinstance(block_data, dict):
        if any(entry.get("is_hallucination") for entry in entries):
            return {
                "action": "DELETE_BLOCK",
                "reason": "Validation flagged this block as a hallucination. Safe option is to remove it.",
            }
        if len(fatal_entries) >= 2:
            detail = fatal_entries[0]["text"]
            return {
                "action": "DELETE_BLOCK",
                "reason": (
                    f"{len(fatal_entries)} fatal validation warnings reference this block "
                    f"(e.g., '{detail}')."
                ),
            }
        return None

    if isinstance(block_data, list):
        list_length = len(block_data)
        if list_length == 0:
            return None
        invalid_indices: list[int] = []
        for entry in entries:
            normalized_path = canonicalize_path(entry.get("path"))
            if not normalized_path:
                continue
            idx_matches = re.findall(r"\[(\d+)\]", normalized_path)
            for match in idx_matches:
                idx_val = int(match)
                if idx_val < list_length:
                    invalid_indices.append(idx_val)
        invalid_indices = sorted(set(invalid_indices))

        if invalid_indices:
            if len(invalid_indices) >= list_length:
                return {
                    "action": "DELETE_BLOCK",
                    "invalid_indices": invalid_indices,
                    "reason": "All entries in this block failed validation checks.",
                }
            return {
                "action": "CLEAN_ENTRIES",
                "invalid_indices": invalid_indices,
                "reason": f"Validation marked list indices {invalid_indices} as invalid.",
            }

        direct_block_hit = any(
            canonicalize_path(entry.get("path")) == canonicalize_path(block_path)
            for entry in fatal_entries
        )
        if direct_block_hit:
            return {
                "action": "DELETE_BLOCK",
                "reason": "Validation marked the entire list as invalid.",
            }
    return None
