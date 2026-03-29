"""Data loading and PDB discovery for the CSV generator.

Handles reading aggregated JSON results, voting logs, validation logs,
and tracking which PDBs have been processed.
"""

import json
from datetime import UTC
from typing import Any

from gpcr_tools.config import (
    LOGS_DIR,
    OUTPUT_DIR,
    PROCESSED_LOG_FILE,
    RESULTS_DIR,
    VALIDATION_DIR,
)
from gpcr_tools.csv_generator.ui import console


def load_processed_log() -> dict:
    """Load the processed PDB tracking log."""
    if not PROCESSED_LOG_FILE.exists():
        return {}
    try:
        with open(PROCESSED_LOG_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        console.print(f"[bold red]Error loading processed log: {e}[/bold red]")
        return {}


def update_processed_log(pdb_id: str, status: str = "completed") -> None:
    """Record a PDB as processed."""
    from datetime import datetime

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_data = load_processed_log()
    log_data[pdb_id] = {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        with open(PROCESSED_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
    except Exception as e:
        console.print(f"[bold red]FATAL: Failed to update log: {e}[/bold red]")


def get_pending_pdbs() -> tuple[list[str], list[str], int]:
    """Scan DATA_DIR for JSON files and return (pending, skipped, total_count).

    Only PDBs with status == "completed" are excluded from all queues.
    PDBs with status == "failed" are returned in the pending list (auto-retry).
    PDBs with status == "skipped" are returned in a separate list so the user
    can be prompted whether to re-review them.

    Returns:
        (pending_ids, skipped_ids, total_count)
    """
    if not RESULTS_DIR.is_dir():
        return [], [], 0
    all_pdb_ids = sorted([f.stem for f in RESULTS_DIR.glob("*.json")])
    log = load_processed_log()

    pending_ids: list[str] = []
    skipped_ids: list[str] = []

    for pid in all_pdb_ids:
        entry = log.get(pid)
        if entry is None:
            # Never seen before
            pending_ids.append(pid)
        elif entry.get("status") == "completed":
            # Truly done — skip
            continue
        elif entry.get("status") == "skipped":
            # Previously skipped — let the user decide
            skipped_ids.append(pid)
        else:
            # "failed" or any unknown status — auto-retry
            pending_ids.append(pid)

    return pending_ids, skipped_ids, len(all_pdb_ids)


def load_pdb_data(
    pdb_id: str,
) -> tuple[dict[str, Any] | None, dict, dict]:
    """Load main data, controversy map, and validation data for a PDB.

    Returns:
        (main_data, controversy_map, validation_data) or (None, None, None) on error.
    """
    main_path = RESULTS_DIR / f"{pdb_id}.json"
    log_path = LOGS_DIR / f"{pdb_id}_voting_log.json"
    val_path = VALIDATION_DIR / f"{pdb_id}_validation.json"

    try:
        with open(main_path, encoding="utf-8") as f:
            data = json.load(f)
            main_data = data if isinstance(data, dict) else None
    except Exception as e:
        console.print(f"[bold red]Error loading {pdb_id}.json: {e}[/bold red]")
        return None, {}, {}

    controversy_map: dict = {}
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                clist = json.load(f)
                controversy_map = {item["path"]: item for item in clist}
        except Exception:
            pass

    validation_data: dict = {"critical_warnings": [], "algo_conflicts": []}
    if val_path.exists():
        try:
            with open(val_path, encoding="utf-8") as f:
                data = json.load(f)
                validation_data = data if isinstance(data, dict) else validation_data
        except Exception:
            pass

    return main_data, controversy_map, validation_data
