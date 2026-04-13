"""Data loading and PDB discovery for the CSV generator.

Handles reading aggregated JSON results, voting logs, validation logs,
and tracking which PDBs have been processed.
"""

import json
from datetime import UTC
from typing import Any

from gpcr_tools.config import AGG_STATUS_COMPLETED, AGG_STATUS_SKIPPED, get_config
from gpcr_tools.csv_generator.ui import console


def load_processed_log() -> dict:
    """Load the processed PDB tracking log."""
    cfg = get_config()
    if not cfg.processed_log_file.exists():
        return {}
    try:
        with open(cfg.processed_log_file, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        console.print(f"[bold red]Error loading processed log: {e}[/bold red]")
        return {}


def update_processed_log(pdb_id: str, status: str = AGG_STATUS_COMPLETED) -> None:
    """Record a PDB as processed."""
    from datetime import datetime

    cfg = get_config()
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    log_data = load_processed_log()
    log_data[pdb_id] = {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        with open(cfg.processed_log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
    except Exception as e:
        console.print(f"[bold red]FATAL: Failed to update log: {e}[/bold red]")


def get_pending_pdbs() -> tuple[list[str], list[str], int]:
    """Scan aggregated dir for JSON files and return (pending, skipped, total_count).

    Only PDBs with status == "completed" are excluded from all queues.
    PDBs with status == "failed" are returned in the pending list (auto-retry).
    PDBs with status == "skipped" are returned in a separate list so the user
    can be prompted whether to re-review them.

    Returns:
        (pending_ids, skipped_ids, total_count)
    """
    cfg = get_config()
    results_dir = cfg.aggregated_dir
    if not results_dir.is_dir():
        return [], [], 0
    all_pdb_ids = sorted([f.stem for f in results_dir.glob("*.json")])
    log = load_processed_log()

    pending_ids: list[str] = []
    skipped_ids: list[str] = []

    for pid in all_pdb_ids:
        entry = log.get(pid)
        if entry is None:
            pending_ids.append(pid)
        elif entry.get("status") == AGG_STATUS_COMPLETED:
            continue
        elif entry.get("status") == AGG_STATUS_SKIPPED:
            skipped_ids.append(pid)
        else:
            pending_ids.append(pid)

    return pending_ids, skipped_ids, len(all_pdb_ids)


def load_pdb_data(
    pdb_id: str,
) -> tuple[dict[str, Any] | None, dict, dict]:
    """Load main data, controversy map, and validation data for a PDB.

    Returns:
        (main_data, controversy_map, validation_data), where main_data is ``None``
        on error and the other values are empty or default dictionaries.
    """
    cfg = get_config()
    results_dir = cfg.aggregated_dir
    logs_dir = cfg.aggregated_dir / "logs"
    validation_dir = cfg.aggregated_dir / "validation_logs"

    main_path = results_dir / f"{pdb_id}.json"
    log_path = logs_dir / f"{pdb_id}_voting_log.json"
    val_path = validation_dir / f"{pdb_id}_validation.json"

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
