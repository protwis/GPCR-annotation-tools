"""Audit trail logging for the CSV generator.

Records every accept/edit/delete action the expert takes, written
as a JSONL file for full traceability.
"""

import json
from datetime import UTC, datetime
from typing import Any

from gpcr_tools.config import AUDIT_TRAIL_FILE, OUTPUT_DIR
from gpcr_tools.csv_generator.ui import console


def log_audit_trail(pdb_id: str, path: str, action: str, orig_val: Any, final_val: Any) -> None:
    """Append a single audit entry to the JSONL audit trail file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "pdb_id": pdb_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "field_path": path,
        "action": action,
        "original_value": orig_val,
        "final_value": final_val,
    }
    try:
        with open(AUDIT_TRAIL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        console.print(f"[bold red]FATAL: Failed to write audit trail: {e}[/bold red]")
