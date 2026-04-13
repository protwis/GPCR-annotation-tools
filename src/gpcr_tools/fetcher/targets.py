"""Target file reader — parse PDB ID lists from text files.

Used by the ``fetch`` command and available to all commands via ``--targets``.
Format: one PDB ID per line, ``#`` comments and blank lines ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_targets(path: Path) -> list[str]:
    """Read PDB IDs from a text file.

    Return a list of uppercased PDB IDs, preserving order, skipping
    comments (``#``) and blank lines.  Duplicates are removed while
    preserving first-occurrence order.
    """
    if not path.exists():
        logger.error("Target file not found: %s", path)
        return []

    seen: set[str] = set()
    result: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pdb_id = stripped.upper()
            if pdb_id not in seen:
                seen.add(pdb_id)
                result.append(pdb_id)

    return result
