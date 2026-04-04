"""Persistent cache layer for API validation results.

Both :class:`ValidationCache` and :class:`SequenceCache` use **atomic writes**
(tempfile + ``os.replace``) — Blood Lesson 2.

Caches are saved once per PDB (batch), not after every API hit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ValidationCache:
    """Persistent cache for UniProt/PubChem existence checks.

    Keys follow the pattern ``"uniprot:{name}"`` or ``"pubchem:{cid}"``.
    Values are ``bool`` — ``set(key, None)`` is disallowed so that
    ``get()`` returning ``None`` unambiguously means cache miss (Review 5).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, bool] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data = {k: bool(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read validation cache %s: %s", self._path, exc)

    def get(self, key: str) -> bool | None:
        """Return cached value, or ``None`` on cache miss."""
        return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def set(self, key: str, value: bool) -> None:
        """Store a validation result.  *value* must be ``bool``."""
        self._data[key] = value

    def save(self) -> None:
        """Persist cache to disk using atomic write (Blood Lesson 2)."""
        _atomic_json_write(self._path, self._data)


class SequenceCache:
    """Persistent cache for UniProt FASTA sequences.

    Keys are UniProt accessions, values are sequence strings.
    Uses the same atomic write pattern as :class:`ValidationCache`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data = {k: str(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read sequence cache %s: %s", self._path, exc)

    def get(self, key: str) -> str | None:
        """Return cached sequence, or ``None`` on cache miss."""
        return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def set(self, key: str, value: str) -> None:
        """Store a sequence string."""
        self._data[key] = value

    def save(self) -> None:
        """Persist cache to disk using atomic write (Blood Lesson 2)."""
        _atomic_json_write(self._path, self._data)


def _atomic_json_write(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* atomically.

    Uses ``tempfile.NamedTemporaryFile`` in the same directory + ``os.replace``.
    The ``finally`` block guarantees cleanup of the temp file on failure
    (Blood Lesson 2, Trap B).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(path.parent),
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fd:
            tmp_path = fd.name
            json.dump(data, fd, indent=2)
        os.replace(tmp_path, str(path))
        tmp_path = None  # committed, no cleanup needed
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
