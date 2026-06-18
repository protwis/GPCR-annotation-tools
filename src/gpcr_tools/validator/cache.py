"""Persistent cache layer for API validation results.

Both :class:`ValidationCache` and :class:`SequenceCache` use **atomic writes**
(tempfile + ``os.replace``).

Caches are saved once per PDB (batch), not after every API hit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from gpcr_tools.config import POLYMER_FEATURES_CACHE_TTL_DAYS, SEQUENCE_CACHE_TTL_DAYS

logger = logging.getLogger(__name__)


class ValidationCache:
    """Persistent cache for UniProt/PubChem existence checks.

    Keys follow the pattern ``"uniprot:{name}"`` or ``"pubchem:{cid}"``.
    Values are ``bool`` — ``set(key, None)`` is disallowed so that
    ``get()`` returning ``None`` unambiguously means cache miss.
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
        """Persist cache to disk using atomic write."""
        _atomic_json_write(self._path, self._data)


class SequenceCache:
    """Persistent, time-bounded cache for UniProt FASTA sequences.

    Keys are UniProt accessions. Each entry stores the sequence plus the epoch
    time it was fetched; an entry older than ``ttl_days`` is treated as a miss so
    a drifted upstream reference is eventually refetched rather than persisting
    forever. Legacy plain-string entries (pre-TTL caches) have an unknown age and
    are treated as expired, so they refresh once on next use. Atomic writes.
    """

    def __init__(self, path: Path, ttl_days: int = SEQUENCE_CACHE_TTL_DAYS) -> None:
        self._path = path
        self._ttl_seconds = ttl_days * 86400
        self._data: dict[str, dict[str, Any]] = {}
        # Accessions that transiently failed to fetch THIS run. In-memory only:
        # never loaded or saved, so a transient outage is never frozen as a
        # cached fact and a fresh run re-probes. Lets one run skip re-requesting
        # a reference that already failed, instead of re-hitting the dead
        # endpoint once per PDB across the whole batch. Assumes serial use (the
        # detect stage runs PDBs sequentially); not thread/process-safe.
        self._unavailable: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read sequence cache %s: %s", self._path, exc)
            return
        if not isinstance(raw, dict):
            return
        for key, value in raw.items():
            if isinstance(value, dict) and "seq" in value:
                self._data[key] = {
                    "seq": str(value["seq"]),
                    "fetched_at": float(value.get("fetched_at") or 0.0),
                }
            else:
                # Legacy plain-string entry: unknown age -> treat as expired.
                self._data[key] = {"seq": str(value), "fetched_at": 0.0}

    def get(self, key: str, *, now: float | None = None) -> str | None:
        """Return cached sequence, or ``None`` on cache miss or expiry."""
        entry = self._data.get(key)
        if entry is None:
            return None
        current = time.time() if now is None else now
        if current - entry["fetched_at"] > self._ttl_seconds:
            return None
        return str(entry["seq"])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def set(self, key: str, value: str, *, now: float | None = None) -> None:
        """Store a sequence string, stamped with the fetch time."""
        self._data[key] = {"seq": value, "fetched_at": time.time() if now is None else now}

    def mark_unavailable(self, key: str) -> None:
        """Record that *key* transiently failed to fetch this run (in-memory)."""
        self._unavailable.add(key)

    def is_unavailable(self, key: str) -> bool:
        """Whether *key* already transiently failed to fetch this run."""
        return key in self._unavailable

    def save(self) -> None:
        """Persist cache to disk using atomic write (the unavailable set is not saved)."""
        _atomic_json_write(self._path, self._data)


class PolymerFeaturesCache:
    """Persistent, time-bounded cache for RCSB polymer-feature responses.

    Keys are PDB ids; each entry stores the GraphQL ``entry`` dict (the
    transmembrane-helix annotations the oligomer 7TM analysis reads) plus the
    epoch time it was fetched. An entry older than ``ttl_days`` is treated as a
    miss so an upstream revision is eventually refetched. The cache stores only
    successful fetches; a failed (``None``) fetch is never written, so a
    transient outage is not frozen as a fact. Atomic writes.
    """

    def __init__(self, path: Path, ttl_days: int = POLYMER_FEATURES_CACHE_TTL_DAYS) -> None:
        self._path = path
        self._ttl_seconds = ttl_days * 86400
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read polymer-features cache %s: %s", self._path, exc)
            return
        if not isinstance(raw, dict):
            return
        for key, value in raw.items():
            if isinstance(value, dict) and isinstance(value.get("entry"), dict):
                self._data[key] = {
                    "entry": value["entry"],
                    "fetched_at": float(value.get("fetched_at") or 0.0),
                }

    def get(self, key: str, *, now: float | None = None) -> dict[str, Any] | None:
        """Return the cached ``entry`` dict, or ``None`` on cache miss or expiry."""
        entry = self._data.get(key)
        if entry is None:
            return None
        current = time.time() if now is None else now
        if current - entry["fetched_at"] > self._ttl_seconds:
            return None
        value = entry["entry"]
        return value if isinstance(value, dict) else None

    def __contains__(self, key: str) -> bool:
        # TTL-aware so membership matches get(): an expired entry reads as absent,
        # never as present-but-then-None.
        return self.get(key) is not None

    def set(self, key: str, value: dict[str, Any], *, now: float | None = None) -> None:
        """Store a polymer-feature ``entry`` dict, stamped with the fetch time."""
        self._data[key] = {"entry": value, "fetched_at": time.time() if now is None else now}

    def save(self) -> None:
        """Persist cache to disk using atomic write."""
        _atomic_json_write(self._path, self._data)


def _atomic_json_write(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* atomically.

    Uses ``tempfile.NamedTemporaryFile`` in the same directory + ``os.replace``.
    The ``finally`` block guarantees cleanup of the temp file on failure.
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
