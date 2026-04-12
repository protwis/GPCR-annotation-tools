"""Generalized JSON cache utility for enrichment caches.

Each cache is a JSON file stored under ``config.cache_dir``.  Caches include
a ``_version`` key; when the schema version changes, the stale cache is
discarded and replaced.

Cache files are written atomically (tempfile + ``os.replace``) to prevent
corruption on interrupted writes.

Thread safety: caches are designed for **single-writer** use within one
pipeline invocation.  Never share a cache instance between concurrent writers.
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


class JsonCache:
    """In-memory dict backed by an atomic-write JSON file.

    Parameters
    ----------
    path
        Absolute path to the cache file.
    version
        Schema version.  If the on-disk version differs, the cache is reset.
    """

    def __init__(self, path: Path, *, version: int = 1) -> None:
        self._path = path
        self._version = version
        self._data: dict[str, Any] = {}
        self._dirty = False
        self._load()

    # -- public API ----------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if absent."""
        return self._data.get(key)

    def has(self, key: str) -> bool:
        """Check if *key* is in the cache."""
        return key in self._data

    def set(self, key: str, value: Any) -> None:
        """Store *value* for *key* and mark cache dirty."""
        self._data[key] = value
        self._dirty = True

    def save(self) -> None:
        """Persist to disk via atomic write.  No-op if not dirty."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        envelope: dict[str, Any] = {"_version": self._version, **self._data}
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self._path.parent),
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as fd:
                tmp_path = fd.name
                json.dump(envelope, fd, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(self._path))
            tmp_path = None
            self._dirty = False
        except OSError as exc:
            logger.warning("Failed to save cache %s: %s", self._path.name, exc)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    # -- internals -----------------------------------------------------------

    def _load(self) -> None:
        """Load from disk, discarding stale versions."""
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load cache %s: %s", self._path.name, exc)
            return

        if not isinstance(raw, dict):
            return

        disk_version = raw.pop("_version", None)
        if disk_version != self._version:
            logger.info(
                "Cache %s version mismatch (disk=%s, expected=%s), resetting",
                self._path.name,
                disk_version,
                self._version,
            )
            return

        self._data = raw
