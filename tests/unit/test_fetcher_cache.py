"""Tests for fetcher/cache.py — JsonCache with atomic writes."""

from __future__ import annotations

import json
from pathlib import Path

from gpcr_tools.fetcher.cache import JsonCache


class TestJsonCacheBasics:
    """Basic get/set/has operations."""

    def test_empty_cache(self, tmp_path: Path) -> None:
        cache = JsonCache(tmp_path / "cache.json")
        assert cache.get("foo") is None
        assert not cache.has("foo")

    def test_set_and_get(self, tmp_path: Path) -> None:
        cache = JsonCache(tmp_path / "cache.json")
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.has("key1")

    def test_set_none_value(self, tmp_path: Path) -> None:
        cache = JsonCache(tmp_path / "cache.json")
        cache.set("key1", None)
        assert cache.has("key1")
        assert cache.get("key1") is None


class TestJsonCachePersistence:
    """Save/load round-trip."""

    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache1 = JsonCache(path)
        cache1.set("a", 1)
        cache1.set("b", [1, 2, 3])
        cache1.save()

        cache2 = JsonCache(path)
        assert cache2.get("a") == 1
        assert cache2.get("b") == [1, 2, 3]

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "cache.json"
        cache = JsonCache(path)
        cache.set("x", "y")
        cache.save()
        assert path.exists()

    def test_no_op_save_when_clean(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache = JsonCache(path)
        cache.save()  # should not create file if nothing was set
        assert not path.exists()

    def test_atomic_write_includes_version(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache = JsonCache(path, version=42)
        cache.set("k", "v")
        cache.save()

        raw = json.loads(path.read_text())
        assert raw["_version"] == 42
        assert raw["k"] == "v"


class TestJsonCacheVersioning:
    """Version mismatch handling."""

    def test_stale_version_resets_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"

        # Save with version 1
        c1 = JsonCache(path, version=1)
        c1.set("old_key", "old_val")
        c1.save()

        # Load with version 2 — should discard
        c2 = JsonCache(path, version=2)
        assert c2.get("old_key") is None

    def test_same_version_preserves_data(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"

        c1 = JsonCache(path, version=3)
        c1.set("key", "val")
        c1.save()

        c2 = JsonCache(path, version=3)
        assert c2.get("key") == "val"

    def test_missing_version_on_disk_resets(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"key": "val"}))

        cache = JsonCache(path, version=1)
        assert cache.get("key") is None  # reset because no _version


class TestJsonCacheCorruption:
    """Corrupt/invalid cache files."""

    def test_corrupt_json_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("{invalid json")

        cache = JsonCache(path)
        assert cache.get("anything") is None

    def test_non_dict_json_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("[1, 2, 3]")

        cache = JsonCache(path)
        assert cache.get("anything") is None
