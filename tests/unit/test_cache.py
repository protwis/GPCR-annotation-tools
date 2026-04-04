"""Tests for the persistent cache layer (Epic 4).

Covers: read/write/save cycle, cache miss, file persistence, atomic writes,
and no orphaned temp files on simulated crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpcr_tools.validator.cache import SequenceCache, ValidationCache, _atomic_json_write


class TestValidationCache:
    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        assert cache.get("uniprot:missing") is None

    def test_set_and_get(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        cache.set("uniprot:drd2_human", True)
        assert cache.get("uniprot:drd2_human") is True

    def test_contains(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "cache.json")
        assert "uniprot:x" not in cache
        cache.set("uniprot:x", False)
        assert "uniprot:x" in cache

    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache1 = ValidationCache(path)
        cache1.set("uniprot:test", True)
        cache1.set("pubchem:123", False)
        cache1.save()

        cache2 = ValidationCache(path)
        assert cache2.get("uniprot:test") is True
        assert cache2.get("pubchem:123") is False

    def test_load_from_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"uniprot:a": True}), encoding="utf-8")
        cache = ValidationCache(path)
        assert cache.get("uniprot:a") is True

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("{bad", encoding="utf-8")
        cache = ValidationCache(path)
        assert cache.get("any") is None

    def test_load_missing_file(self, tmp_path: Path) -> None:
        cache = ValidationCache(tmp_path / "nonexistent.json")
        assert cache.get("any") is None


class TestSequenceCache:
    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        assert cache.get("P12345") is None

    def test_set_and_get(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        cache.set("P12345", "MDEFGH")
        assert cache.get("P12345") == "MDEFGH"

    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "seq.json"
        cache1 = SequenceCache(path)
        cache1.set("P12345", "ACDEF")
        cache1.save()

        cache2 = SequenceCache(path)
        assert cache2.get("P12345") == "ACDEF"

    def test_contains(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        assert "P999" not in cache
        cache.set("P999", "ABC")
        assert "P999" in cache


class TestAtomicWrite:
    def test_file_written(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        _atomic_json_write(path, {"key": "value"})
        assert path.exists()
        with path.open() as f:
            assert json.load(f) == {"key": "value"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "out.json"
        _atomic_json_write(path, {"ok": True})
        assert path.exists()

    def test_no_orphaned_temp_on_failure(self, tmp_path: Path) -> None:
        """Blood Lesson 2: temp files must be cleaned up on write failure."""
        path = tmp_path / "out.json"

        class BadSerializable:
            def __init__(self) -> None:
                pass

        with pytest.raises(TypeError):
            _atomic_json_write(path, BadSerializable())

        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned temp files: {tmp_files}"
        assert not path.exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        _atomic_json_write(path, {"v": 1})
        _atomic_json_write(path, {"v": 2})
        with path.open() as f:
            assert json.load(f) == {"v": 2}

    def test_atomic_no_partial_write(self, tmp_path: Path) -> None:
        """If json.dump fails mid-write, original file is untouched."""
        path = tmp_path / "out.json"
        _atomic_json_write(path, {"original": True})

        # Simulate failure during dump
        with (
            patch("gpcr_tools.validator.cache.json.dump", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            _atomic_json_write(path, {"new": True})

        # Original file should be intact
        with path.open() as f:
            assert json.load(f) == {"original": True}
        # No orphaned temp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []
