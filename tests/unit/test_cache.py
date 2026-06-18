"""Tests for the persistent cache layer (Epic 4).

Covers: read/write/save cycle, cache miss, file persistence, atomic writes,
and no orphaned temp files on simulated crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpcr_tools.validator.cache import (
    PolymerFeaturesCache,
    SequenceCache,
    ValidationCache,
    _atomic_json_write,
)


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

    def test_expired_entry_is_a_miss(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json", ttl_days=30)
        cache.set("P12345", "MDEFGH", now=1000.0)
        # Fresh up to and including exactly the TTL (strict > boundary); expired after.
        assert cache.get("P12345", now=1000.0 + 29 * 86400) == "MDEFGH"
        assert cache.get("P12345", now=1000.0 + 30 * 86400) == "MDEFGH"
        assert cache.get("P12345", now=1000.0 + 31 * 86400) is None

    def test_legacy_plain_string_entry_treated_as_expired(self, tmp_path: Path) -> None:
        # A pre-TTL cache stored bare strings; unknown age -> refetch (miss).
        path = tmp_path / "seq.json"
        path.write_text(json.dumps({"P12345": "MDEFGH"}))
        cache = SequenceCache(path)
        assert "P12345" in cache  # the key is present
        assert cache.get("P12345") is None  # but treated as expired

    def test_reload_preserves_freshness(self, tmp_path: Path) -> None:
        path = tmp_path / "seq.json"
        cache1 = SequenceCache(path)
        cache1.set("P12345", "ACDEF", now=2000.0)
        cache1.save()
        cache2 = SequenceCache(path)
        assert cache2.get("P12345", now=2000.0 + 1 * 86400) == "ACDEF"
        assert cache2.get("P12345", now=2000.0 + 31 * 86400) is None

    def test_mark_unavailable_default_false(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        assert cache.is_unavailable("P12345") is False

    def test_mark_unavailable_is_in_memory_and_not_persisted(self, tmp_path: Path) -> None:
        # The transient-failure guard is per-run: never saved, so a fresh run
        # re-probes the accession rather than freezing an outage as a cached fact.
        path = tmp_path / "seq.json"
        cache = SequenceCache(path)
        cache.mark_unavailable("P12345")
        assert cache.is_unavailable("P12345") is True
        cache.save()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert "P12345" not in on_disk
        assert SequenceCache(path).is_unavailable("P12345") is False


class TestPolymerFeaturesCache:
    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = PolymerFeaturesCache(tmp_path / "pf.json")
        assert cache.get("7RKF") is None

    def test_set_and_get_roundtrip(self, tmp_path: Path) -> None:
        cache = PolymerFeaturesCache(tmp_path / "pf.json")
        entry = {"polymer_entities": [{"x": 1}]}
        cache.set("7RKF", entry)
        assert cache.get("7RKF") == entry

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "pf.json"
        entry = {"polymer_entities": [{"x": 1}]}
        cache = PolymerFeaturesCache(path)
        cache.set("7RKF", entry)
        cache.save()
        assert PolymerFeaturesCache(path).get("7RKF") == entry

    def test_expired_entry_is_a_miss(self, tmp_path: Path) -> None:
        # An entry older than the TTL is refetched rather than persisting forever.
        cache = PolymerFeaturesCache(tmp_path / "pf.json", ttl_days=30)
        cache.set("7RKF", {"polymer_entities": []}, now=0.0)
        # 31 days later (in seconds) the entry is past its TTL.
        assert cache.get("7RKF", now=31 * 86400) is None

    def test_fresh_entry_within_ttl_is_a_hit(self, tmp_path: Path) -> None:
        cache = PolymerFeaturesCache(tmp_path / "pf.json", ttl_days=30)
        entry = {"polymer_entities": []}
        cache.set("7RKF", entry, now=0.0)
        assert cache.get("7RKF", now=29 * 86400) == entry

    def test_legacy_or_malformed_entry_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "pf.json"
        path.write_text(json.dumps({"7RKF": "not-a-dict"}), encoding="utf-8")
        assert PolymerFeaturesCache(path).get("7RKF") is None

    def test_contains_is_ttl_aware(self, tmp_path: Path) -> None:
        # Membership must match get(): an expired entry reads as absent, never as
        # present-but-then-None (which would let a caller trust a stale key).
        cache = PolymerFeaturesCache(tmp_path / "pf.json", ttl_days=30)
        cache.set("7RKF", {"polymer_entities": []})  # stamped "now" -> fresh
        assert "7RKF" in cache
        # Re-set with a deterministic stamp, then verify expiry by reloading and
        # confirming the get()/__contains__ pair agree once past the TTL.
        expired = PolymerFeaturesCache(tmp_path / "expired.json", ttl_days=0)
        expired.set("7RKF", {"polymer_entities": []}, now=0.0)
        # ttl_days=0 -> any positive age is expired; the stored entry is in _data
        # but must read as absent through both get() and membership.
        assert expired._data.get("7RKF") is not None
        assert expired.get("7RKF") is None
        assert "7RKF" not in expired


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
