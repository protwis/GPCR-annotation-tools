"""Tests for validator.geometry pure helpers (no coordinate I/O or network).

The burial metric is the load-bearing separator between a ligand enclosed in a
pocket and one lying on the membrane-facing surface, so it is exercised directly
on synthetic environments.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
import requests

from gpcr_tools.config import API_MAX_RETRIES
from gpcr_tools.validator import geometry as geom
from gpcr_tools.validator.geometry import (
    _SPHERE_DIRECTIONS,
    LigandCopyGeometry,
    _burial,
    centroid,
    fetch_structure,
    fibonacci_directions,
    load_structure,
)


class TestFibonacciDirections:
    def test_count_and_unit_length(self) -> None:
        dirs = fibonacci_directions(200)
        assert len(dirs) == 200
        for d in dirs:
            assert abs(d.length() - 1.0) < 1e-6


class TestBurial:
    def test_fully_enclosed_is_one(self) -> None:
        # An environment atom in every sampled direction shields the centroid fully.
        origin = gemmi.Position(0.0, 0.0, 0.0)
        env = [gemmi.Position(d.x * 3.0, d.y * 3.0, d.z * 3.0) for d in _SPHERE_DIRECTIONS]
        assert _burial(origin, env) == 1.0

    def test_one_sided_is_low(self) -> None:
        # Atoms only on one hemisphere (a surface-exposed copy) cover well under half.
        origin = gemmi.Position(0.0, 0.0, 0.0)
        env = [
            gemmi.Position(d.x * 3.0, d.y * 3.0, d.z * 3.0) for d in _SPHERE_DIRECTIONS if d.x > 0.5
        ]
        assert _burial(origin, env) < 0.8

    def test_no_environment_is_zero(self) -> None:
        assert _burial(gemmi.Position(0.0, 0.0, 0.0), []) == 0.0


class TestCentroid:
    def test_mean_position(self) -> None:
        atoms = []
        for x in (0.0, 2.0):
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(x, 0.0, 0.0)
            atoms.append(atom)
        c = centroid(atoms)
        assert (c.x, c.y, c.z) == (1.0, 0.0, 0.0)


class TestLigandCopyGeometry:
    def test_primary_chain_and_residue_numbers(self) -> None:
        copy = LigandCopyGeometry(
            auth_chain="R",
            seq_id=601,
            burial=0.99,
            pocket_residues=frozenset({("R", 104), ("R", 107), ("A", 12)}),
            contacts_partner=True,
        )
        assert copy.n_pocket_residues == 3
        assert copy.primary_gpcr_chain() == "R"  # the chain with the most residues
        assert copy.residue_numbers_on("R") == frozenset({104, 107})
        assert copy.residue_numbers_on("A") == frozenset({12})

    def test_primary_chain_none_when_empty(self) -> None:
        copy = LigandCopyGeometry("R", 1, 0.0, frozenset(), False)
        assert copy.primary_gpcr_chain() is None


class _FakeResp:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        pass


class TestFetchStructure:
    def test_cache_hit_does_not_hit_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cached = tmp_path / "structure_files" / "9iix.cif.gz"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")

        def boom(*a: object, **k: object) -> None:
            raise AssertionError("network must not be used on a cache hit")

        monkeypatch.setattr(requests, "get", boom)
        assert fetch_structure("9IIX", tmp_path) == cached

    def test_downloads_and_caches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(b"coords"))
        path = fetch_structure("9IIX", tmp_path)
        assert path is not None and path.read_bytes() == b"coords"

    def test_download_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*a: object, **k: object) -> None:
            raise requests.RequestException("boom")

        monkeypatch.setattr(requests, "get", fail)
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        assert fetch_structure("9IIX", tmp_path) is None

    def test_transient_status_then_success_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A transient 503 on the first attempt must be retried; the next 200 wins.
        responses = [_FakeResp(b"", status_code=503), _FakeResp(b"coords", status_code=200)]
        monkeypatch.setattr(requests, "get", lambda *a, **k: responses.pop(0))
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        path = fetch_structure("9IIX", tmp_path)
        assert path is not None and path.read_bytes() == b"coords"
        assert responses == []  # both responses consumed (one retry happened)

    def test_404_returns_none_without_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A definitive 404 is not retried -- the request is made exactly once.
        calls = {"n": 0}

        def get(*a: object, **k: object) -> _FakeResp:
            calls["n"] += 1
            return _FakeResp(b"", status_code=404)

        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        assert fetch_structure("9IIX", tmp_path) is None
        assert calls["n"] == 1

    def test_all_attempts_transient_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def get(*a: object, **k: object) -> _FakeResp:
            calls["n"] += 1
            return _FakeResp(b"", status_code=503)

        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        assert fetch_structure("9IIX", tmp_path) is None
        assert calls["n"] == API_MAX_RETRIES


class TestLoadStructure:
    def test_missing_download_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(geom, "fetch_structure", lambda *a, **k: None)
        assert load_structure("9IIX", tmp_path) is None

    def test_unparseable_file_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bogus = tmp_path / "bogus.cif.gz"
        bogus.write_bytes(b"not a structure")
        monkeypatch.setattr(geom, "fetch_structure", lambda *a, **k: bogus)
        assert load_structure("9IIX", tmp_path) is None
