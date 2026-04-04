"""Tests for AI results loader (Epic 2)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpcr_tools.aggregator.ai_results_loader import (
    get_pending_pdb_ids,
    load_ai_runs,
)
from gpcr_tools.config import reset_config


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    reset_config()


@pytest.fixture()
def workspace(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    ai_dir = tmp_path / "ai_results"
    ai_dir.mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    return tmp_path


class TestLoadAiRuns:
    def test_multi_run_directory(self, workspace: Any) -> None:
        pdb_dir = workspace / "ai_results" / "5G53"
        pdb_dir.mkdir()
        for i in range(1, 4):
            (pdb_dir / f"run_{i:02d}.json").write_text(json.dumps({"run": i}), encoding="utf-8")
        runs = load_ai_runs("5G53")
        assert len(runs) == 3
        assert runs[0]["run"] == 1
        assert runs[2]["run"] == 3

    def test_ordering(self, workspace: Any) -> None:
        pdb_dir = workspace / "ai_results" / "XORD"
        pdb_dir.mkdir()
        (pdb_dir / "run_03.json").write_text(json.dumps({"r": 3}), encoding="utf-8")
        (pdb_dir / "run_01.json").write_text(json.dumps({"r": 1}), encoding="utf-8")
        (pdb_dir / "run_02.json").write_text(json.dumps({"r": 2}), encoding="utf-8")
        runs = load_ai_runs("XORD")
        assert [r["r"] for r in runs] == [1, 2, 3]

    def test_skip_non_json(self, workspace: Any) -> None:
        pdb_dir = workspace / "ai_results" / "SKP1"
        pdb_dir.mkdir()
        (pdb_dir / "run_01.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (pdb_dir / "notes.txt").write_text("not a run", encoding="utf-8")
        (pdb_dir / "run_02.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        runs = load_ai_runs("SKP1")
        assert len(runs) == 2

    def test_skip_corrupt_json(self, workspace: Any) -> None:
        pdb_dir = workspace / "ai_results" / "BAD1"
        pdb_dir.mkdir()
        (pdb_dir / "run_01.json").write_text("{invalid json", encoding="utf-8")
        (pdb_dir / "run_02.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        runs = load_ai_runs("BAD1")
        assert len(runs) == 1
        assert runs[0]["ok"] is True

    def test_missing_directory(self, workspace: Any) -> None:
        runs = load_ai_runs("NONEXIST")
        assert runs == []

    def test_empty_directory(self, workspace: Any) -> None:
        pdb_dir = workspace / "ai_results" / "EMPTY"
        pdb_dir.mkdir()
        runs = load_ai_runs("EMPTY")
        assert runs == []

    def test_skip_non_dict_top_level(self, workspace: Any) -> None:
        pdb_dir = workspace / "ai_results" / "LIST"
        pdb_dir.mkdir()
        (pdb_dir / "run_01.json").write_text("[1, 2, 3]", encoding="utf-8")
        runs = load_ai_runs("LIST")
        assert runs == []


class TestGetPendingPdbIds:
    def test_all_pending(self, workspace: Any) -> None:
        for pid in ("AAA", "BBB"):
            d = workspace / "ai_results" / pid
            d.mkdir()
            (d / "run_01.json").write_text(json.dumps({}), encoding="utf-8")
        result = get_pending_pdb_ids()
        assert result == ["AAA", "BBB"]

    def test_skip_already_processed(self, workspace: Any) -> None:
        for pid in ("AAA", "BBB", "CCC"):
            d = workspace / "ai_results" / pid
            d.mkdir()
            (d / "run_01.json").write_text(json.dumps({}), encoding="utf-8")
        log_path = workspace / "state" / "aggregate_log.json"
        log_path.write_text(json.dumps({"AAA": "done"}), encoding="utf-8")
        result = get_pending_pdb_ids()
        assert result == ["BBB", "CCC"]

    def test_no_ai_results_dir(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        reset_config()
        result = get_pending_pdb_ids()
        assert result == []
