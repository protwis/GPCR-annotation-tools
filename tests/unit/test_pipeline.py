"""Tests for end-to-end pipeline orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpcr_tools import pipeline
from gpcr_tools.config import SUPPORTED_CONTRACT_VERSION, get_config, reset_config


@pytest.fixture()
def cfg(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    c = get_config()
    c.enriched_dir.mkdir(parents=True, exist_ok=True)
    # The pipeline now validates the storage contract before running any stage,
    # so a valid contract must exist in the test workspace.
    c.contract_file.parent.mkdir(parents=True, exist_ok=True)
    c.contract_file.write_text(json.dumps({"storage_contract_version": SUPPORTED_CONTRACT_VERSION}))
    return c


def _patch_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr("gpcr_tools.fetcher.runner.run_fetch", lambda **k: calls.append("fetch"))
    monkeypatch.setattr(
        "gpcr_tools.papers.runner.run_fetch_papers", lambda **k: calls.append("fetch-papers")
    )
    monkeypatch.setattr(
        "gpcr_tools.detector.stage.run_detect_stage", lambda **k: calls.append("detect")
    )
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.run_annotation_stage", lambda **k: calls.append("annotate")
    )
    monkeypatch.setattr(
        "gpcr_tools.aggregator.runner.aggregate_all",
        lambda **k: (calls.append("aggregate"), [])[1],
    )
    monkeypatch.setattr(
        "gpcr_tools.aggregator.runner.aggregate_pdb",
        lambda pdb_id, **k: (
            calls.append(f"aggregate_pdb:{pdb_id}"),
            SimpleNamespace(success=True),
        )[1],
    )


def test_dry_run_executes_nothing(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(dry_run=True)
    assert calls == []


def test_runs_stages_in_dependency_order(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (cfg.enriched_dir / "7W55.json").write_text("{}")  # fetch prerequisite
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline()
    assert calls == ["fetch", "fetch-papers", "detect", "annotate", "aggregate"]


def test_batch_mode_stops_after_submit(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (cfg.enriched_dir / "7W55.json").write_text("{}")
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(batch=True)
    # Batch results arrive asynchronously -> do not aggregate in the same run.
    assert calls == ["fetch", "fetch-papers", "detect", "annotate"]


def test_skip_fetch_papers(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (cfg.enriched_dir / "7W55.json").write_text("{}")
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(skip_fetch_papers=True)
    assert calls == ["fetch", "detect", "annotate", "aggregate"]


def test_stops_when_no_enriched_data(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # enriched dir is empty -> nothing to annotate; stop after fetch.
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline()
    assert calls == ["fetch"]


def test_single_pdb_aggregates_only_that_pdb(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """`pipeline <PDB>` must aggregate just that PDB, not sweep every pending
    one (which would also mark unrelated PDBs processed)."""
    (cfg.enriched_dir / "7W55.json").write_text("{}")
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(pdb_id="7W55")
    assert "aggregate_pdb:7W55" in calls
    assert "aggregate" not in calls  # aggregate_all (sweep-all) must NOT run


def test_runs_count_flows_to_dry_run_plan(
    cfg: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """--runs must reach the plan (it was previously unwired in the CLI)."""
    import logging

    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    with caplog.at_level(logging.INFO, logger="gpcr_tools.pipeline"):
        pipeline.run_pipeline(num_runs=3, dry_run=True)
    assert "runs=3" in caplog.text


def test_dry_run_logs_the_plan(
    cfg: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    with caplog.at_level(logging.INFO, logger="gpcr_tools.pipeline"):
        pipeline.run_pipeline(dry_run=True)
    assert "would run" in caplog.text  # the preview is emitted (not swallowed)
    assert calls == []


def test_stale_contract_fails_fast_before_any_stage(
    cfg: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale storage contract must fail fast BEFORE any (expensive) stage
    runs -- not crash late at the curate step after AI annotation completed.
    """
    # Overwrite the valid contract with an unsupported (stale) version.
    cfg.contract_file.write_text(json.dumps({"storage_contract_version": 1}))
    (cfg.enriched_dir / "7W55.json").write_text("{}")
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)

    with pytest.raises(SystemExit):
        pipeline.run_pipeline()

    # The gate fired before fetch -- no stage ran.
    assert calls == []


def test_dry_run_skips_contract_gate(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run only prints the plan, so it must not fail on a stale contract."""
    cfg.contract_file.write_text(json.dumps({"storage_contract_version": 1}))
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(dry_run=True)  # must not raise
    assert calls == []
