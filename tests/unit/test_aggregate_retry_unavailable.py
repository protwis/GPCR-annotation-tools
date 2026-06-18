"""Tests for --retry-unavailable.

``_pdbs_with_api_unavailable`` selects exactly the PDBs whose last validation
report recorded a transient API abstention (``[API_UNAVAILABLE]``); ``aggregate_all``
re-runs only those (and only when their AI results are still present), so a retry
re-hits just the spots that failed -- cached definitive results are reused. The
retry is incompatible with ``skip_api_checks`` (which would skip the very checks
the retry exists to re-run).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gpcr_tools.aggregator.runner as runner
from gpcr_tools.aggregator.runner import AggregateResult, _pdbs_with_api_unavailable


def _write_report(vdir, pdb: str, warnings: list[str]) -> None:
    (vdir / f"{pdb}_validation.json").write_text(
        json.dumps({"critical_warnings": warnings}), encoding="utf-8"
    )


# --- _pdbs_with_api_unavailable -------------------------------------------------


def test_selects_only_pdbs_with_api_unavailable(tmp_path) -> None:
    vdir = tmp_path / "validation_logs"
    vdir.mkdir()
    _write_report(vdir, "AAAA", ["[API_UNAVAILABLE] at 'signaling_partners': UniProt down"])
    _write_report(vdir, "BBBB", ["[CHIMERIC G-PROTEIN] at 'signaling_partners': confirm"])
    _write_report(vdir, "CCCC", [])
    cfg = SimpleNamespace(aggregated_dir=tmp_path)
    assert _pdbs_with_api_unavailable(cfg) == ["AAAA"]


def test_no_validation_logs_dir_returns_empty(tmp_path) -> None:
    cfg = SimpleNamespace(aggregated_dir=tmp_path)  # no validation_logs/ subdir
    assert _pdbs_with_api_unavailable(cfg) == []


def test_malformed_report_is_skipped(tmp_path) -> None:
    vdir = tmp_path / "validation_logs"
    vdir.mkdir()
    (vdir / "AAAA_validation.json").write_text("{not valid json", encoding="utf-8")
    _write_report(vdir, "BBBB", ["[API_UNAVAILABLE] at 'ligands': PubChem down"])
    cfg = SimpleNamespace(aggregated_dir=tmp_path)
    assert _pdbs_with_api_unavailable(cfg) == ["BBBB"]


# --- aggregate_all(retry_unavailable=...) --------------------------------------


def _stub_aggregate_all(monkeypatch, tmp_path, *, flagged, result_warnings=None):
    """Stub aggregate_all's environment so only the selection + summary logic runs.

    Returns the list that records which PDB ids aggregate_pdb was called with.
    """
    cfg = SimpleNamespace(cache_dir=tmp_path, ai_results_dir=tmp_path, aggregated_dir=tmp_path)
    monkeypatch.setattr(runner, "get_config", lambda: cfg)
    monkeypatch.setattr("gpcr_tools.workspace.validate_contract", lambda c: None)
    monkeypatch.setattr(runner, "ValidationCache", lambda *a, **k: MagicMock())
    monkeypatch.setattr(runner, "SequenceCache", lambda *a, **k: MagicMock())
    monkeypatch.setattr(runner, "JsonCache", lambda *a, **k: MagicMock())
    monkeypatch.setattr(runner, "_pdbs_with_api_unavailable", lambda c: list(flagged))
    monkeypatch.setattr(runner, "pdb_has_runs", lambda p: True)
    monkeypatch.setattr(runner, "_update_aggregate_log", lambda *a, **k: None)

    called: list[str] = []

    def _spy(pdb_id, **kwargs):
        called.append(pdb_id)
        return AggregateResult(pdb_id=pdb_id, success=True, warnings=list(result_warnings or []))

    monkeypatch.setattr(runner, "aggregate_pdb", _spy)
    return called


def test_retry_unavailable_runs_only_flagged_pdbs(tmp_path, monkeypatch) -> None:
    called = _stub_aggregate_all(monkeypatch, tmp_path, flagged=["AAAA", "BBBB"])
    results = runner.aggregate_all(retry_unavailable=True)
    assert sorted(called) == ["AAAA", "BBBB"]
    assert {r.pdb_id for r in results} == {"AAAA", "BBBB"}


def test_summary_warns_when_abstention_persists(tmp_path, monkeypatch, caplog) -> None:
    _stub_aggregate_all(
        monkeypatch,
        tmp_path,
        flagged=["AAAA"],
        result_warnings=["[API_UNAVAILABLE] at 'ligands': still down"],
    )
    with caplog.at_level("WARNING"):
        runner.aggregate_all(retry_unavailable=True)
    assert "transient API failure" in caplog.text
    assert "AAAA" in caplog.text


def test_skip_api_checks_combo_is_rejected() -> None:
    with pytest.raises(ValueError, match="skip_api_checks"):
        runner.aggregate_all(retry_unavailable=True, skip_api_checks=True)
