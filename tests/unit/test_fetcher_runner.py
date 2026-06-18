"""Tests for fetcher/runner.py — fetch/enrich orchestration seams."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Minimal workspace with the dirs run_fetch's caches expect."""
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))

    from gpcr_tools.config import reset_config

    reset_config()
    (tmp_path / "cache").mkdir()
    yield tmp_path
    reset_config()


def test_failed_raw_download_skips_enrichment(workspace: Path) -> None:
    # When the raw download returns None — e.g. a PARTIAL GraphQL response that
    # is not persisted, or any transient download failure — the runner must skip
    # enrichment for that PDB rather than enrich a non-existent raw record.
    from gpcr_tools.fetcher import runner

    # The runner exits non-zero when any PDB fails; the seam under test is that
    # enrichment is skipped, not the exit code.
    with (
        patch.object(runner, "fetch_single_pdb", return_value=None) as m_fetch,
        patch.object(runner, "enrich_single_pdb") as m_enrich,
        pytest.raises(SystemExit),
    ):
        runner.run_fetch(pdb_id="7W55")

    m_fetch.assert_called_once()
    m_enrich.assert_not_called()


def test_successful_raw_download_proceeds_to_enrichment(workspace: Path) -> None:
    # The contrast case: a non-None raw download proceeds to enrichment.
    from gpcr_tools.fetcher import runner

    with (
        patch.object(runner, "fetch_single_pdb", return_value={"data": {"entry": {}}}),
        patch.object(runner, "enrich_single_pdb", return_value=True) as m_enrich,
    ):
        runner.run_fetch(pdb_id="7W55")

    m_enrich.assert_called_once()
