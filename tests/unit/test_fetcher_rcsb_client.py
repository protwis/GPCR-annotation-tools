"""Tests for fetcher/rcsb_client.py — raw RCSB download with mocked HTTP."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


def _session_returning(payload: dict[str, Any]) -> MagicMock:
    """A mock session whose .post returns a 200-style response with *payload*."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    session = MagicMock()
    session.post.return_value = resp
    return session


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
def test_nonexistent_pdb_is_not_written(_sleep, configure_paths) -> None:
    """RCSB answers an unknown PDB with HTTP 200 and a null entry. That must be
    treated as 'not found' — no hollow file, no success — rather than a record
    that masquerades as a real structure downstream."""
    from gpcr_tools.config import get_config
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    result = fetch_single_pdb("ZZZZ", session=_session_returning({"data": {"entry": None}}))

    assert result is None
    raw_path = get_config().raw_pdb_json_dir / "ZZZZ.json"
    assert not raw_path.exists()


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
def test_valid_pdb_is_written(_sleep, configure_paths) -> None:
    """A real entry is written verbatim and returned (guard must not reject it)."""
    from gpcr_tools.config import get_config
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    payload = {"data": {"entry": {"rcsb_id": "7W55"}}}
    result = fetch_single_pdb("7W55", session=_session_returning(payload))

    assert result == payload
    raw_path = get_config().raw_pdb_json_dir / "7W55.json"
    assert raw_path.exists()


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
def test_partial_data_with_errors_is_not_persisted(_sleep, configure_paths) -> None:
    """HTTP 200 with BOTH a populated entry AND a top-level errors[] is a PARTIAL
    response (a sub-resolver failed): the entry is missing fields it would
    normally carry. Persisting it would freeze that partial metadata as a
    complete record, so it must be treated as transient — return None and write
    nothing, leaving the entry to be re-fetched on the next run."""
    from gpcr_tools.config import get_config
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    payload = {"data": {"entry": {"rcsb_id": "7W55"}}, "errors": [{"message": "soft"}]}
    result = fetch_single_pdb("7W55", session=_session_returning(payload))

    assert result is None
    raw_path = get_config().raw_pdb_json_dir / "7W55.json"
    assert not raw_path.exists()


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
def test_errors_without_entry_is_failure(_sleep, configure_paths) -> None:
    """Errors with no usable entry remain a hard failure."""
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    payload = {"data": {"entry": None}, "errors": [{"message": "bad id"}]}
    result = fetch_single_pdb("ZZZZ", session=_session_returning(payload))

    assert result is None


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
def test_uses_passed_session_not_bare_requests(_sleep, configure_paths) -> None:
    """The primary download must go through the (retry-enabled) session it is
    handed, so a transient 429/5xx is retried like every other RCSB call."""
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    session = _session_returning({"data": {"entry": {"rcsb_id": "7W55"}}})
    fetch_single_pdb("7W55", session=session)

    session.post.assert_called_once()
