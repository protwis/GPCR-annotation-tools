"""Tests for the Gemini API client with rate limiting."""

import time
import warnings
from collections import deque
from unittest.mock import MagicMock

import pytest
from google.genai import Client

from gpcr_tools.annotator import gemini_client


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch):
    """Ensure each test starts with a clean global client."""
    monkeypatch.delenv("GPCR_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GPCR_GEMINI_API_KEYS", raising=False)
    gemini_client.reset_client()
    yield
    gemini_client.reset_client()


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    def test_new_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_GEMINI_API_KEY", "new_key_abc")
        assert gemini_client._resolve_api_key() == "new_key_abc"

    def test_legacy_env_var_with_deprecation_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_GEMINI_API_KEYS", "legacy1,legacy2")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            key = gemini_client._resolve_api_key()
        assert key == "legacy1"
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()

    def test_new_takes_priority_over_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_GEMINI_API_KEY", "new_key")
        monkeypatch.setenv("GPCR_GEMINI_API_KEYS", "legacy_key")
        assert gemini_client._resolve_api_key() == "new_key"

    def test_missing_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="GPCR_GEMINI_API_KEY"):
            gemini_client._resolve_api_key()

    def test_empty_string_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_GEMINI_API_KEY", "  ")
        with pytest.raises(RuntimeError, match="GPCR_GEMINI_API_KEY"):
            gemini_client._resolve_api_key()


# ---------------------------------------------------------------------------
# RateLimitedClient
# ---------------------------------------------------------------------------


class TestRateLimitedClient:
    def test_get_client_returns_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock(spec=Client)
        monkeypatch.setattr(
            "gpcr_tools.annotator.gemini_client.Client",
            lambda api_key: mock_client,
        )

        rl = gemini_client.RateLimitedClient("test_key")
        result = rl.get_client()
        assert result is mock_client

    def test_records_timestamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock(spec=Client)
        monkeypatch.setattr(
            "gpcr_tools.annotator.gemini_client.Client",
            lambda api_key: mock_client,
        )

        rl = gemini_client.RateLimitedClient("test_key")
        rl.get_client()
        rl.get_client()
        assert len(rl._timestamps) == 2

    def test_rate_limit_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When RPM_LIMIT is reached, get_client blocks until window clears."""
        mock_client = MagicMock(spec=Client)
        monkeypatch.setattr(
            "gpcr_tools.annotator.gemini_client.Client",
            lambda api_key: mock_client,
        )

        rl = gemini_client.RateLimitedClient("test_key")

        # Fill the window to capacity with timestamps that will expire soon
        now = time.time()
        expire_soon = now - gemini_client.GEMINI_WINDOW_SECONDS + 0.15
        rl._timestamps = deque([expire_soon] * gemini_client.GEMINI_RPM_LIMIT)

        # This call should block briefly then succeed after cleanup
        start = time.time()
        rl.get_client()
        elapsed = time.time() - start

        # Should have slept at least a tiny bit (the 0.15s remaining)
        assert elapsed >= 0.1
        # Timestamps should have been cleaned up
        assert len(rl._timestamps) <= 2


# ---------------------------------------------------------------------------
# Module-level get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_initializes_on_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_GEMINI_API_KEY", "test_key")
        mock_client = MagicMock(spec=Client)
        monkeypatch.setattr(
            "gpcr_tools.annotator.gemini_client.Client",
            lambda api_key: mock_client,
        )

        assert gemini_client._rate_limiter is None
        result = gemini_client.get_client()
        assert result is mock_client
        assert gemini_client._rate_limiter is not None

    def test_reset_clears_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPCR_GEMINI_API_KEY", "test_key")
        mock_client = MagicMock(spec=Client)
        monkeypatch.setattr(
            "gpcr_tools.annotator.gemini_client.Client",
            lambda api_key: mock_client,
        )

        gemini_client.get_client()
        assert gemini_client._rate_limiter is not None

        gemini_client.reset_client()
        assert gemini_client._rate_limiter is None
