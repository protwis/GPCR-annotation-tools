"""Gemini API client with per-key sliding-window rate limiting.

Reads a single API key from the ``GPCR_GEMINI_API_KEY`` environment variable.
The legacy ``GPCR_GEMINI_API_KEYS`` (comma-separated) is still accepted for
backward compatibility — the **first** key is used and a deprecation warning
is emitted.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from collections import deque
from threading import Lock

from google.genai import Client

from gpcr_tools.config import (
    GEMINI_API_KEY_ENV,
    GEMINI_API_KEY_ENV_LEGACY,
    GEMINI_RPM_LIMIT,
    GEMINI_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)


def _resolve_api_key() -> str:
    """Resolve the Gemini API key from environment variables.

    Priority:
      1. ``GPCR_GEMINI_API_KEY`` (new, preferred)
      2. ``GPCR_GEMINI_API_KEYS`` (legacy, first key used, deprecation warning)

    Raises ``RuntimeError`` if neither variable provides a usable key.
    """
    # New env var — single key
    key = (os.environ.get(GEMINI_API_KEY_ENV) or "").strip()
    if key:
        return key

    # Legacy env var — comma-separated, take first
    legacy = (os.environ.get(GEMINI_API_KEY_ENV_LEGACY) or "").strip()
    if legacy:
        first_key = next((k.strip() for k in legacy.split(",") if k.strip()), "")
        if first_key:
            warnings.warn(
                f"{GEMINI_API_KEY_ENV_LEGACY} is deprecated. "
                f"Use {GEMINI_API_KEY_ENV} with a single API key instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return first_key

    raise RuntimeError(
        f"{GEMINI_API_KEY_ENV} environment variable is required.\nSet it to your Gemini API key."
    )


class RateLimitedClient:
    """Wraps a single Gemini API key with sliding-window rate limiting.

    Ensures no more than ``GEMINI_RPM_LIMIT`` requests are made within
    any ``GEMINI_WINDOW_SECONDS``-second window.  When the limit is
    reached, the caller blocks until capacity is available.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def get_client(self) -> Client:
        """Return a configured ``Client``, blocking if rate-limited."""
        with self._lock:
            self._wait_for_capacity()
            self._timestamps.append(time.time())
            return Client(api_key=self._api_key)

    def _wait_for_capacity(self) -> None:
        """Block until the sliding window has room for one more request."""
        now = time.time()
        self._cleanup(now)

        if len(self._timestamps) < GEMINI_RPM_LIMIT:
            return

        # Window is full — sleep until the oldest entry expires
        oldest = self._timestamps[0]
        sleep_time = max(0.1, (oldest + GEMINI_WINDOW_SECONDS) - now)
        logger.info(
            "Rate limit reached (%d/%d RPM). Sleeping %.1f s...",
            len(self._timestamps),
            GEMINI_RPM_LIMIT,
            sleep_time,
        )
        time.sleep(sleep_time)
        now = time.time()
        self._cleanup(now)

    def _cleanup(self, now: float) -> None:
        """Remove timestamps older than the sliding window."""
        while self._timestamps and self._timestamps[0] < now - GEMINI_WINDOW_SECONDS:
            self._timestamps.popleft()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_rate_limiter: RateLimitedClient | None = None


def _init_client() -> None:
    """Initialize the global rate-limited client from environment variables."""
    global _rate_limiter
    if _rate_limiter is None:
        api_key = _resolve_api_key()
        _rate_limiter = RateLimitedClient(api_key)


def get_client() -> Client:
    """Return a rate-limited Gemini client, initializing on first call."""
    if _rate_limiter is None:
        _init_client()
    assert _rate_limiter is not None
    return _rate_limiter.get_client()


def reset_client() -> None:
    """Reset the global client (for testing)."""
    global _rate_limiter
    _rate_limiter = None
