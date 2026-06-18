"""Tests for the code-version resolver (env-baked -> git -> unknown)."""

from __future__ import annotations

import re
import shutil

import pytest

from gpcr_tools import code_version
from gpcr_tools.code_version import CODE_VERSION_ENV, get_code_version


@pytest.fixture(autouse=True)
def _clear_git_cache():
    # _git_version is lru_cached; clear it around every test so the real-git test
    # cannot leave a primed cache that masks a mocked _git_version in another test.
    code_version._git_version.cache_clear()
    yield
    code_version._git_version.cache_clear()


def test_baked_env_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    # A baked image sets the env var; it wins without consulting git.
    monkeypatch.setenv(CODE_VERSION_ENV, "deadbee")
    monkeypatch.setattr(code_version, "_git_version", lambda: "should-not-be-used")
    assert get_code_version() == "deadbee"


@pytest.mark.parametrize("value", ["", "unknown"])
def test_empty_or_unknown_env_falls_through_to_git(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # An empty or placeholder env var (plain `docker build` with no build-arg)
    # must not mask a real git checkout.
    monkeypatch.setenv(CODE_VERSION_ENV, value)
    monkeypatch.setattr(code_version, "_git_version", lambda: "gitsha9")
    assert get_code_version() == "gitsha9"


def test_no_env_uses_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    monkeypatch.setattr(code_version, "_git_version", lambda: "gitsha9")
    assert get_code_version() == "gitsha9"


def test_resolver_always_returns_nonempty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    # No env and git unavailable -> "unknown", never empty/None.
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    monkeypatch.setattr(code_version, "_git_version", lambda: "unknown")
    result = get_code_version()
    assert isinstance(result, str) and result


def test_real_git_version_happy_path() -> None:
    # Exercise the actual subprocess (not mocked): in a checkout it must return a
    # short hash, optionally -dirty; "unknown" only if git/.git is somehow absent.
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = code_version._git_version()
    assert result == "unknown" or re.fullmatch(r"[0-9a-f]{7,}(-dirty)?", result), result
