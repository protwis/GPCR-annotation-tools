"""Resolve the code version (git commit) that produced an annotation.

Recorded in each result's ``_provenance`` so an annotation can be traced back to
the exact code that generated it. The value is resolved in three steps:

1. The ``GPCR_CODE_VERSION`` env var, baked into a built image at ``docker build``
   time (and set by CI). Inside a built image this is the only option -- the
   ``.git`` directory and the ``git`` binary are both absent there.
2. ``git rev-parse`` in a local checkout (the conda development path).
3. ``"unknown"`` when neither is available.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

CODE_VERSION_ENV = "GPCR_CODE_VERSION"
_UNKNOWN = "unknown"


def _git(*args: str) -> str | None:
    """Run a git command in the package dir; return stdout, or None on any failure.

    cwd is the package directory; git searches upward for the enclosing ``.git``,
    so this resolves in an editable checkout and fails cleanly (caller -> "unknown")
    when the package is installed without a repo, e.g. inside a built image.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


@lru_cache(maxsize=1)
def _git_version() -> str:
    """Short commit hash from git, or ``"unknown"`` (cached: git is the slow path).

    A ``-dirty`` suffix is appended when the tracked tree has uncommitted edits, so
    provenance never claims a clean commit for code that was modified before it ran.
    """
    head = _git("rev-parse", "--short", "HEAD")
    sha = (head or "").strip()
    if not sha:
        return _UNKNOWN
    # Tracked modifications mean the running code differs from the named commit
    # (untracked files are ignored, matching `git describe --dirty`).
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status and status.strip():
        return f"{sha}-dirty"
    return sha


def get_code_version() -> str:
    """The baked-in version if present, else git, else ``"unknown"``.

    The env var is read on every call (cheap) so a baked image is authoritative;
    only the git lookup is cached.
    """
    baked = os.environ.get(CODE_VERSION_ENV)
    if baked and baked != _UNKNOWN:
        return baked
    return _git_version()
