"""Read-only git provenance: the HEAD sha and whether the tree is dirty. Never raises."""
from __future__ import annotations

import subprocess
from pathlib import Path

_TIMEOUT_SEC = 2.0


def head_sha(cwd: Path) -> str | None:
    """Return `git rev-parse HEAD` for the checkout containing `cwd`, or None outside one."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=cwd,
            timeout=_TIMEOUT_SEC,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError,
            OSError):
        return None
    sha = out.decode("ascii", errors="replace").strip()
    return sha or None


def is_dirty(cwd: Path) -> bool | None:
    """Return whether the checkout containing `cwd` has tracked changes, or None outside one."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL, cwd=cwd, timeout=_TIMEOUT_SEC,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError,
            OSError):
        return None
    return bool(out.strip())
