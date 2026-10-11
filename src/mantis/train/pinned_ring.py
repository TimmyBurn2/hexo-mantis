"""A config-pinned replay ring: the file a config names, loaded only once its sha256 re-derives to the pin."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis.util.hashing import sha256_file


class PinnedRingError(RuntimeError):
    """A config-pinned ring hashes otherwise than its pin, or loads no row."""


def load_pinned_ring(buffer: Any, path: Path, sha256: str, *, key: str) -> int:
    """Load `path` into `buffer` once its sha256 equals the pin, returning the rows; Raises: FileNotFoundError — no file at `path` (the engine would read it as empty); PinnedRingError — another sha, or no row; ValueError — the engine refuses the encoding, geometry or capacity."""
    if not path.exists():
        raise FileNotFoundError(f"{key} not found: {path}")
    actual = sha256_file(path)
    if actual != sha256:
        raise PinnedRingError(f"{key} {path} hashes sha256 {actual}, but the config declares {sha256}")
    rows = int(buffer.load_from_path(str(path)))
    if rows < 1:
        raise PinnedRingError(f"{key} {path} loaded no row")
    return rows


__all__ = ["PinnedRingError", "load_pinned_ring"]
