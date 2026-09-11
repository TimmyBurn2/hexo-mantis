"""File hashing — the one `sha256_file` the bundle, the receipts and the puller share."""
from __future__ import annotations

import hashlib
from pathlib import Path

_HASH_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's bytes, streamed.

    Raises:
        OSError: the file could not be opened or read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["sha256_file"]
