"""`sha256_file` is the tree's one file hash: its digest must equal a whole-file sha256."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from mantis.util.hashing import sha256_file

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_every_pinned_fixture_hashes_to_its_manifest_sha() -> None:
    rows = tomllib.loads((_FIXTURES / "manifest.toml").read_text(encoding="utf-8"))["required"]
    multi_chunk = [r for r in rows if (_FIXTURES / r["path"]).stat().st_size > 1 << 20]
    assert multi_chunk, "no pinned fixture spans more than one read chunk"
    wrong = {r["path"]: sha256_file(_FIXTURES / r["path"]) for r in rows
             if sha256_file(_FIXTURES / r["path"]) != r["sha256"]}
    assert not wrong, f"sha256_file disagrees with the manifest's pinned digests: {wrong}"


def test_a_file_spanning_partial_chunks_matches_a_whole_read(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * ((5 << 20) // 256 + 3))
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(str(path)) == sha256_file(path)


def test_an_empty_file_is_the_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()
