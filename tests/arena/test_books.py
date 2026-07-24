"""⊕ WP11-A arena — opening-book authority (LAW-15 sha-pin; design §a.2 books.py,
§b arena/test_books.py).

RED-at-import until IMPL writes `mantis.arena.books`. `tools/mint_opening_book.py` (the
dev-only deterministic minter) is also new — the reproducibility test invokes it as a
subprocess and is RED today via a nonzero/missing-file failure, not an import error,
since the minter is a standalone script outside this suite's import surface.

ORACLE-CHOSEN SEAM: `mantis.arena.books` functions accept an explicit `books_dir: Path`
kwarg (defaulting, per the design, to the repo-packaged `src/mantis/arena/books/`) so this
suite can point the loader at a throwaway tmp manifest for the sha-tamper / unknown-id
oracles without touching the frozen packaged book. This is the minimal testable surface
for the behaviour DESIGN.md §a.2 describes (sha verification at load); it is not a
redesign of the book format.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mantis.arena.books import BookError, paired_openings

_REPO = Path(__file__).resolve().parents[2]
_PACKAGED_BOOKS_DIR = _REPO / "src" / "mantis" / "arena" / "books"
_MINTER = _REPO / "tools" / "mint_opening_book.py"

_BOOK_V1_ID = "book_v1_s20260625_p4"


def _write_manifest(tmp_path: Path, *, book_id: str, openings: list, sha_override: str | None = None):
    book_file = tmp_path / f"{book_id}.json"
    payload = {"openings": openings}
    book_file.write_text(json.dumps(payload))
    real_sha = hashlib.sha256(book_file.read_bytes()).hexdigest()
    sha = sha_override if sha_override is not None else real_sha
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f'[books."{book_id}"]\nfile = "{book_file.name}"\nsha256 = "{sha}"\n'
    )
    return tmp_path


def test_book_sha_mismatch_bites(tmp_path):
    openings = [{"id": 0, "moves": [[0, 0], [1, 1], [0, 1], [1, 0]]}]
    tampered = "0" * 64  # deliberately wrong sha256
    books_dir = _write_manifest(tmp_path, book_id="tampered_book", openings=openings,
                                 sha_override=tampered)
    with pytest.raises(BookError) as exc:
        paired_openings("tampered_book", n_pairs=1, seed=1, books_dir=books_dir)
    assert "tampered_book" in str(exc.value), "BookError must NAME the offending book id"


def test_unknown_book_id_is_a_named_error(tmp_path):
    openings = [{"id": 0, "moves": [[0, 0], [1, 1], [0, 1], [1, 0]]}]
    books_dir = _write_manifest(tmp_path, book_id="a_real_book", openings=openings)
    with pytest.raises(BookError) as exc:
        paired_openings("does_not_exist_book", n_pairs=1, seed=1, books_dir=books_dir)
    assert "does_not_exist_book" in str(exc.value)


def test_book_v1_reproducible_from_minter_args(tmp_path):
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    args = ["--seed", "20260625", "--plies", "4", "--n", "512"]
    result_a = subprocess.run(
        [sys.executable, str(_MINTER), *args, "--out", str(out_a)],
        capture_output=True, text=True, timeout=60,
    )
    result_b = subprocess.run(
        [sys.executable, str(_MINTER), *args, "--out", str(out_b)],
        capture_output=True, text=True, timeout=60,
    )
    assert result_a.returncode == 0, f"minter failed: {result_a.stderr}"
    assert result_b.returncode == 0, f"minter failed: {result_b.stderr}"
    assert out_a.read_bytes() == out_b.read_bytes(), (
        "identical --seed/--plies/--n/--out args must reproduce byte-identical output"
    )


def test_openings_are_legal_move_sequences():
    from mantis._engine import Board

    openings = paired_openings(_BOOK_V1_ID, n_pairs=8, seed=20260625,
                                books_dir=_PACKAGED_BOOKS_DIR)
    assert len(openings) > 0
    for opening in openings:
        board = Board.with_encoding_name("gnn_axis_v1")
        for q, r in opening.moves:
            legal = board.legal_moves()
            assert (q, r) in legal, f"opening {opening.opening_id} plays an illegal move ({q}, {r})"
            board.apply_move(q, r)
