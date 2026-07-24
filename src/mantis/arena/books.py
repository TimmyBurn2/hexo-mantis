"""Opening-book authority (LAW-15 sha-pin; design §a.2 books.py).

Loads `manifest.toml` (book id -> repo-packaged file + sha256), VERIFIES the sha at load
(`BookError` on mismatch/unknown id), and hands back `paired_openings(book_id, n_pairs,
seed)` — a seeded selection of openings from the book. Each selected opening is later
played exactly twice (colors swapped) by `mantis.arena.match.play_paired_match` — the
paired-game law lives there, not here.
"""
from __future__ import annotations

import hashlib
import json
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_BOOKS_DIR = Path(__file__).resolve().parent / "books"


class BookError(ValueError):
    """A book failed to resolve: unknown id, missing file, or a sha256 mismatch (tamper)."""


@dataclass(frozen=True)
class Opening:
    opening_id: str
    moves: list[tuple[int, int]]


def _load_manifest(books_dir: Path) -> dict:
    manifest_path = books_dir / "manifest.toml"
    if not manifest_path.is_file():
        raise BookError(f"no manifest.toml under {books_dir}")
    with manifest_path.open("rb") as handle:
        return tomllib.load(handle)


def _load_book_openings(book_id: str, books_dir: Path) -> list[dict]:
    manifest = _load_manifest(books_dir)
    books = manifest.get("books", {})
    if book_id not in books:
        raise BookError(f"unknown book id {book_id!r} (known: {sorted(books)})")
    entry = books[book_id]
    book_file = books_dir / entry["file"]
    if not book_file.is_file():
        raise BookError(f"book {book_id!r} file missing: {book_file}")
    raw = book_file.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = str(entry["sha256"])
    if actual_sha != expected_sha:
        raise BookError(
            f"book {book_id!r} sha256 mismatch: expected {expected_sha}, got {actual_sha} "
            f"(tampered or stale book file: {book_file})"
        )
    payload = json.loads(raw)
    return payload["openings"]


def paired_openings(
    book_id: str, n_pairs: int, seed: int, *, books_dir: "Path | str | None" = None
) -> list[Opening]:
    """Deterministically select up to `n_pairs` openings from `book_id` (seeded).

    Each returned `Opening` is played exactly twice (colors swapped) by
    `mantis.arena.match.play_paired_match` — the selection here is single-count, the
    doubling is match.py's law.
    """
    directory = Path(books_dir) if books_dir is not None else _DEFAULT_BOOKS_DIR
    raw_openings = _load_book_openings(book_id, directory)
    rng = random.Random(seed)
    n = min(n_pairs, len(raw_openings))
    chosen = rng.sample(raw_openings, n) if raw_openings else []
    return [
        Opening(opening_id=str(o["id"]), moves=[tuple(m) for m in o["moves"]])
        for o in chosen
    ]


__all__ = ["BookError", "Opening", "paired_openings"]
