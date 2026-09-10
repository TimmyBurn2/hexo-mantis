"""Opening-book authority: resolve a book id to openings, sha-verified at load.

`manifest.toml` maps a book id to a repo-packaged file and its sha256, which is verified on
every load. Each selected opening is later played exactly twice with colours swapped by
`mantis.arena.match.play_paired_match`; the paired-game law lives there, not here.
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
    book_id: str, n_pairs: int, seed: int, *, books_dir: Path | str | None = None
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


def round_openings(
    book_id: str,
    *,
    n_pairs: int,
    seed_base: int,
    round_index: int,
    books_dir: Path | str | None = None,
) -> list[Opening]:
    """Return one eval round's openings: a non-overlapping window over a seeded permutation.

    `seed_base` permutes the whole book once; round `r` takes the `n_pairs`-wide window at
    `r * n_pairs`, modulo the book size, so consecutive rounds are disjoint by construction
    rather than by luck. Past `len(book) / n_pairs` rounds the window wraps into a fresh
    alignment of the same permutation, and the disjointness holds for consecutive rounds only.

    Args:
        book_id: the book to draw from.
        n_pairs: openings in this round (each is later played twice, colours swapped).
        seed_base: the run's own `gate.seed_base` — the permutation's only entropy.
        round_index: the round's ordinal, monotone within a run.
        books_dir: override for the packaged book directory (tests).

    Returns:
        `n_pairs` distinct openings, or the whole book when it is smaller than `n_pairs`.

    Raises:
        BookError: unknown id, missing file, or a sha256 mismatch.
        ValueError: `n_pairs` or `round_index` is negative.
    """
    if n_pairs < 0 or round_index < 0:
        raise ValueError(
            f"round_openings: n_pairs={n_pairs} and round_index={round_index} must both be "
            ">= 0; a negative window has no meaning and would silently wrap backwards"
        )
    directory = Path(books_dir) if books_dir is not None else _DEFAULT_BOOKS_DIR
    raw_openings = _load_book_openings(book_id, directory)
    total = len(raw_openings)
    if total == 0:
        return []
    order = list(range(total))
    random.Random(seed_base).shuffle(order)
    take = min(n_pairs, total)
    start = (round_index * take) % total
    # `% total` per index, not a slice: the window must stay `take` wide when it runs off
    # the end, where a slice would silently return a short round.
    chosen = [raw_openings[order[(start + i) % total]] for i in range(take)]
    return [
        Opening(opening_id=str(o["id"]), moves=[tuple(m) for m in o["moves"]])
        for o in chosen
    ]


__all__ = ["BookError", "Opening", "paired_openings", "round_openings"]
