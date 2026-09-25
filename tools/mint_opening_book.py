"""Dev-only deterministic opening-book minter (design §a.2); `--exclude-book` mints the BOOK_V2 pool.

CLI: uv run python tools/mint_opening_book.py --seed S --plies P --n N --out PATH.json
Openings are uniform-random over `board.legal_moves()` for `plies` plies from an empty board (run3's
gate-opening mechanics), off ONE `random.Random(seed)` stream in generation order, so identical args
reproduce byte-identical output (tests/arena/test_books.py::test_book_v1_reproducible_from_minter_args).
`--exclude-book PATH.json` (repeatable) runs the same stream but SKIPS any opening whose move list is in
an excluded book or earlier in the pool — an EXACT match, since the Python package has no D6 symmetry
helper (tests/tools/test_mint_opening_book.py). The pool is the balance measurement's INPUT, in
manifest.toml only because the eval worker resolves books by id; its consumers: the box's anchor-vs-itself cells and tools/select_balanced_book.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from mantis._engine import Board

#: The board encoding minted openings are generated (and later legality-replayed) against.
_MINT_ENCODING = "gnn_axis_v1"

MovesList = list[list[int]]


def _mint_one_opening(*, plies: int, rng: random.Random) -> MovesList:
    board = Board.with_encoding_name(_MINT_ENCODING)
    moves: MovesList = []
    for _ in range(plies):
        legal = board.legal_moves()
        if not legal:
            break
        q, r = rng.choice(legal)
        board.apply_move(q, r)
        moves.append([q, r])
    return moves


def mint_book(*, seed: int, plies: int, n: int) -> dict:
    """Deterministic book payload: `{"openings": [{"id": i, "moves": [...]}]}`."""
    rng = random.Random(seed)
    openings = [
        {"id": i, "moves": _mint_one_opening(plies=plies, rng=rng)} for i in range(n)
    ]
    return {"openings": openings}


def _key(moves: MovesList) -> str:
    return json.dumps([[int(q), int(r)] for q, r in moves])


def mint_pool(*, seed: int, plies: int, n: int, exclude: list[MovesList]) -> dict:
    """The candidate pool: `mint_book`'s stream with every opening in `exclude` (exact move list) and every repeat skipped, ids `0..n-1` in generation order, plus a `provenance` block. Raises: ValueError when `n` is not positive."""
    if n <= 0:
        raise ValueError(f"mint_pool: n={n} must be positive")
    excluded = {_key(m) for m in exclude}
    seen: set[str] = set()
    rng = random.Random(seed)
    openings: list[dict] = []
    while len(openings) < n:
        moves = _mint_one_opening(plies=plies, rng=rng)
        key = _key(moves)
        if key in excluded or key in seen:
            continue
        seen.add(key)
        openings.append({"id": len(openings), "moves": moves})
    provenance = {"minter": "tools/mint_opening_book.py", "seed": seed, "plies": plies, "n": n,
                  "excluded_openings": len(excluded)}
    return {"openings": openings, "provenance": provenance}


def _read_book_moves(path: Path) -> list[MovesList]:
    """Every opening's move list from a book JSON. Raises: OSError when the file cannot be read, ValueError when it is not a book payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "openings" not in payload:
        raise ValueError(f"{path}: not a book payload (no top-level 'openings')")
    return [o["moves"] for o in payload["openings"]]


def main(argv: list[str] | None = None) -> int:
    """Mint a book, or with `--exclude-book` a candidate pool, to `--out`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--plies", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-book", action="append", default=[],
                        help="book JSON whose openings the POOL may not contain (repeatable)")
    args = parser.parse_args(argv)

    if args.exclude_book:
        exclude = [m for book in args.exclude_book for m in _read_book_moves(Path(book))]
        payload = mint_pool(seed=args.seed, plies=args.plies, n=args.n, exclude=exclude)
    else:
        payload = mint_book(seed=args.seed, plies=args.plies, n=args.n)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
