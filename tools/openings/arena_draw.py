"""The arena's opening draw, ported exactly (hexarena `packages/rules/src/opening.ts` at cf28a07), and the cell book minter.

An opening of an odd count of plies: the origin, then each later ply on a cell within hex distance 2 of it that the index
source picks among those still empty, owned by the ply's player; a draw with four of one player and none of the other in
six consecutive cells of an axis is discarded whole and redrawn from ply 1, so the result is uniform over balanced draws.
The arena's (x, y) are our axial (q, r), and its player 0 (the origin's) is our Player One.

CLI: python tools/openings/arena_draw.py --seed S --plies P --n N --out PATH.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

#: The longest opening a draw accepts; longer ones let forced wins survive the balance check.
MAX_OPENING_PLIES = 9
#: Stones after the origin land at most this far from it.
OPENING_RADIUS = 2
WINNING_LINE_LENGTH = 6
#: A turn places two stones, so four of one player in a window the other has not touched is a threat.
THREAT_STONES = WINNING_LINE_LENGTH - 2
LINE_AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
ORIGIN = (0, 0)

Cell = tuple[int, int]
#: (x, y, player), player 0 owning the origin.
Stone = tuple[int, int, int]
IndexSource = Callable[[int], int]


def hex_distance(a: Cell, b: Cell) -> int:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (abs(dx) + abs(dy) + abs(dx + dy)) // 2


def _region() -> tuple[Cell, ...]:
    cells = []
    for x in range(-OPENING_RADIUS, OPENING_RADIUS + 1):
        for y in range(-OPENING_RADIUS, OPENING_RADIUS + 1):
            if 0 < hex_distance((x, y), ORIGIN) <= OPENING_RADIUS:
                cells.append((x, y))
    return tuple(cells)


#: The 18 cells a drawn stone may take, in the order an index source indexes: x ascending, then y.
OPENING_REGION: tuple[Cell, ...] = _region()


def owner_of_ply(ply: int) -> int:
    """Turn 0 is ply 0; turn t >= 1 is plies 2t-1 and 2t; odd turns are player 1's."""
    turn = (ply + 1) // 2
    return 1 if turn % 2 == 1 else 0


def is_balanced_opening(stones: Sequence[Stone]) -> bool:
    """No six consecutive cells on any axis hold four or more stones of one player and none of the other."""
    for ax, ay in LINE_AXES:
        lines: dict[int, list[tuple[int, int]]] = {}
        for x, y, player in stones:
            lines.setdefault(x * ay - y * ax, []).append((y if ax == 0 else x, player))
        for members in lines.values():
            if len(members) >= THREAT_STONES and _holds_threat(members):
                return False
    return True


def _holds_threat(members: list[tuple[int, int]]) -> bool:
    for along, _player in members:
        for start in range(along - WINNING_LINE_LENGTH + 1, along + 1):
            counts = [0, 0]
            for other_along, other_player in members:
                if start <= other_along < start + WINNING_LINE_LENGTH:
                    counts[other_player] += 1
            zero, one = counts
            if (zero >= THREAT_STONES and one == 0) or (one >= THREAT_STONES and zero == 0):
                return True
    return False


def draw_opening(plies: int, index: IndexSource) -> list[Stone]:
    """The arena's `drawOpening`: `plies` stones in ply order. Raises: ValueError — not an odd count in [1, 9], or
    an index outside the bound it was asked for."""
    if not isinstance(plies, int) or isinstance(plies, bool) or plies < 1 or plies > MAX_OPENING_PLIES or plies % 2 == 0:
        raise ValueError(f"an opening is an odd count of plies from 1 to {MAX_OPENING_PLIES}, not {plies!r}")
    while True:
        drawn = _draw_once(plies, index)
        if is_balanced_opening(drawn):
            return drawn


def _draw_once(plies: int, index: IndexSource) -> list[Stone]:
    stones: list[Stone] = [(*ORIGIN, owner_of_ply(0))]
    while len(stones) < plies:
        taken = {(x, y) for x, y, _ in stones}
        empty = [cell for cell in OPENING_REGION if cell not in taken]
        pick = index(len(empty))
        if not 0 <= pick < len(empty):
            raise ValueError(f"the index source drew {pick} outside [0, {len(empty)})")
        stones.append((*empty[pick], owner_of_ply(len(stones))))
    return stones


class Mulberry32:
    """The arena tests' seeded index source (`test/helpers/prng.ts`): `int(bound)` is floor(next() * bound)."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self._state = (self._state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self._state
        t = _imul(t ^ (t >> 15), t | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def int(self, bound: int) -> int:
        return int(self.next() * bound)


def _imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


def mint_arena_book(*, seed: int, plies: int, n: int) -> dict:
    """`n` openings drawn in sequence off one `Mulberry32(seed)`, moves in ply order as our `[q, r]`. Raises:
    ValueError — `n` not positive, or `plies` not a draw's length."""
    if n <= 0:
        raise ValueError(f"mint_arena_book: n={n} must be positive")
    rng = Mulberry32(seed)
    openings = [{"id": i, "moves": [[x, y] for x, y, _ in draw_opening(plies, rng.int)]} for i in range(n)]
    positions = {frozenset((x, y, owner_of_ply(k)) for k, (x, y) in enumerate(o["moves"])) for o in openings}
    provenance = {"minter": "tools/openings/arena_draw.py", "protocol": "hexarena packages/rules/src/opening.ts@cf28a07",
                  "index_source": "mulberry32", "seed": seed, "plies": plies, "n": n,
                  "distinct_positions": len(positions)}
    return {"openings": openings, "provenance": provenance}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--plies", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    payload = mint_arena_book(seed=args.seed, plies=args.plies, n=args.n)
    args.out.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(payload["provenance"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
