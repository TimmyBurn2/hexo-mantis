"""Forced-move tactics on a Hex Tac Toe position (6-in-a-row, three axes, 2-stone turns)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
PAD = 6
Cell = tuple[int, int]


def hex_ball(radius: int) -> np.ndarray:
    """Every axial offset within hex distance `radius`, as an (n, 2) int array."""
    offs = [(dq, dr) for dq in range(-radius, radius + 1) for dr in range(-radius, radius + 1)
            if abs(dq + dr) <= radius]
    return np.array(offs, dtype=np.int64)


BALL8 = hex_ball(8)


@dataclass
class RowTactics:
    """One position's tactical reading for the mover (GAME_QUALITY_CENSUS_2026-09-14.md's terms)."""

    #: W1: cells completing six for the mover now.
    w1: set[Cell] = field(default_factory=set)
    #: W2: cells starting a two-stone completion (a mover window with 4 stones and 2 empties).
    w2: set[Cell] = field(default_factory=set)
    #: A FOUR of the opponent: a 6-cell window with >= 4 opponent stones and no mover stone.
    fours: list[frozenset[Cell]] = field(default_factory=list)
    opp_fives: set[Cell] = field(default_factory=set)
    #: B(k): cells c such that the fours c does not hit have a hitting set of size <= k-1.
    block: set[Cell] = field(default_factory=set)
    #: B_strict(k): B(k) restricted to cells that hit at least one four.
    block_strict: set[Cell] = field(default_factory=set)
    #: True when the fours share a cell, so at k=2 EVERY first stone is a safe one.
    block_any: bool = False
    #: Empty cells within `radius` of any stone — the legal set (25 on an empty board).
    n_legal: int = 0

    @property
    def check(self) -> bool:
        """True when the opponent has at least one four."""
        return bool(self.fours)

    def win_set(self, k: int) -> set[Cell]:
        """The mover's winning first stones with `k` stones left this turn."""
        return self.w1 | self.w2 if k == 2 else set(self.w1)

    def forced(self, k: int) -> tuple[str, set[Cell]]:
        """('win', W) | ('block', B) | ('lost1', {}) — B(k) empty under check — | ('quiet', {})."""
        w = self.win_set(k)
        if w:
            return "win", w
        if self.fours:
            return ("block", self.block) if self.block else ("lost1", set())
        return "quiet", set()


def _window_sums(grid: np.ndarray, dq: int, dr: int) -> tuple[np.ndarray, int, int]:
    """Sum over the 6 cells of every window start; returns (sums, q_lo, r_lo) of start (0,0)."""
    hq, hr = grid.shape
    nq = hq - 5 * dq
    r_lo = 5 if dr < 0 else 0
    nr = hr - 5 * abs(dr)
    acc = np.zeros((nq, nr), dtype=np.int16)
    for i in range(6):
        q0 = i * dq
        r0 = r_lo + i * dr
        acc += grid[q0 : q0 + nq, r0 : r0 + nr]
    return acc, 0, r_lo


def _window_cells(q0: int, r0: int, dq: int, dr: int) -> list[Cell]:
    return [(q0 + i * dq, r0 + i * dr) for i in range(6)]


def legal_count(q: np.ndarray, r: np.ndarray, radius: int = 8) -> int:
    """Number of empty cells within `radius` of any stone; 25 on an empty board."""
    if q.size == 0:
        return 25
    ball = BALL8 if radius == 8 else hex_ball(radius)
    cq = (q[:, None] + ball[None, :, 0]).ravel()
    cr = (r[:, None] + ball[None, :, 1]).ravel()
    packed = (cq + 4096) * 8192 + (cr + 4096)
    return int(np.unique(packed).size) - int(q.size)


def hitting_sets(fours: list[frozenset[Cell]], k: int) -> tuple[set[Cell], set[Cell], bool]:
    """(B literal, B strict, any-first-stone-safe) for the fours' empty cells under k stones left."""
    if not fours:
        return set(), set(), False
    union: set[Cell] = set().union(*fours)
    common = frozenset.intersection(*fours)
    if k == 1:
        return set(common), set(common), False
    literal: set[Cell] = set()
    strict: set[Cell] = set()
    for c in union:
        rest = [f for f in fours if c not in f]
        if not rest or bool(frozenset.intersection(*rest)):
            literal.add(c)
            strict.add(c)
    return literal, strict, bool(common)


def analyze(q: np.ndarray, r: np.ndarray, p: np.ndarray, mover: int, k: int,
            radius: int = 8) -> RowTactics:
    """Tactics for the mover with `k` stones left this turn; stones as parallel int arrays."""
    out = RowTactics(n_legal=legal_count(q, r, radius))
    if q.size == 0:
        return out
    qmin, rmin = int(q.min()) - PAD, int(r.min()) - PAD
    hq, hr = int(q.max()) - qmin + PAD + 1, int(r.max()) - rmin + PAD + 1
    mine = np.zeros((hq, hr), dtype=np.int16)
    theirs = np.zeros((hq, hr), dtype=np.int16)
    qi, ri = q - qmin, r - rmin
    own = p == mover
    mine[qi[own], ri[own]] = 1
    theirs[qi[~own], ri[~own]] = 1
    fours: set[frozenset[Cell]] = set()
    for dq, dr in AXES:
        sm, _, r_lo = _window_sums(mine, dq, dr)
        so, _, _ = _window_sums(theirs, dq, dr)
        for cls, mask in (("w1", (sm >= 5) & (so == 0)), ("w2", (sm == 4) & (so == 0)),
                          ("four", (so >= 4) & (sm == 0))):
            if not mask.any():
                continue
            for a, b in np.argwhere(mask):
                cells = _window_cells(int(a), int(b) + r_lo, dq, dr)
                empt = [(cq + qmin, cr + rmin) for cq, cr in cells
                        if mine[cq, cr] == 0 and theirs[cq, cr] == 0]
                if cls == "w1":
                    out.w1.update(empt)
                elif cls == "w2":
                    out.w2.update(empt)
                else:
                    fours.add(frozenset(empt))
                    if len(empt) == 1:
                        out.opp_fives.update(empt)
    out.fours = sorted(fours, key=lambda f: sorted(f))
    out.block, out.block_strict, out.block_any = hitting_sets(out.fours, k)
    return out
