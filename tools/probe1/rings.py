"""Ring-only readings: the one-hot DECOMPOSER (reading 1) and the row selectors the net/solver readings draw from."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mantis._engine import Board
from mantis.diagnostics.ring_audit import ONE_HOT_H, reconstruct_moves
from mantis.diagnostics.ring_reader import Ring, explicit_entropy, load_ring
from mantis.util.constants import is_alpha_full


def _share(mask: np.ndarray, one_hot: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    return {"n": n, "one_hot_share": (float(one_hot[mask].mean()) if n else None)}


def decompose(ring: Ring) -> dict[str, Any]:
    """The full-arm one-hot share as audited, then with tail-only (α = 1.0) rows EXCLUDED, then by `moves_remaining` — never a band."""
    full = ring.is_full_search != 0
    h = explicit_entropy(ring)
    one_hot = h < ONE_HOT_H
    tail_only = np.array([is_alpha_full(float(a)) for a in ring.tail_mass], dtype=bool)
    kept = full & ~tail_only
    out: dict[str, Any] = {
        "rows": int(ring.header.size),
        "full_arm": {**_share(full, one_hot), "tail_only": int((full & tail_only).sum())},
        # The audit's number: tail-only rows read H = 0 and count as one-hots (ring_reader.explicit_entropy).
        "one_hot_share_full_as_audited": _share(full, one_hot)["one_hot_share"],
        "one_hot_share_full_excl_tail_only": _share(kept, one_hot)["one_hot_share"],
        "by_moves_remaining": {},
        "h_full_median_excl_tail_only": (float(np.median(h[kept])) if kept.any() else None),
    }
    for k in sorted({int(v) for v in ring.moves_remaining[full]}):
        sel = kept & (ring.moves_remaining == k)
        sel_all = full & (ring.moves_remaining == k)
        out["by_moves_remaining"][str(k)] = {
            "as_audited": _share(sel_all, one_hot), "excl_tail_only": _share(sel, one_hot),
            "tail_only": int((sel_all & tail_only).sum()),
        }
    return out


def decompose_rings(paths: list[Path]) -> list[dict[str, Any]]:
    """`decompose` per ring, newest-independent: each row names its file."""
    return [{"ring": p.name, **decompose(load_ring(p))} for p in paths]


def full_arm_rows(ring: Ring, *, seed: int, n: int, moves_remaining: int | None = None) -> np.ndarray:
    """`n` full-search row indices drawn without replacement (fewer if the arm is smaller), optionally at one `mr`."""
    mask = ring.is_full_search != 0
    if moves_remaining is not None:
        mask &= ring.moves_remaining == moves_remaining
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=min(n, idx.size), replace=False))


def target_argmax(ring: Ring, i: int) -> tuple[int, int] | None:
    """The row's max-mass explicit child, or None on a tail-only row."""
    visits = ring.row_visits(i)
    if visits.size == 0:
        return None
    j = int(np.argmax(visits["prob"]))
    return int(visits["q"][j]), int(visits["r"][j])


def reconstructed(ring: Ring, rows: np.ndarray) -> list[tuple[int, Any, list[tuple[int, int]]]]:
    """`(row, Board, moves)` for every row the cadence search rebuilds; the rows it cannot are counted by the caller."""
    out = []
    for i in rows.tolist():
        found = reconstruct_moves(ring, int(i))
        if found is not None:
            out.append((int(i), found[0], found[1]))
    return out


def fence_legal_moves(ring: Ring, i: int) -> list[tuple[int, int]] | None:
    """Row `i`'s stones as a FENCE-LEGAL cadence move list (`apply_move` enforces no fence, `build_board` does); greedy, both first movers tried."""
    stones = ring.row_stones(i)
    mover, k = int(ring.current_player[i]), int(ring.moves_remaining[i])
    ids = sorted({int(st["p"]) for st in stones})
    for p1 in ids:
        p2 = next((x for x in ids if x != p1), None)
        left = {p1: [(int(st["q"]), int(st["r"])) for st in stones if int(st["p"]) == p1],
                p2: [(int(st["q"]), int(st["r"])) for st in stones if int(st["p"]) == p2]}
        board = Board.with_encoding_name(ring.header.encoding)
        seq: list[tuple[int, int]] = []
        ok = True
        while left[p1] or left[p2]:
            owner = p1 if board.current_player == 1 else p2
            pick = next((c for c in left[owner] if board.is_legal(*c)), None)
            if pick is None:
                ok = False
                break
            left[owner].remove(pick)
            board.apply_move(*pick)
            seq.append(pick)
        if not ok or board.check_win():
            continue
        if (p1 if board.current_player == 1 else p2) == mover and board.moves_remaining == k:
            return seq
    return None


def spread_positions(ring: Ring, *, seed: int, n: int, ply_bins: int = 6) -> list[dict[str, Any]]:
    """`n` fence-legal full-arm rows at `moves_remaining` 2, drawn round-robin over `ply_bins` quantile bins of `ply_index`."""
    mask = (ring.is_full_search != 0) & (ring.moves_remaining == 2)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    plies = ring.ply_index[idx]
    edges = np.unique(np.quantile(plies, np.linspace(0.0, 1.0, ply_bins + 1))[1:-1])
    rng = np.random.default_rng(seed)
    pools: list[list[int]] = [[] for _ in range(edges.size + 1)]
    for i, b in zip(idx.tolist(), np.digitize(plies, edges).tolist(), strict=True):
        pools[b].append(i)
    for pool in pools:
        rng.shuffle(pool)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(pools):
        for pool in pools:
            while pool and len(out) < n:
                i = pool.pop()
                moves = fence_legal_moves(ring, int(i))
                if moves is None:
                    continue
                out.append({"row": int(i), "ply": int(ring.ply_index[i]), "n_stones": int(ring.n_stones[i]),
                            "moves": [[q, r] for q, r in moves]})
                break
    return out


__all__ = ["decompose", "decompose_rings", "fence_legal_moves", "full_arm_rows", "reconstructed", "spread_positions",
           "target_argmax"]
