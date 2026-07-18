"""Perf smoke for chain-plane computation (Q13).

`_compute_chain_planes` uses numpy shift+zero-pad (NOT np.roll) and must stay
cheap. This is a portable SMOKE, not a host-pinned gate: the CI ceiling is a
generous 300µs/call ceiling with ample headroom over the reference measurement
(~163µs on a mid-range laptop core), so it flags an algorithmic regression
without failing on a slow shared CI runner. The board size is registry-derived.
"""
from __future__ import annotations
import time
import numpy as np

from mantis.env.game_state import _compute_chain_planes
from mantis.encoding import lookup as _lookup_encoding

BOARD_SIZE: int = _lookup_encoding("v6").board_size


def _make_50_stone_position() -> tuple[np.ndarray, np.ndarray]:
    """50-stone mixed position spread across the window.

    Deterministic: 25 cur stones + 25 opp stones on a pseudo-random,
    non-overlapping interleave seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=2613)
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    flat_indices = rng.choice(BOARD_SIZE * BOARD_SIZE, size=50, replace=False)
    for i, flat in enumerate(flat_indices):
        q, r = divmod(int(flat), BOARD_SIZE)
        if i < 25:
            cur[q, r] = 1.0
        else:
            opp[q, r] = 1.0
    return cur, opp


def test_compute_chain_planes_ci_budget(capsys):
    """Portable CI ceiling: 300µs/call (generous headroom over ~163µs reference).

    Kept as a soft perf smoke — a regression here signals an algorithmic
    change, not host jitter. The absolute number is deliberately loose.
    """
    cur, opp = _make_50_stone_position()
    # Warm up numpy kernels.
    for _ in range(10):
        _compute_chain_planes(cur, opp)

    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _compute_chain_planes(cur, opp)
    elapsed = time.perf_counter() - t0
    us_per_call = (elapsed / n_iters) * 1e6

    with capsys.disabled():
        print(
            f"\n_compute_chain_planes: {us_per_call:.1f} µs/call "
            f"({n_iters} iters, 50-stone position, portable CI ceiling <300µs)"
        )

    assert us_per_call < 300.0, (
        f"_compute_chain_planes took {us_per_call:.1f}µs/call, "
        f"exceeds the 300µs portable CI ceiling."
    )
