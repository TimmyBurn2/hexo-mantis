"""The teacher over the corpus: its raw heads per row, written to disk-backed arrays beside the corpus."""
from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .planes import CELLS, PLANES, six_planes
from .teacher import Teacher

#: The files a labelled set holds, row-aligned with the corpus.
LABEL_FILES = ("policy.npy", "value_logits.npy", "score.npy", "center.npy")


def row_moves(corpus: dict[str, np.ndarray], i: int) -> np.ndarray:
    """Row `i`'s stones in ply order."""
    g, k = int(corpus["game"][i]), int(corpus["k"][i])
    start = int(corpus["games_move_off"][g])
    return corpus["games_moves"][start:start + k]


#: The corpus columns a forked featuriser reads, set before the pool forks so no task pickles them.
_SHARED: dict[str, Any] = {}


def _featurise(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rows [start, stop)'s planes as uint8 (every plane is 0/1) and their crop centres."""
    start, stop = bounds
    game, k, move_off, moves, radius = (_SHARED[key] for key in ("game", "k", "move_off", "moves", "radius"))
    planes = np.zeros((stop - start, PLANES, 25, 25), np.uint8)
    centers = np.zeros((stop - start, 2), np.int32)
    for j, i in enumerate(range(start, stop)):
        first = int(move_off[game[i]])
        p, centers[j] = six_planes(moves[first:first + int(k[i])], radius)
        planes[j] = p.astype(np.uint8)
    return planes, centers


def label(corpus: dict[str, np.ndarray], teacher: Teacher, out_dir: Path, *, radius: int, batch: int = 2048,
          workers: int = 8, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write the teacher's raw heads for every corpus row under `out_dir`; Raises: FileExistsError — a labelled set is already there."""
    n = len(corpus["game"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / name for name in LABEL_FILES]
    if any(p.exists() for p in paths):
        raise FileExistsError(f"{out_dir} already holds a labelled set")
    open_memmap = np.lib.format.open_memmap
    policy = open_memmap(paths[0], mode="w+", dtype=np.float32, shape=(n, CELLS))
    value = open_memmap(paths[1], mode="w+", dtype=np.float32, shape=(n, 2))
    score = open_memmap(paths[2], mode="w+", dtype=np.float32, shape=(n,))
    center = open_memmap(paths[3], mode="w+", dtype=np.int32, shape=(n, 2))
    starts = list(range(0, n, batch))
    _SHARED.update(game=corpus["game"], k=corpus["k"], move_off=corpus["games_move_off"],
                   moves=corpus["games_moves"], radius=radius)
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        for j, (planes, centers) in enumerate(pool.imap(_featurise, [(b, min(b + batch, n)) for b in starts])):
            s = starts[j]
            out = teacher.evaluate(planes.astype(np.float32))
            policy[s:s + len(planes)] = out.policy
            value[s:s + len(planes)] = out.value_logits
            score[s:s + len(planes)] = out.score
            center[s:s + len(planes)] = centers
            if j % 50 == 0:
                log(f"labelled {s + len(planes)}/{n} rows, {time.time() - t0:.0f} s")
    for arr in (policy, value, score, center):
        arr.flush()
    return {"rows": n, "wall_s": round(time.time() - t0, 1), "teacher": teacher.record(), "radius": radius,
            "batch": batch}


def open_labels(out_dir: Path) -> dict[str, np.ndarray]:
    """A labelled set's arrays, memory-mapped read-only."""
    return {name.removesuffix(".npy"): np.load(out_dir / name, mmap_mode="r") for name in LABEL_FILES}
