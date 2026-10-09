"""Frozen HEXG rings the production trainer reads: the teacher's targets, the run's own targets (the control), and the teacher's labels permuted across positions (the known-bad)."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from mantis._engine import Board, HexgBuffer
from mantis.util.hashing import sha256_file

from .corpus import SRC_RING
from .label import row_moves
from .planes import crop_cells, crop_index, near_stones, player_for_stone

#: The slot counts the target's K is chosen from, smallest first.
SLOT_CHOICES = (64, 96, 128, 192, 256)
#: K keeps this much of a row's mass on the row at the QUANTILE (the 99.9th-percentile row).
COVERAGE = 0.9999
QUANTILE = 0.001
#: Our empty board's legal set: the 5×5 axial square Board::new() seeds.
EMPTY_BOARD_HALF = 2
#: Beside every ring written here: what it holds and the teachers its labels came from.
PROVENANCE_SUFFIX = ".provenance.json"
#: Teacher-labelled rings start at this ply: Six's engine never evaluates its net on the empty board.
TEACHER_MIN_PLY = 1


def legal_support(moves: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    """Crop indices of OUR legal cells inside the crop: empty and within `radius` of a stone; the 5×5 square on an empty board."""
    if len(moves) == 0:
        d = np.arange(-EMPTY_BOARD_HALF, EMPTY_BOARD_HALF + 1)
        q, r = np.meshgrid(d, d, indexing="ij")
        idx = crop_index(q.reshape(-1), r.reshape(-1), center)
        return np.sort(idx[idx >= 0])
    q, r = crop_cells(center)
    near = near_stones(q, r, moves, radius)
    occupied = crop_index(moves[:, 0], moves[:, 1], center)
    near[occupied[occupied >= 0]] = False
    return np.nonzero(near)[0]


def top_targets(logits: np.ndarray, support: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """The softmax over `support` (float64), its top `kmax` as (crop indices, masses) by mass descending, ties by index."""
    x = logits[support].astype(np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    order = np.lexsort((support, -p))[:kmax]
    return support[order], p[order]


_SHARED: dict[str, Any] = {}


def _targets_chunk(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows [start, stop) of `_SHARED["rows"]`: top-`kmax` crop indices (-1 pad), masses (0 pad), support sizes."""
    start, stop = bounds
    corpus, policy, center, rows, source, radius, kmax = (
        _SHARED[k] for k in ("corpus", "policy", "center", "rows", "source", "radius", "kmax"))
    idx = np.full((stop - start, kmax), -1, np.int16)
    mass = np.zeros((stop - start, kmax), np.float32)
    n_support = np.zeros(stop - start, np.int32)
    for j, i in enumerate(rows[start:stop]):
        support = legal_support(row_moves(corpus, int(i)), tuple(center[i]), radius)
        top_i, top_p = top_targets(np.asarray(policy[source[start + j]]), support, kmax)
        idx[j, :len(top_i)] = top_i
        mass[j, :len(top_p)] = top_p
        n_support[j] = len(support)
    return idx, mass, n_support


def compute_targets(corpus: dict[str, np.ndarray], labels: dict[str, np.ndarray], rows: np.ndarray, *,
                    radius: int, source_rows: np.ndarray | None = None, kmax: int = SLOT_CHOICES[-1],
                    workers: int = 8, chunk: int = 4096, log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """Each row's teacher target on its own legal cells, read from `source_rows`' logits (the known-bad's permutation) or its own; Raises: OSError — a worker pool cannot fork."""
    _SHARED.update(corpus=corpus, policy=labels["policy"], center=labels["center"], rows=rows,
                   source=rows if source_rows is None else source_rows, radius=radius, kmax=kmax)
    bounds = [(s, min(s + chunk, len(rows))) for s in range(0, len(rows), chunk)]
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        for j, part in enumerate(pool.imap(_targets_chunk, bounds)):
            parts.append(part)
            if j % 50 == 0:
                log(f"targets {bounds[j][1]}/{len(rows)}, {time.time() - t0:.0f} s")
    return {"idx": np.concatenate([p[0] for p in parts]), "mass": np.concatenate([p[1] for p in parts]),
            "n_support": np.concatenate([p[2] for p in parts])}


def choose_slots(mass: np.ndarray) -> tuple[int, dict[str, float]]:
    """The smallest K in SLOT_CHOICES keeping >= COVERAGE of the mass on the QUANTILE row, with every choice's quantile."""
    cum = np.cumsum(mass.astype(np.float64), axis=1)
    reads = {str(k): float(np.quantile(cum[:, k - 1], QUANTILE)) for k in SLOT_CHOICES}
    for k in SLOT_CHOICES:
        if reads[str(k)] >= COVERAGE:
            return k, reads
    return SLOT_CHOICES[-1], reads


def teacher_value(labels: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """v = tanh(½(l_win − l_loss)), the mover's frame."""
    vl = np.asarray(labels["value_logits"][rows], np.float64)
    return np.tanh(0.5 * (vl[:, 0] - vl[:, 1])).astype(np.float32)


def _stones(moves: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(q), int(r), player_for_stone(i)) for i, (q, r) in enumerate(moves)]


def write_teacher_ring(path: Path, corpus: dict[str, np.ndarray], labels: dict[str, np.ndarray], rows: np.ndarray,
                       targets: dict[str, np.ndarray], values: np.ndarray, slots: int, *, encoding: str,
                       value_valid: np.ndarray | None = None) -> dict[str, float]:
    """One teacher-labelled ring: the top-`slots` masses renormalised, α 0, every row a policy row, `values` as z (valid unless `value_valid` says not); returns the dropped-mass reads."""
    buf = HexgBuffer(max(len(rows), 8), encoding, slots)
    kept = targets["mass"][:, :slots].astype(np.float64).sum(axis=1)
    for j, i in enumerate(rows):
        moves = row_moves(corpus, int(i))
        center = tuple(labels["center"][i])
        sel = targets["idx"][j, :slots]
        live = sel >= 0
        cq, cr = crop_cells(center)
        cells = sel[live].astype(np.int64)
        probs = targets["mass"][j, :slots][live].astype(np.float64) / kept[j]
        visits = [(int(cq[c]), int(cr[c]), float(p)) for c, p in zip(cells, probs, strict=True)]
        buf.push_graph_position(_stones(moves), visits, int(corpus["current_player"][i]),
                                int(corpus["moves_remaining"][i]), int(corpus["k"][i]), True, float(values[j]),
                                True if value_valid is None else bool(value_valid[j]), int(corpus["game_length"][i]),
                                int(corpus["game"][i]), 0.0)
    buf.save_to_path(str(path))
    dropped = 1.0 - kept
    return {"rows": int(len(rows)), "slots": slots, "dropped_mass_mean": float(dropped.mean()),
            "dropped_mass_p999": float(np.quantile(dropped, 1 - QUANTILE)), "dropped_mass_max": float(dropped.max())}


def write_control_ring(path: Path, corpus: dict[str, np.ndarray], rows: np.ndarray, *, encoding: str) -> dict[str, int]:
    """The ring rows with their own targets exactly as the run stored them (explicit masses, α, z and validity); Raises: ValueError — a row that is not a ring row."""
    if not np.all(corpus["source"][rows] == SRC_RING):
        raise ValueError("the control ring holds ring rows only: an opening has no own targets")
    off = np.concatenate([[0], np.cumsum(corpus["n_visits"])]).astype(np.int64)
    slots = int(corpus["n_visits"][rows].max())
    buf = HexgBuffer(max(len(rows), 8), encoding, slots)
    for i in rows:
        a, b = int(off[i]), int(off[i + 1])
        visits = [(int(q), int(r), float(p)) for q, r, p in zip(corpus["visit_q"][a:b], corpus["visit_r"][a:b],
                                                                 corpus["visit_p"][a:b], strict=True)]
        buf.push_graph_position(_stones(row_moves(corpus, int(i))), visits, int(corpus["current_player"][i]),
                                int(corpus["moves_remaining"][i]), int(corpus["k"][i]),
                                bool(corpus["is_full_search"][i]), float(corpus["outcome"][i]),
                                bool(corpus["value_valid"][i]), int(corpus["game_length"][i]), int(corpus["game"][i]),
                                float(corpus["tail_mass"][i]), float(corpus["root_value"][i]),
                                bool(corpus["root_value_valid"][i]))
    buf.save_to_path(str(path))
    return {"rows": int(len(rows)), "slots": slots}


def write_provenance(ring: Path, *, kind: str, lineage: list[str], encoding: str, rows: int, slots: int,
                     record: dict[str, Any]) -> None:
    """The sidecar beside `ring`: the ring-provenance keys pretrain reads, plus its kind, the teachers its labels came from and its sha256; Raises: OSError — the ring cannot be read or the sidecar written."""
    payload = {"encoding": encoding, "ring_capacity": rows, "ring_visit_capacity": slots, "plies": rows, "kind": kind,
               "lineage": lineage, "ring_sha256": sha256_file(ring), **record}
    Path(str(ring) + PROVENANCE_SUFFIX).write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")


def ring_lineage(ring: Path) -> tuple[str, ...]:
    """The teachers a ring's labels came from, read from the provenance written for exactly this ring; Raises: FileNotFoundError — no provenance beside it; ValueError — a malformed one, or one written for other bytes."""
    prov = json.loads(Path(str(ring) + PROVENANCE_SUFFIX).read_text(encoding="utf-8"))
    lineage = prov.get("lineage") if isinstance(prov, dict) else None
    if not isinstance(lineage, list) or not all(isinstance(t, str) and t for t in lineage):
        raise ValueError(f"{ring.name}: its provenance names no lineage list")
    if prov.get("ring_sha256") != sha256_file(ring):
        raise ValueError(f"{ring.name}: its provenance was written for other bytes")
    return tuple(lineage)


def legal_check(corpus: dict[str, np.ndarray], labels: dict[str, np.ndarray], rows: np.ndarray, *, radius: int,
                encoding: str) -> dict[str, float]:
    """On `rows`, the engine's own legal set against `legal_support` inside the crop (must be equal), and its cells outside the crop; Raises: ValueError — a row where they differ."""
    outside = np.zeros(len(rows), np.int64)
    total = np.zeros(len(rows), np.int64)
    for j, i in enumerate(rows):
        moves = row_moves(corpus, int(i))
        board = Board.with_encoding_name(encoding)
        for q, r in moves:
            board.apply_move(int(q), int(r))
        legal = np.asarray(board.legal_moves(), dtype=np.int64).reshape(-1, 2)
        center = tuple(labels["center"][i])
        idx = crop_index(legal[:, 0], legal[:, 1], center)
        mine = legal_support(moves, center, radius)
        if not np.array_equal(np.sort(idx[idx >= 0]), mine):
            raise ValueError(f"row {int(i)}: the engine's in-crop legal set differs from the target's support")
        outside[j] = int((idx < 0).sum())
        total[j] = len(legal)
    return {"rows": int(len(rows)), "share_rows_with_legal_outside_crop": float((outside > 0).mean()),
            "mean_legal_outside_crop": float(outside.mean()), "mean_legal": float(total.mean()),
            "share_legal_cells_outside_crop": float(outside.sum() / max(total.sum(), 1))}


def empty_board_mass(labels: dict[str, np.ndarray], rows: np.ndarray, radius: int) -> dict[str, float]:
    """On empty-board rows: Six's legal-plane softmax mass outside our 5×5 square (renormalised away)."""
    if len(rows) == 0:
        return {"rows": 0}
    q, r = crop_cells((0, 0))
    six_legal = np.nonzero(near_stones(q, r, np.zeros((0, 2), np.int64), radius))[0]
    ours = set(legal_support(np.zeros((0, 2), np.int64), (0, 0), radius).tolist())
    out = []
    for i in rows:
        x = np.asarray(labels["policy"][i])[six_legal].astype(np.float64)
        p = np.exp(x - x.max())
        p /= p.sum()
        out.append(float(p[[c not in ours for c in six_legal]].sum()))
    return {"rows": int(len(rows)), "mass_outside_ours_mean": float(np.mean(out)), "mass_outside_ours_max": float(np.max(out))}
