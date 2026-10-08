"""The labelled positions: ring rows put back in move order by their game records (a ring row keeps its stones as a set), a by-game held-out split, arena openings."""
from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from mantis.diagnostics.ring_reader import Ring, load_ring

from .planes import player_for_stone, stones_left_before

U64 = np.uint64
SRC_RING, SRC_OPENING = 0, 1
#: Fields copied from each ring row, the control's own targets among them.
#: Up to three stones the coloured set fixes everything Six's planes read of the order.
ORDER_FREE_PLIES = 3
RING_FIELDS = ("current_player", "moves_remaining", "ply_index", "is_full_search", "value_valid", "outcome",
               "game_length", "tail_mass", "root_value", "root_value_valid")


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, elementwise on uint64."""
    z = x.astype(U64) * U64(0x9E3779B97F4A7C15)
    z ^= z >> U64(30)
    z *= U64(0xBF58476D1CE4E5B9)
    z ^= z >> U64(27)
    z *= U64(0x94D049BB133111EB)
    z ^= z >> U64(31)
    return z


def stone_hash(q: np.ndarray, r: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """A 64-bit hash per (q, r, colour) stone; a position's key sums them, so order does not matter."""
    packed = ((np.asarray(q, np.int64) & 0xFFFF) | ((np.asarray(r, np.int64) & 0xFFFF) << 16)
              | ((np.asarray(colour) > 0).astype(np.int64) + 1) << 32)
    return _mix(packed.astype(U64))


def position_key(stone_sum: np.ndarray, ply: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """The match key: the stone-set hash with the ply and the game's turn count folded in."""
    with np.errstate(over="ignore"):
        return _mix(stone_sum + np.asarray(ply, U64) * U64(0x632BE59BD9B4E019) + np.asarray(turns, U64) * U64(0x8CB92BA72F3D8DD7))


def turns_of(plies: np.ndarray | int) -> np.ndarray:
    """The ring's game_length for a game of `plies` stones: ceil(plies / 2), as the producer stores it."""
    return (np.asarray(plies) + 1) // 2


@dataclass
class Games:
    """Self-play game records, moves flat in ply order."""

    uid: list[str]
    step: np.ndarray
    winner: np.ndarray  # +1 the opener, -1 the other, 0 neither
    move_off: np.ndarray
    n_moves: np.ndarray
    moves: np.ndarray  # [M, 2] int32

    def game_moves(self, g: int, k: int | None = None) -> np.ndarray:
        """Game `g`'s first `k` moves (all when None)."""
        start = int(self.move_off[g])
        return self.moves[start:start + (int(self.n_moves[g]) if k is None else k)]


def load_games(paths: Iterable[Path], *, run_id: str, step_lo: int, step_hi: int) -> Games:
    """Every self-play record of `run_id` whose actor step lies in [step_lo, step_hi]; Raises: OSError, ValueError — an unreadable shard or a malformed line."""
    uid: list[str] = []
    step: list[int] = []
    winner: list[int] = []
    n_moves: list[int] = []
    chunks: list[np.ndarray] = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("channel") != "selfplay" or rec.get("run_id") != run_id:
                    continue
                s = int(rec.get("step", -1))
                if not step_lo <= s <= step_hi:
                    continue
                mv = np.asarray(rec["moves"], dtype=np.int32).reshape(-1, 2)
                uid.append(str(rec["game_id"]))
                step.append(s)
                winner.append({"p1": 1, "p2": -1}.get(str(rec.get("result")), 0))
                n_moves.append(len(mv))
                chunks.append(mv)
    n = np.asarray(n_moves, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64) if len(n) else np.zeros(0, np.int64)
    moves = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.int32)
    return Games(uid=uid, step=np.asarray(step, np.int64), winner=np.asarray(winner, np.int8), move_off=off,
                 n_moves=n, moves=moves)


def game_position_keys(games: Games) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (game, ply k in 0..n) position's key, with its game index and k."""
    keys: list[np.ndarray] = []
    gidx: list[np.ndarray] = []
    kk: list[np.ndarray] = []
    for g in range(len(games.uid)):
        mv = games.game_moves(g)
        n = len(mv)
        colour = np.array([player_for_stone(i) for i in range(n)], dtype=np.int64)
        with np.errstate(over="ignore"):
            sums = np.concatenate([[U64(0)], np.cumsum(stone_hash(mv[:, 0], mv[:, 1], colour), dtype=U64)])
        ks = np.arange(n + 1)
        keys.append(position_key(sums, ks, np.full(n + 1, turns_of(n))))
        gidx.append(np.full(n + 1, g, np.int64))
        kk.append(ks)
    if not keys:
        return np.zeros(0, U64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(keys), np.concatenate(gidx), np.concatenate(kk)


def ring_row_keys(ring: Ring) -> np.ndarray:
    """Each ring row's key from its stone set, ply and game_length."""
    seg = np.repeat(np.arange(ring.header.size), ring.n_stones)
    sums = np.zeros(ring.header.size, U64)
    with np.errstate(over="ignore"):
        np.add.at(sums, seg, stone_hash(ring.stones["q"], ring.stones["r"], ring.stones["p"]))
    return position_key(sums, ring.ply_index, ring.game_length)


def six_signature(moves: np.ndarray) -> tuple[Any, ...]:
    """What Six's planes read of the order beyond the stone set: the last four stones, the turn's first stone, the opponent's last turn."""
    n = len(moves)
    last4 = tuple(sorted(map(tuple, moves[-4:].tolist())))
    second = n > 0 and stones_left_before(n) == 1
    first = tuple(moves[n - 1].tolist()) if second else None
    start = n - 1 if second else n
    opp = tuple(sorted(tuple(moves[i].tolist()) for i in (start - 2, start - 1) if i >= 0))
    return last4, first, opp


@dataclass
class Match:
    """Per ring row: its game (-1 unmatched), its ply, how many games share its key, and whether they disagree on the order Six reads."""

    game: np.ndarray
    k: np.ndarray
    candidates: np.ndarray
    order_ambiguous: np.ndarray


def match_ring(ring: Ring, games: Games, index: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Match:
    """Match every ring row to a game position; among several, prefer the game whose winner agrees with the row's z, then the lowest index."""
    keys, gidx, ks = index
    order = np.argsort(keys, kind="stable")
    skeys = keys[order]
    rkeys = ring_row_keys(ring)
    lo = np.searchsorted(skeys, rkeys, side="left")
    hi = np.searchsorted(skeys, rkeys, side="right")
    n = ring.header.size
    game = np.full(n, -1, np.int64)
    kout = np.full(n, -1, np.int64)
    cand = (hi - lo).astype(np.int64)
    ambiguous = np.zeros(n, bool)
    single = cand == 1
    game[single] = gidx[order[lo[single]]]
    kout[single] = ks[order[lo[single]]]
    for i in np.nonzero(cand > 1)[0]:
        options = order[lo[i]:hi[i]]
        row_winner = 0
        if ring.value_valid[i] and ring.outcome[i] != 0:
            row_winner = int(ring.current_player[i]) * (1 if ring.outcome[i] > 0 else -1)
        agreeing = [o for o in options if games.winner[gidx[o]] == row_winner] or list(options)
        pick = min(agreeing, key=lambda o: int(gidx[o]))
        game[i], kout[i] = gidx[pick], ks[pick]
        if ring.ply_index[i] > ORDER_FREE_PLIES:
            sigs = {six_signature(games.game_moves(int(gidx[o]), int(ks[o]))) for o in agreeing}
            ambiguous[i] = len(sigs) > 1
    return Match(game=game, k=kout, candidates=cand, order_ambiguous=ambiguous)


def _arena_draw() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "openings" / "arena_draw.py"
    spec = importlib.util.spec_from_file_location("distill_arena_draw", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arena_openings(*, seed: int, plies: int, n: int) -> list[np.ndarray]:
    """`n` openings drawn as the arena draws them (`tools/openings/arena_draw.py`), each `[plies, 2]` in ply order."""
    book = _arena_draw().mint_arena_book(seed=seed, plies=plies, n=n)
    return [np.asarray(o["moves"], dtype=np.int32) for o in book["openings"]]


def heldout_games(row_game: np.ndarray, *, rows: int, seed: int) -> np.ndarray:
    """Whole games, in a seeded order, until their rows reach `rows`: the held-out set's games."""
    games, counts = np.unique(row_game[row_game >= 0], return_counts=True)
    perm = np.random.default_rng(seed).permutation(len(games))
    total = np.cumsum(counts[perm])
    take = int(np.searchsorted(total, rows, side="left")) + 1
    return np.sort(games[perm[:take]])


def build_corpus(ring_paths: Sequence[Path], game_paths: Sequence[Path], *, run_id: str, heldout_rows: int,
                 split_seed: int, openings_n: int, openings_seed: int, openings_plies: int,
                 step_margin: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The corpus arrays and its summary; Raises: ValueError — a ring that is not a self-play ring of `run_id`'s steps or a row its game does not account for."""
    steps = [int(p.name.split("_")[1]) for p in ring_paths]
    games = load_games(game_paths, run_id=run_id, step_lo=min(steps) - step_margin, step_hi=max(steps))
    index = game_position_keys(games)
    cols: dict[str, list[np.ndarray]] = {f: [] for f in RING_FIELDS}
    game_l: list[np.ndarray] = []
    k_l: list[np.ndarray] = []
    src_ring: list[np.ndarray] = []
    src_row: list[np.ndarray] = []
    v_q: list[np.ndarray] = []
    v_r: list[np.ndarray] = []
    v_p: list[np.ndarray] = []
    n_vis: list[np.ndarray] = []
    summary: dict[str, Any] = {"rings": [], "games_loaded": len(games.uid)}
    for j, path in enumerate(ring_paths):
        ring = load_ring(path)
        m = match_ring(ring, games, index)
        ok = m.game >= 0
        if not np.array_equal(ring.n_stones, ring.ply_index):
            raise ValueError(f"{path.name}: a row whose stone count is not its ply")
        summary["rings"].append({"ring": path.name, "rows": int(ring.header.size), "matched": int(ok.sum()),
                                 "multi_candidate": int((m.candidates > 1).sum()),
                                 "order_ambiguous": int(m.order_ambiguous.sum()),
                                 "full_search": int(ring.is_full_search[ok].sum())})
        sel = np.nonzero(ok)[0]
        for f in RING_FIELDS:
            cols[f].append(np.asarray(getattr(ring, f))[sel])
        game_l.append(m.game[sel])
        k_l.append(m.k[sel])
        src_ring.append(np.full(len(sel), j, np.int64))
        src_row.append(sel)
        n_vis.append(ring.n_visits[sel])
        rows_vis = np.repeat(np.isin(np.arange(ring.header.size), sel), ring.n_visits)
        v_q.append(ring.visits["q"][rows_vis])
        v_r.append(ring.visits["r"][rows_vis])
        v_p.append(ring.visits["prob"][rows_vis])
    out: dict[str, np.ndarray] = {f: np.concatenate(v) for f, v in cols.items()}
    out["game"] = np.concatenate(game_l)
    out["k"] = np.concatenate(k_l)
    if not np.array_equal(out["k"], out["ply_index"]):
        raise ValueError("a matched row's ply differs from its game position's")
    mover = np.array([player_for_stone(int(k)) for k in out["k"]], dtype=np.int64)
    if not np.array_equal(mover, out["current_player"]):
        raise ValueError("a matched row's player to move is not its ply's")
    out["src_ring"] = np.concatenate(src_ring)
    out["src_row"] = np.concatenate(src_row)
    out["n_visits"] = np.concatenate(n_vis)
    out["visit_q"] = np.concatenate(v_q)
    out["visit_r"] = np.concatenate(v_r)
    out["visit_p"] = np.concatenate(v_p)
    out["source"] = np.full(len(out["game"]), SRC_RING, np.int8)
    held = heldout_games(out["game"], rows=heldout_rows, seed=split_seed)
    out["heldout"] = np.isin(out["game"], held)
    opening_moves = arena_openings(seed=openings_seed, plies=openings_plies, n=openings_n)
    out = _append_openings(out, games, opening_moves)
    out["games_move_off"] = games.move_off
    out["games_n_moves"] = games.n_moves
    out["games_moves"] = games.moves
    out["games_step"] = games.step
    distinct = {tuple(map(tuple, mv.tolist())) for mv in opening_moves}
    summary.update(rows=int(len(out["game"])), ring_rows=int((out["source"] == SRC_RING).sum()),
                   opening_rows=int(openings_n), opening_distinct_orders=len(distinct),
                   heldout_rows=int(out["heldout"].sum()), heldout_games=int(len(held)),
                   games_used=int(len(np.unique(out["game"][out["source"] == SRC_RING]))))
    return out, {"summary": summary, "game_uid": games.uid}


def _append_openings(out: dict[str, np.ndarray], games: Games, openings: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Each opening becomes its own one-row game appended to the games table; its targets are the teacher's alone."""
    n = len(openings)
    plies = np.array([len(o) for o in openings], np.int64)
    base = len(games.uid)
    games.uid.extend(f"opening-{i}" for i in range(n))
    games.step = np.concatenate([games.step, np.full(n, -1, np.int64)])
    games.winner = np.concatenate([games.winner, np.zeros(n, np.int8)])
    games.move_off = np.concatenate([games.move_off, len(games.moves) + np.concatenate([[0], np.cumsum(plies)[:-1]])])
    games.n_moves = np.concatenate([games.n_moves, plies])
    games.moves = np.concatenate([games.moves, *openings]) if n else games.moves
    add: dict[str, np.ndarray] = {
        "current_player": np.array([player_for_stone(int(p)) for p in plies], np.int64),
        "moves_remaining": np.array([stones_left_before(int(p)) for p in plies], np.int64),
        "ply_index": plies, "is_full_search": np.ones(n, np.int64), "value_valid": np.zeros(n, np.int64),
        "outcome": np.zeros(n, np.float32), "game_length": np.zeros(n, np.int64), "tail_mass": np.zeros(n, np.float32),
        "root_value": np.zeros(n, np.float32), "root_value_valid": np.zeros(n, np.int64),
        "game": base + np.arange(n, dtype=np.int64), "k": plies, "src_ring": np.full(n, -1, np.int64),
        "src_row": np.full(n, -1, np.int64), "n_visits": np.zeros(n, np.int64), "source": np.full(n, SRC_OPENING, np.int8),
        "heldout": np.zeros(n, bool),
    }
    return {key: (np.concatenate([val, add[key]]) if key in add else val) for key, val in out.items()}
