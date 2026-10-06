"""A deploy player's per-game exact evaluation cache: a position evaluated once in a game is served from here after."""
from __future__ import annotations

from array import array
from typing import Any

#: One served position in f32 and i32 arrays (exactly the f32 the net returned): the in-window policy, the off-window
#: cells flattened (q, r, q, r, ...) with their probabilities, the value and the builder's window centre.
_Entry = tuple[array, array, array, float, tuple[int, int]]
#: Policy rows, off-window rows, values and window centres, as the engine's `infer_batch_ls` returns them.
_Served = tuple[list[list[float]], list[list[tuple[tuple[int, int], float]]], list[float], list[tuple[int, int]]]

#: A game's entries at most (~5 KB each at radius 8); past it the oldest goes, and a miss is evaluated again.
MAX_ENTRIES = 8192

_ROWS = ("calls", "positions", "hits", "served", "in_batch_repeats", "all_hit_calls", "evicted")


class GameEvalCache:
    """One player's evaluations in the game in play, keyed exactly as self-play's eval cache keys a leaf.

    A hit returns the outputs of the position's first evaluation in the game, which a later pop reproduces within the
    served path's own spread (exactly, in DEPLOY-1's sm_86 dumps).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._entries: dict[str, _Entry] = {}
        self._rows: dict[str, int] = dict.fromkeys(_ROWS, 0)

    def new_game(self) -> None:
        """Empty the cache: no game reads another's evaluations."""
        self._entries.clear()

    def counters(self) -> dict[str, int]:
        """`positions` = `hits` + `served` + `in_batch_repeats`; `all_hit_calls` evaluated nothing; `evicted` hit the cap."""
        return dict(self._rows)

    def infer_batch_ls(self, boards: list[Any]) -> _Served:
        """The engine's `infer_batch_ls`, each unseen key evaluated once. Raises: RuntimeError (closed), ValueError."""
        positions = self._engine.positions_of(boards)
        keys = self._engine.leaf_keys(positions)
        fresh: dict[str, int] = {}
        submit: list[Any] = []
        hits = 0
        for key, position in zip(keys, positions, strict=True):
            if key in self._entries:
                hits += 1
            elif key not in fresh:
                fresh[key] = len(submit)
                submit.append(position)
        served: dict[str, _Entry] = {}
        if submit:
            dense, overflow, values, centers = self._engine.infer_positions_ls(submit)
            for key, i in fresh.items():
                served[key] = (array("f", dense[i]), array("i", [c for (cell, _p) in overflow[i] for c in cell]),
                               array("f", [p for _cell, p in overflow[i]]), float(values[i]), tuple(centers[i]))
        out = _unpack([served[key] if key in served else self._entries[key] for key in keys])
        self._remember(served)
        rows = self._rows
        rows["calls"] += 1
        rows["positions"] += len(keys)
        rows["hits"] += hits
        rows["served"] += len(fresh)
        rows["in_batch_repeats"] += len(keys) - hits - len(fresh)
        rows["all_hit_calls"] += int(bool(keys) and not fresh)
        return out

    def _remember(self, served: dict[str, _Entry]) -> None:
        for key, entry in served.items():
            if len(self._entries) >= MAX_ENTRIES:
                del self._entries[next(iter(self._entries))]
                self._rows["evicted"] += 1
            self._entries[key] = entry


def _unpack(entries: list[_Entry]) -> _Served:
    dense, overflow, values, centers = [], [], [], []
    for d, cells, probs, value, center in entries:
        dense.append(d.tolist())
        overflow.append([((cells[2 * j], cells[2 * j + 1]), p) for j, p in enumerate(probs.tolist())])
        values.append(value)
        centers.append(center)
    return dense, overflow, values, centers


__all__ = ["MAX_ENTRIES", "GameEvalCache"]
