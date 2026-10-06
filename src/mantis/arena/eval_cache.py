"""A deploy player's per-game exact evaluation cache: a position evaluated once in a game is served from here after."""
from __future__ import annotations

from array import array
from typing import Any

#: One served position: the in-window policy (f32 storage holds the f32 the net returned), the off-window half, the
#: value and the builder's window centre.
_Entry = tuple[array, tuple[tuple[tuple[int, int], float], ...], float, tuple[int, int]]

_ROWS = ("calls", "positions", "hits", "served", "in_batch_repeats", "all_hit_calls")


class GameEvalCache:
    """One player's evaluations in the game in play, keyed exactly as self-play's eval cache keys a leaf.

    A served output does not depend on the pop it rides in, so a hit returns what a fresh evaluation would.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._entries: dict[str, _Entry] = {}
        self._rows: dict[str, int] = dict.fromkeys(_ROWS, 0)

    def new_game(self) -> None:
        self._entries.clear()

    def counters(self) -> dict[str, int]:
        """`positions` = `hits` + `served` + `in_batch_repeats`; `all_hit_calls` were answered with no evaluation."""
        return dict(self._rows)

    def infer_batch_ls(self, boards: list[Any]) -> tuple[list[list[float]], list[list[tuple[tuple[int, int], float]]],
                                                         list[float], list[tuple[int, int]]]:
        """The engine's `infer_batch_ls` for `boards`, evaluating only keys this game has not seen, each once."""
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
        if submit:
            dense, overflow, values, centers = self._engine.infer_positions_ls(submit)
            for key, i in fresh.items():
                self._entries[key] = (array("f", dense[i]), tuple(overflow[i]), float(values[i]), tuple(centers[i]))
        rows = self._rows
        rows["calls"] += 1
        rows["positions"] += len(keys)
        rows["hits"] += hits
        rows["served"] += len(fresh)
        rows["in_batch_repeats"] += len(keys) - hits - len(fresh)
        rows["all_hit_calls"] += int(bool(keys) and not fresh)
        entries = [self._entries[key] for key in keys]
        return ([e[0].tolist() for e in entries], [list(e[1]) for e in entries], [e[2] for e in entries],
                [e[3] for e in entries])


__all__ = ["GameEvalCache"]
