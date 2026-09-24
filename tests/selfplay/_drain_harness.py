"""The scripted clock and golden-games loader the drain suites share."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ScriptedTime:
    """`pool_drain`'s `time`: replays `sequence` (then holds its last value), records sleeps."""

    def __init__(self, sequence: Iterable[float]) -> None:
        self.sequence = list(sequence)
        self.i = 0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        value = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        return value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def games_from_golden(golden: dict[str, Any]) -> list[tuple]:
    """The drain golden's scripted `drain_game_results()` rows, moves back to tuple-of-tuples."""
    games = []
    for row in golden["_constants"]["games_batch"]:
        plies, winner_code, moves, *rest = row
        games.append((plies, winner_code, [tuple(m) for m in moves], *rest))
    return games
