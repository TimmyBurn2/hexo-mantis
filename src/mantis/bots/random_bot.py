"""RandomBot — the locally-resolving ladder floor (port of hexo_rl/hexo_rl/bots/random_bot.py).

Uniform over `board.legal_moves()`, driven by a seeded `random.Random` — deterministic
given the seed + the sequence of positions it is asked to move from.
"""
from __future__ import annotations

import random
from typing import Any


class RandomBot:
    """Uniform-random legal-move bot. Stateless beyond its own seeded RNG stream."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def name(self) -> str:
        return "random"

    def new_game(self) -> None:
        return None

    def select_move(self, board: Any) -> tuple[int, int]:
        legal = board.legal_moves()
        return self._rng.choice(legal)


__all__ = ["RandomBot"]
