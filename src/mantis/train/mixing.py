"""Per-round selfplay -> training-step budgeting (WP10 §a.4 PORT; old training/mixing.py).

`_compute_pretrained_weight` and the corpus-anchor mixing schedule it served are DELETED with
`train.mixing_*` (R346(f)): the mixed arm was dense-only and no graph route was ever carded.
"""
from __future__ import annotations

import math


def _steps_budget(
    new_games: int, training_steps_per_game: float, max_train_burst: int, carry: float,
) -> tuple[int, float]:
    """Per-burst training-step budget `(min(max(1, floor(carry + games * ratio)), burst), the fraction floored off)`."""
    # The carry is what makes a fractional ratio hold long-run: 98 % of run8's bursts saw ONE new
    # game, where a per-burst round() realised integers only (2.5 -> 2). The ceiling drops, never owes.
    total = round(carry + new_games * training_steps_per_game, 9)  # 2.9999999999999996 floors to 3
    want = math.floor(total)
    return min(max(1, want), max_train_burst), total - want
