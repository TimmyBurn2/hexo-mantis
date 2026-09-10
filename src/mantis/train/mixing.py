"""Per-round selfplay -> training-step budgeting (WP10 §a.4 PORT; old training/mixing.py).

`_compute_pretrained_weight` and the corpus-anchor mixing schedule it served are DELETED with
`train.mixing_*` (R346(f)): the mixed arm was dense-only and no graph route was ever carded.
"""
from __future__ import annotations


def _steps_budget(new_games: int, training_steps_per_game: float, max_train_burst: int) -> int:
    """Per-round training-step budget from newly-completed self-play games."""
    return min(max(1, round(new_games * training_steps_per_game)), max_train_burst)
