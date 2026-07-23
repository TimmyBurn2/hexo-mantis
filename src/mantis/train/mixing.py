"""Pretrain/selfplay mixing helpers (WP10 §a.4 PORT; old training/mixing.py).

Pure functions for AlphaZero-style pretrained-weight decay and per-round
selfplay → training-step budgeting. The ``w_pre`` schedule is the context-law
corpus-anchor mix (kept per §e); no falsified lever. Behaviour-exact.
"""
from __future__ import annotations

import math


def _compute_pretrained_weight(step: int, initial_w: float, min_w: float, decay_steps: float) -> float:
    """``w_pre = max(min_w, initial_w · exp(-step / decay_steps))`` — the pretrained
    corpus-anchor mixing weight (context law). Verbatim from old mixing.py."""
    return max(min_w, initial_w * math.exp(-step / decay_steps))


def _steps_budget(new_games: int, training_steps_per_game: float, max_train_burst: int) -> int:
    """Per-round training-step budget from newly-completed self-play games."""
    return min(max(1, round(new_games * training_steps_per_game)), max_train_burst)
