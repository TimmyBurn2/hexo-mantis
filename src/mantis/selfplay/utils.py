"""Shared temperature utilities for the self-play pipeline.

`get_temperature` is the mode-based resolver the Python bot paths use; the training path
resolves through `SelfPlayHParams` in `hparams.py` instead. Both funnel into
`quarter_cosine_temperature`, the one shared schedule shape.
"""
from __future__ import annotations

import math
from typing import Any


def quarter_cosine_temperature(compound_move: int, threshold: int, temp_min: float) -> float:
    """Return the within-game quarter-cosine temperature.

    Mirrors the Rust training-path `mantis_search::compute_move_temperature` exactly:

        tau(cm) = max(temp_min, cos(pi/2 * cm / threshold))   for cm < threshold
                = temp_min                                     for cm >= threshold

    ``threshold == 0`` => the schedule is OFF: a constant ``temp_min`` at every
    compound move (no div-by-zero — the divide lives inside the ``cm < threshold``
    branch, which ``cm < 0`` never enters).
    """
    if threshold > 0 and compound_move < threshold:
        return max(temp_min, math.cos(math.pi / 2 * compound_move / threshold))
    return temp_min


def get_temperature(ply: int, mode: str, config: dict[str, Any]) -> float:
    """Return the MCTS sampling temperature for the current game state.

    Args:
        ply:    Total half-moves played so far (board.ply).
        mode:   "training"   → compound-turn quarter-cosine (see
                               :func:`quarter_cosine_temperature`); identical
                               shape + clock as the Rust training path.
                "evaluation" → tau=0.0 (argmax, deterministic).
                "bootstrap"  → tau=0.5 (moderate, for minimax corpus games).
        config: Config dict. The "training" branch reads
                ``temperature_threshold_compound_moves`` + ``temp_min`` from the
                ``mcts`` sub-dict (else top-level). The legacy
                ``temperature_threshold_ply`` is honoured as an eval/bot alias,
                auto-converted plies → compound-turns. Missing keys ⇒ schedule
                OFF (threshold 0 ⇒ constant ``temp_min``, default 0.5).

    Returns:
        Sampling temperature as a float.
    """
    if mode == "evaluation":
        return 0.0
    if mode == "bootstrap":
        return 0.5
    # Training/exploration mode: the shared compound-turn quarter-cosine.
    mcts_cfg = config.get("mcts", config)

    def _get(key: str) -> Any:
        return mcts_cfg.get(key, config.get(key))

    threshold = _get("temperature_threshold_compound_moves")
    if threshold is None:
        # Legacy eval/bot alias: a ply-clock threshold converts to compound turns.
        legacy_ply = _get("temperature_threshold_ply")
        threshold = (int(legacy_ply) + 1) // 2 if legacy_ply is not None else 0
    threshold = int(threshold)

    tm = _get("temp_min")
    temp_min = float(tm if tm is not None else 0.5)
    compound_move = 0 if ply == 0 else (ply + 1) // 2
    return quarter_cosine_temperature(compound_move, threshold, temp_min)


__all__ = ["get_temperature", "quarter_cosine_temperature"]
