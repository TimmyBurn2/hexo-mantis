"""mantis.arena — the EVALFAIR instrument: deploy-matched, paired-book arena play.

Public API: `RegimeKey`/`MixedRegimeError`, `BookError`/`Opening`/`paired_openings`,
`GameRecord`/`play_paired_match`, `DeployHeadPlayer`/`select_argmax_child`,
`PlyCapAdjudicator`/`PlyCapVerdict`/`TERMINAL_REASONS`.
"""
from __future__ import annotations

from mantis.arena.adjudicate import (
    PLY_CAP_CRITERIA,
    TERMINAL_REASONS,
    PlyCapAdjudicator,
    PlyCapCriterionError,
    PlyCapVerdict,
)
from mantis.arena.books import BookError, Opening, paired_openings
from mantis.arena.deploy_head import DeployHeadPlayer, select_argmax_child
from mantis.arena.match import GameRecord, play_paired_match
from mantis.arena.regime import MixedRegimeError, RegimeKey

__all__ = [
    "PLY_CAP_CRITERIA",
    "TERMINAL_REASONS",
    "BookError",
    "DeployHeadPlayer",
    "GameRecord",
    "MixedRegimeError",
    "Opening",
    "PlyCapAdjudicator",
    "PlyCapCriterionError",
    "PlyCapVerdict",
    "RegimeKey",
    "paired_openings",
    "play_paired_match",
    "select_argmax_child",
]
