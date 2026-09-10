"""Ply-cap adjudication — the seam that decides a game which ran out of PLIES, not of moves.

The cap is the one loop exit that is a NON-RESULT; collapsing it into `"draw"` once made the
eval instrument's whole outcome channel a constant. `min_margin` arrives from
`eval.ply_cap_adjudication.min_margin` with no code-side default, and absent that block
`PlyCapAdjudicator` is never constructed, so arming is a property of the config VALUE alone.

Both criteria count PLIES while the game is played in compound two-stone TURNS, and
`immediate_win_margin` is not seat-neutral: at the cap one side is to move and never plays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A CLOSED vocabulary stamped onto every `GameRecord`: a win detected at exactly `max_plies`
#: and a capped non-result both report `plies == max_plies`, so `(winner, plies)` cannot separate
#: them.
TERMINAL_WIN = "win"
TERMINAL_EXHAUSTED = "exhausted"
TERMINAL_PLY_CAP = "ply_cap"
#: A coordinate outside the board's legal set — NOT a draw and NOT a win, because a climbing
#: forfeit rate is a broken head rather than a strong one.
TERMINAL_FORFEIT = "forfeit"
TERMINAL_REASONS: tuple[str, ...] = (
    TERMINAL_WIN, TERMINAL_EXHAUSTED, TERMINAL_PLY_CAP, TERMINAL_FORFEIT,
)

#: The criteria this module implements, a CLOSED set checked at construction: an unimplemented
#: name is a loud refusal, since a fall-through would read ARMED in the config and be absent in
#: effect.
CRITERION_LONGEST_RUN = "longest_run_margin"
CRITERION_IMMEDIATE_WIN = "immediate_win_margin"
PLY_CAP_CRITERIA: tuple[str, ...] = (CRITERION_LONGEST_RUN, CRITERION_IMMEDIATE_WIN)


class PlyCapCriterionError(ValueError):
    """A criterion name this module does not implement. Raised, never defaulted."""


@dataclass(frozen=True)
class PlyCapVerdict:
    """One adjudication decision, with the SIGNED candidate-minus-opponent margin in the
    criterion's own unit, so an operator can see how close the call was."""

    winner: str  # "candidate" | "opponent" | "draw"
    criterion: str
    margin: int

    def as_payload(self) -> dict[str, Any]:
        return {"winner": self.winner, "criterion": self.criterion, "margin": self.margin}


def longest_run(board: Any, player: int, *, ceiling: int) -> int:
    """The player's longest line, DERIVED by probing the engine rather than transcribing a win
    length. `has_player_long_run` is monotone in `k` so no `WIN_LENGTH` literal is needed, and
    `ceiling` bounds the loop structurally. Public because the acceptance witness reads FINISHED
    boards through it, making this the ONE run-length derivation in Python.

    Args:
        board: an engine board.
        player: the stone colour to measure, in the engine's own `+1`/`-1` vocabulary.
        ceiling: the structural upper bound on the answer (plies played).

    Returns:
        The length of `player`'s longest line, `0` if they own no stone.
    """
    length = 0
    while length < ceiling and board.has_player_long_run(player, length + 1):
        length += 1
    return length


class PlyCapAdjudicator:
    """Resolve a ply-capped game by a DECLARED criterion and an operator-declared margin;
    constructed only when `eval.ply_cap_adjudication` is a block."""

    def __init__(self, criterion: str, min_margin: int) -> None:
        if criterion not in PLY_CAP_CRITERIA:
            raise PlyCapCriterionError(
                f"ply-cap criterion {criterion!r} is not implemented; the closed set is "
                f"{list(PLY_CAP_CRITERIA)}. Refusing to fall through to a draw — a criterion "
                f"chosen by fall-through reads armed in the config and is absent in effect."
            )
        self._criterion = criterion
        self._min_margin = int(min_margin)
        # The lever counts its OWN fires: this is the one object that sees every capped game,
        # and it exists only when the posture is armed.
        self._tally: dict[str, int] = {
            "adjudicated": 0, "candidate": 0, "opponent": 0, "draw": 0,
        }

    @property
    def criterion(self) -> str:
        return self._criterion

    @property
    def min_margin(self) -> int:
        return self._min_margin

    def tally(self) -> dict[str, int]:
        """Capped games seen, and how they were resolved. A COPY — mutating the live counter
        would silently rewrite the fire rate the caller was asked to report."""
        return dict(self._tally)

    def measure(self, board: Any, *, candidate_color: int, plies: int) -> int:
        """The signed candidate-minus-opponent margin in the criterion's own unit."""
        opponent_color = -candidate_color
        if self._criterion == CRITERION_LONGEST_RUN:
            return (
                longest_run(board, candidate_color, ceiling=plies)
                - longest_run(board, opponent_color, ceiling=plies)
            )
        return (
            int(board.count_winning_moves(candidate_color))
            - int(board.count_winning_moves(opponent_color))
        )

    def adjudicate(self, board: Any, *, candidate_color: int, plies: int) -> PlyCapVerdict:
        """Award the capped game iff `|margin| >= min_margin`; otherwise it stays a draw — the
        criterion replaces the unconditional collapse, it does not abolish the draw, and the
        schema's `min_margin >= 1` means a margin of exactly 0 can never award a game."""
        margin = self.measure(board, candidate_color=candidate_color, plies=plies)
        if margin >= self._min_margin:
            winner = "candidate"
        elif -margin >= self._min_margin:
            winner = "opponent"
        else:
            winner = "draw"
        self._tally["adjudicated"] += 1
        self._tally[winner] += 1
        return PlyCapVerdict(winner=winner, criterion=self._criterion, margin=margin)


__all__ = [
    "CRITERION_IMMEDIATE_WIN",
    "CRITERION_LONGEST_RUN",
    "PLY_CAP_CRITERIA",
    "PlyCapAdjudicator",
    "PlyCapCriterionError",
    "PlyCapVerdict",
    "TERMINAL_EXHAUSTED",
    "TERMINAL_FORFEIT",
    "TERMINAL_PLY_CAP",
    "TERMINAL_REASONS",
    "TERMINAL_WIN",
    "longest_run",
]
