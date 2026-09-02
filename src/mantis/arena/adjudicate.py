"""Ply-cap adjudication — the seam that decides a game which ran out of PLIES, not of moves.

The arena's `_play_one_game` loop ends on three disjoint conditions: a win, an exhausted
legal-move set, and the ply cap. Only the third one is a NON-RESULT: the board is unbounded
(CLAUDE.md), the cap exists solely so a degenerate pairing terminates, and until this module
the loop collapsed that non-result into `"draw"` — the same label a genuinely balanced
finished game gets. F-R-P2B-5's companion measurement is what makes the collapse load-bearing
rather than cosmetic: on the live shakedown burn every game reached the 128-move cap and
`draw_rate` sat at 1.0, so at early strength the eval instrument's entire outcome channel was
one constant. A constant carries no signal, and a promotion bar reading it cannot separate
"the two players are equal" from "neither player can finish".

WHAT THIS MODULE SHIPS, AND WHAT IT DELIBERATELY DOES NOT. It ships the CLASS and the
criteria — each one a pure query against the engine board, so no criterion introduces a
number this layer would have to own. It ships NO threshold: `min_margin` arrives from
`eval.ply_cap_adjudication.min_margin`, an operator-owned mint-prereg value, and there is no
code-side default for it anywhere (R1). Absent config, `PlyCapAdjudicator` is never
constructed and the loop's legacy `"draw"` arm runs untouched — the arming posture is a
property of the config VALUE (`None` vs a block), never of a boolean beside it (the R79
shape `train.draw_rate_abort` established).

MEASUREMENT UNITS (LAW-03), stated because both criteria are counted in PLIES while the game
is played in compound two-stone TURNS: `longest_run_margin` counts CELLS in a line and
`immediate_win_margin` counts single-stone completing moves. Neither is a turn-level quantity
and neither should be read as one.

DISCLOSED ASYMMETRY, for the operator's prereg and not resolved here: at the cap exactly one
side is to move and never gets to play. `immediate_win_margin` is therefore not seat-neutral —
a side-to-move with a completing move would have won on the next ply, so the same position
scores differently depending on whose turn the cap interrupted. `longest_run_margin` is a
property of the placed stones alone and carries no such seat term. The two criteria are
offered side by side precisely so the choice is made with that difference visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The three ways `_play_one_game` can end, as a CLOSED vocabulary stamped onto every
#: `GameRecord`. It exists because "reached the cap" and "was won on the cap ply" were
#: previously indistinguishable downstream: a win detected at exactly `max_plies` and a
#: capped non-result both report `plies == max_plies`, so any consumer deriving decisiveness
#: from `(winner, plies)` alone misclassifies the first as the second. The strength-floor
#: probe reads exactly this field, which is why it is recorded rather than re-derived.
TERMINAL_WIN = "win"
TERMINAL_EXHAUSTED = "exhausted"
TERMINAL_PLY_CAP = "ply_cap"
TERMINAL_REASONS: tuple[str, ...] = (TERMINAL_WIN, TERMINAL_EXHAUSTED, TERMINAL_PLY_CAP)

#: The criteria this module implements, in the order the schema `Literal` declares them. A
#: CLOSED set, checked at construction: a criterion name nothing here implements must be a
#: loud refusal, never a silent fall-through to `"draw"` — a fall-through would read ARMED in
#: the config and be absent in effect, which is the exact defect class R1/LAW-08 exist to kill.
CRITERION_LONGEST_RUN = "longest_run_margin"
CRITERION_IMMEDIATE_WIN = "immediate_win_margin"
PLY_CAP_CRITERIA: tuple[str, ...] = (CRITERION_LONGEST_RUN, CRITERION_IMMEDIATE_WIN)


class PlyCapCriterionError(ValueError):
    """A criterion name this module does not implement. Raised, never defaulted."""


@dataclass(frozen=True)
class PlyCapVerdict:
    """One adjudication decision, with the evidence that produced it.

    `margin` is the SIGNED candidate-minus-opponent measurement in the criterion's own unit,
    kept beside the verdict so an operator reading the round can see how close the call was
    rather than only which way it went (LAW-18's reason for wanting fire-rate over a flag).
    """

    winner: str  # "candidate" | "opponent" | "draw"
    criterion: str
    margin: int

    def as_payload(self) -> dict[str, Any]:
        return {"winner": self.winner, "criterion": self.criterion, "margin": self.margin}


def longest_run(board: Any, player: int, *, ceiling: int) -> int:
    """The player's longest line, DERIVED by probing the engine rather than compared against a
    transcribed win length.

    `has_player_long_run(player, k)` is monotone in `k`, so the loop needs no `WIN_LENGTH`
    literal to stop (a second copy of that constant in Python is the drift surface R192(e)
    names): on an unfinished board the probe turns False strictly below the engine's own win
    length, and on a FINISHED one it turns False just above the winning line, returning that
    line's length. `ceiling` is a structural bound, not a policy one: a player cannot own a
    line longer than the number of plies played, so the loop terminates even if the engine
    predicate were ever to stop being monotone.

    Callers are the ply-cap adjudicator, which only ever sees unfinished boards, and
    `mantis.diagnostics.acceptance_witness`, which reads finished ones — the second is why
    this is the ONE run-length derivation in Python and why it is public: a witness that
    reconstructed stone colour itself would be a second authority over whose line it is, and
    that is exactly the defect the first witness shipped with.

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
    """Resolve a ply-capped game by a DECLARED criterion and an operator-declared margin.

    Constructed only when `eval.ply_cap_adjudication` is a block; the `None` posture never
    builds one, which is what makes the legacy `"draw"` arm bit-identical pre-arm.
    """

    def __init__(self, criterion: str, min_margin: int) -> None:
        if criterion not in PLY_CAP_CRITERIA:
            raise PlyCapCriterionError(
                f"ply-cap criterion {criterion!r} is not implemented; the closed set is "
                f"{list(PLY_CAP_CRITERIA)}. Refusing to fall through to a draw — a criterion "
                f"chosen by fall-through reads armed in the config and is absent in effect."
            )
        self._criterion = criterion
        self._min_margin = int(min_margin)
        # LAW-18: the lever counts its OWN fires. The adjudicator is the one object that sees
        # every capped game, so the fire rate belongs here rather than being re-derived by a
        # consumer downstream — and because the object exists only when the posture is armed,
        # the counter cannot exist in a run that never armed it.
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
        """Capped games seen, and how they were resolved. A COPY — a caller that mutated the
        live counter would silently rewrite the fire rate it was asked to report."""
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
        """Award the capped game iff `|margin| >= min_margin`; otherwise it stays a draw.

        A draw is still a REPRESENTABLE outcome here and deliberately so: the criterion
        replaces the unconditional collapse, it does not abolish the draw. `min_margin >= 1`
        is a schema bound, so a margin of exactly 0 — the two sides measured equal — can never
        award a game to either of them.
        """
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
    "TERMINAL_PLY_CAP",
    "TERMINAL_REASONS",
    "TERMINAL_WIN",
    "longest_run",
]
