"""Ply-cap adjudication — the criterion class, the arena seam, and the INERT default.

The suite's spine is one pair of tests that must be read together: the DISARMED arm proves a
ply-capped game still scores `"draw"` exactly as it did before this seam existed, and the
ARMED arm proves that the SAME position under an adjudicator scores differently. Without the
second, the first is vacuous — it would pass equally against a seam that could never fire,
which is the phantom-lever class LAW-07 exists to prevent.

The second thing pinned here is the `terminal` field's REASON FOR EXISTING: a genuine win
found on the cap ply and a capped non-result both report `plies == max_plies`, so any
consumer that derives decisiveness from `(winner, plies)` misreads the first as the second.
The strength-floor probe is exactly such a consumer, so the distinction is pinned as
something the record itself carries.

POSITIONS ARE PLANTED THROUGH THE OPENING, never through a bot script. `_play_one_game`
replays an opening onto the board with `board.apply_move` directly, so the sequence is
deterministic and independent of the compound two-stone turn order (LAW-03: the engine hands
the first stone to player 1 and then alternates in PAIRS, so ply index and player are not the
same alternation and a bot-scripted line silently splits between the two sides).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from mantis._engine import Board
from mantis.arena.adjudicate import (
    CRITERION_IMMEDIATE_WIN,
    CRITERION_LONGEST_RUN,
    PLY_CAP_CRITERIA,
    TERMINAL_PLY_CAP,
    TERMINAL_WIN,
    PlyCapAdjudicator,
    PlyCapCriterionError,
)
from mantis.arena.match import play_paired_match
from mantis.arena.regime import RegimeKey

_ENCODING = "v6_live2_ls"

#: A LOPSIDED, unfinished position, ten plies long. Player 1's stones are collinear and
#: player -1's are scattered, so both criteria measure a non-zero margin — asserted below
#: rather than assumed, because a balanced position would make every award test vacuous. The
#: cell assignment follows the engine's own compound-turn order (1, then pairs), verified by
#: the two `longest`/`count_winning_moves` assertions in `test_the_planted_position_is_lopsided`.
_PLANTED = [
    (0, 0),          # ply 0  -> player  1
    (9, 9), (0, 9),  # plies 1,2 -> player -1
    (1, 0), (2, 0),  # plies 3,4 -> player  1
    (2, 9), (4, 9),  # plies 5,6 -> player -1
    (3, 0), (4, 0),  # plies 7,8 -> player  1
    (6, 9),          # ply 9  -> player -1
]
#: The same position CONTINUED to a genuine win on the last permitted ply: player -1 takes
#: plies 9,10 and player 1 completes its six at ply 11, so the game ends `plies == 12` WITH a
#: winner — the shape that is indistinguishable from a cap without the `terminal` field.
_WIN_ON_CAP = [*_PLANTED, (8, 9), (5, 0)]


@dataclass(frozen=True)
class _Opening:
    opening_id: str
    moves: list


class _FirstLegalBot:
    """Plays the first legal move forever. Never consulted when the opening fills the cap."""

    def new_game(self) -> None:
        return None

    def select_move(self, board):
        return board.legal_moves()[0]

    def name(self) -> str:
        return "first_legal_bot"


def _board_factory():
    return Board.with_encoding_name(_ENCODING)


def _planted_board() -> Board:
    board = _board_factory()
    for q, r in _PLANTED:
        board.apply_move(q, r)
    return board


def _longest_run(board, player: int) -> int:
    n = 0
    while board.has_player_long_run(player, n + 1):
        n += 1
    return n


def _regime_key() -> RegimeKey:
    return RegimeKey(
        bot="candidate", variant="test", model_sims=1, opponent_spec="fixed",
        opening_book="test_book", deploy_matched=True, encoding=_ENCODING,
    )


def _play(moves: list, *, max_plies: int, adjudicator):
    return play_paired_match(
        _FirstLegalBot(), _FirstLegalBot(),
        [_Opening(opening_id="planted", moves=list(moves))],
        regime_key=_regime_key(), board_factory=_board_factory, record_sink=None,
        max_plies=max_plies, adjudicator=adjudicator,
    )


# ── the fixture's own premise ──────────────────────────────────────────────────────────
def test_the_planted_position_is_lopsided_and_unfinished() -> None:
    """Guard the premise every award test below rests on. If the engine's turn order or line
    geometry ever moved, these assertions fail HERE — naming the fixture — instead of turning
    the award tests into passes against a balanced board."""
    board = _planted_board()
    assert not board.check_win(), "the planted position must be unfinished"
    assert board.ply == len(_PLANTED)
    assert _longest_run(board, 1) > _longest_run(board, -1), (
        "player 1 must own the strictly longer line, or longest_run_margin measures nothing"
    )
    assert board.count_winning_moves(1) > board.count_winning_moves(-1), (
        "player 1 must own the strictly greater immediate-win count"
    )


# ── the criterion class ────────────────────────────────────────────────────────────────
def test_an_unimplemented_criterion_refuses_loudly_instead_of_defaulting_to_draw() -> None:
    """A criterion this module cannot honour must RAISE. A fall-through to `"draw"` would
    read ARMED in the config and be absent in effect — the exact silently-disabled-lever
    class R1/LAW-08 exist to kill."""
    with pytest.raises(PlyCapCriterionError) as ei:
        PlyCapAdjudicator("centre_control", 1)
    assert "centre_control" in str(ei.value)
    assert CRITERION_LONGEST_RUN in str(ei.value), "the refusal must name the closed set"


def test_the_criterion_set_is_closed_and_matches_the_schema_literal() -> None:
    """The adjudicator's closed set and the schema `Literal` are ONE fact. Two authorities
    for it is how a config-legal criterion becomes a round-time refusal."""
    from typing import get_args

    from mantis.config.schema import PlyCapAdjudicationConfig

    schema_names = set(get_args(
        PlyCapAdjudicationConfig.model_fields["criterion"].annotation
    ))
    assert schema_names == set(PLY_CAP_CRITERIA)
    assert PLY_CAP_CRITERIA == (CRITERION_LONGEST_RUN, CRITERION_IMMEDIATE_WIN)


@pytest.mark.parametrize("criterion", PLY_CAP_CRITERIA)
def test_every_criterion_is_seat_neutral_in_its_MEASUREMENT(criterion: str) -> None:
    """The signed margin must invert exactly when the seat swaps, for both criteria.

    This is a claim about the MEASUREMENT, not about the position: the seat asymmetry the
    module discloses for `immediate_win_margin` is about which side never moves again after
    the cap, not about the arithmetic being lopsided in one direction.
    """
    board = _planted_board()
    adj = PlyCapAdjudicator(criterion, 1)
    plus = adj.measure(board, candidate_color=1, plies=board.ply)
    minus = adj.measure(board, candidate_color=-1, plies=board.ply)
    assert plus == -minus
    assert plus != 0, "the planted position must be measurably unequal on every criterion"


def test_each_criterion_reads_the_engine_rather_than_a_transcribed_number() -> None:
    """Both margins are re-derived here from the engine's own queries at the point of use, so
    the test cannot drift away from what the adjudicator computes (R192(e))."""
    board = _planted_board()
    assert PlyCapAdjudicator(CRITERION_LONGEST_RUN, 1).measure(
        board, candidate_color=1, plies=board.ply
    ) == _longest_run(board, 1) - _longest_run(board, -1)
    assert PlyCapAdjudicator(CRITERION_IMMEDIATE_WIN, 1).measure(
        board, candidate_color=1, plies=board.ply
    ) == board.count_winning_moves(1) - board.count_winning_moves(-1)


def test_a_margin_below_the_bar_stays_a_draw_and_one_at_the_bar_awards() -> None:
    """`min_margin` is a real threshold in both directions, and equality AWARDS (`>=`)."""
    board = _planted_board()
    measured = PlyCapAdjudicator(CRITERION_LONGEST_RUN, 1).measure(
        board, candidate_color=1, plies=board.ply
    )
    at_bar = PlyCapAdjudicator(CRITERION_LONGEST_RUN, measured).adjudicate(
        board, candidate_color=1, plies=board.ply
    )
    above_bar = PlyCapAdjudicator(CRITERION_LONGEST_RUN, measured + 1).adjudicate(
        board, candidate_color=1, plies=board.ply
    )
    assert at_bar.winner == "candidate" and at_bar.margin == measured
    assert above_bar.winner == "draw", "a margin under the bar must not award the game"


def test_the_adjudicator_counts_its_own_fires() -> None:
    """LAW-18: the lever reports its fire rate. The tally is the count the round's event
    carries, and it must move once per capped game and split by outcome."""
    board = _planted_board()
    adj = PlyCapAdjudicator(CRITERION_LONGEST_RUN, 1)
    assert adj.tally() == {"adjudicated": 0, "candidate": 0, "opponent": 0, "draw": 0}
    adj.adjudicate(board, candidate_color=1, plies=board.ply)
    adj.adjudicate(board, candidate_color=-1, plies=board.ply)
    tally = adj.tally()
    assert tally["adjudicated"] == 2
    assert tally["candidate"] == 1 and tally["opponent"] == 1, (
        "the two seats on one lopsided position must award opposite ways"
    )

    tally["adjudicated"] = 99
    assert adj.tally()["adjudicated"] == 2, "tally() must hand back a COPY, never the ring"


# ── the arena seam: INERT by default, and the mutation that proves it is not vacuous ────
def test_a_capped_game_is_a_draw_when_no_adjudicator_is_armed() -> None:
    """THE INERTNESS ARM. `adjudicator=None` is what every committed config produces, and on
    it the capped game keeps the label it had before this seam existed."""
    records = _play(_PLANTED, max_plies=len(_PLANTED), adjudicator=None)
    assert records, "the fixture must produce games or the assertion below is vacuous"
    for rec in records:
        assert rec.terminal == TERMINAL_PLY_CAP
        assert rec.winner == "draw"
        assert rec.adjudication is None


def test_the_same_capped_game_is_awarded_once_a_criterion_is_armed() -> None:
    """THE MUTATION ARM — without it the test above passes against a dead seam."""
    disarmed = _play(_PLANTED, max_plies=len(_PLANTED), adjudicator=None)
    adj = PlyCapAdjudicator(CRITERION_LONGEST_RUN, 1)
    armed = _play(_PLANTED, max_plies=len(_PLANTED), adjudicator=adj)

    assert [r.winner for r in disarmed] == ["draw"] * len(disarmed)
    assert all(r.adjudication is not None for r in armed)
    assert all(r.terminal == TERMINAL_PLY_CAP for r in armed)
    assert adj.tally()["adjudicated"] == len(armed), (
        "every capped game must reach the adjudicator exactly once"
    )
    assert [r.winner for r in armed] != [r.winner for r in disarmed], (
        "the armed posture must CHANGE at least one outcome, or the inertness arm above is "
        "passing against a seam that can never fire"
    )


def test_a_win_found_on_the_cap_ply_is_recorded_as_a_win_not_a_cap() -> None:
    """The misclassification `terminal` exists to prevent, pinned directly.

    A game whose winning move lands exactly at `max_plies` reports `plies == max_plies` —
    indistinguishable from a capped non-result to any consumer reading `(winner, plies)`.
    The recorded reason separates them, and such a game must NOT reach adjudication.
    """
    adj = PlyCapAdjudicator(CRITERION_LONGEST_RUN, 1)
    records = _play(_WIN_ON_CAP, max_plies=len(_WIN_ON_CAP), adjudicator=adj)
    assert records
    for rec in records:
        assert rec.plies == len(_WIN_ON_CAP), (
            "the fixture must land its win ON the cap ply, or it tests a different shape"
        )
        assert rec.terminal == TERMINAL_WIN
        assert rec.winner in ("candidate", "opponent")
        assert rec.adjudication is None, (
            "a game won under the rules must never be routed through adjudication"
        )
    assert adj.tally()["adjudicated"] == 0
