"""Ply-cap adjudication: the criterion class, the arena seam, and the INERT default.

The disarmed and armed arms are read together — without the armed one, the disarmed arm passes
equally against a seam that can never fire. `terminal` exists because a genuine win on the cap
ply and a capped non-result both report `plies == max_plies`.

Positions are planted through the OPENING, not a bot script: the engine hands the first stone
to player 1 and then alternates in PAIRS, so a bot-scripted line splits between the two sides.
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

_ENCODING = "gnn_axis_v1"

#: A LOPSIDED, unfinished, and LEGALLY REACHABLE position, ten plies long — every stone within
#: the encoding's `legal_move_radius`, or the replay refuses it. Its properties are re-derived
#: from the engine below, never asserted here.
_PLANTED = [
    (0, 0),          # ply 0  -> player  1
    (0, 4), (2, 4),  # plies 1,2 -> player -1
    (1, 0), (2, 0),  # plies 3,4 -> player  1
    (4, 4), (6, 4),  # plies 5,6 -> player -1
    (3, 0), (4, 0),  # plies 7,8 -> player  1
    (8, 4),          # ply 9  -> player -1
]
#: The same position continued to a genuine win on the last permitted ply — the shape that is
#: indistinguishable from a cap without the `terminal` field.
_WIN_ON_CAP = [*_PLANTED, (10, 4), (5, 0)]


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


def test_the_planted_position_is_lopsided_and_unfinished() -> None:
    """Guard the premise every award test rests on, so a moved turn order fails HERE."""
    board = _planted_board()
    assert not board.check_win(), "the planted position must be unfinished"
    assert board.ply == len(_PLANTED)
    assert _longest_run(board, 1) > _longest_run(board, -1), (
        "player 1 must own the strictly longer line, or longest_run_margin measures nothing"
    )
    assert board.count_winning_moves(1) > board.count_winning_moves(-1), (
        "player 1 must own the strictly greater immediate-win count"
    )


def test_an_unimplemented_criterion_refuses_loudly_instead_of_defaulting_to_draw() -> None:
    """Prove an unhonourable criterion RAISES rather than reading armed and being inert."""
    with pytest.raises(PlyCapCriterionError) as ei:
        PlyCapAdjudicator("centre_control", 1)
    assert "centre_control" in str(ei.value)
    assert CRITERION_LONGEST_RUN in str(ei.value), "the refusal must name the closed set"


def test_the_criterion_set_is_closed_and_matches_the_schema_literal() -> None:
    """Prove the closed set and the schema `Literal` are ONE fact, not two authorities."""
    from typing import get_args

    from mantis.config.schema import PlyCapAdjudicationConfig

    schema_names = set(get_args(
        PlyCapAdjudicationConfig.model_fields["criterion"].annotation
    ))
    assert schema_names == set(PLY_CAP_CRITERIA)
    assert PLY_CAP_CRITERIA == (CRITERION_LONGEST_RUN, CRITERION_IMMEDIATE_WIN)


@pytest.mark.parametrize("criterion", PLY_CAP_CRITERIA)
def test_every_criterion_is_seat_neutral_in_its_MEASUREMENT(criterion: str) -> None:
    """Prove the signed margin inverts exactly when the seat swaps, for both criteria."""
    board = _planted_board()
    adj = PlyCapAdjudicator(criterion, 1)
    plus = adj.measure(board, candidate_color=1, plies=board.ply)
    minus = adj.measure(board, candidate_color=-1, plies=board.ply)
    assert plus == -minus
    assert plus != 0, "the planted position must be measurably unequal on every criterion"


def test_each_criterion_reads_the_engine_rather_than_a_transcribed_number() -> None:
    """Re-derive both margins from the engine at the point of use, so the test cannot drift."""
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
    """Prove the tally moves once per capped game, splits by outcome, and hands back a copy."""
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


def test_a_capped_game_is_a_draw_when_no_adjudicator_is_armed() -> None:
    """THE INERTNESS ARM: `adjudicator=None`, what every committed config produces, still draws."""
    records = _play(_PLANTED, max_plies=len(_PLANTED), adjudicator=None)
    assert records, "the fixture must produce games or the assertion below is vacuous"
    for rec in records:
        assert rec.terminal == TERMINAL_PLY_CAP
        assert rec.winner == "draw"
        assert rec.adjudication is None


def test_the_same_capped_game_is_awarded_once_a_criterion_is_armed() -> None:
    """THE MUTATION ARM: without it the inertness arm passes against a dead seam."""
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
    """Pin the misclassification `terminal` exists to prevent: a win at `max_plies` is a win."""
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
