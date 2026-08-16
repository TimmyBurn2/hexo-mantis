"""play_paired_match — drives compound-turn paired games on the engine Board (design §a.2
match.py). ARGMAX ONLY: there is no softmax-knob parameter anywhere in this signature —
structurally unrepresentable (dispatch item 7).

Every opening is played exactly TWICE (colors swapped) — the paired-game law
(hexo_rl `_play_pair` :331-360 parity). Every `GameRecord` is stamped with `regime_key`,
`opening_id`, `colors`, `trajectory_hash` (sha256 over the move list — LAW-04 dedupe
input), `winner`, `plies`.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from mantis.arena.adjudicate import (
    TERMINAL_EXHAUSTED,
    TERMINAL_PLY_CAP,
    TERMINAL_WIN,
    PlyCapAdjudicator,
    PlyCapVerdict,
)
from mantis.arena.regime import RegimeKey

#: `colors` shape: {"candidate": <player int>, "opponent": <player int>} — engine
#: convention (1 / -1). ORACLE-CHOSEN SEAM (tests/arena/test_match_fairness.py docstring).
Colors = dict[str, int]


@dataclass(frozen=True)
class GameRecord:
    regime_key: RegimeKey
    opening_id: str
    colors: Colors
    trajectory_hash: str
    winner: str  # "candidate" | "opponent" | "draw"
    plies: int
    moves: tuple[tuple[int, int], ...]
    #: WHICH of the loop's three exits ended this game — `mantis.arena.adjudicate.
    #: TERMINAL_REASONS`. RECORDED rather than re-derived because `(winner, plies)` cannot
    #: recover it: a genuine win found on the cap ply and a capped non-result both report
    #: `plies == max_plies`, so any consumer deriving decisiveness from those two fields
    #: misreads the first as the second. The strength-floor probe's decisive-rate is exactly
    #: such a consumer, which is why the fact is written down at the one place that knows it.
    terminal: str
    #: The adjudication that produced `winner`, or `None` when no adjudicator was armed —
    #: i.e. `None` on every game every shipped config plays. It is `None` on non-capped games
    #: even under an armed adjudicator, because those were decided by the rules.
    adjudication: PlyCapVerdict | None


def _trajectory_hash(moves: Iterable[tuple[int, int]]) -> str:
    h = hashlib.sha256()
    for q, r in moves:
        h.update(f"{q},{r};".encode())
    return h.hexdigest()


#: Hard ply cap — the board is UNBOUNDED (CLAUDE.md), so a game between two
#: non-adversarial/degenerate bots (a fixed-move-then-first-legal-move stub, an untrained
#: net) is not guaranteed to ever complete a 6-in-a-row: without a cap the loop below would
#: never terminate. Mirrors the production self-play default
#: (`SelfPlayRunnerConfig.max_moves_per_game=128`, _engine.pyi) — reaching it ends the game
#: a draw, exactly like a real self-play worker's own cap.
_DEFAULT_MAX_PLIES = 128


def _play_one_game(
    candidate_player: Any,
    opponent_bot: Any,
    opening_moves: list[tuple[int, int]],
    *,
    candidate_color: int,
    board_factory: Callable[[], Any],
    max_plies: int = _DEFAULT_MAX_PLIES,
    adjudicator: PlyCapAdjudicator | None = None,
) -> tuple[str, int, tuple[tuple[int, int], ...], str, PlyCapVerdict | None]:
    """Play one game from `opening_moves`; return
    `(winner, plies, all_moves, terminal, adjudication)`.

    `winner` is `"candidate"`, `"opponent"`, or `"draw"`; `terminal` is one of
    `mantis.arena.adjudicate.TERMINAL_REASONS`. Both players' `new_game()` fire before the
    opening is replayed; play then alternates argmax move selection.

    A game that reaches `max_plies` without a winner ends a draw when `adjudicator is None`
    — the pre-existing behaviour, and the behaviour every shipped config still gets, because
    the disarmed `eval.ply_cap_adjudication: null` posture never constructs an adjudicator.
    With one armed, that ONE branch consults it instead (see `_DEFAULT_MAX_PLIES` and
    `mantis.arena.adjudicate`). The exhausted-legal-moves exit is deliberately NOT routed
    through adjudication: that is a finished game under the rules, not a budget expiry.
    """
    board = board_factory()
    candidate_player.new_game()
    opponent_bot.new_game()

    moves: list[tuple[int, int]] = []
    for q, r in opening_moves:
        board.apply_move(q, r)
        moves.append((q, r))

    while (
        not board.check_win()
        and board.legal_move_count() > 0
        and len(moves) < max_plies
    ):
        current = board.current_player
        mover = candidate_player if current == candidate_color else opponent_bot
        q, r = mover.select_move(board)
        board.apply_move(q, r)
        moves.append((q, r))

    plies = len(moves)
    adjudication: PlyCapVerdict | None = None
    if board.check_win():
        winning_player = board.winner()
        if winning_player == candidate_color:
            winner = "candidate"
        else:
            winner = "opponent"
        terminal = TERMINAL_WIN
    elif plies >= max_plies:
        terminal = TERMINAL_PLY_CAP
        if adjudicator is None:
            winner = "draw"
        else:
            adjudication = adjudicator.adjudicate(
                board, candidate_color=candidate_color, plies=plies
            )
            winner = adjudication.winner
    else:
        winner = "draw"
        terminal = TERMINAL_EXHAUSTED
    return winner, plies, tuple(moves), terminal, adjudication


def play_paired_match(
    candidate_player: Any,
    opponent_bot: Any,
    openings: Iterable[Any],
    *,
    regime_key: RegimeKey,
    board_factory: Callable[[], Any],
    record_sink: Any = None,
    max_plies: int = _DEFAULT_MAX_PLIES,
    adjudicator: PlyCapAdjudicator | None = None,
) -> list[GameRecord]:
    """Play every opening TWICE (colors swapped); return one `GameRecord` per game.

    `openings` items need only `.opening_id` and `.moves` (duck-typed — this module does
    not import `mantis.arena.books`, so it stays decoupled from the book package).

    `adjudicator` is the resolved `eval.ply_cap_adjudication` posture, forwarded unchanged;
    `None` (every shipped config) keeps the legacy capped-game draw.
    """
    records: list[GameRecord] = []
    for opening in openings:
        for candidate_color in (1, -1):
            opponent_color = -candidate_color
            winner, plies, moves, terminal, adjudication = _play_one_game(
                candidate_player, opponent_bot, list(opening.moves),
                candidate_color=candidate_color, board_factory=board_factory,
                max_plies=max_plies, adjudicator=adjudicator,
            )
            record = GameRecord(
                regime_key=regime_key,
                opening_id=opening.opening_id,
                colors={"candidate": candidate_color, "opponent": opponent_color},
                trajectory_hash=_trajectory_hash(moves),
                winner=winner,
                plies=plies,
                moves=moves,
                terminal=terminal,
                adjudication=adjudication,
            )
            records.append(record)
            if record_sink is not None:
                record_sink(record)
    return records


__all__ = ["Colors", "GameRecord", "play_paired_match"]
