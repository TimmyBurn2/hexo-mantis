# >300 justify (R8): one game record and the loop that produces it. `_play_one_game`,
# `_record_one` and the two concurrency arms of `play_paired_match` share ONE record
# construction on purpose — two copies would be two authorities over `trajectory_hash`, the
# strength dedupe input — so they stay in one file.
"""play_paired_match — drives compound-turn paired games on the engine Board.

ARGMAX ONLY: there is no softmax-knob parameter anywhere in this signature, so it is
structurally unrepresentable. Every opening is played exactly TWICE with colours swapped, and
every `GameRecord` is stamped with `regime_key`, `opening_id`, `colors`, `trajectory_hash` (a
sha256 over the move list, the dedupe input), `winner` and `plies`.
"""
from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from mantis.arena.adjudicate import (
    TERMINAL_EXHAUSTED,
    TERMINAL_FORFEIT,
    TERMINAL_PLY_CAP,
    TERMINAL_WIN,
    PlyCapAdjudicator,
    PlyCapVerdict,
)
from mantis.arena.regime import RegimeKey


class IllegalOpeningError(ValueError):
    """A book opening did not replay: one of its moves is outside the board's legal set. FATAL,
    and deliberately not the forfeit a player's illegal move gets — the book is shared by every
    game in the round, and the colour-swap law would make a forfeit score BOTH sides in turn,
    producing a perfectly balanced 50% win rate out of a broken instrument."""


#: `colors` shape: {"candidate": <player int>, "opponent": <player int>}, engine convention.
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
    #: WHICH of the loop's three exits ended this game. RECORDED rather than re-derived because
    #: `(winner, plies)` cannot recover it: a genuine win found on the cap ply and a capped
    #: non-result both report `plies == max_plies`, so a consumer deriving decisiveness from
    #: those two fields misreads the first as the second.
    terminal: str
    #: The adjudication that produced `winner`, or `None` when no adjudicator was armed —
    #: i.e. `None` on every game every shipped config plays. It is `None` on non-capped games
    #: even under an armed adjudicator, because those were decided by the rules.
    adjudication: PlyCapVerdict | None
    #: Per-position search stats for every ply whose MOVER exposed a search root, each entry
    #: naming the side that produced it. `None` when no player exposed a root at all, and `None`
    #: and `()` are different facts that both occur: `None` is "nobody could produce these", `()`
    #: is "nobody who could, moved".
    search_stats: tuple[dict[str, Any], ...] | None


def _trajectory_hash(moves: Iterable[tuple[int, int]]) -> str:
    h = hashlib.sha256()
    for q, r in moves:
        h.update(f"{q},{r};".encode())
    return h.hexdigest()


#: Hard ply cap — the board is UNBOUNDED, so a game between two degenerate bots is not
#: guaranteed ever to complete a 6-in-a-row and the loop below would not terminate. NO LONGER A
#: DEFAULT: it used to mirror a copy of a minted key while the eval worker passed none of them,
#: so a re-mint would have left eval capping at 128 with no config diff. It is the TEST
#: fixture's cap now, and production threads the run's own `max_plies`.
DEFAULT_MAX_PLIES = 128


def _play_one_game(
    candidate_player: Any,
    opponent_bot: Any,
    opening_moves: list[tuple[int, int]],
    *,
    candidate_color: int,
    board_factory: Callable[[], Any],
    max_plies: int,
    opening_id: str,
    adjudicator: PlyCapAdjudicator | None = None,
) -> tuple[str, int, tuple[tuple[int, int], ...], str, PlyCapVerdict | None, tuple[dict[str, Any], ...] | None]:
    """Play one game from `opening_moves`; return
    `(winner, plies, all_moves, terminal, adjudication, search_stats)`.

    Both players' `new_game()` fire before the opening is replayed; play then alternates argmax
    move selection. A game reaching `max_plies` without a winner ends a draw when `adjudicator is
    None`, which is what every shipped config gets. The exhausted-legal-moves exit is deliberately
    NOT adjudicated — that is a finished game under the rules, not a budget expiry. Every move is
    checked against the board's legal set before it is applied: a player's illegal move forfeits,
    an opening's raises.

    Raises:
        IllegalOpeningError: a move in `opening_moves` is not in the board's legal set.
    """
    board = board_factory()
    candidate_player.new_game()
    opponent_bot.new_game()

    moves: list[tuple[int, int]] = []
    for q, r in opening_moves:
        if not board.is_legal(q, r):
            raise IllegalOpeningError(
                f"opening {opening_id!r} does not replay: move ({q}, {r}) at ply "
                f"{len(moves)} is not in the board's legal set under encoding geometry "
                f"(occupied, or outside every stone's legal_move_radius ball). The board "
                f"would have accepted an off-radius cell — `apply_move` refuses only an "
                f"occupied one — so this is checked here or nowhere."
            )
        board.apply_move(q, r)
        moves.append((q, r))

    #: Collected iff a mover EXPOSES a root — structural, not a flag: a knob here would be a
    #: way to silently turn the record's stats half off. A bot that cannot produce them yields
    #: `None`, which says so.
    stats: list[dict[str, Any]] = []
    saw_a_root = False
    while (
        not board.check_win()
        and board.legal_move_count() > 0
        and len(moves) < max_plies
    ):
        current = board.current_player
        mover = candidate_player if current == candidate_color else opponent_bot
        q, r = mover.select_move(board)
        # THE LEGALITY BOUNDARY, checked BEFORE `apply_move`, because `apply_move` refuses an
        # occupied cell and nothing else: an off-radius coordinate was accepted, played on and
        # scored. Cheap here and only here — the `while` condition has just rebuilt the engine's
        # legal-set cache, so this is an O(1) lookup on a clean one.
        if not board.is_legal(q, r):
            forfeiting = "candidate" if mover is candidate_player else "opponent"
            winner = "opponent" if forfeiting == "candidate" else "candidate"
            return (winner, len(moves), tuple(moves), TERMINAL_FORFEIT, None,
                    tuple(stats) if saw_a_root else None)
        root = getattr(mover, "last_root", None)
        if root is not None:
            saw_a_root = True
            root_value, children = root
            # ONLY the visited children: a zero-visit child carries none of the distribution's
            # information while being most of the bytes at radius 8, so the record stores the
            # support and a reader takes absence as zero.
            stats.append({
                "ply": len(moves),
                # WHICH SIDE searched. On the promotion channel both players are deploy heads,
                # so a stats list covers EVERY ply from BOTH sides; elsewhere only the
                # candidate's plies appear, and without this field the two are indistinguishable.
                "by": "candidate" if mover is candidate_player else "opponent",
                "root_value": root_value,
                "visits": [[int(c[0][0]), int(c[0][1]), int(c[3])]
                           for c in children if c[3] > 0],
            })
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
    return (winner, plies, tuple(moves), terminal, adjudication,
            tuple(stats) if saw_a_root else None)


def _record_one(
    candidate_player: Any,
    opponent_bot: Any,
    opening: Any,
    candidate_color: int,
    *,
    regime_key: RegimeKey,
    board_factory: Callable[[], Any],
    max_plies: int,
    adjudicator: PlyCapAdjudicator | None,
) -> GameRecord:
    """One slot of the paired match: play it and stamp its `GameRecord`. Extracted so the serial
    loop and the in-flight arm share ONE record construction.

    Raises:
        Exception: whatever the players' `select_move` or the board raises; nothing is caught
            here, so a defect in one game is not converted into a silently missing record.
    """
    winner, plies, moves, terminal, adjudication, stats = _play_one_game(
        candidate_player, opponent_bot, list(opening.moves),
        candidate_color=candidate_color, board_factory=board_factory,
        max_plies=max_plies, opening_id=str(opening.opening_id), adjudicator=adjudicator,
    )
    return GameRecord(
        regime_key=regime_key,
        opening_id=opening.opening_id,
        colors={"candidate": candidate_color, "opponent": -candidate_color},
        trajectory_hash=_trajectory_hash(moves),
        winner=winner,
        plies=plies,
        moves=moves,
        terminal=terminal,
        adjudication=adjudication,
        search_stats=stats,
    )


def play_paired_match(
    candidate_player: Any,
    opponent_bot: Any,
    openings: Iterable[Any],
    *,
    regime_key: RegimeKey,
    board_factory: Callable[[], Any],
    record_sink: Any = None,
    max_plies: int,
    adjudicator: PlyCapAdjudicator | None = None,
    player_factory: Callable[[], tuple[Any, Any]] | None = None,
    concurrency: int = 1,
) -> list[GameRecord]:
    """Play every opening TWICE (colours swapped); return one `GameRecord` per game.

    `openings` items need only `.opening_id` and `.moves`, so this stays decoupled from the book
    package; `adjudicator` is forwarded unchanged, and `None` keeps the capped-game draw.

    `concurrency` > 1 runs that many games IN FLIGHT, one thread each, sharing whatever inference
    server the players hold, and requires `player_factory` because the players carry per-game
    state that cannot be shared. **THE DEFAULT IS UNCHANGED AND UNARMED**: `concurrency = 1` is
    the serial loop on the same objects in the same order, and no shipped config sets it.
    Records return and `record_sink` fires in LOOP ORDER under every concurrency.

    Raises:
        ValueError: `concurrency < 1`, or `concurrency > 1` without a `player_factory`.
    """
    if concurrency < 1:
        raise ValueError(
            f"play_paired_match: concurrency={concurrency} must be >= 1; 1 is the serial "
            "arm and is what every shipped config runs."
        )
    if concurrency == 1:
        records: list[GameRecord] = []
        for opening in openings:
            for candidate_color in (1, -1):
                record = _record_one(
                    candidate_player, opponent_bot, opening, candidate_color,
                    regime_key=regime_key, board_factory=board_factory,
                    max_plies=max_plies, adjudicator=adjudicator,
                )
                records.append(record)
                if record_sink is not None:
                    record_sink(record)
        return records

    if player_factory is None:
        raise ValueError(
            f"play_paired_match: concurrency={concurrency} needs a `player_factory`. The two "
            "players carry per-game state (search tree, `new_game()`), so G games in flight "
            "need G pairs; sharing one pair across threads would interleave two searches on "
            "one tree."
        )

    slots = [(o, c) for o in openings for c in (1, -1)]
    local = threading.local()

    def _run(slot: tuple[Any, int]) -> GameRecord:
        pair = getattr(local, "pair", None)
        if pair is None:
            pair = local.pair = player_factory()
        opening, candidate_color = slot
        return _record_one(
            pair[0], pair[1], opening, candidate_color,
            regime_key=regime_key, board_factory=board_factory,
            max_plies=max_plies, adjudicator=adjudicator,
        )

    # `Executor.map` yields IN SUBMISSION ORDER regardless of completion order, which keeps the
    # game index stable, and re-raises a worker's exception at the yield, so a failed game halts
    # the match rather than shortening it.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(_run, slots))
    if record_sink is not None:
        for record in records:
            record_sink(record)
    return records


__all__ = ["DEFAULT_MAX_PLIES", "Colors", "GameRecord", "IllegalOpeningError",
           "play_paired_match"]
