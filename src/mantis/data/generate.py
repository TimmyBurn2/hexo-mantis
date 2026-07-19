"""Generate bot self-play corpus for bootstrap pretraining.

Library surface (DAG-clean — takes a bot as a parameter, no ``mantis.bots``
import):
  - :func:`generate_bot_games` — play + persist unique self-play games.
  - :func:`load_cached_bot_games` — load cached bot games as move sequences.

The bot is duck-typed via the LOCAL structural :class:`BotLike` protocol below —
only the three methods the generator calls (``reset`` / ``get_move`` / ``name``).
A concrete WP12 bot satisfies it structurally with no nominal dependency; the
CLI ``_make_bot``/``main`` that instantiates concrete bots is deferred to WP12
(the only place a concrete bot is constructed). Path defaults resolve in the
caller (no code-side default path; CLAUDE.md R1).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mantis._engine import Board
from mantis.data._log import get_logger
from mantis.env.game_state import GameState

log = get_logger(__name__)

MAX_MOVES_PER_GAME = 500


@runtime_checkable
class BotLike(Protocol):
    """Structural protocol for a self-play bot — the 3 methods the generator calls."""

    def reset(self) -> None: ...

    def get_move(self, state: GameState, board: Board) -> tuple[int, int]: ...

    def name(self) -> str: ...


def _play_one_game(
    bot: BotLike,
    game_idx: int,
    rng_seed: int = 0,
    n_random_opening: int = 1,
    use_human_seeding: bool = False,
    human_corpus_dir: str | None = None,
    human_seeding_min_move: int = 10,
    human_seeding_max_move: int = 25,
) -> dict[str, Any] | None:
    """Play one self-play game using bot for both sides.

    To ensure opening diversity (deterministic bots at fixed depth always play
    the same game otherwise), the first n_random_opening moves are random. The
    bot takes over after that.

    When use_human_seeding=True, the opening is instead drawn from a real human
    game mid-position via sample_human_midgame_position().

    Returns a dict with keys: moves, winner, plies, bot_name.
    Returns None if the game ends without a winner (capped).
    """
    bot.reset()
    board = Board()
    state = GameState.from_board(board)
    moves: list[tuple[int, int]] = []

    rng = random.Random(rng_seed + game_idx)

    if use_human_seeding and human_corpus_dir:
        # Try human-seeded opening; fall back to random on failure
        try:
            from mantis.data.human_seeding import sample_human_midgame_position

            opening_moves = sample_human_midgame_position(
                corpus_dir=human_corpus_dir,
                min_move=human_seeding_min_move,
                max_move=human_seeding_max_move,
                rng=rng,
            )
            for q, r in opening_moves:
                if board.check_win() or board.legal_move_count() == 0:
                    break
                state = state.apply_move(board, q, r)
                moves.append((q, r))
        except Exception as exc:  # noqa: BLE001 — human-seeding is best-effort; fall back to random
            log.warning(
                "human_seeding_fallback",
                game=game_idx,
                error=str(exc),
                fallback="random_opening",
            )
            # Fall through to random opening below
            use_human_seeding = False

    if not use_human_seeding or not moves:
        # Random opening moves for diversity (original behaviour)
        for _ in range(n_random_opening):
            legal = board.legal_moves()
            if not legal or board.check_win():
                break
            q, r = rng.choice(legal)
            state = state.apply_move(board, q, r)
            moves.append((q, r))

    while (not board.check_win() and board.legal_move_count() > 0
           and len(moves) < MAX_MOVES_PER_GAME):
        try:
            q, r = bot.get_move(state, board)
        except Exception as exc:  # noqa: BLE001 — a bot move error ends this game, not the run
            log.warning("bot_move_error", game=game_idx, ply=len(moves), error=str(exc))
            break
        state = state.apply_move(board, q, r)
        moves.append((q, r))

    winner = board.winner()
    if winner is None:
        return None

    return {
        "moves": [{"x": q, "y": r} for q, r in moves],
        "winner": int(winner),
        "plies": len(moves),
        "bot_name": bot.name(),
    }


def _game_hash(moves: list[dict[str, Any]]) -> str:
    """SHA-256 of the move sequence, truncated to 16 hex chars."""
    key = json.dumps(moves, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_bot_games(
    bot: BotLike,
    n_games: int,
    output_dir: Path,
    rng_seed: int = 42,
    n_random_opening: int = 1,
    use_human_seeding: bool = False,
    human_corpus_dir: str | None = None,
    human_seeding_min_move: int = 10,
    human_seeding_max_move: int = 25,
) -> int:
    """Generate n_games unique self-play games and save to output_dir.

    Games are named by a hash of their move sequence, so:
    - Re-running never overwrites existing games with different content
    - Duplicate games (identical move sequences) are detected and skipped

    Returns the number of new games saved (excludes duplicates and pre-existing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = set(p.stem for p in output_dir.glob("*.json"))
    saved = 0
    dupes = 0
    t0 = time.monotonic()

    for i in range(n_games):
        result = _play_one_game(
            bot, i, rng_seed=rng_seed,
            n_random_opening=n_random_opening,
            use_human_seeding=use_human_seeding,
            human_corpus_dir=human_corpus_dir,
            human_seeding_min_move=human_seeding_min_move,
            human_seeding_max_move=human_seeding_max_move,
        )
        if result is None:
            log.info("game_no_winner", game=i, status="skipped")
            continue

        move_payload: list[dict[str, Any]] = result["moves"]
        name = _game_hash(move_payload)
        if name in existing:
            dupes += 1
            continue

        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(result, f)
        existing.add(name)
        saved += 1

        if saved % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = saved / elapsed if elapsed > 0 else 0
            log.info("corpus_progress", saved=saved, total=n_games,
                     dupes=dupes, rate_per_min=f"{rate * 60:.1f}")

    elapsed = time.monotonic() - t0
    log.info("corpus_generation_complete",
             saved=saved, dupes=dupes, attempted=n_games,
             total_on_disk=len(existing), elapsed_min=f"{elapsed / 60:.1f}")
    return saved


def load_cached_bot_games(bot_dir: Path) -> list[list[tuple[int, int]]]:
    """Load all cached bot games from disk as move sequences.

    Args:
        bot_dir: Directory containing game JSON files (searched recursively).

    Returns:
        List of move sequences, each a list of (q, r) tuples.
    """
    if not bot_dir.exists():
        log.info("no_bot_games_dir", path=str(bot_dir))
        return []

    games: list[list[tuple[int, int]]] = []
    json_files = sorted(bot_dir.rglob("*.json"))

    for p in json_files:
        try:
            with open(p) as f:
                data: dict[str, Any] = json.load(f)
            moves = [(m["x"], m["y"]) for m in data["moves"]]
            games.append(moves)
        except Exception:  # noqa: BLE001 — skip unreadable/malformed cached game JSON
            continue

    log.info("loaded_cached_bot_games", count=len(games), dir=str(bot_dir))
    return games
