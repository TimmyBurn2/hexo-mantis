"""`GameRecorder` — the self-play channel's GAME-RECORD-1 producer (R344(b)).

It fills `mantis.selfplay.pool_hooks.RecorderLike`, a seam that has been injected into the
worker pool and defaulted to `NullRecorder` since WP13-A, described in its own docstring as
*"a display surface that does not exist in this tree"*. R344(b) is that surface's producer
half, so this is a seam being filled rather than a seam being cut.

IT LIVES UNDER `monitor/` AND IMPORTS NO `selfplay`. The Protocol is structural, so
satisfying it needs no import, and the DAG edge stays absent in both directions: the pool is
handed an instance by the composition root and never reaches for this module.

`latest_replay_path()` returns `None` and that is not an oversight — this recorder writes a
sharded record STORE, not one replay file per game, so there is no "latest replay path" for
it to name. Returning the newest shard would answer a different question than the one the
method asks.
"""
from __future__ import annotations

from pathlib import Path

from mantis.monitor.game_record import GameRecordWriter, selfplay_record

#: Rust `winner_code` -> the record's seat vocabulary. `0` is a DRAW, `1` is the first mover
#: and `2` the second (`pool_drain._WINNER_NAMES = ("draw", "x", "o")`). Anything else is
#: `"unknown"` and NEVER a draw: AUDIT-1 F-28/C04 is the finding that an undecodable outcome
#: reported as a measured draw is worse than an outcome reported as unread.
_SEAT_BY_WINNER_CODE = {0: "draw", 1: "p1", 2: "p2"}


class GameRecorder:
    """Write one game record per completed self-play game.

    Raises:
        RunIdError: `run_id` cannot safely become part of a filename.
        GameRecordError: no shard could be claimed at construction.
    """

    def __init__(self, *, record_dir: Path | str, run_id: str, seed: int) -> None:
        self._writer = GameRecordWriter(record_dir=record_dir, run_id=run_id)
        self._run_id = str(run_id)
        self._seed = int(seed)
        #: The ACTOR step — the training step whose weights the pool's net is running.
        #: `-1` until `ActorSync` forwards the first one, which is honest: a game drained
        #: before the first sync was played by weights no step has been attributed to yet,
        #: and `0` would name a step that did play games.
        self._step = -1

    def set_step(self, step: int) -> None:
        """Forwarded by `ActorSync` on every sync (`WorkerPool.update_checkpoint_step`)."""
        self._step = int(step)

    def maybe_record(
        self,
        *,
        game_id: str,
        moves: list[tuple[int, int]],
        winner_code: int,
        plies: int,
        worker_id: int,
        terminal_reason: str,
        game_id_byte_hash: str,
        served_sims: int,
    ) -> None:
        """Write this game. Never raises — the writer owns the failure posture."""
        self._writer.write(selfplay_record(
            game_id=game_id,
            run_id=self._run_id,
            step=self._step,
            moves=moves,
            result=_SEAT_BY_WINNER_CODE.get(int(winner_code), "unknown"),
            plies=plies,
            termination=terminal_reason,
            worker_id=worker_id,
            seed=self._seed,
            served_sims=served_sims,
            game_id_byte_hash=game_id_byte_hash,
        ))

    def latest_replay_path(self) -> Path | None:
        """`None`: this recorder writes a record store, not per-game replay files."""
        return None

    def stop(self) -> None:
        """Close the open shard and index it (`WorkerPool.stop` calls this)."""
        self._writer.close()

    @property
    def games_written(self) -> int:
        return self._writer.games_written

    @property
    def persist_errors_total(self) -> int:
        return self._writer.persist_errors_total
