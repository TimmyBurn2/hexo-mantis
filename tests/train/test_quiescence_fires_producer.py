"""`iteration_complete.mcts_quiescence_fires` — A-2's LAW-18 counter reaches the one channel (R355(a))."""
from __future__ import annotations

import dataclasses
from typing import Any

from mantis.selfplay.pool_hooks import RunnerStats


def _iteration_complete(rstats: Any) -> dict[str, Any]:
    from mantis.train.events import emit_iteration_complete_event

    events: list[dict[str, Any]] = []

    class _Sink:
        def emit(self, event: Any) -> None:
            events.append(dict(event))

    class _Buffer:
        size = 0
        capacity = 1024

    class _Pool:
        sims_per_sec = None
        avg_game_length = None
        search_kind = "gumbel"
        x_winrate = 0.0
        o_winrate = 0.0
        draw_rate = 0.0
        batch_fill_pct = 0.0
        inference_batch_timing = None

    emit_iteration_complete_event(
        train_step=0, w_pre=0.0, games_played=0, last_iter_games=0, pool=_Pool(),
        buffer=_Buffer(), config={}, mcts_config={}, capacity=1024,
        games_per_hour_fn=lambda: None, steps_per_hour_fn=None,
        target_integrity={}, rstats=rstats, sink=_Sink(),
    )
    assert len(events) == 1, events
    return events[0]


def _snapshot(fires: int) -> RunnerStats:
    return RunnerStats(games_completed=3, positions_generated=90, x_wins=1, o_wins=2, draws=0,
                       model_version=1, mcts_quiescence_fires=fires, mcts_mean_depth=2.5,
                       mcts_mean_root_concentration=0.4)


def test_the_runner_snapshot_carries_the_counter_by_name() -> None:
    assert "mcts_quiescence_fires" in {f.name for f in dataclasses.fields(RunnerStats)}


def test_the_payload_carries_the_cumulative_fire_count() -> None:
    assert _iteration_complete(_snapshot(10))["mcts_quiescence_fires"] == 10


def test_a_measured_zero_survives_as_zero() -> None:
    payload = _iteration_complete(_snapshot(0))
    assert payload["mcts_quiescence_fires"] == 0 and payload["mcts_quiescence_fires"] is not None


def test_a_snapshot_without_the_counter_reads_absent_not_zero() -> None:
    class _NoCounter:
        mcts_mean_depth = None
        mcts_mean_root_concentration = None

    assert _iteration_complete(_NoCounter())["mcts_quiescence_fires"] is None
