"""`iteration_complete.sym_draws` (R358(b), R266): the LAW-18 augmentation-group counter, ring → row → audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mantis import _engine
from mantis.diagnostics import ring_audit as A
from mantis.selfplay.pool_hooks import RunnerStats
from mantis.train.events import emit_iteration_complete_event

_ENCODING = "gnn_axis_r8"
N_SYMS = 12
_BATCH, _BATCHES = 48, 25


def _filled_ring(n_rows: int = 64, *, empty_every: int = 0) -> Any:
    buf = _engine.HexgBuffer(256, _ENCODING, 16)
    for i in range(n_rows):
        n = 6 + i % 11
        if empty_every and i % empty_every == 0:
            stones: list[tuple[int, int, int]] = []
            visits = [(0, 0, 0.6), (1, 0, 0.4)]
        else:
            stones = [(q, (q % 3) - 1, 1 if q % 2 == 0 else -1) for q in range(n)]
            visits = [(-1, 0, 0.6), (n, 0, 0.4)]
        buf.push_graph_position(stones, visits, 1 if i % 2 == 0 else -1, 2, i % 50, True, 1.0, True, 40, i, 0.0)
    buf.seed_sampler(20260918)
    return buf


def _sample(buf: Any, *, augment: bool) -> int:
    for _ in range(_BATCHES):
        buf.sample_graph_batch(_BATCH, augment=augment, recent_frac=0.0, n_threads=1)
    return _BATCH * _BATCHES


def test_a_fresh_ring_reads_zero_draws_and_zero_samples() -> None:
    buf = _filled_ring()
    assert buf.sym_draw_counts() == ([0] * N_SYMS, 0)
    assert buf.samples_consumed_total() == 0


def test_augment_true_populates_all_twelve_bins_and_bin_zero_is_under_twice_the_mean() -> None:
    buf = _filled_ring()
    n = _sample(buf, augment=True)
    bins, skipped = buf.sym_draw_counts()
    assert len(bins) == N_SYMS and all(b > 0 for b in bins), bins
    assert sum(bins) == n and skipped == 0
    assert bins[0] <= 2 * (n / N_SYMS), bins
    assert buf.samples_consumed_total() == n


def test_augment_false_puts_every_draw_in_bin_zero() -> None:
    buf = _filled_ring()
    n = _sample(buf, augment=False)
    assert buf.sym_draw_counts() == ([n] + [0] * (N_SYMS - 1), 0)


def test_an_empty_board_row_is_skipped_and_counted_beside_the_bins() -> None:
    buf = _filled_ring(empty_every=4)
    n = _sample(buf, augment=True)
    bins, skipped = buf.sym_draw_counts()
    assert skipped > 0 and sum(bins) + skipped == n


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))


class _Pool:
    sims_per_sec = None
    avg_game_length = None
    search_kind = "gumbel"
    x_winrate = o_winrate = draw_rate = 0.0
    batch_fill_pct = 0.0
    inference_batch_timing = None


def _rstats(positions: int) -> RunnerStats:
    return RunnerStats(games_completed=3, positions_generated=positions, x_wins=1, o_wins=2, draws=0,
                       model_version=1, mcts_quiescence_fires=0, mcts_mean_depth=2.5,
                       mcts_mean_root_concentration=0.4, pcr_full_moves=0, pcr_quick_moves=0, gumbel_round_leaves=0, gumbel_rounds=0)


def _iteration_complete(buffer: Any, rstats: Any = None) -> dict[str, Any]:
    sink = _Sink()
    emit_iteration_complete_event(
        train_step=0, games_played=0, last_iter_games=0, pool=_Pool(), buffer=buffer,
        config={}, mcts_config={}, capacity=1024, games_per_hour_fn=lambda: None,
        steps_per_hour_fn=None, target_integrity={}, rstats=rstats if rstats is not None else _rstats(90),
        sink=sink, search_levers={},
    )
    assert len(sink.events) == 1
    return sink.events[0]


def test_the_row_carries_the_bins_and_the_skips_read_off_the_real_ring() -> None:
    buf = _filled_ring(empty_every=4)
    n = _sample(buf, augment=True)
    row = _iteration_complete(buf)
    block = row["sym_draws"]
    assert list(block) == ["bins", "empty_skipped"]
    assert len(block["bins"]) == N_SYMS and sum(block["bins"]) + block["empty_skipped"] == n
    assert row["samples_consumed_total"] == n


def test_a_buffer_with_no_producer_reads_absent_not_zero() -> None:
    class _NoCounter:
        size = 0
        capacity = 1024

    row = _iteration_complete(_NoCounter())
    assert row["sym_draws"] is None and row["samples_consumed_total"] is None


def _events(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(ts: float, samples: int, positions: int, bins: list[int], skipped: int = 0) -> dict[str, Any]:
    return {"event": "iteration_complete", "ts": ts, "step": int(samples / 256),
            "samples_consumed_total": samples, "positions_produced_total": positions,
            "sym_draws": {"bins": bins, "empty_skipped": skipped}}


def test_the_audit_reads_the_uniformity_row_off_the_last_row_with_a_block(tmp_path: Path) -> None:
    uniform = [100] * N_SYMS
    events = _events(tmp_path / "events.jsonl", [_row(0.0, 0, 0, [0] * N_SYMS),
                                                 _row(3600.0, 1200, 400, uniform, 7)])
    rows = {r.key: r for r in A.event_rows(events, ring_size=1000)}
    assert rows["sym_bin0_over_mean"].value == pytest.approx(1.0)
    assert rows["sym_bin0_over_mean"].n == 1200
    assert "empty_skipped 7" in rows["sym_bin0_over_mean"].note


def test_a_planted_stuck_rng_reds_the_uniformity_band(tmp_path: Path) -> None:
    """LAW-07: a draw stuck on the identity is a row with every draw in bin 0; the band must red."""
    stuck = [1200] + [0] * (N_SYMS - 1)
    events = _events(tmp_path / "events.jsonl", [_row(0.0, 0, 0, [0] * N_SYMS), _row(3600.0, 1200, 400, stuck)])
    rows = A.event_rows(events, ring_size=1000)
    misses, unknown = A.check_bands(rows, {"sym_bin0_over_mean": ("le", 2.0)})
    assert unknown == [] and len(misses) == 1 and misses[0].startswith("sym_bin0_over_mean: 12 not le 2")


def test_without_events_the_two_rows_say_not_measured_and_name_the_flag(tmp_path: Path) -> None:
    rows = {r.key: r for r in A.event_rows(None, ring_size=1000)}
    assert rows["sym_bin0_over_mean"].value is None and "--events" in rows["sym_bin0_over_mean"].note
    assert rows["replay_ratio"].value is None and "--events" in rows["replay_ratio"].note
