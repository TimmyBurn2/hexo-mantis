"""The inference-seam counter reaches the RUN's OWN STREAM.

The Rust legs prove `inference_failures_total` FIRES and a sibling suite proves it reaches
`RunnerStats`; this file owns the last stage, the `iteration_complete.target_integrity` block a
live run emits, because a counter only a test can read is not an in-run instrument.

MUTATIONS THAT RED IT: drop the counter from `_TARGET_INTEGRITY_COUNTERS`; omit zero-valued
counters (the idle-at-0 row is load-bearing for a RUN-FATAL counter, which reads 0 in every run
that survives to emit); or publish the defects counter's value in the seam slot.
"""
from __future__ import annotations

from mantis._engine import HexgBuffer

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.selfplay.pool_hooks import RunnerStats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through; each file that needs one builds
    it, because cross-test imports are barred."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



#: The declaration a `StepCoordinator` reads on the graph route: the identity it dispatches on
#: plus the sections the route's own resolvers read. The caps are non-binding — nothing here
#: exercises a split.
_GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}


_REPO = Path(__file__).resolve().parents[2]
_DEV_CONFIG = load_config(_REPO / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)
_GATE_INTERVAL = _DEV_CONFIG.monitor.gate_interval

#: Transcribed, not derived from the payload under test: an oracle that read its expectation off
#: its own subject would be satisfied by any consistent renaming.
_SEAM = "inference_failures_total"
_DEFECTS = "target_integrity_defects"
_SLOTS = ("total", "delta", "per_position")
_PAYLOAD_KEY = "target_integrity"


def _stats(*, positions: int, seam: int, defects: int) -> RunnerStats:
    """A REAL `RunnerStats` with the three load-bearing numbers supplied EXPLICITLY."""
    return RunnerStats(
        games_completed=0, positions_generated=positions, x_wins=0, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=5.0,
        mcts_mean_root_concentration=0.1,
        export_offwindow_mass_moves=0,
        target_integrity_defects=defects, inference_failures_total=seam,
    )


class _Pool:
    search_kind = "gumbel"
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # the third outcome share.
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    def __init__(self, stats: RunnerStats) -> None:
        self._games = 0
        self.recent_move_histories: list = []
        self.current = stats

    @property
    def games_completed(self) -> int:
        # A step only runs when new games have arrived, so a CONSTANT count would silently
        # collapse a two-emit drive into one.
        self._games += 1
        return self._games

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> RunnerStats:
        return self.current

    def sync_inference_weights(self, state_dict: Any) -> None:
        return None

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        return self.train_step_from_tensors()

    def save_checkpoint(self, loss_info: Any) -> None:
        return None


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, path: Any) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # DELEGATED to a real `HexgBuffer`: the dispatcher collates the wire for real before the
        # trainer stub sees it, so a hand-built payload would be a second wire format.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]


def _drive(*snapshots: RunnerStats) -> list[dict]:
    """Drive a REAL `StepCoordinator` once per snapshot at `log_interval=1` and return the
    `iteration_complete` payloads, in order."""
    assert snapshots, "a drive with no snapshot measures nothing"
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        **{"eval_interval": 10**9, "log_interval": 1, "gate_interval": 1,
           "min_buf_size": 10},
    )
    pool = _Pool(snapshots[0])
    sink = _SpySink()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=config,
        full_config=_GRAPH_FULL_CONFIG,
        train_cfg={}, mixing_cfg={}, sink=sink, monitor_cfg=MonitorConfig(),
    )
    for snapshot in snapshots:
        pool.current = snapshot
        coord.step()
    payloads = sink.named("iteration_complete")
    assert len(payloads) == len(snapshots), (
        f"premise: one `iteration_complete` per driven step at log_interval=1; drove "
        f"{len(snapshots)} and saw {len(payloads)}"
    )
    return payloads


def _integrity(payload: dict) -> dict:
    return payload[_PAYLOAD_KEY]


def test_iteration_complete_carries_the_inference_seam_counter() -> None:
    """The emission leg: a run that dies at the seam must say so IN ITS OWN STREAM. Before it, a
    failed inference produced no event at all — only a target-integrity refusal a hundred plies
    downstream that named neither the failure nor the leaf."""
    block = _integrity(_drive(
        _stats(positions=1200, seam=0, defects=0),
        _stats(positions=2400, seam=1, defects=0),
    )[-1])

    assert _SEAM in block, (
        f"the seam counter never reached the stream — LAW-18 is not satisfied by a counter "
        f"only `runner_stats(pool)` can read (R164). Keys: {sorted(block)}"
    )
    absent = [slot for slot in _SLOTS if slot not in block[_SEAM]]
    assert absent == [], (
        f"{_SEAM} is missing {absent} — a cumulative `total` alone cannot be attributed to "
        f"an interval, which is what LAW-18 asks for. Got {block[_SEAM]}"
    )
    assert block[_SEAM]["total"] == 1, f"cumulative total must ride: {block[_SEAM]}"
    assert block[_SEAM]["delta"] == 1, f"interval delta must ride: {block[_SEAM]}"


def test_the_idle_seam_counter_is_visible_at_zero() -> None:
    """The idle-at-0 posture, which for THIS counter is the normal case: the seam latch is
    run-fatal, so the counter reads 0 in every run that survives to emit. That permanent zero is
    what distinguishes "no inference has failed" from "nobody is counting"."""
    block = _integrity(_drive(
        _stats(positions=500, seam=0, defects=0),
        _stats(positions=1000, seam=0, defects=0),
    )[-1])

    assert _SEAM in block, "a zero-valued counter was omitted — absence now reads as 'no producer'"
    assert block[_SEAM]["total"] == 0
    assert block[_SEAM]["delta"] == 0
    assert block[_SEAM]["per_position"] == 0.0, (
        "a real zero rate over a non-zero denominator is a MEASUREMENT and must be published "
        f"as 0.0, not None: {block[_SEAM]}"
    )


def test_the_two_conjunct_counters_are_distinct_in_the_stream() -> None:
    """The reason the seam conjunct got its OWN counter: seam advanced with defects at 0 says the
    run died BEFORE any target was built, and the reverse says the seam held and the exporter
    caught something else. One shared counter makes those two readings identical."""
    block = _integrity(_drive(
        _stats(positions=100, seam=0, defects=0),
        _stats(positions=200, seam=7, defects=0),
    )[-1])

    assert block[_SEAM]["total"] == 7, f"the seam value landed elsewhere: {block[_SEAM]}"
    assert block[_DEFECTS]["total"] == 0, (
        "the target-integrity counter advanced on a SEAM fire — the two conjuncts share a "
        f"counter and the diagnosis they exist to separate is gone: {block[_DEFECTS]}"
    )
