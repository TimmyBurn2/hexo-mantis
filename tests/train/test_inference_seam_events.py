"""F-816-9 Phase C — the SEAM counter reaches the RUN's OWN STREAM (R275(b), LAW-18/R164).

`inference_failures_total` is the SEAM conjunct's in-run instrument. The Rust legs
(`crates/mantis-selfplay/tests/search_seam_fatal.rs`) prove the counter FIRES and that a
drain shutdown does not fire it; `tests/selfplay/test_inference_seam_counter.py` proves it
reaches the Python `RunnerStats` surface. This file owns the last stage: surface → the
`iteration_complete.target_integrity` block a live run emits. LAW-18's own text is that a
post-hoc offline probe cannot distinguish "starved" from "ineffective", and R164 is the
ruling that a counter readable only by a test calling `runner_stats(pool)` is not an in-run
instrument at all.

WHY THIS FILE CARRIES ITS OWN HARNESS rather than adding a row to
`tests/train/test_target_counter_events.py`, which already drives a `StepCoordinator`: that
file is a ⊕ frozen Phase-O oracle bank and editing it is an R43 queue event. The duplication
is the ~60 lines of injected doubles, and it is disclosed here rather than hidden. The claim
is also genuinely a different one: the frozen bank's `_COUNTERS` is transcribed to pin THE
THREE Phase-T counters, and a fourth counter appended to that tuple would have widened an
oracle whose stated subject is those three.

MUTATIONS THAT RED THIS FILE:
  * M-SEAMEV-1 — drop `inference_failures_total` from
    `mantis.train.coordinator.step._TARGET_INTEGRITY_COUNTERS`; the key vanishes from the
    block and the counter is back to being readable only by a test.
  * M-SEAMEV-2 — omit zero-valued counters from the report; the idle-at-0 row goes RED, and
    that row is the load-bearing one for a RUN-FATAL counter, which reads 0 in every run
    that survives to emit.
  * M-SEAMEV-3 — publish `target_integrity_defects`' value in the seam counter's slot; the
    distinctness row goes RED and nothing else here notices.

Real: the shipped `StepCoordinator`, its real `_run_log_interval` boundary, the real event
payloads, and the real `RunnerStats` dataclass — so a field rename in `pool_hooks` reds this
file. Fake: the pool/trainer/buffer seam, and the counter VALUES (a real advance needs a
live Rust runner, which the Rust legs own).
"""
from __future__ import annotations

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

_REPO = Path(__file__).resolve().parents[2]
_DEV_CONFIG = load_config(_REPO / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)
_GATE_INTERVAL = _DEV_CONFIG.monitor.gate_interval

#: Transcribed, not derived from the payload under test: an oracle that read its expectation
#: off its own subject would be satisfied by any consistent renaming (R81).
_SEAM = "inference_failures_total"
_DEFECTS = "target_integrity_defects"
_SLOTS = ("total", "delta", "per_position")
_PAYLOAD_KEY = "target_integrity"


def _stats(*, positions: int, seam: int, defects: int) -> RunnerStats:
    """A REAL `RunnerStats` with the three load-bearing numbers supplied EXPLICITLY."""
    return RunnerStats(
        games_completed=0, positions_generated=positions, x_wins=0, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=5.0,
        mcts_mean_root_concentration=0.1, cluster_value_std_mean=0.0,
        cluster_policy_disagreement_mean=0.0, cluster_variance_sample_count=0,
        export_offwindow_mass_moves=0, gridls_zero_policy_rows=0,
        target_integrity_defects=defects, inference_failures_total=seam,
    )


class _Pool:
    gumbel_mcts = True
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # F-816-2: the third outcome share.
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
        # collapse a two-emit drive into one (the house rig).
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

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, path: Any) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


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
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}},
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
    """The emission leg (M-SEAMEV-1).

    A run that dies at the seam must be able to say so IN ITS OWN STREAM. Pre-R275(b) a
    failed inference produced no event at all — it produced a silently degraded search, and
    the only trace was a target-integrity refusal a hundred plies downstream that named
    neither the failure nor the leaf."""
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
    """M-SEAMEV-2 — the idle-at-0 posture, which for THIS counter is the normal case.

    The seam latch is run-fatal, so `inference_failures_total` reads 0 in every run that
    survives to emit. That permanent zero is the posture, not an unproduced field, and it is
    the only thing that distinguishes "no inference has failed" from "nobody is counting"."""
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
    """M-SEAMEV-3 — the whole reason the SEAM conjunct got its own counter.

    R275(b) splits the F-816-9 class into two conjuncts, and the split only buys anything if
    a reader can tell which one fired. Seam advanced with defects at 0 says the run died
    BEFORE any target was built; the reverse says the seam held and the exporter caught
    something else. One shared counter would make those two readings identical."""
    block = _integrity(_drive(
        _stats(positions=100, seam=0, defects=0),
        _stats(positions=200, seam=7, defects=0),
    )[-1])

    assert block[_SEAM]["total"] == 7, f"the seam value landed elsewhere: {block[_SEAM]}"
    assert block[_DEFECTS]["total"] == 0, (
        "the target-integrity counter advanced on a SEAM fire — the two conjuncts share a "
        f"counter and the diagnosis they exist to separate is gone: {block[_DEFECTS]}"
    )
