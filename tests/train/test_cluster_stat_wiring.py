# >300 justify (R8): one seam, four holes, and one production `StepCoordinator` that costs
# most of the file. The coordinator fakes below (pool / trainer / buffer / eval-pipeline /
# sink) exist to drive the REAL `_emit_iteration_complete`, which is the only way to pin the
# caller half of this seam; the getter-crosswiring, wheel-compat-default and stub-type pins
# are the other end of the SAME payload's journey and share the constants and the fixture
# vocabulary. Splitting them would fork the coordinator harness for one test and leave the
# remaining three homeless, and the four holes were found together for the same reason —
# every one of them is a place the cluster block travels where nothing was watching.
"""ADJ-D32 / R249 + R250 — the WIRING the payload pins cannot see.

`tests/train/test_cluster_stat_absence.py` drives the real `emit_iteration_complete_event`
through a spy sink and asserts on the emitted payload, so a mutation in the builder or its
helper reds there. It stops one level short at BOTH ends of the seam the payload travels,
and this file closes those ends:

  H-1  the CALLER. `StepCoordinator._emit_iteration_complete` is what hands the builder the
       config `is_graph_run` reads. Pass `{}` there and R250 (absence) silently degrades to
       R249 (zero-count drop): `cluster_variance_sample_count: 0` ships in every
       `iteration_complete` of a graph run, for an instrument that does not exist on that
       arm. The payload pins cannot see it — they choose the config themselves. Nor does
       `test_full_config_carries_the_real_config_not_an_empty_dict` (O-S1b): it asserts on
       `coordinator.full_config`, the ATTRIBUTE, and says nothing about what is passed on.
  H-2  the PRODUCER. `pool_hooks.runner_stats` reads the two cluster getters by name.
       Swapping them transposes two live telemetry series permanently and invisibly — in
       aggregate they simply trade places for the whole run, and nothing recovers them
       post-hoc. Every other pin in this card's set drives both means `None`, and a swap of
       `None` for `None` passes them all. DISTINCT values are the only instrument that sees
       it. `test_winrates_are_computed_from_the_right_counters` makes exactly this argument
       for the winrate pair; the cluster pair got the shape change without the instrument.
  H-3  the TYPE AUTHORITY. Both `_engine.pyi` twins are the only thing pyright reads for the
       FFI getters — never the compiled module — so a stub still saying `-> float` lets a
       consumer write `runner.cluster_value_std_mean + 1.0` with gate 14 at ZERO and fail at
       runtime on precisely the arm this card is about.
  H-4  the WHEEL-COMPAT DEFAULT. R249 changed `getattr(r, <mean>, 0.0)` to
       `getattr(..., None)` so an engine build predating the getter reports absence rather
       than the fabricated zero the card removes. Reverting it reds nothing today.

The snapshot→getter half of the same crosswiring question is pinned in Rust
(`runner.rs::tests::cluster_means_read_their_own_accumulators`) — it is unreachable from
Python, since nothing outside `mantis-selfplay` can seed the atomics.
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
from mantis.selfplay.pool_hooks import runner_stats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

REPO_ROOT = Path(__file__).resolve().parents[2]

CLUSTER_KEYS = ("cluster_value_std_mean", "cluster_policy_disagreement_mean",
                "cluster_variance_sample_count")
CLUSTER_MEANS = CLUSTER_KEYS[:2]

# Minted-config-derived knobs (the WPMINT Phase K-A/K-B precedent every coordinator test in
# this directory follows — no hand-restated numbers).
_CONFIG = load_config(REPO_ROOT / "configs" / "dev_example.yaml")
GRAPH_CONFIG: dict[str, Any] = {"identity": {"encoding": "gnn_axis_v1",
                                             "representation": "graph"}}
GRID_CONFIG: dict[str, Any] = {"identity": {"encoding": "v6_live2_ls",
                                            "representation": "grid"}}


# ── coordinator collaborators (the same minimal surface as
#    tests/train/test_iteration_complete_decoupling.py's fakes) ──────────────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = None
    cluster_policy_disagreement_mean = None
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self.games_completed = 5
        self.gumbel_mcts = False        # PUCT — run5's arm, where the zeros were observed
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def save_checkpoint(self, loss_info) -> None:
        return None


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


class _EvalPipeline:
    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        return {"status": "skipped"}

    def drain_pending(self):
        return None

    def poll_completed(self):
        return None


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _coordinator(full_config: dict[str, Any]):
    cfg = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=resolve_drain_caps(_CONFIG.monitor),
                                 gate_interval=_CONFIG.monitor.gate_interval,
                                 knobs=resolve_coordinator_knobs(_CONFIG.train)),
        eval_interval=1, log_interval=1000, gate_interval=1000, min_buf_size=10,
    )
    sink = _SpySink()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=_Pool(), eval_pipeline=_EvalPipeline(),
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=cfg,
        full_config=full_config, train_cfg={}, mixing_cfg={}, sink=sink,
        monitor_cfg=MonitorConfig(),
    )
    return coord, cfg, sink


def _one_iteration_complete(sink: _SpySink) -> dict[str, Any]:
    events = sink.named("iteration_complete")
    assert len(events) == 1, f"expected exactly one iteration_complete, got {len(events)}"
    return events[0]


# ═══ H-1 — the coordinator's own emit seam, on a GRAPH run ═══
def test_a_real_coordinator_emits_no_cluster_key_on_a_graph_run() -> None:
    """R250 asserted on the event stream a PRODUCTION `StepCoordinator` produces.

    Driven through `_emit_iteration_complete` — the one site that hands the builder its
    `config` argument — rather than a full `step()`, because a graph declaration additionally
    routes the training step through `dispatch` and demands a graph-capable buffer
    (`RepresentationRouteError`). That is a different seam; the grid test below carries the
    full-`step()` evidence that the production loop really reaches this method.

    FALSIFYING MUTATION: pass `{}` (or any config without `identity`) as the builder's
    `config` argument in `coordinator/step.py::_emit_iteration_complete`. `is_graph_run` then
    reads non-graph, the graph arm is never taken, and `cluster_variance_sample_count: 0`
    reaches the ONE channel on a run whose producer does not exist.
    """
    coord, cfg, sink = _coordinator(GRAPH_CONFIG)
    coord._emit_iteration_complete(cfg)
    payload = _one_iteration_complete(sink)

    for key in CLUSTER_KEYS:
        assert key not in payload, (
            f"R250: {key} reached the sink as {payload.get(key)!r} from a real coordinator "
            f"on a graph run — the coordinator must hand the builder the run's OWN config, "
            f"which is the declaration is_graph_run reads."
        )
    assert "mcts_root_concentration" in payload, (
        "mcts_root_concentration is live on the graph path and must survive the drop"
    )


def test_a_real_coordinator_step_on_a_grid_run_still_reports_the_count() -> None:
    """The same wiring on a GRID run, through a full `StepCoordinator.step()`: R249's
    zero-count rules apply, so the truthful count is published and the two means are dropped.

    Asserted end-to-end so the graph pin above cannot be satisfied by an emitter that has
    simply stopped publishing the block, and so the production loop's route to
    `_emit_iteration_complete` is itself exercised.
    """
    coord, _cfg, sink = _coordinator(GRID_CONFIG)
    coord.step()
    payload = _one_iteration_complete(sink)

    assert payload["cluster_variance_sample_count"] == 0
    for key in CLUSTER_MEANS:
        assert key not in payload, f"R249: {key} published as {payload.get(key)!r}"


# ═══ H-2 — the two cluster means are not crosswired at the snapshot layer ═══
def test_runner_stats_reads_each_cluster_mean_from_its_own_getter() -> None:
    """DISTINCT values threaded getter → snapshot field, because that is the only shape of
    test a transposition cannot survive.

    FALSIFYING MUTATION: swap the two `getattr(r, ...)` names in `pool_hooks.runner_stats`.
    """
    runner = SimpleNamespace(
        cluster_value_std_mean=0.125,
        cluster_policy_disagreement_mean=0.875,
        cluster_variance_sample_count=4,
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
    )

    rstats = runner_stats(SimpleNamespace(_runner=runner))

    assert rstats.cluster_value_std_mean == 0.125, (
        "cluster_value_std_mean must read the value-spread getter, not the disagreement one "
        "— a swap is permanent, silent, and unrecoverable after the run"
    )
    assert rstats.cluster_policy_disagreement_mean == 0.875, (
        "cluster_policy_disagreement_mean must read the disagreement getter"
    )
    assert rstats.cluster_variance_sample_count == 4


# ═══ H-4 — the wheel-compat default is ABSENCE, not a fabricated zero ═══
def test_the_wheel_compat_default_for_a_missing_cluster_getter_is_absence() -> None:
    """`runner_stats`'s `getattr` defaults exist for an engine build that predates a counter.
    R249 changed the two cluster means' default from `0.0` to `None` so such a build reports
    "no reading" instead of the very fabrication the card removes, one layer up.

    FALSIFYING MUTATION: restore either default to `0.0`.
    """
    old_wheel = SimpleNamespace(
        cluster_variance_sample_count=0,
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
    )  # NO cluster-mean getters at all

    rstats = runner_stats(SimpleNamespace(_runner=old_wheel))

    assert rstats.cluster_value_std_mean is None, (
        "an engine build without the getter must report absence, not a fabricated 0.0"
    )
    assert rstats.cluster_policy_disagreement_mean is None


# ═══ H-3 — the shipped type stubs must not claim the getters are non-optional ═══
def test_both_engine_stubs_declare_the_cluster_means_optional() -> None:
    """The two `_engine.pyi` twins are the ONLY type authority for the FFI getters — pyright
    reads the stub, never the compiled module — and nothing else in the repo compares them or
    checks either against the getter's real nullability.

    FALSIFYING MUTATION: revert either twin's two cluster getters to `-> float`.
    """
    twins = (REPO_ROOT / "src" / "mantis" / "_engine.pyi",
             REPO_ROOT / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi")

    for twin in twins:
        text = twin.read_text(encoding="utf-8")
        for name in CLUSTER_MEANS:
            assert f"def {name}(self) -> float | None: ..." in text, (
                f"{twin.relative_to(REPO_ROOT)} must declare {name} as `float | None` "
                f"(R249): the getter returns None at zero cluster-variance samples, and a "
                f"stub saying `float` type-checks a consumer that crashes on the graph arm."
            )
