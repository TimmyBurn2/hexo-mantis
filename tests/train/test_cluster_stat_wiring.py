# The R8 justification this file carried is RETIRED with two of its four holes (R346(f)): it
# argued that four seam pins plus one production `StepCoordinator` were one unit, and the
# file is now under the cap. The coordinator fakes below (pool / trainer / buffer /
# eval-pipeline / sink) still exist to drive the REAL `_emit_iteration_complete`, which is
# the only way to pin the caller half of this seam.
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
  H-2  the PRODUCER — RETIRED with the getters (R346(f)); see the block below.
  H-3  the TYPE AUTHORITY. Both `_engine.pyi` twins are the only thing pyright reads for the
       FFI getters — never the compiled module — so a stub still saying `-> float` lets a
       consumer write `runner.cluster_value_std_mean + 1.0` with gate 14 at ZERO and fail at
       runtime on precisely the arm this card is about.
  H-4  the WHEEL-COMPAT DEFAULT — RETIRED with the fields (R346(f)); see the block below.

The snapshot→getter half of the same crosswiring question is pinned in Rust
(`runner.rs::tests::cluster_means_read_their_own_accumulators`) — it is unreachable from
Python, since nothing outside `mantis-selfplay` can seed the atomics.
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
from mantis.selfplay.pool_hooks import runner_stats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through (R5 bars cross-test imports,
    so each file that needs one builds it)."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



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
        self.search_kind = "puct"        # PUCT — run5's arm, where the zeros were observed
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
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # The graph route's sampler. DELEGATED to a real `HexgBuffer` rather than faked: the
        # dispatcher collates the wire for real before the trainer stub ever sees it, so a
        # hand-built payload would be a second wire format for the collate to disagree with.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


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


# ═══ H-2 and H-4 — RETIRED WITH THE FIELDS THEY GUARDED (R346(f)) ═══
# H-2 threaded two DISTINCT values through the snapshot so a transposition of the two
# `getattr` names could not survive; H-4 pinned that a wheel without the getters reported
# absence rather than a fabricated 0.0. Both measured `RunnerStats.cluster_value_std_mean` /
# `cluster_policy_disagreement_mean` / `cluster_variance_sample_count`, which are DELETED:
# the engine exposes no getter, so the snapshot would have carried the wheel-compat default
# forever and the "absence" H-4 asserted would have been the only reading it could ever
# produce — a pin on a constant. The surviving claim is that the fields do not come back
# without their producers, and it is asserted where the snapshot's field set is asserted
# exactly (`tests/selfplay/test_pool_surface.py`). H-3 below is untouched: it is about the
# TYPE STUBS, and a stub re-declaring a getter the engine lacks is still a live hazard.


# ═══ H-3 — the shipped type stubs must not re-declare the retired cluster getters ═══
def test_neither_engine_stub_declares_a_cluster_mean_getter() -> None:
    """INVERTED by R346(f). The two `_engine.pyi` twins are the ONLY type authority for the FFI
    getters — pyright reads the stub, never the compiled module — and this row used to pin that
    both declared the cluster means `float | None` rather than `float`, because the getter
    returned None at zero cluster-variance samples.

    The cluster-variance accumulators only ever ran on the dense arm's `*k >= 2` branch, so the
    getters went with it and the engine exposes neither. What is pinned now is that a stub does
    not RE-declare one: a typed getter with no compiled counterpart is a phantom the checker
    would bless and every reader would trust.

    FALSIFYING MUTATION: add either getter back to either twin.
    """
    twins = (REPO_ROOT / "src" / "mantis" / "_engine.pyi",
             REPO_ROOT / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi")

    for twin in twins:
        text = twin.read_text(encoding="utf-8")
        for name in CLUSTER_MEANS:
            assert f"def {name}(" not in text, (
                f"{twin.relative_to(REPO_ROOT)} declares {name}, which the compiled engine no "
                f"longer exposes (R346(f)) — a stub with no counterpart type-checks a consumer "
                f"that gets None at runtime."
            )
