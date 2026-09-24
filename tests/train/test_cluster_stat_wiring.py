"""The retired cluster-variance fields stay retired: no `cluster_*` key reaches a real
coordinator's `iteration_complete`, and the `_engine.pyi` stub declares no cluster-mean getter.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _drivable import DrivablePoolStub, RunnerStats
from _graph_drive import DEV_DRAIN_CAPS, DEV_GATE_INTERVAL, DEV_KNOBS, GraphSampleBuffer
from _monitor_config import monitor_config
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from _spy import SpyEventSink


REPO_ROOT = Path(__file__).resolve().parents[2]

CLUSTER_KEYS = ("cluster_value_std_mean", "cluster_policy_disagreement_mean",
                "cluster_variance_sample_count")
CLUSTER_MEANS = CLUSTER_KEYS[:2]

GRAPH_CONFIG: dict[str, Any] = {"identity": {"encoding": "gnn_axis_v1",
                                             "representation": "graph"}}


class _ClusterCarryingStats(RunnerStats):
    """A snapshot still carrying the retired cluster fields, so absence is the builder's doing."""

    cluster_value_std_mean = None
    cluster_policy_disagreement_mean = None
    cluster_variance_sample_count = 0


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


class _EvalPipeline:
    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        return {"status": "skipped"}

    def drain_pending(self):
        return None

    def poll_completed(self):
        return None


def _coordinator(full_config: dict[str, Any]):
    cfg = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
                                 drain_caps=DEV_DRAIN_CAPS, gate_interval=DEV_GATE_INTERVAL,
                                 knobs=DEV_KNOBS),
        eval_interval=1, log_interval=1000, gate_interval=1000, min_buf_size=10,
    )
    sink = SpyEventSink()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=GraphSampleBuffer(),
        pool=DrivablePoolStub(games=5, search_kind="puct", rstats=_ClusterCarryingStats()), eval_pipeline=_EvalPipeline(),
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), config=cfg,
        full_config=full_config, sink=sink,
        monitor_cfg=monitor_config(),
    )
    return coord, cfg, sink


def _one_iteration_complete(sink: SpyEventSink) -> dict[str, Any]:
    events = sink.named("iteration_complete")
    assert len(events) == 1, f"expected exactly one iteration_complete, got {len(events)}"
    return events[0]


def test_a_real_coordinator_emits_no_cluster_key_on_a_graph_run() -> None:
    """Absence on the event stream a real `StepCoordinator` emits from a snapshot that still carries
    the fields. Planted break: any `cluster_*` entry in `emit_iteration_complete_event`'s payload reds."""
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


# The snapshot-crosswiring and wheel-compat pins retired with the fields they guarded: with no
# cluster-mean getter, both would be pins on a constant. That the fields do not come back
# without their producers is asserted in `tests/selfplay/test_pool_surface.py`.
def test_neither_engine_stub_declares_a_cluster_mean_getter() -> None:
    """The `_engine.pyi` stub is the ONLY type authority for the FFI getters — pyright
    reads the stub, never the compiled module — so a stub declaring a getter the engine no
    longer exposes is a phantom the checker blesses and every reader trusts.
    FALSIFYING MUTATION: add either getter back to the stub."""
    stub = REPO_ROOT / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi"
    text = stub.read_text(encoding="utf-8")
    for name in CLUSTER_MEANS:
        assert f"def {name}(" not in text, (
            f"{stub.relative_to(REPO_ROOT)} declares {name}, which the compiled engine no "
            f"longer exposes (R346(f)) — a stub with no counterpart type-checks a consumer "
            f"that gets None at runtime."
        )
