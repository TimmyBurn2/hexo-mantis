"""The held-out gap witness (R366(c), v37): the rows resolve, the slice is pinned and FROZEN, the coordinator reads it at its own cadence, and the planted break (an un-seeded read) is caught."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.heldout_gap import HeldoutGapSpec, MissingHeldoutGapError, resolve_heldout_gap
from mantis.config.schema import RunConfig
from _monitor_config import monitor_config
from mantis.run import _step_coordinator_config
from mantis.train.coordinator import StepCoordinator
from mantis.train.heldout import HeldoutSlice, HeldoutSliceError
from mantis.util.hashing import sha256_file
from mantis.train.lifecycle.signals import ShutdownState
from _coordinator_pool import CoordinatorPoolStub
from _drivable import DrivableTrainerStub

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "dev_example.yaml"
_ENC = "gnn_axis_v1"


def _ring(path: Path, *, n: int = 32, visits: int = 128) -> HexgBuffer:
    hb = HexgBuffer(64, _ENC, visits)
    for i in range(n):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 1, -1), (1, 2, 1)][: 2 + (i % 4)]
        policy = [(3, 0, 0.5 + 0.01 * (i % 5)), (1, 1, 0.5 - 0.01 * (i % 5))]
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    hb.save_to_path(str(path))
    return hb


def _spec(path: Path, **over: Any) -> HeldoutGapSpec:
    base = dict(ring=str(path), ring_sha256=sha256_file(path), batches=2, seed=7, interval=2)
    base.update(over)
    return HeldoutGapSpec(**base)


def test_the_rows_resolve_and_absence_is_named() -> None:
    dump = load_config(_CONFIG).model_dump()
    assert resolve_heldout_gap(dump) is None, "the template mints the explicit OFF"
    dump["train"]["heldout_gap"] = {"ring": "r.bin", "ring_sha256": "a" * 64, "batches": 12, "seed": 1, "interval": 3000}
    spec = resolve_heldout_gap(RunConfig.model_validate(dump).model_dump())
    assert spec == HeldoutGapSpec("r.bin", "a" * 64, 12, 1, 3000)
    del dump["train"]["heldout_gap"]
    with pytest.raises(MissingHeldoutGapError, match="is absent"):
        resolve_heldout_gap(dump)
    dump["train"]["heldout_gap"] = {"ring": "r.bin", "ring_sha256": "a" * 64, "batches": 1, "seed": 1,
                                    "interval": dump["train"]["max_train_steps"]}
    with pytest.raises(ValueError, match="heldout_gap.interval"):
        RunConfig.model_validate(dump)


def test_the_slice_is_pinned_by_sha_and_refuses_a_foreign_geometry(tmp_path: Path) -> None:
    path = tmp_path / "held.ring.bin"
    _ring(path)
    spec = _spec(path)
    opened = HeldoutSlice.open(spec, encoding=_ENC, visit_capacity=128, capacity=64)
    assert opened.rows == 32 and opened.ring_path == path
    with pytest.raises(HeldoutSliceError, match="hashes sha256"):
        HeldoutSlice.open(dataclasses.replace(spec, ring_sha256="b" * 64), encoding=_ENC, visit_capacity=128, capacity=64)
    with pytest.raises(ValueError, match="slot-geometry mismatch"):
        HeldoutSlice.open(spec, encoding=_ENC, visit_capacity=64, capacity=64)
    with pytest.raises(FileNotFoundError):
        HeldoutSlice.open(dataclasses.replace(spec, ring=str(tmp_path / "missing.bin")), encoding=_ENC, visit_capacity=128, capacity=64)


def test_two_reads_at_the_same_weights_are_the_same_rows_and_the_planted_break_is_caught(tmp_path: Path, monkeypatch) -> None:
    """The slice is FROZEN by re-seeding; an un-seeded read (the planted break) drifts between reads."""
    import _microbatch_harness as H  # noqa: PLC0415 — the tests/train rootdir harness
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec

    path = tmp_path / "held.ring.bin"
    _ring(path)
    opened = HeldoutSlice.open(_spec(path, batches=2), encoding=_ENC, visit_capacity=128, capacity=64)
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    kwargs = dict(batch_size=4, caps_provider=lambda: MicrobatchCapsSpec(10**8, 10**6),
                  sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)
    first = opened.read(trainer, H.GSPEC, **kwargs)
    second = opened.read(trainer, H.GSPEC, **kwargs)
    assert first == second and opened.reads == 2
    assert first["policy_loss"] > 0.0 and first["value_loss"] > 0.0
    real = opened.buffer

    class _Unseeded:
        sample_graph_batch = real.sample_graph_batch

        def seed_sampler(self, _seed: int) -> None:
            return None

    monkeypatch.setattr(opened, "buffer", _Unseeded())
    drifting = [opened.read(trainer, H.GSPEC, **kwargs) for _ in range(6)]
    assert any(d != drifting[0] for d in drifting), "un-seeded reads must drift, or the freeze is not what pins them"


class _Trainer(DrivableTrainerStub):
    """The drivable stub with a policy loss that RISES by step (so the gap has a sign) and the eval step the witness reads."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        super().__init__(model=torch.nn.Linear(1, 1))
        self.bundle_publisher = None

    def loss_info(self) -> dict[str, float]:
        return {**super().loss_info(), "policy_loss": 2.0 + self.step}

    def eval_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        return {"loss": 3.0, "policy_loss": 2.5, "value_loss": 0.6}


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def test_the_coordinator_reads_the_slice_at_its_own_cadence_and_reports_the_gap(tmp_path: Path) -> None:
    """Producer test: `heldout_gap` lands at every `interval` boundary with the train mean since the last read."""
    path = tmp_path / "held.ring.bin"
    _ring(path)
    opened = HeldoutSlice.open(_spec(path, batches=1, interval=2), encoding=_ENC, visit_capacity=128, capacity=64)
    cfg = load_config(_CONFIG)
    base = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
        drain_caps=resolve_drain_caps(cfg.monitor), gate_interval=100, knobs=resolve_coordinator_knobs(cfg.train))
    config = dataclasses.replace(base, eval_interval=0, log_interval=100, min_buf_size=1, max_train_burst=1,
                                 training_steps_per_game=1.0, hard_gn_threshold=1e9, batch_size=4)
    sink = _Sink()
    buffer = SimpleNamespace(size=100, capacity=1000, resize=lambda n: None, save_to_path=lambda p: None,
                             sample_graph_batch=opened.buffer.sample_graph_batch)
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=buffer, pool=CoordinatorPoolStub(search_kind="puct"),
        eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None), shutdown=ShutdownState(),
        eval_model=object(), config=config,
        full_config={"identity": {"encoding": _ENC, "representation": "graph"},
                     "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000}, "fast_policy_weight": 0.0},
                     "selfplay": {"n_workers": 1}},
        sink=sink, heartbeat=None, monitor_cfg=monitor_config(), heldout=opened)
    for _ in range(4):
        coord.pool.games_completed += 1
        coord.step()
    events = sink.named("heldout_gap")
    assert [e["step"] for e in events] == [2, 4]
    first = events[0]
    assert (first["policy_loss"], first["value_loss"]) == (2.5, 0.6)
    assert first["train_steps"] == 2 and first["train_policy_loss"] == pytest.approx((3.0 + 4.0) / 2)
    assert first["gap_policy"] == pytest.approx(2.5 - 3.5) and first["gap_value"] == pytest.approx(0.6 - 0.4)
    assert first["ring"] == path.name and first["rows"] == 32 and first["read_index"] == 1
    assert events[1]["train_steps"] == 2 and events[1]["train_policy_loss"] == pytest.approx((5.0 + 6.0) / 2)
