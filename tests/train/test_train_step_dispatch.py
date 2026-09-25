"""The declared training-step dispatcher.

The straight self-play arm routes through `run_declared_train_step`, keyed on the resolved
`EncodingSpec.representation` and never on a buffer sniff. Pinned: a REAL train step executes
end-to-end from the coordinator path for the GRAPH representation; an unknown representation
raises and an UNDECLARED encoding raises `MissingEncodingError` from THE resolver; removing the
trainer-side implementation reds THIS suite, not the conformance gate; and the graph arm threads
`recency_weight` in as `recent_frac`."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from _monitor_config import monitor_config
from _drivable import DrivablePoolStub
from _graph_drive import filled_hexg
from mantis.encoding import lookup
from mantis.encoding.resolvers import MissingEncodingError
from mantis.model import GnnArch, build_net
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.train.coordinator.dispatch import (
    RepresentationRouteError,
    resolve_step_spec,
    run_declared_train_step,
)
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from mantis.train.trainer.core import Trainer

GRAPH_ENCODING = "gnn_axis_v1"
_GSPEC = lookup(GRAPH_ENCODING)


def _coord_cfg(**over: Any) -> StepCoordinatorConfig:
    base: dict[str, Any] = dict(
        # The three interval knobs are 0: this drive is about the training step, not a boundary.
        eval_interval=0, log_interval=0, gate_interval=0, min_buf_size=1,
        capacity=64, buffer_schedule=(), training_steps_per_game=1.0, max_train_burst=1,
        batch_size=4, augment=False, recency_weight=0.0, hard_gn_threshold=1e9,
        hard_gn_min_steps=10_000, stop_step=None, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
        final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
        eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0,
        terminal_eval_enabled=False,
        selfplay_stall_timeout_sec=1800.0,
    )
    base.update(over)
    return StepCoordinatorConfig(**base)


def _tiny_graph_trainer(tmp_path, mk_config) -> Trainer:
    torch.manual_seed(20260729)
    arch = GnnArch(in_dim=_GSPEC.node_feat_dim, edge_dim=_GSPEC.edge_feat_dim, hidden=16,
                   num_layers=1, policy_hidden=16, value_hidden=16)
    return Trainer(build_net(arch), mk_config(GRAPH_ENCODING, "graph"), arch=arch,
                   checkpoint_dir=tmp_path / "ckpt", device=torch.device("cpu"))


#: A REQUIRED zero-arg `caps_provider` the GRAPH arm alone invokes: no default, since a caller
#: that forgot it would silently get an UNCAPPED step. These caps are far past anything the
#: fixtures can build, so every assertion below is about ROUTING, not a split.
def _NON_BINDING_CAPS() -> MicrobatchCapsSpec:
    return MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)


class _RecordingTypedTrainer:
    """A double conforming to the DECLARED seam (both typed entry points; no train_step)."""

    def __init__(self) -> None:
        self.step = 0
        self.model = None
        self.device = torch.device("cpu")
        self.tensor_calls: list[dict[str, Any]] = []
        self.graph_calls: list[dict[str, Any]] = []

    def train_step_from_tensors(self, states, policies, outcomes, **kw) -> dict[str, float]:
        self.step += 1
        self.tensor_calls.append({"n": int(np.asarray(states).shape[0]), **kw})
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1, "lr": 1e-3}

    def train_step_from_graph_batch(self, **kw) -> dict[str, float]:
        self.step += 1
        self.graph_calls.append(kw)
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1, "lr": 1e-3}

    def save_checkpoint(self, loss_info) -> None:  # pragma: no cover - not driven here
        pass


def _coordinator(trainer, buffer, full_config, cfg=None, **over) -> StepCoordinator:
    return StepCoordinator(
        monitor_cfg=monitor_config(),
        trainer=trainer, buffer=buffer, pool=over.pop("pool", DrivablePoolStub(games=3)),
        eval_pipeline=None, subsystems=None, anchor_state=None, shutdown=ShutdownState(),
        eval_model=None, config=cfg or _coord_cfg(), full_config=full_config,
        **over,
    )


def test_graph_train_step_end_to_end_from_coordinator(tmp_path, mk_config) -> None:
    """A REAL gradient step executes through step() → the straight self-play arm → the declared
    dispatcher → `train_step_from_graph_batch`, on the graph representation."""
    trainer = _tiny_graph_trainer(tmp_path, mk_config)
    coord = _coordinator(trainer, filled_hexg(), mk_config(GRAPH_ENCODING, "graph"))
    out = coord.step()
    assert out.in_warmup is False and out.waiting_for_games is False
    assert out.steps_run >= 1
    assert trainer.step >= 1
    assert coord._last_loss_info is not None
    for key in ("loss", "policy_loss", "value_loss", "grad_norm", "lr"):
        assert key in coord._last_loss_info, f"loss_info missing {key!r}"
    assert np.isfinite(coord._last_loss_info["loss"])


def test_graph_step_advances_trainer_step_counter(tmp_path, mk_config) -> None:
    trainer = _tiny_graph_trainer(tmp_path, mk_config)
    before = trainer.step
    run_declared_train_step(
        trainer, filled_hexg(), _GSPEC,
        batch_size=4, augment=False, recency_weight=0.0,
        caps_provider=_NON_BINDING_CAPS,
        sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
    )
    assert trainer.step == before + 1


def test_unknown_representation_raises_named_error() -> None:
    class _AlienSpec:
        name = "alien_v0"
        representation = "voxel"

    with pytest.raises(RepresentationRouteError, match="voxel"):
        run_declared_train_step(_RecordingTypedTrainer(), filled_hexg(), _AlienSpec(),
                                batch_size=2, augment=False, recency_weight=0.0,
                                caps_provider=_NON_BINDING_CAPS, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)


def test_undeclared_encoding_raises_from_the_one_resolver() -> None:
    """A thin veneer over `resolve_from_config`: an undeclared encoding raises, never defaults."""
    with pytest.raises(MissingEncodingError):
        resolve_step_spec({})
    with pytest.raises(MissingEncodingError):
        resolve_step_spec({"identity": {}})


def test_resolver_veneer_agrees_with_identity_declaration(mk_config) -> None:
    spec = resolve_step_spec(mk_config(GRAPH_ENCODING, "graph"))
    assert spec.name == GRAPH_ENCODING and spec.representation == "graph"


def test_missing_graph_entry_point_dies_loud_on_the_graph_route() -> None:
    class _HalfTrainer:
        step = 0
        model = None
        device = torch.device("cpu")

        def train_step_from_tensors(self, *a, **kw):  # pragma: no cover - must not be hit
            raise AssertionError("dense entry point must not absorb the graph route")

        def save_checkpoint(self, loss_info) -> None:  # pragma: no cover
            pass

    with pytest.raises(AttributeError, match="train_step_from_graph_batch"):
        run_declared_train_step(_HalfTrainer(), filled_hexg(), _GSPEC,
                                batch_size=2, augment=False, recency_weight=0.0,
                                caps_provider=_NON_BINDING_CAPS, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)


def test_graph_arm_threads_recency_weight_as_recent_frac() -> None:
    real = filled_hexg()
    seen: list[dict[str, Any]] = []

    class _RecordingHexg:
        size = real.size
        capacity = real.capacity

        def sample_graph_batch(self, batch_size, augment=False, recent_frac=0.0,
                               n_threads=1):
            seen.append({"batch_size": batch_size, "augment": augment,
                         "recent_frac": recent_frac})
            return real.sample_graph_batch(batch_size, augment=augment,
                                           recent_frac=recent_frac)

    rec = _RecordingTypedTrainer()
    run_declared_train_step(rec, _RecordingHexg(), _GSPEC,
                            batch_size=2, augment=False, recency_weight=0.25,
                            caps_provider=_NON_BINDING_CAPS, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)
    assert seen == [{"batch_size": 2, "augment": False, "recent_frac": 0.25}]
    assert len(rec.graph_calls) == 1
    kw = rec.graph_calls[0]
    # The graph entry point takes a PARTITION plus the step's denominators, not eleven loose
    # tensors; those names moved down onto what each part materialises, asserted below.
    assert set(kw) == {"parts", "policy_denominator", "value_denominator", "total_edges",
                       "total_nodes", "caps_max_edges", "caps_max_nodes",
                       # What the sampled batch was made of; `{}` on a buffer with no reading.
                       "batch_composition"}
    assert len(kw["parts"]) >= 1
    inputs = kw["parts"][0]()
    for name in ("x", "edge_index", "edge_attr", "legal_index", "stone_mask",
                 "node_offsets", "legal_offsets", "policy_target", "outcomes",
                 "value_valid", "policy_row_weight", "explicit_mask", "tail_mass"):
        assert getattr(inputs, name, None) is not None, (
            f"a materialised micro-batch is missing {name!r}")


def test_the_caps_provider_is_invoked_exactly_once_per_graph_step() -> None:
    """The caps are a PROVIDER called ONCE by the graph arm, not a value: Python evaluates every
    argument before the call."""
    rec = _RecordingTypedTrainer()
    invoked: list[int] = []

    def _counting() -> MicrobatchCapsSpec:
        invoked.append(1)
        return _NON_BINDING_CAPS()

    run_declared_train_step(rec, filled_hexg(), _GSPEC, batch_size=2, augment=False,
                            recency_weight=0.0, caps_provider=_counting,
                            sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)
    assert invoked == [1], "the graph arm must invoke the provider exactly once"


def test_the_sample_threads_provider_is_invoked_exactly_once_per_graph_step() -> None:
    """`resolve_sample_threads` reads `full_config["selfplay"]`, and passing the RESOLVED VALUE
    would evaluate it before the call — the provider keeps the derivation where it belongs."""
    rec = _RecordingTypedTrainer()
    invoked: list[int] = []

    def _counting() -> int:
        invoked.append(1)
        return 1

    run_declared_train_step(rec, filled_hexg(), _GSPEC, batch_size=2, augment=False,
                            recency_weight=0.0,
                            caps_provider=_NON_BINDING_CAPS,
                            sample_threads_provider=_counting,
                            fast_policy_weight_provider=lambda: 0.0)
    assert invoked == [1], "the graph arm must invoke the provider exactly once"
