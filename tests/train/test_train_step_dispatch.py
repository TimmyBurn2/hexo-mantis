# >300 justify (R8), measured at write: the WPTS Phase T oracle suite (O-T1..O-T7) — the
# end-to-end graph/grid step oracles, the typed-route unreachability oracles and the mutation
# "alone" arms are one cohesive contract over ONE seam (the declared training-step dispatcher);
# splitting it would scatter the mutation matrix the dispatch requires reading as a unit.
"""WPTS Phase T oracles — TD-1 / CARD-TRAINSTEP-ADAPTER (R102).

The straight self-play arm (`step.py::_run_training_step`) routes through the DECLARED
training-step dispatcher (`coordinator/dispatch.py::run_declared_train_step`), keyed on the
resolved `EncodingSpec.representation` — never a buffer sniff. These oracles pin:

- O-T1: a REAL train step executes end-to-end from the coordinator path for the GRAPH
  representation (run5's) on CPU-scale data — real `HexgBuffer`, real `Trainer` (tiny GnnNet).
- O-T2: the dense route is TYPE-UNREACHABLE from a graph config (oracle, not assumption).
- O-T3: the grid route end-to-end + the old-side recency-mix contract.
- O-T4: the mixed arm's dense-only feed is typed — a graph spec + pretrained buffer RAISES
  (CENSUS_C C-2b), never reaches `assemble_mixed_batch`.
- O-T5: closed match — unknown representation raises; an UNDECLARED encoding raises
  `MissingEncodingError` from THE resolver (LAW-11 re-pin at this new consumer).
- O-T6(b): removing the trainer-side implementation reds THIS suite (the step oracle), not the
  conformance gate (which never imports a trainer) — the "alone" separation, R86.
- O-T7: the graph arm refuses a non-None recent_buffer and threads
  `recent_frac=recency_weight` into `sample_graph_batch` (old-side commit-B parity).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from mantis._engine import HexgBuffer, ReplayBuffer
from mantis.encoding import lookup
from mantis.encoding.resolvers import MissingEncodingError
from mantis.model import GnnArch, build_net
from mantis.train.coordinator.config import StepCoordinatorConfig
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
GRID_ENCODING = "v6_live2_ls"
_DSPEC = lookup(GRID_ENCODING)


# ── builders ─────────────────────────────────────────────────────────────────────────────
def _coord_cfg(**over: Any) -> StepCoordinatorConfig:
    base: dict[str, Any] = dict(
        eval_interval=0, log_interval=0, checkpoint_interval=0, min_buf_size=1,
        capacity=64, buffer_schedule=(), training_steps_per_game=1.0, max_train_burst=1,
        batch_size=4, augment=False, recency_weight=0.0, mixing_initial_w=0.0,
        mixing_min_w=0.0, mixing_decay_steps=1.0, hard_gn_threshold=1e9,
        hard_gn_min_steps=10_000, stop_step=None, draw_rate_abort=None,
        final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
        eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0,
        terminal_eval_enabled=False, bot_batch_share=0.0,
        selfplay_stall_timeout_sec=1800.0,
    )
    base.update(over)
    return StepCoordinatorConfig(**base)


def _graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real HexgBuffer fed through the real graph push path (the pyclass-roundtrip record
    shape: stones, policy, current_player, moves_remaining, ply_index, is_full_search,
    outcome, value_valid, game_length)."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        policy = [(2, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


def _dense_buffer(n_records: int = 8, capacity: int = 64) -> ReplayBuffer:
    rb = ReplayBuffer(capacity, GRID_ENCODING)
    s = int(_DSPEC.board_size)
    n_cells = s * s
    for i in range(n_records):
        state = np.zeros((int(_DSPEC.n_planes), s, s), dtype=np.float16)
        state[0, 0, i % s] = 1.0
        chain = np.zeros((6, s, s), dtype=np.float16)
        policy = np.zeros(int(_DSPEC.policy_stride), dtype=np.float32)
        policy[i % n_cells] = 1.0
        own = np.zeros(n_cells, dtype=np.uint8)
        wl = np.zeros(n_cells, dtype=np.uint8)
        rb.push(state, chain, policy, 1.0 if i % 2 == 0 else -1.0, own, wl)
    return rb


def _tiny_graph_trainer(tmp_path, mk_config) -> Trainer:
    torch.manual_seed(20260729)
    arch = GnnArch(in_dim=_GSPEC.node_feat_dim, edge_dim=_GSPEC.edge_feat_dim, hidden=16,
                   num_layers=1, policy_hidden=16, value_hidden=16)
    return Trainer(build_net(arch), mk_config(GRAPH_ENCODING, "graph"), arch=arch,
                   checkpoint_dir=tmp_path / "ckpt", device=torch.device("cpu"))


def _tiny_dense_trainer(tmp_path, mk_config, tiny_arch) -> Trainer:
    torch.manual_seed(20260729)
    return Trainer(build_net(tiny_arch), mk_config(), arch=tiny_arch,
                   checkpoint_dir=tmp_path / "ckpt", device=torch.device("cpu"))


class _Pool:
    """Minimal WorkerPoolLike stand-in for driving step() past O4/O5 (not the subject)."""

    def __init__(self, games_completed: int = 3) -> None:
        self.games_completed = games_completed
        self.n_workers = 1


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
        trainer=trainer, buffer=buffer, pretrained_buffer=over.pop("pretrained_buffer", None),
        recent_buffer=over.pop("recent_buffer", None), pool=over.pop("pool", _Pool()),
        eval_pipeline=None, subsystems=None, anchor_state=None, shutdown=ShutdownState(),
        eval_model=None, bufs=None, config=cfg or _coord_cfg(), full_config=full_config,
        **over,
    )


# ── O-T1: the end-to-end GRAPH step from the coordinator path ────────────────────────────
def test_graph_train_step_end_to_end_from_coordinator(tmp_path, mk_config) -> None:
    """run5's representation: a REAL gradient step executes through step() → the straight
    self-play arm → the declared dispatcher → `train_step_from_graph_batch`. The rehearsal's
    wall (rc 40 behind the warmup gate) can no longer hide a missing learner half: this drive
    IS the training step the CPU box never reached."""
    trainer = _tiny_graph_trainer(tmp_path, mk_config)
    coord = _coordinator(trainer, _graph_buffer(), mk_config(GRAPH_ENCODING, "graph"))
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
        trainer, _graph_buffer(), _GSPEC,
        batch_size=4, augment=False, recency_weight=0.0, recent_buffer=None,
    )
    assert trainer.step == before + 1


# ── O-T2: dense route type-unreachable from a graph config ───────────────────────────────
def test_graph_spec_never_calls_the_dense_entry_point() -> None:
    rec = _RecordingTypedTrainer()
    run_declared_train_step(rec, _graph_buffer(), _GSPEC,
                            batch_size=2, augment=False, recency_weight=0.0,
                            recent_buffer=None)
    assert len(rec.graph_calls) == 1
    assert rec.tensor_calls == [], "dense entry point must be unreachable from a graph spec"


def test_graph_spec_over_a_dense_buffer_raises_named_error() -> None:
    """Declaration↔object mismatch is a NAMED wiring error (the BufferKindMismatch posture),
    never a silent fall-through to the dense arm."""
    rec = _RecordingTypedTrainer()
    with pytest.raises(RepresentationRouteError, match="graph"):
        run_declared_train_step(rec, _dense_buffer(), _GSPEC,
                                batch_size=2, augment=False, recency_weight=0.0,
                                recent_buffer=None)
    assert rec.tensor_calls == [] and rec.graph_calls == []


def test_grid_spec_over_a_graph_buffer_raises_named_error() -> None:
    rec = _RecordingTypedTrainer()
    with pytest.raises(RepresentationRouteError, match="grid"):
        run_declared_train_step(rec, _graph_buffer(), _DSPEC,
                                batch_size=2, augment=False, recency_weight=0.0,
                                recent_buffer=None)
    assert rec.tensor_calls == [] and rec.graph_calls == []


# ── O-T3: the grid route ─────────────────────────────────────────────────────────────────
def test_grid_train_step_end_to_end_from_coordinator(tmp_path, mk_config, tiny_arch) -> None:
    trainer = _tiny_dense_trainer(tmp_path, mk_config, tiny_arch)
    coord = _coordinator(trainer, _dense_buffer(), mk_config())
    out = coord.step()
    assert out.steps_run >= 1
    assert trainer.step >= 1
    assert coord._last_loss_info is not None and "loss" in coord._last_loss_info


def test_grid_recency_mix_contract_matches_old_side() -> None:
    """Old-side `train_step` dense-arm parity: n_recent = max(1, round(bs*rw)); recent aux
    rows reshape from flat (n, s*s) to (n, s, s); recent ply-index rows are ZERO-filled
    (§S181-AUDIT 4B-impl-3); the remainder is one uniform `sample_batch_with_pos` draw."""
    s = 5
    n_cells = s * s

    class _RecentBuf:
        size = 4

        def sample(self, n: int):
            st = np.zeros((n, 3, s, s), dtype=np.float16)
            ch = np.zeros((n, 6, s, s), dtype=np.float16)
            po = np.zeros((n, n_cells + 1), dtype=np.float32)
            oc = np.ones(n, dtype=np.float32)
            own = np.zeros((n, n_cells), dtype=np.uint8)
            wl = np.zeros((n, n_cells), dtype=np.uint8)
            ifs = np.ones(n, dtype=bool)
            vv = np.ones(n, dtype=np.uint8)
            return st, ch, po, oc, own, wl, ifs, vv

    class _DenseBuf:
        size = 32
        sampled: list[tuple[int, bool]] = []

        def sample_batch_with_pos(self, n: int, augment: bool):
            self.sampled.append((n, augment))
            st = np.zeros((n, 3, s, s), dtype=np.float16)
            ch = np.zeros((n, 6, s, s), dtype=np.float16)
            po = np.zeros((n, n_cells + 1), dtype=np.float32)
            oc = -np.ones(n, dtype=np.float32)
            own = np.zeros((n, s, s), dtype=np.uint8)
            wl = np.zeros((n, s, s), dtype=np.uint8)
            ifs = np.ones(n, dtype=bool)
            pos = np.arange(n, dtype=np.uint16)
            vv = np.ones(n, dtype=np.uint8)
            return st, ch, po, oc, own, wl, ifs, pos, vv

    rec = _RecordingTypedTrainer()
    run_declared_train_step(rec, _DenseBuf(), _DSPEC,
                            batch_size=8, augment=False, recency_weight=0.25,
                            recent_buffer=_RecentBuf())
    assert len(rec.tensor_calls) == 1
    call = rec.tensor_calls[0]
    assert call["n"] == 8, "recent + uniform rows must concatenate to the full batch"
    assert call["n_recent"] == 2, "n_recent = max(1, round(8 * 0.25))"
    assert call["n_pretrain"] == 0
    assert _DenseBuf.sampled == [(6, False)], "uniform remainder = batch_size - n_recent"
    pos = np.asarray(call["position_indices"])
    assert pos.shape == (8,) and (pos[:2] == 0).all(), "recent rows carry ZERO ply-index"


# ── O-T4: the mixed arm's dense-only feed is typed (CENSUS_C C-2b) ───────────────────────
def test_mixed_arm_with_graph_spec_raises_at_the_route(tmp_path, mk_config) -> None:
    class _Pretrained:
        size = 4

    rec = _RecordingTypedTrainer()
    coord = _coordinator(rec, _graph_buffer(), mk_config(GRAPH_ENCODING, "graph"),
                         pretrained_buffer=_Pretrained())
    with pytest.raises(RepresentationRouteError, match="mixed"):
        coord._run_training_step(coord.config)
    assert rec.tensor_calls == [] and rec.graph_calls == []


# ── O-T5: closed match (LAW-11 posture) ──────────────────────────────────────────────────
def test_unknown_representation_raises_named_error() -> None:
    class _AlienSpec:
        name = "alien_v0"
        representation = "voxel"

    with pytest.raises(RepresentationRouteError, match="voxel"):
        run_declared_train_step(_RecordingTypedTrainer(), _graph_buffer(), _AlienSpec(),
                                batch_size=2, augment=False, recency_weight=0.0,
                                recent_buffer=None)


def test_undeclared_encoding_raises_from_the_one_resolver() -> None:
    """`resolve_step_spec` is a thin veneer over `resolve_from_config` — an undeclared
    encoding raises `MissingEncodingError`, never a default (LAW-11 re-pin at the new
    consumer; the TD-4 lesson applied to TD-1's fix)."""
    with pytest.raises(MissingEncodingError):
        resolve_step_spec({})
    with pytest.raises(MissingEncodingError):
        resolve_step_spec({"identity": {}})


def test_resolver_veneer_agrees_with_identity_declaration(mk_config) -> None:
    spec = resolve_step_spec(mk_config(GRAPH_ENCODING, "graph"))
    assert spec.name == GRAPH_ENCODING and spec.representation == "graph"


# ── O-T6(b): implementation removal reds the STEP oracle (this suite), gate stays blind ──
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
        run_declared_train_step(_HalfTrainer(), _graph_buffer(), _GSPEC,
                                batch_size=2, augment=False, recency_weight=0.0,
                                recent_buffer=None)


# ── O-T7: graph-arm recency semantics (old-side commit-B parity) ─────────────────────────
def test_graph_arm_refuses_a_dense_recent_buffer() -> None:
    class _RecentBuf:
        size = 4

    with pytest.raises(RepresentationRouteError, match="recent_buffer"):
        run_declared_train_step(_RecordingTypedTrainer(), _graph_buffer(), _GSPEC,
                                batch_size=2, augment=False, recency_weight=0.0,
                                recent_buffer=_RecentBuf())


def test_graph_arm_threads_recency_weight_as_recent_frac() -> None:
    real = _graph_buffer()
    seen: list[dict[str, Any]] = []

    class _RecordingHexg:
        size = real.size
        capacity = real.capacity

        def sample_graph_batch(self, batch_size, augment=False, recent_frac=0.0):
            seen.append({"batch_size": batch_size, "augment": augment,
                         "recent_frac": recent_frac})
            return real.sample_graph_batch(batch_size, augment=augment,
                                           recent_frac=recent_frac)

    rec = _RecordingTypedTrainer()
    run_declared_train_step(rec, _RecordingHexg(), _GSPEC,
                            batch_size=2, augment=False, recency_weight=0.25,
                            recent_buffer=None)
    assert seen == [{"batch_size": 2, "augment": False, "recent_frac": 0.25}]
    assert len(rec.graph_calls) == 1
    kw = rec.graph_calls[0]
    for name in ("x", "edge_index", "edge_attr", "legal_mask", "stone_mask",
                 "node_offsets", "legal_offsets", "policy_target", "outcomes",
                 "value_valid", "is_full_search"):
        assert name in kw, f"graph entry point kwarg {name!r} missing"
