"""CARD-SERVER-OWNED-COPY (R366(b)): the actors serve a copy the sync writes and the learner never reads; with EMA on the actors STILL serve the learner while deploy, gate and follower read the EMA shadow, which rides the envelope and survives a resume."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.eval.snapshot import load_model_snapshot, write_model_snapshot
from mantis.model import GnnArch, build_net, net_param_hash, state_dict_param_hash
from mantis.selfplay.pool import WorkerPool, served_copy
from mantis.train.actor_sync import ActorSync
from mantis.train.checkpoints import deploy_state, load_checkpoint, resume_trainer
from mantis.train.trainer.core import Trainer

_REPO = Path(__file__).resolve().parents[2]
_ENCODING = "gnn_axis_v1"


def _pool_cfg() -> dict[str, Any]:
    selfplay: dict[str, Any] = {
        "search": {"kind": "puct"}, "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 32,
        "c_visit": 50.0, "c_scale": 1.0, "q_rescale": True, "gumbel_m": 4, "gumbel_explore_moves": 10,
        "search_stats_every": 8, "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": False,
        "mcts": {"n_simulations": 8, "c_puct": 1.5, "fpu_reduction": 0.25, "quiescence_enabled": True,
                 "quiescence_blend_2": 0.3, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 8, "fast_prob": 0.0, "standard_sims": 0, "full_search_prob": 0.0,
                        "n_sims_quick": 0, "n_sims_full": 0, "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    inference = {"inference_batch_size": 4, "inference_max_wait_ms": 10, "edge_geometry_check": "inline",
                 "compile_trunk": False, "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921}}
    return {"encoding": _ENCODING, "deploy": {"search": {"kind": "puct"}}, "selfplay": selfplay,
            "inference": inference, "train": {"draw_reward": -0.5, "ply_cap_value": -0.5}}


def _arch() -> GnnArch:
    spec = lookup(_ENCODING)
    return GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), hidden=16, num_layers=1,
                   policy_hidden=16, value_hidden=16)


def test_the_pool_serves_its_own_copy_and_the_learner_never_reads_it() -> None:
    torch.manual_seed(1)
    arch = _arch()
    learner = build_net(arch)
    pool = WorkerPool(learner, _pool_cfg(), torch.device("cpu"),
                      HexgBuffer(capacity=64, encoding=_ENCODING, visit_capacity=128), arch=arch)
    served = pool._inference_server.model
    assert served is not learner and pool.model is served and pool.learner is learner
    assert net_param_hash(served) == net_param_hash(learner), "the copy is seeded from the learner"
    before = net_param_hash(learner)
    with torch.no_grad():
        next(served.parameters()).add_(1.0)
    assert net_param_hash(learner) == before, "a write to the copy never reaches the learner"
    fresh = {k: torch.zeros_like(v) for k, v in learner.state_dict().items()}
    pool.sync_inference_weights(fresh)
    assert state_dict_param_hash(served.state_dict()) == state_dict_param_hash(fresh)
    assert net_param_hash(learner) == before, "the sync writes the copy, not the learner"


def test_a_pool_with_no_declared_arch_still_never_serves_the_module_it_was_handed() -> None:
    """The test seam `arch=None` must not re-open the B-1 hazard: the served net is a copy either way."""
    torch.manual_seed(1)
    learner = build_net(_arch())
    pool = WorkerPool(learner, _pool_cfg(), torch.device("cpu"),
                      HexgBuffer(capacity=64, encoding=_ENCODING, visit_capacity=128), arch=None)
    assert pool.model is not learner and pool.learner is learner
    assert net_param_hash(pool.model) == net_param_hash(learner)
    with torch.no_grad():
        next(pool.model.parameters()).add_(1.0)
    assert net_param_hash(pool.model) != net_param_hash(learner)


def test_served_copy_refuses_weights_of_another_shape() -> None:
    torch.manual_seed(2)
    arch = _arch()
    with pytest.raises(RuntimeError):
        served_copy(build_net(GnnArch(in_dim=arch.in_dim, edge_dim=arch.edge_dim, hidden=8, num_layers=1)), arch)


def _step(trainer, buf):
    import _microbatch_harness as H  # noqa: PLC0415 — the tests/train rootdir harness
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import run_declared_train_step

    replay = H.ReplayWireBuffer(buf, 8)
    return run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=8, augment=False, recency_weight=0.0, recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
        sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)


def test_with_ema_on_the_actors_serve_the_learner_and_deploy_reads_the_shadow(tmp_path: Path) -> None:
    """THE HASH WITNESS: after real steps the synced copy hashes to the learner, never to the EMA."""
    import _microbatch_harness as H  # noqa: PLC0415

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    trainer = H.ema_graph_trainer(tmp_path, sink=H.SpySink(), update_every=1)
    served: dict[str, Any] = {}

    class _Target:
        def sync_inference_weights(self, sd):
            served["state"] = {k: v.detach().clone() for k, v in sd.items()}
        def update_checkpoint_step(self, step):
            served["step"] = step

    sync = ActorSync(target=_Target(), state_dict_fn=trainer.actor_state_dict, step_fn=lambda: trainer.step,
                     cadence_steps=1, sink=H.SpySink(), run_id="ema")
    for _ in range(3):
        _step(trainer, buf)
        sync.maybe_sync(trainer.step)
    learner = state_dict_param_hash(trainer.model.state_dict())
    ema = state_dict_param_hash(trainer.ema_model.state_dict())
    assert learner != ema, "premise: after three steps the shadow lags the learner"
    assert state_dict_param_hash(served["state"]) == learner, "the actors serve the LEARNER"
    assert state_dict_param_hash(trainer.inference_state_dict()) == ema, "deploy/gate/promotion read the EMA"
    view = trainer.deploy_module()
    assert view is not trainer.model and type(view.arch) is type(trainer.arch)
    write_model_snapshot(view, tmp_path / "cand.pt")
    assert net_param_hash(load_model_snapshot(tmp_path / "cand.pt")) == ema, "the gate's candidate is the EMA net"


def test_the_shadow_rides_the_envelope_and_a_resume_restores_it(tmp_path: Path) -> None:
    import _microbatch_harness as H  # noqa: PLC0415

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    trainer = H.ema_graph_trainer(tmp_path, sink=H.SpySink(), update_every=1)
    for _ in range(2):
        _step(trainer, buf)
    path = trainer.save_checkpoint()
    ck = load_checkpoint(path)
    assert ck.ema_state is not None and set(ck.ema_state) == set(ck.model_state)
    state, which = deploy_state(ck)
    assert which == "ema" and state_dict_param_hash(state) == state_dict_param_hash(trainer.ema_model.state_dict())
    assert state_dict_param_hash(ck.model_state) == state_dict_param_hash(trainer.model.state_dict())
    resumed = resume_trainer(Trainer, path)
    assert resumed.ema_model is not None
    assert state_dict_param_hash(resumed.ema_model.state_dict()) == state_dict_param_hash(trainer.ema_model.state_dict())

    off = H.tiny_graph_trainer(tmp_path / "off", sink=H.SpySink())
    _step(off, buf)
    off_ck = load_checkpoint(off.save_checkpoint())
    assert off_ck.ema_state is None and deploy_state(off_ck)[1] == "learner"


def test_every_deploy_reader_rebuilds_the_shadow_from_an_ema_stamp(tmp_path: Path) -> None:
    """One deploy reader: the gate's anchor and the warm start's parent are the EMA shadow when the stamp carries one, never the learner."""
    import _microbatch_harness as H  # noqa: PLC0415
    from mantis.train.anchor import _build_anchor_model
    from mantis.train.warmstart import BcWarmStart, WarmStartIdentityError, apply_bc_warm_start

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    trainer = H.ema_graph_trainer(tmp_path, sink=H.SpySink(), update_every=1)
    for _ in range(2):
        _step(trainer, buf)
    path = trainer.save_checkpoint()
    ema_hash = net_param_hash(trainer.deploy_module())
    assert ema_hash != net_param_hash(trainer.model), "premise: the shadow and the learner differ"
    anchor, _ck = _build_anchor_model(path, declared_encoding=None, device=torch.device("cpu"))
    assert net_param_hash(anchor) == ema_hash
    child = build_net(trainer.arch)
    spec = lookup(_ENCODING)
    with pytest.raises(WarmStartIdentityError, match="net_param_hash"):
        apply_bc_warm_start(child, BcWarmStart(path, net_param_hash(trainer.model), ()), spec=spec)
    apply_bc_warm_start(child, BcWarmStart(path, ema_hash, ()), spec=spec)
    assert net_param_hash(child) == ema_hash


def test_a_resume_from_a_shadowless_stamp_reseeds_and_says_so(tmp_path: Path) -> None:
    import _microbatch_harness as H  # noqa: PLC0415

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    off = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    _step(off, buf)
    path = off.save_checkpoint()
    sink = H.SpySink()
    on = resume_trainer(Trainer, path, config_overrides={"train": {"ema": {"enabled": True, "decay": 0.9, "update_every": 1}}},
                        sink=sink)
    assert on.ema_model is not None
    assert state_dict_param_hash(on.ema_model.state_dict()) == state_dict_param_hash(on.model.state_dict())
    assert [e["event"] for e in sink.events if e["event"] == "ema_shadow_reseeded"] == ["ema_shadow_reseeded"]


def test_an_armed_ema_row_now_mints_and_the_lag_row_is_back() -> None:
    from mantis.config.armed_aborts import MANIFEST, Status
    from mantis.config.census import production_configs

    for path in production_configs(_REPO):
        dump = load_config(path).model_dump()
        dump["train"]["ema"] = {"enabled": True, "decay": 0.999, "update_every": 10}
        assert RunConfig.model_validate(dump).train.ema.enabled is True, path.name
    rows = {row.name: row for row in MANIFEST}
    assert rows["actor_lag"].status is Status.REQUIRED
