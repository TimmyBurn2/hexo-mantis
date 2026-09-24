"""Continuous actor sync runs unimpaired with the gate, promotion and eval machinery ABSENT.

`eval_enabled=False` means none of the deploy side is constructed in this process, so sync
provably needs nothing it provides.

Bounded by construction: the coordinator's `stop_step` terminates the loop, no thread starts
(`build_run_safety` is faked), and no sleep is reached — the buffer is above floor and every
step sees fresh games.
"""
from __future__ import annotations

from types import SimpleNamespace

import mantis.run
import mantis.train.actor_sync  # noqa: F401 — import anchor
from _drivable import DrivablePoolStub, DrivableTrainerStub

_STOP_STEP = 5


def test_compose_run_syncs_actor_on_cadence_without_eval(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """The dependency-absence proof: with no eval pipeline in the process at all, the pool
    still records cadence-consistent weight pushes and the actor's recorded step ends inside
    the cadence bound of the learner's."""
    pool = DrivablePoolStub(game_per_read=True)
    trainer = DrivableTrainerStub()

    def _fake_build_run_safety(**kwargs):
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_build_run_safety)
    # `stop_step` is config-authored, so this drives the PRODUCTION coordinator config with no
    # monkeypatch — which couples the step counts below to that builder's other knobs.

    handles = mantis.run.compose_run(
        # Reachability bound: cadence < threshold < max_train_steps, so _STOP_STEP >= 3.
        config=smoke_run_config(
            train={"actor_sync_cadence_steps": 1, "max_train_steps": _STOP_STEP,
                   # Real graph route per step; the minted 256 batch is drag.
                   "batch_size": 8},
            monitor={"actor_lag_threshold_steps": _STOP_STEP - 1},
            # The eval posture is the config's fact; no parameter can force it.
            eval_enabled=False),
        trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert handles.eval_pipeline is None, (
        "harness precondition: the deploy side must not exist in this process"
    )
    assert trainer.step >= 1, "harness precondition: at least one real training step ran"
    assert len(pool.sync_payloads) >= 1, (
        "with the gate/promotion machinery ABSENT, sync must run unimpaired — zero pushes "
        "means actor sync still depends on something the deploy side provides (R49 breach)"
    )
    assert all(sd is trainer.actor_sd for sd in pool.sync_payloads), (
        "every push carries trainer.actor_state_dict()'s result — the LEARNER's weights; the deploy "
        "view (inference_state_dict, the EMA shadow when on) never reaches the actors (R366(b))"
    )
    assert not any(sd is trainer.inference_sd for sd in pool.sync_payloads)
    assert pool.step_calls == sorted(set(pool.step_calls)), (
        f"recorded sync steps must be strictly increasing: {pool.step_calls}"
    )
    cadence = 1  # the composed config's cadence: the most-synced world
    assert trainer.step - pool.step_calls[-1] < cadence + 1, (
        f"actor_ckpt_step {pool.step_calls[-1]} must track learner_step {trainer.step} "
        f"within the cadence bound"
    )
    gaps = [b - a for a, b in zip(pool.step_calls, pool.step_calls[1:])]
    assert all(g <= cadence for g in gaps), (
        f"consecutive syncs must never be further apart than the cadence: {pool.step_calls}"
    )
