"""The composed lag-watchdog callables read the LIVE engine and the LIVE trainer.

REVIEW-impl MF-1. Every lag oracle injects its own callables, and every `compose_run` test
fakes `build_run_safety` — so the two lambdas the composition root actually hands to the
watchdog (`run.py`: `actor_ckpt_step_fn`, `learner_step_fn`) were pinned by **nothing**.
Replacing either with `lambda: 0`, or swapping them, passed all 1681 tests while blinding
or false-firing the exit-45 actor-lag invariant at run5.

That is the F-10 / LAW-07 phantom-gate class — a gate fed by nothing — on the very
invariant this WP ships. R4 is explicit: no gate input without a producer test. These are
that producer test.

Deliberately NOT frozen: written after ORACLE-WRITE, in response to a review finding.

WPAX R67: the RED-TEAM F-2 signature census that used to close this file has been FOLDED into
`tests/train/test_actor_lag_watchdog.py`'s parametrized no-defaults census, which is the one
authority for that rule (LAW-08). It lived here only because that file was byte-frozen and the
fix pass that found F-2 held no R43 event; R67 was that event. Nothing replaces it here.

>300 justify (R8): the RED-TEAM F-1 pins at the end are the SAME subject as this file's
existing ones — what the composition root hands `build_run_safety`, and what that builder
does with it — and R5 bars cross-test imports, so a second file would fork a fourth copy of
the drivable pool/trainer/buffer fakes above.
"""
from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run
from mantis.monitor.heartbeat import HEARTBEAT_SOURCES
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.subsystems import build_run_safety

_STOP_STEP = 3


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self._games = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...
    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None: ...
    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.inference_sd = {"w": "SENTINEL"}

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...
    def save_to_path(self, p) -> None: ...


#: The UNPATCHED production builder, captured at import so the patch below can delegate to
#: it without re-entering itself (WPMINT Phase K-A stage 0).
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


def _bounded_config(**kwargs) -> StepCoordinatorConfig:
    """WPMINT Phase K-A stage 0: the harness's own deltas over the REAL builder, not a
    24-kwarg restatement of it. `draw_rate_abort` (and every other config-authored value)
    is passed THROUGH untouched; `stop_step` stays the harness's own bound, which is this
    patch's stated reason for existing."""
    return dataclasses.replace(_PRODUCTION_BUILDER(**kwargs),
                               eval_interval=0, log_interval=1, stop_step=_STOP_STEP)


def _compose_capturing_lag_fns(tmp_path, monkeypatch, smoke_run_config, *, abort_enabled=None):
    """Run `compose_run` and return (captured_kwargs, pool, trainer).

    Captures what the composition root ACTUALLY hands `build_run_safety`, rather than
    what a test injects in its place.

    WPAX S-1: the config is a REAL minted `RunConfig` (the strict gate rejects the
    `SimpleNamespace()` this used to pass), bounded by co-overriding all three step-clock
    knobs — the reachability validator spans `cadence < threshold < max_train_steps`, so
    `_STOP_STEP` must stay >= 3 for the chain to hold.
    """
    captured: dict = {}
    pool, trainer = _Pool(), _Trainer()
    monitor_overrides: dict = {"actor_lag_threshold_steps": _STOP_STEP - 1}
    if abort_enabled is not None:
        monitor_overrides["actor_lag_abort_enabled"] = abort_enabled

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _capture)
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _bounded_config)
    mantis.run.compose_run(
        config=smoke_run_config(
            train={"actor_sync_cadence_steps": 1, "max_train_steps": _STOP_STEP},
            monitor=monitor_overrides),
        trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        eval_enabled=False,
    )
    return captured, pool, trainer


def test_composition_root_supplies_both_lag_callables(tmp_path, monkeypatch, smoke_run_config):
    captured, _pool, _trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch, smoke_run_config)
    for name in ("actor_ckpt_step_fn", "learner_step_fn"):
        assert name in captured, f"composition root did not supply {name}"
        assert callable(captured[name])


def test_learner_step_fn_reads_the_live_trainer(tmp_path, monkeypatch, smoke_run_config):
    """Not a captured snapshot: mutating the trainer must move the reading.

    A build-time snapshot would freeze `learner_step` and the lag would never grow, so the
    invariant could never fire however far the actor fell behind.
    """
    captured, _pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch, smoke_run_config)
    trainer.step = 4242
    assert captured["learner_step_fn"]() == 4242


def test_actor_ckpt_step_fn_reads_the_live_sync_engine(tmp_path, monkeypatch, smoke_run_config):
    """It must report the actor's real synced step, not a constant and not the trainer."""
    captured, pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch, smoke_run_config)
    assert pool.step_calls, "harness precondition: the actor synced at least once"
    assert captured["actor_ckpt_step_fn"]() == pool.step_calls[-1]

    # Moving the LEARNER must not move the ACTOR reading — that would hide all lag.
    before = captured["actor_ckpt_step_fn"]()
    trainer.step = 99999
    assert captured["actor_ckpt_step_fn"]() == before


def test_the_two_lag_callables_are_not_swapped(tmp_path, monkeypatch, smoke_run_config):
    """Swapping them inverts the invariant into a permanent false-negative.

    `learner_step − actor_ckpt_step` would go negative rather than positive, so a starved
    actor would read as healthy no matter how far behind it fell.
    """
    captured, pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch, smoke_run_config)
    trainer.step = pool.step_calls[-1] + 500

    learner = captured["learner_step_fn"]()
    actor = captured["actor_ckpt_step_fn"]()
    assert learner == trainer.step, "learner_step_fn is not reading the trainer"
    assert actor == pool.step_calls[-1], "actor_ckpt_step_fn is not reading the engine"
    assert learner - actor == 500, (
        f"lag must be learner-minus-actor and positive when the actor is behind; "
        f"got learner={learner} actor={actor}"
    )


def test_neither_callable_is_a_constant(tmp_path, monkeypatch, smoke_run_config):
    """Kills the `lambda: 0` stub directly — the cheapest way to blind the gate."""
    captured, pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch, smoke_run_config)
    trainer.step = 7
    assert captured["learner_step_fn"]() == 7
    trainer.step = 8
    assert captured["learner_step_fn"]() == 8, "learner_step_fn returns a constant"

    # The actor reading must be the engine's real synced step. The harness guarantees at
    # least one sync, so a zero here means the callable is a stub rather than a reading.
    actor = captured["actor_ckpt_step_fn"]()
    assert actor == pool.step_calls[-1] and actor > 0, (
        f"actor_ckpt_step_fn returned {actor}; expected the live synced step "
        f"{pool.step_calls[-1]}"
    )


@pytest.mark.parametrize("armed", [True, False])
def test_the_composed_monitor_cfg_carries_the_declared_arming(
    tmp_path, monkeypatch, smoke_run_config, armed
):
    """The arming value must survive the WHOLE chain, not just its first arrow.

    WPAX REVIEW-impl found the one live resurrection route this card left open: the
    registry chain `resolve_monitor_config -> build_run_safety ->
    ActorLagSpec.abort_enabled` was pinned at the FIRST arrow only. A disarm written into
    `compose_run` after `_resolve_monitor_cfg` returns left the suite 1725-green while an
    ARMED run5 config reached `build_run_safety` as `False` — the silent disarm this card
    exists to make impossible, surviving one hop past where anything was looking.

    This is the same LAW-07 / R4 phantom-gate class as MF-1 above (a gate input pinned by
    nothing), one arrow further down, and it matters more since WPAX `0ef05ff` armed the
    abort for run5.

    BOTH directions are driven deliberately: the pin is on the TRANSPORT, not on a value.
    Hardcoding either `True` or `False` anywhere in the chain fails exactly one arm — an
    assertion that only checked `is True` would be satisfied by a hardcoded `True`.
    """
    captured, _pool, _trainer = _compose_capturing_lag_fns(
        tmp_path, monkeypatch, smoke_run_config, abort_enabled=armed)
    assert captured["monitor_cfg"].actor_lag_abort_enabled is armed, (
        f"config declared actor_lag_abort_enabled={armed} but build_run_safety received "
        f"{captured['monitor_cfg'].actor_lag_abort_enabled} — the arming was lost in transit"
    )


@pytest.mark.parametrize("armed", [True, False])
def test_the_REAL_build_run_safety_carries_the_declared_arming_into_ActorLagSpec(
    tmp_path, smoke_run_config, armed
):
    """The LAST arrow of the arming chain, driven through the REAL builder (RED-TEAM F-1).

    The pin above monkeypatches `build_run_safety` and asserts on the kwarg it RECEIVES, so
    it covers `config -> resolve_monitor_config -> the build_run_safety kwarg` and stops
    there. The arrow both consumer registries name LAST —
    `build_run_safety -> ActorLagSpec.abort_enabled`, at `subsystems.py`'s
    `ActorLagSpec(abort_enabled=cfg.actor_lag_abort_enabled)` — is outside it, and was
    measured to have exactly one producer test, which hand-builds an ARMED `MonitorConfig`
    and asserts a fire. Hardcoding `abort_enabled=True` at that site therefore left the
    whole suite green while arming the exit-45 hard abort for all four minted configs that
    deliberately ship it DISARMED — a healthy-run false 45 on every non-production config
    (WPUF F-3's failure mode), through the arrow this card declared closed.

    So this test does what the one above cannot: NO monkeypatch on `build_run_safety`. It
    composes the monitor config from a REAL minted `RunConfig` through the production
    resolver, hands it to the REAL builder, and reads the arming back off the
    `ActorLagSpec` the watchdog actually holds.

    BOTH directions are parametrized for the same reason as above, and here the reason is
    load-bearing rather than stylistic: a hardcoded `True` at the wiring site fails ONLY the
    `[False]` arm and a hardcoded `False` fails ONLY the `[True]` arm. An assertion on one
    value would be satisfied by the constant it is asserting. The assertion is written
    against `monitor_cfg`'s own field, not against `armed`, because the pin is on the
    TRANSPORT; the `armed` assertion beside it closes the loop back to the config.

    The watchdog is left UNSTARTED and unarmed — this reads composition, not behaviour, and
    a started daemon thread in a unit test is a flake surface. The fire path from an armed
    spec is the frozen `test_build_run_safety_wires_actor_lag_from_monitor_config`'s job.
    """
    cfg = smoke_run_config(
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _STOP_STEP},
        monitor={"actor_lag_threshold_steps": _STOP_STEP - 1,
                 "actor_lag_abort_enabled": armed},
    )
    monitor_cfg = mantis.run._resolve_monitor_cfg(cfg)
    assert monitor_cfg.actor_lag_abort_enabled is armed, (
        "harness precondition: the resolver must already carry the declared arming (that "
        "arrow is pinned above; if THIS fails, the failure is upstream of the subject)"
    )

    run_safety = build_run_safety(
        log_dir=tmp_path, run_id="lag_arrow",
        buffer=SimpleNamespace(save_to_path=lambda p: None),
        buffer_persist_path=tmp_path / "replay_buffer.bin",
        wired_sources=list(HEARTBEAT_SOURCES),
        monitor_cfg=monitor_cfg,
        actor_ckpt_step_fn=lambda: 0, learner_step_fn=lambda: 0,
        exit_fn=lambda code: None,
    )

    spec = run_safety.watchdog._actor_lag
    assert spec is not None, (
        "build_run_safety built a watchdog with NO ActorLagSpec: the actor-lag invariant "
        "is not merely disarmed, it is absent"
    )
    assert spec.abort_enabled is monitor_cfg.actor_lag_abort_enabled, (
        f"monitor_cfg said actor_lag_abort_enabled="
        f"{monitor_cfg.actor_lag_abort_enabled} but the composed ActorLagSpec carries "
        f"{spec.abort_enabled} — the arming is HARDCODED at the wiring site, not "
        "transported. A constant here arms exit-45 on every config that ships it disarmed "
        "(or disarms run5, which ships it ARMED)"
    )
    assert spec.abort_enabled is armed, (
        f"the config declared actor_lag_abort_enabled={armed}; the ActorLagSpec the "
        f"watchdog holds says {spec.abort_enabled}"
    )
    assert spec.threshold_steps == monitor_cfg.actor_lag_threshold_steps, (
        f"the sibling field on the same dataclass is also transport, not a constant: "
        f"monitor_cfg says {monitor_cfg.actor_lag_threshold_steps}, spec says "
        f"{spec.threshold_steps}"
    )
