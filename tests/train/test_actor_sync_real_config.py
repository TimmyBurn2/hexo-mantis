"""`compose_run` with a `RunConfig` whose own payload mints the cadence under test.

Two axes stay pinned because no test can vary them: `_step_coordinator_config` is
monkeypatched and `build_run_safety` is faked.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mantis.run
from mantis.config.schema.core import RunConfig
from _drivable import DrivablePoolStub, DrivableTrainerStub, with_deltas

_CADENCE = 2
_STOP_STEP = 6


def _frozen_payload():
    """Load the frozen schema oracle's payload builder by path, since `tests` is not a package.

    Built from the payload rather than a shipped config so this pins the resolver's behaviour
    and not one config's current values.
    """
    path = Path(__file__).resolve().parents[1] / "config" / "test_actor_sync_schema.py"
    spec = importlib.util.spec_from_file_location("_frozen_schema_for_fa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._payload


#: The unpatched production builder, captured at import so the patch below can delegate to it
#: without re-entering itself.
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


#: The production builder bounded so no eval round runs; config-authored values pass through.
_bounded_config = with_deltas(_PRODUCTION_BUILDER, terminal_eval_enabled=False, eval_interval=1000,
                              log_interval=1, stop_step=_STOP_STEP)


def _real_run_config() -> RunConfig:
    payload = _frozen_payload()(
        # The reachability bound binds on the run length, so the threshold must fit inside
        # `cadence < threshold < max_train_steps`.
        train_over={"actor_sync_cadence_steps": _CADENCE, "max_train_steps": _STOP_STEP},
        monitor_over={"actor_lag_threshold_steps": _CADENCE + 2},
    )
    # The frozen payload mints `eval_enabled: True`, this file's production posture.
    assert payload["eval_enabled"] is True
    return RunConfig(**payload)


def _drive(monkeypatch, *, eval_enabled: bool = True):
    """Compose with a real RunConfig; return (captured build_run_safety kwargs, pool, trainer).

    `eval_enabled` is a config fact — `compose_run` has no such parameter — so the posture
    travels on the payload `_real_run_config` builds.
    """
    import mantis.train.anchor as _anchor

    captured: dict = {}
    pool, trainer = DrivablePoolStub(game_per_read=True), DrivableTrainerStub()

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _capture)
    # What this patch buys is `terminal_eval_enabled=False`: the terminal eval round would
    # reach eval/snapshot.py's `.arch` read on a fake model, and that knob has no config key.
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _bounded_config)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="graph"),
    )
    return captured, pool, trainer, _real_run_config()


def test_sync_follows_the_configured_cadence_under_a_real_config(tmp_path, monkeypatch, mk_graph_buffer):
    """Kill a resolver unit-slip on its own: a `* 1000` slip makes the cadence unreachable
    inside the run, so the actor takes one unconditional first sync and then freezes."""
    captured, pool, trainer, cfg = _drive(monkeypatch)
    mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert trainer.step >= _STOP_STEP - 1, "harness precondition: the run actually stepped"
    expected_min = trainer.step // _CADENCE
    assert len(pool.sync_payloads) >= expected_min, (
        f"actor synced {len(pool.sync_payloads)}x over {trainer.step} steps at cadence "
        f"{_CADENCE}; expected at least {expected_min}. A single sync then silence is the "
        f"frozen actor this WP removed"
    )
    gaps = [b - a for a, b in zip(pool.step_calls, pool.step_calls[1:], strict=False)]
    assert all(g <= _CADENCE for g in gaps), (
        f"sync gaps {gaps} exceed the configured cadence {_CADENCE}"
    )


