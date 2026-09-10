"""ORACLE — the composition root (`src/mantis/run.py`).

Sits at the tests/ TOP LEVEL, mirroring a module deliberately ABOVE both `mantis.train` and
`mantis.eval`. Covers the pool-then-watchdog start order, the `wired_sources` declaration, the
never-started-pool `on_drained` closure, and the `train -> eval` lazy-import ban, which passes
GREEN by vacancy while `mantis.eval` does not exist.

>300 justify (R8): the monitor-config producer test, the drivable fakes and the re-validation
pins are folded in here rather than given their own files — same subject, and the bar on
cross-test imports means each new file would fork another copy of the same fakes.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run  # noqa: F401 — RED-at-import anchor: this module does not exist yet
from mantis.config.resolve.composition import (
    UnvalidatedConfigError,
    require_run_config,
    revalidate_run_config,
)
from mantis.config.schema import RunConfig
from mantis.monitor.config import MonitorConfig
from mantis.train.coordinator.config import StepCoordinatorConfig

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "mantis"

#: `stop_step` is config-authored, so every `compose_run` call below drives a REAL bounded
#: burst. The reachability validator spans all three step-clock knobs, so they are co-overridden
#: together; 3 is the smallest legal run at cadence 1.
_DRIVE_STEPS = 3


def _bounded(smoke_run_config, *, eval_enabled: bool = False, **monitor_over):
    """A REAL minted `RunConfig`, bounded so a drive terminates: the strict gate means no bare
    namespace reaches this root, so smoke runs get smoke CONFIGS. `eval_enabled` is the
    CONFIG's fact — `compose_run` has no such parameter — so each drive declares its posture
    here rather than at the call."""
    return smoke_run_config(
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               # WPTS/TD-1: the drive runs the real graph route; minted 256 batch is drag.
               "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1, **monitor_over},
        eval_enabled=eval_enabled,
    )


#: The UNPATCHED production builder, captured at import so the patch below can delegate to
#: it without re-entering itself (WPMINT Phase K-A stage 0).
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


def _no_terminal_eval_config(**kwargs) -> StepCoordinatorConfig:
    """The ONE builder patch the `eval_enabled=True` drives still need: the production builder
    defaults `terminal_eval_enabled=True`, so `close_out` runs a terminal eval round that
    reaches a `.arch` read on a fake model, and that knob has no config key. A ONE-KNOB DELTA
    over the real builder, so `**kwargs` forwards every CONFIG-AUTHORED value untouched —
    `stop_step` above all, or a patched builder dictating run length would hide it."""
    return dataclasses.replace(_PRODUCTION_BUILDER(**kwargs), terminal_eval_enabled=False)


def _patch_eval_side(monkeypatch, capture: dict | None = None):
    """Fake the eval pipeline AND the anchor seed for an `eval_enabled=True` drive. Both are
    harness, not subject: `run_training_loop` seeds the anchor from `trainer.model`, which
    reads `.arch` off it and blows up on any fake model."""
    import mantis.train.anchor as _anchor

    def _fake_build_eval_pipeline(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return SimpleNamespace(
            run_evaluation=lambda *a, **k: {"kicked": False, "reason": None},
            poll_completed=lambda: None, drain_pending=lambda: None,
            apply_gate_decision=lambda *a, **k: None, stop=lambda: None,
        )

    monkeypatch.setattr(mantis.run, "build_eval_pipeline", _fake_build_eval_pipeline)
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _no_terminal_eval_config)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )


# ── census helpers (operate on today's tree; no import of not-yet-existing modules) ──────
def _train_sources() -> list[Path]:
    return sorted((_SRC / "train").rglob("*.py"))


def test_no_train_module_imports_eval_even_lazily() -> None:
    """Token-level census over EVERY file under `src/mantis/train`: no `mantis.eval` substring
    anywhere, top-level OR inside any function body — a lazy import is exactly what a substring
    scan catches where an AST top-level walk would not. GREEN TODAY BY VACANCY, so this is a
    forward-held regression guard rather than an oracle for unbuilt behaviour."""
    violations: list[str] = []
    for path in _train_sources():
        text = path.read_text()
        if "mantis.eval" in text or "from mantis import eval" in text:
            violations.append(str(path.relative_to(_SRC)))
    assert violations == [], f"train/** must never reference mantis.eval: {violations}"


# ── fakes shared by the compose_run tests (built only when mantis.run is importable) ─────
class _OrderSpy:
    """Records the name of every call, in order, across multiple collaborators."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, name: str):
        def _fn(*a, **k):
            self.calls.append(name)
        return _fn


class FakeWatchdog:
    def __init__(self, order: _OrderSpy) -> None:
        self._order = order
        self.disarm_staleness = order.record("watchdog.disarm_staleness")

    def start(self) -> None:
        self._order.calls.append("watchdog.start")


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class FakePoolNeverStarted:
    """Models the real hazard: `WorkerPool.stop()` joins the inference server, and
    `Thread.join` on a never-started thread raises `RuntimeError`, so only a caller that GUARDS
    on "was start() ever called" may call `.stop()` safely. Also DRIVABLE, because every
    compose_run call here runs a real burst: `games_completed` yields one fresh game per read,
    so each `step()` runs exactly one burst."""

    def __init__(self, order: _OrderSpy | None = None) -> None:
        self._order = order
        self._started = False
        self._games = 0
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self.sync_payloads: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self._started = True
        if self._order is not None:
            self._order.calls.append("pool.start")

    def stop(self) -> None:
        if not self._started:
            raise RuntimeError("cannot join thread before it is started")
        if self._order is not None:
            self._order.calls.append("pool.stop")

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


class _DrivableTrainer:
    """WPTS/TD-1 re-point (R90a): conforms to the DECLARED seam — typed entry points +
    `device`; the dead `train_step` fake is gone with the card."""

    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self.inference_sd: dict = {}

    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def save_checkpoint(self, loss_info) -> None:
        return None


class _DrivableBuffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None:
        return None

    def save_to_path(self, p) -> None:
        return None


def test_compose_run_publishes_its_boot_identity_first_through_the_one_authority(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """`compose_run` must emit `run_boot_identity` carrying the identity sha of the config it
    actually runs, and it must land BEFORE `pool.start` — the witness has to exist even if the
    burst later wedges. The sha is asserted against the ONE authority, so a second hashing
    expression cannot drift in silently."""
    from mantis.config.loader import config_identity_sha256

    mantis_run = mantis.run
    order = _OrderSpy()
    pool = FakePoolNeverStarted(order)
    watchdog = FakeWatchdog(order)
    emitted: list[dict] = []

    def _fake_build_run_safety(**kwargs):
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: (emitted.append(e),
                                                 order.calls.append("sink.emit:" + e.get("event", "")))[0]),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=watchdog,
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis_run, "build_run_safety", _fake_build_run_safety)
    config = _bounded(smoke_run_config)
    mantis_run.compose_run(
        config=config, trainer=_DrivableTrainer(),
        pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    identity = [e for e in emitted if e.get("event") == "run_boot_identity"]
    assert len(identity) == 1, f"exactly one boot-identity event, got {len(identity)}"
    assert identity[0]["config_sha256"] == config_identity_sha256(config), (
        "the published identity must be the ONE authority's hash of the composed config"
    )
    first_identity = order.calls.index("sink.emit:run_boot_identity")
    assert first_identity < order.calls.index("pool.start"), (
        f"the identity witness must exist before anything can wedge: {order.calls[:6]}"
    )


def test_compose_run_calls_build_run_safety_once_and_starts_watchdog_after_pool(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """Spy order: pool.start() then watchdog.start(), which the composition root is the ONE
    place to enforce; `build_run_safety` must be called exactly once."""
    mantis_run = mantis.run
    order = _OrderSpy()
    pool = FakePoolNeverStarted(order)
    watchdog = FakeWatchdog(order)
    build_calls = {"n": 0}

    def _fake_build_run_safety(**kwargs):
        build_calls["n"] += 1
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=watchdog,
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis_run, "build_run_safety", _fake_build_run_safety)
    handles = mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=_DrivableTrainer(),
        pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert build_calls["n"] == 1, "build_run_safety must be called exactly once"
    assert "pool.start" in order.calls and "watchdog.start" in order.calls, (
        f"both pool.start and watchdog.start must fire: {order.calls}"
    )
    assert order.calls.index("pool.start") < order.calls.index("watchdog.start"), (
        f"pool must start BEFORE the watchdog (subsystems.py contract): {order.calls}"
    )
    assert handles is not None


def test_wired_sources_include_eval_round_iff_pipeline_built(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """`wired_sources` passed to `build_run_safety` includes "eval_round" iff an eval
    pipeline is actually built (`eval_enabled=True`); absent when `eval_enabled=False`."""
    mantis_run = mantis.run
    seen: dict[str, list[str]] = {}

    def _make_fake_build_run_safety(key: str):
        def _fake(**kwargs):
            seen[key] = list(kwargs.get("wired_sources", []))
            return SimpleNamespace(
                sink=SimpleNamespace(emit=lambda e: None),
                registry=SimpleNamespace(beat=lambda s: None),
                watchdog=FakeWatchdog(_OrderSpy()),
                heartbeat=lambda s: None,
            )
        return _fake

    monkeypatch.setattr(mantis_run, "build_run_safety", _make_fake_build_run_safety("with_eval"))
    _patch_eval_side(monkeypatch)
    mantis_run.compose_run(
        config=_bounded(smoke_run_config, eval_enabled=True), trainer=_DrivableTrainer(),
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert "eval_round" in seen["with_eval"], (
        f"eval_enabled=True must declare eval_round wired: {seen['with_eval']}"
    )

    monkeypatch.setattr(mantis_run, "build_run_safety", _make_fake_build_run_safety("no_eval"))
    mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=_DrivableTrainer(),
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert "eval_round" not in seen["no_eval"], (
        f"eval_enabled=False must NOT declare eval_round wired: {seen['no_eval']}"
    )


def test_close_out_with_never_started_pool_does_not_raise() -> None:
    """The never-started-pool closure. The risk is OPEN-BY-VACANCY at HEAD — no in-repo caller
    passes `on_drained=pool.stop` — so `_stop_pool_if_start_attempted` must be the first real
    caller AND ship the guard. Its predicate is "start was CALLED", not "start RETURNED", because
    a `start()` that raises partway leaves earlier sub-starts alive and a set-after flag would
    report a silent worker leak. Mutation arm: an UNGUARDED closure DOES raise here, which is
    what proves the fake models the real hazard rather than a tautology."""
    mantis_run = mantis.run
    pool = FakePoolNeverStarted()  # never call .start()

    guarded = mantis_run._stop_pool_if_start_attempted(pool, start_attempted=False)
    guarded()  # must NOT raise

    with pytest.raises(RuntimeError):
        pool.stop()  # the mutation arm: calling stop() unconditionally DOES raise


def test_sink_and_heartbeat_are_threaded_to_pipeline_and_coordinator(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """The sink and heartbeat built by `build_run_safety` must reach BOTH the eval pipeline (if
    built) and the `StepCoordinator`, asserted via identity spies through the injection points."""
    mantis_run = mantis.run
    sink = SimpleNamespace(emit=lambda e: None)
    heartbeat_fn = lambda s: None  # noqa: E731

    def _fake_build_run_safety(**kwargs):
        return SimpleNamespace(
            sink=sink, registry=SimpleNamespace(beat=heartbeat_fn),
            watchdog=FakeWatchdog(_OrderSpy()), heartbeat=heartbeat_fn,
        )

    monkeypatch.setattr(mantis_run, "build_run_safety", _fake_build_run_safety)
    captured: dict[str, Any] = {}
    _patch_eval_side(monkeypatch, captured)

    mantis_run.compose_run(
        config=_bounded(smoke_run_config, eval_enabled=True), trainer=_DrivableTrainer(),
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert captured.get("sink") is sink, "the eval pipeline must receive the SAME sink"
    assert captured.get("heartbeat") is heartbeat_fn, (
        "the eval pipeline must receive the SAME heartbeat fn build_run_safety produced"
    )


# ── F-R-P2B-2 — the TRAINER's sink is composed live, not authored-and-dropped ────────────
def test_trainer_deferred_sink_is_bound_to_run_safety_sink(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """`compose_run` binds a trainer-carried `_DeferredSink` to `run_safety.sink` — the SEPARATE
    bind beside the pool's — asserted by identity through the real adapter. Killer: remove the
    trainer bind and the adapter still holds its pre-bind `NullEventSink`."""
    mantis_run = mantis.run
    sink = SimpleNamespace(emit=lambda e: None)

    def _fake_build_run_safety(**kwargs):
        return SimpleNamespace(
            sink=sink, registry=SimpleNamespace(beat=lambda s: None),
            watchdog=FakeWatchdog(_OrderSpy()), heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis_run, "build_run_safety", _fake_build_run_safety)
    trainer = _DrivableTrainer()
    trainer._sink = mantis_run._DeferredSink()  # the production adapter, on the drivable fake

    mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=trainer,
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert trainer._sink._inner is sink, (
        "the trainer's _DeferredSink must be bound to run_safety.sink by compose_run — an "
        "unbound adapter is the F-R-P2B-2 drop (periodic_checkpoint_save authored, "
        "unit-tested, and absent from every production stream)"
    )


def test_build_run_collaborators_does_not_build_the_trainer_with_sink_none() -> None:
    """SOURCE arm applied to the TRAINER's construction site: the `init_trainer(...)` call in
    `mantis/run.py` passes a real `sink=`, not `None` and not omitted. This bites the exact
    defect in the default tier — `run.py` composing `init_trainer` with `sink=None` while the
    pool got the deferred adapter, so the periodic-checkpoint emission was dropped in
    production. Killer: revert to `sink=None`."""
    import ast

    tree = ast.parse((_SRC / "run.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "init_trainer"
    ]
    assert len(calls) == 1, (
        f"run.py must have exactly ONE init_trainer call site (the builder); found {len(calls)}"
    )
    sink_kw = [kw for kw in calls[0].keywords if kw.arg == "sink"]
    assert sink_kw, "the init_trainer call must pass sink= explicitly (F-R-P2B-2)"
    val = sink_kw[0].value
    # Stronger than an `is not None` refusal: `sink=NullEventSink()` or an adapter nobody binds
    # would satisfy a bare not-None pin while still dropping every event. The site must
    # construct the SAME late-binding adapter the pool's construction does.
    assert isinstance(val, ast.Call) and isinstance(val.func, ast.Name) \
        and val.func.id == "_DeferredSink", (
        "the production trainer's sink must be a _DeferredSink(...) construction — a dead "
        "sink (None, NullEventSink, an unbound stand-in) re-drops every trainer-side event "
        "(periodic_checkpoint_save, trainer_step, aux_chain_loss) in every production run "
        f"(F-R-P2B-2); got {ast.dump(val)}"
    )


def test_deferred_sink_pre_bind_drops_by_design_and_post_bind_delivers() -> None:
    """The named limitation left in place, recorded as deliberate: `_DeferredSink` buffers
    NOTHING pre-bind, so an emission before `bind()` goes to the `NullEventSink` and is GONE.
    On the resume path `init_trainer` runs before `run_safety.sink` exists, so resume-time
    warnings do not reach the production stream. A pre-bind replay buffer was considered and
    NOT added: replayed rows would land ahead of the boot-identity witness and break the
    identity-FIRST stream contract on every resumed run. This pin makes the drop a DECISION a
    future reader can find rather than an accident."""
    mantis_run = mantis.run
    delivered: list[dict] = []
    adapter = mantis_run._DeferredSink()

    adapter.emit({"event": "pre_bind_row"})  # must not raise; must not deliver
    adapter.bind(SimpleNamespace(emit=delivered.append))
    assert delivered == [], "pre-bind emissions are dropped, not buffered (by design)"

    adapter.emit({"event": "post_bind_row"})
    assert [e["event"] for e in delivered] == ["post_bind_row"], (
        "post-bind, the adapter delivers to the bound sink"
    )


# ── STOP CANDIDATE 5 — MonitorConfig production wiring (REV1, DESIGN_P3.md §5.0) ─────────
def test_compose_run_resolves_monitor_cfg_from_a_real_config_monitor_section(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """A real `config.monitor` flows through compose_run's own `resolve_monitor_config` into the
    MonitorConfig handed to `build_run_safety` and `StepCoordinator`, proven via a NON-DEFAULT
    threshold threaded end to end. The negative control was DELETED, not inverted: the
    absent-section fallback to a bare `MonitorConfig()` ceased to exist when the strict gate
    landed, and it had silently DISARMED a hard abort the production config ships armed."""
    mantis_run = mantis.run
    captured: dict[str, Any] = {}

    def _fake_build_run_safety(**kwargs):
        captured["monitor_cfg"] = kwargs.get("monitor_cfg")
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=FakeWatchdog(_OrderSpy()), heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis_run, "build_run_safety", _fake_build_run_safety)
    cfg = _bounded(smoke_run_config, alert_entropy_min=2.75)
    mantis_run.compose_run(
        config=cfg, trainer=_DrivableTrainer(), pool=FakePoolNeverStarted(),
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert captured["monitor_cfg"] is not None
    assert captured["monitor_cfg"].alert_entropy_min == 2.75
    assert captured["monitor_cfg"] != MonitorConfig()  # not the bare-default fallback


# ── RED-TEAM F-3 — the composition root RE-VALIDATES, it does not merely type-check ───────
def test_compose_run_refuses_a_model_copy_the_LOADER_would_reject(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """A genuine `RunConfig` that never re-ran its cross-field validators.

    `model_copy(update=...)` is the idiomatic pydantic-v2 rig: the real class, every typed read
    succeeding, the strict gate accepting it — but validators that never re-ran, so it can carry
    a sync cadence the run never reaches. Measured before this pin: 20 real training steps, ONE
    actor sync, and a lag threshold a 20-step run cannot trip.

    So the closure cannot be a type check. Three parts: the type gate still ACCEPTS the rigged
    config; `compose_run` RAISES, naming the field the loader names; and nothing was built or
    driven, since a re-validation firing after construction would leave a started pool behind.
    """
    base = smoke_run_config()
    rigged = base.model_copy(update={
        "train": base.train.model_copy(update={"max_train_steps": 20,
                                               "actor_sync_cadence_steps": 1000}),
        "monitor": base.monitor.model_copy(update={"actor_lag_threshold_steps": 5000}),
    })

    assert require_run_config(rigged, caller="compose_run") is rigged, (
        "harness precondition: the TYPE gate accepts this object — it is a real RunConfig. "
        "If this ever fails, the type gate grew a validation check and the test below is no "
        "longer measuring re-validation"
    )

    def _must_not_be_called(**_kwargs):
        raise AssertionError(
            "build_run_safety was constructed for a config the loader would reject — "
            "re-validation must happen before any subsystem exists"
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _must_not_be_called)
    trainer, pool = _DrivableTrainer(), FakePoolNeverStarted()

    with pytest.raises(UnvalidatedConfigError, match="must be < train.max_train_steps"):
        mantis.run.compose_run(
            config=rigged, trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        )

    assert trainer.step == 0, (
        f"the rigged config drove {trainer.step} training steps before being refused; the "
        "frozen actor this closes is a RUN, so refusing it after the run is no refusal"
    )
    assert pool.sync_payloads == [], "no run may have started, so no sync may have happened"


def test_revalidation_does_not_over_reject_a_good_config_or_a_validated_subclass(
    smoke_run_config,
) -> None:
    """The other direction, and not optional: a re-validation that rejected everything would
    satisfy the test above while breaking every real run. A loaded minted config must round-trip
    to an EQUAL config, not merely a non-raising one, and a validated SUBCLASS must survive as
    itself, or the type gate and the re-validation disagree about the same object."""
    good = _bounded(smoke_run_config)
    assert revalidate_run_config(good, caller="probe") == good, (
        "re-validating a config the loader produced must return an EQUAL config; a "
        "difference here means composition drives from something other than the config"
    )

    class _Sub(RunConfig):
        pass

    subclass = _Sub.model_validate(good.model_dump())
    out = revalidate_run_config(subclass, caller="probe")
    assert type(out) is _Sub, (
        f"re-validation downgraded a validated subclass to {type(out).__name__}; the type "
        "gate admits subclasses (LSP) and this hop must not undo that"
    )


# ══ AUDIT-1 F-32 / R334(c) SHAPE A — the launch pin DERIVES from `identity.warm_start` ═══
def _capture_anchor(monkeypatch) -> dict:
    """The anchor stub, but CAPTURING: the pin's consumer is `resolve_anchor`, so asserting
    there proves the value crossed every hop rather than that one call site spells the kwarg."""
    import mantis.train.anchor as _anchor

    seen: dict = {}

    def _fake_resolve_anchor(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(best_model=None, best_model_step=None,
                               best_model_path=None, representation="graph")

    monkeypatch.setattr(_anchor, "resolve_anchor", _fake_resolve_anchor)
    return seen


#: A row the RESOLVER accepts without touching the filesystem — it parses and never stats, and
#: the pin's derivation happens before anything opens the artifact. What this drive owns is
#: whether the row's value reaches the guard at all.
_WARM_START_ROW = {"checkpoint": "/nonexistent/bc_of_record.ckpt", "net_hash": "b" * 64}


def test_the_launch_pin_reaches_the_anchor_resolver_from_the_warm_start_row(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """ONE source for the launch pin, with no hand-synced twin and no second config key. The
    audit called `verify_launch_anchor_pin` a refusal nobody could reach, because `run.py`
    never set the value; this is the row that says it does now."""
    _patch_eval_side(monkeypatch)
    seen = _capture_anchor(monkeypatch)
    # RE-VALIDATED, not `model_copy`-patched: a raw dict pushed onto a typed field serialises
    # with a warning and would let this drive assert on a config shape the loader never produces.
    base = _bounded(smoke_run_config, eval_enabled=True).model_dump()
    base["identity"]["warm_start"] = _WARM_START_ROW
    config = RunConfig.model_validate(base)
    assert config.identity.warm_start is not None
    mantis.run.compose_run(
        config=config, trainer=_DrivableTrainer(), pool=FakePoolNeverStarted(_OrderSpy()),
        buffer=mk_graph_buffer(n_records=32), log_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert seen.get("expected_anchor_sha256") == _WARM_START_ROW["net_hash"], (
        "the launch pin must be the warm-start row's own net hash; got "
        f"{seen.get('expected_anchor_sha256')!r}"
    )


def test_no_warm_start_row_means_NO_LAUNCH_PIN(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """The other direction, and it is the one that keeps every pre-row config launchable: an
    absent row is `None`, never a guess and never a pin nothing can satisfy."""
    _patch_eval_side(monkeypatch)
    seen = _capture_anchor(monkeypatch)
    config = _bounded(smoke_run_config, eval_enabled=True)
    assert config.identity.warm_start is None, "the minted smoke config already carries a row"
    mantis.run.compose_run(
        config=config, trainer=_DrivableTrainer(), pool=FakePoolNeverStarted(_OrderSpy()),
        buffer=mk_graph_buffer(n_records=32), log_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert "expected_anchor_sha256" in seen, "the drive never reached the anchor resolver"
    assert seen["expected_anchor_sha256"] is None
