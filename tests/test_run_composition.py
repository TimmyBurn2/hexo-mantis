"""⊕ WP11-A — the composition root (src/mantis/run.py, MUST-FIX 4: RELOCATE ABOVE train/eval).

RED-at-import until IMPL writes `mantis.run` (a top-level module, NOT under `mantis.train`).
ORACLE-FIRST (⊕): the top-level `import mantis.run` raises ModuleNotFoundError before any
port code exists — that failure carries every test below except the two pure-census tests
that operate on today's tree (`test_no_train_module_imports_eval_even_lazily` passes GREEN
today by vacancy: `mantis.eval` does not exist yet, so no `train/**` source can reference it
— a regression guard, not an oracle for unbuilt behavior).

Sits at the tests/ TOP LEVEL (mirrors `src/mantis/run.py`, which is deliberately ABOVE both
`mantis.train` and `mantis.eval` — DESIGN §a.4/§c.6, MUST-FIX 4). Covers: the pool-then-
watchdog start order (subsystems.py contract), the `wired_sources` declaration, the item-11
`on_drained` never-started-pool closure (WP-SP DISPATCH_LOG.md:65-66 — open-by-vacancy at
HEAD, closed here by `_stop_pool_if_started`), the train->eval lazy-import ban (repo_design
§2, un-weakened by the new `run` node). The old actor-lag ABSENCE census (E10) is
discharged by WP-UNFREEZE: the run.py half is deleted (the mechanism lawfully lives
there now) and the eval half survives as the frozen S5 census in
tests/train/test_actor_sync_isolation.py::test_no_actor_lag_mechanism_in_eval.

>300 justify (R8, WPSC Phase 3 SC-B4): STOP CANDIDATE 5's MonitorConfig production-wiring
producer test (DESIGN_P3.md §5.0) is folded in here rather than a new file — same subject
(how compose_run reaches the monitor section) as this file's existing coverage; pushed the
file past the 300-line soft cap. WPAX Phase S added the drivable fakes below for the same
reason (R5 bars cross-test imports, so a second file would fork a third copy of them), and
the WPAX RED-TEAM fix pass added the F-3 re-validation pins at the end for the third time
over: their subject is what `compose_run` may be composed from, they drive the same fakes,
and the alternative is a fourth copy of them.
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

#: WPAX S-4: `stop_step` is config-authored now, so every `compose_run` call below drives a
#: REAL bounded burst instead of terminating at the builder's `stop_step=0`. The reachability
#: validator spans all three step-clock knobs (`cadence < threshold < max_train_steps`), so
#: they are co-overridden together; 3 is the smallest legal run at cadence 1.
_DRIVE_STEPS = 3


def _bounded(smoke_run_config, **monitor_over):
    """A REAL minted `RunConfig`, bounded so a drive terminates (WPAX S-1: the strict gate
    means no `SimpleNamespace()` reaches this root any more — smoke runs get smoke CONFIGS)."""
    return smoke_run_config(
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               # WPTS/TD-1: the drive runs the real graph route; minted 256 batch is drag.
               "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1, **monitor_over},
    )


#: The UNPATCHED production builder, captured at import so the patch below can delegate to
#: it without re-entering itself (WPMINT Phase K-A stage 0).
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


def _no_terminal_eval_config(**kwargs) -> StepCoordinatorConfig:
    """The ONE builder patch the `eval_enabled=True` drives below still need. The production
    builder defaults `terminal_eval_enabled=True`, so `close_out` runs a terminal eval round
    that reaches `eval/snapshot.py`'s `.arch` read on a fake model. That knob has NO config
    key — it is one of the hardcoded `_step_coordinator_config` knobs owned by
    R-TRAINCONFIG-SCHEMA / CARD-COORD-KNOBS (R78).

    WPMINT Phase K-A stage 0: this is now a ONE-KNOB DELTA over the real builder, not a
    24-kwarg restatement of it. The old shape had to be edited by every change to the
    dataclass and, worse, silently pinned 23 knobs to values a reader would take for the
    production posture. `**kwargs` forwards every CONFIG-AUTHORED value untouched — a
    harness builder that swallowed one would be a stand-in dictating a config fact, and
    `stop_step` in particular must keep arriving from `compose_run`'s resolver (WPAX S-4),
    or a patched builder dictating run length would hide the knob.
    """
    return dataclasses.replace(_PRODUCTION_BUILDER(**kwargs), terminal_eval_enabled=False)


def _patch_eval_side(monkeypatch, capture: dict | None = None):
    """Fake the eval pipeline AND the anchor seed for an `eval_enabled=True` drive. Both are
    harness, not subject: `run_training_loop` seeds the anchor from `trainer.model`, which
    reads `.arch` off it (`train/anchor.py`) and blows up on any fake model — the same patch,
    for the same reason, as `tests/train/test_actor_sync_real_config.py::_drive`."""
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
    """(MUST-FIX 4 pin) Token-level census over EVERY file under src/mantis/train: no
    `mantis.eval` / `from mantis import eval` substring anywhere, top-level OR inside any
    function body — a lazy/deferred import is exactly what a substring scan (not just an
    AST top-level walk) catches. GREEN TODAY BY VACANCY (mantis.eval does not exist yet, so
    nothing references it) — this is a forward-held regression guard, not RED-at-import; the
    §2 train-not-imports-eval ban must never regress once mantis.eval exists."""
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
    """Models the real hazard: `WorkerPool.stop()` -> `InferenceServer.join(timeout=5.0)`
    raises RuntimeError when the underlying thread was never started (pool.py:335;
    `threading.Thread.join` on a never-started thread raises `RuntimeError: cannot join
    thread before it is started`). Only a caller that GUARDS on "was start() ever called"
    (compose_run's `_stop_pool_if_started`) may call `.stop()` safely.

    WPAX Phase S: also DRIVABLE. `stop_step` is config-authored now, so every compose_run
    call in this file runs a real burst and the pool must carry the coordinator's read
    surface. `games_completed` yields one fresh game per read so each `step()` runs exactly
    one burst. The never-started guard above is unchanged — it is what
    `test_close_out_with_never_started_pool_does_not_raise` pins."""

    def __init__(self, order: _OrderSpy | None = None) -> None:
        self._order = order
        self._started = False
        self._games = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
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
    """F-B1 closure producer arm (WPCLEAN Phase RES; LAW-07): `compose_run` must emit
    `run_boot_identity` carrying `config_identity_sha256` of the config it actually runs,
    and it must land BEFORE `pool.start` — the witness has to exist even if the burst later
    wedges. The sha is asserted against the ONE authority so a second hashing expression
    cannot drift in silently (the preflight parent compares with the same function)."""
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
        eval_enabled=False,
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
    """Spy order: pool.start() -> watchdog.start() (subsystems.py contract; the composition
    root is the ONE place this order is enforced). `build_run_safety` must be called exactly
    once — patch it with a call-counting spy that returns fakes wired into the order spy."""
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
        eval_enabled=False,
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
        config=_bounded(smoke_run_config), trainer=_DrivableTrainer(),
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
    )
    assert "eval_round" in seen["with_eval"], (
        f"eval_enabled=True must declare eval_round wired: {seen['with_eval']}"
    )

    monkeypatch.setattr(mantis_run, "build_run_safety", _make_fake_build_run_safety("no_eval"))
    mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=_DrivableTrainer(),
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
    )
    assert "eval_round" not in seen["no_eval"], (
        f"eval_enabled=False must NOT declare eval_round wired: {seen['no_eval']}"
    )


def test_close_out_with_never_started_pool_does_not_raise() -> None:
    """The item-11 closure test. Verdict at HEAD (verified against drain.py, already shipped):
    the risk is OPEN-BY-VACANCY — no in-repo caller passes `on_drained=pool.stop` today, so
    nothing ever hits the real hazard (`InferenceServer.join(timeout=5.0)` raising on a
    never-started thread, pool.py:335). `mantis.run`'s `_stop_pool_if_started` must be the
    first real caller AND ship the guard: it calls `pool.stop()` only if `compose_run`'s own
    `pool_started` flag was set. Never-started here + the guard => no raise.

    Mutation arm: an UNGUARDED closure (calls `pool.stop()` unconditionally on a never-started
    pool) DOES raise — proving the fake pool models the real hazard, not a tautology."""
    mantis_run = mantis.run
    pool = FakePoolNeverStarted()  # never call .start()

    guarded = mantis_run._stop_pool_if_started(pool, pool_started=False)
    guarded()  # must NOT raise

    with pytest.raises(RuntimeError):
        pool.stop()  # the mutation arm: calling stop() unconditionally DOES raise


def test_sink_and_heartbeat_are_threaded_to_pipeline_and_coordinator(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """The `run_safety.sink` / `run_safety.heartbeat` built by `build_run_safety` must reach
    BOTH the eval pipeline (if built) and the `StepCoordinator` — asserted via identity spies
    threaded through `compose_run`'s injection points."""
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
        config=_bounded(smoke_run_config), trainer=_DrivableTrainer(),
        pool=FakePoolNeverStarted(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
    )
    assert captured.get("sink") is sink, "the eval pipeline must receive the SAME sink"
    assert captured.get("heartbeat") is heartbeat_fn, (
        "the eval pipeline must receive the SAME heartbeat fn build_run_safety produced"
    )


# ── STOP CANDIDATE 5 — MonitorConfig production wiring (REV1, DESIGN_P3.md §5.0) ─────────
def test_compose_run_resolves_monitor_cfg_from_a_real_config_monitor_section(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """A real `config.monitor` (MonitorSchemaConfig) flows through compose_run's own
    resolve_monitor_config call into the MonitorConfig handed to build_run_safety /
    StepCoordinator. Proven via a NON-DEFAULT threshold value threaded end to end (LAW-07
    mutation-shaped proof, not a did-not-crash check).

    WPAX Phase S DELETED this test's negative control,
    `test_compose_run_falls_back_to_bare_monitor_config_when_config_has_no_monitor_section`.
    Its subject — the absent-monitor-section fallback to a bare `MonitorConfig()` — ceased to
    exist when S-1's gate landed, and a bare `MonitorConfig()` carries
    `actor_lag_abort_enabled=False`, so that fallback silently DISARMED the hard abort
    `configs/run5.yaml` ships armed (ADJ-07). It was deleted rather than inverted: the
    inversion lives in `tests/test_run_strict_composition.py` as an eight-shape corpus, and
    an inverted copy here would be a second authority for one fact (LAW-03).
    """
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
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
    )
    assert captured["monitor_cfg"] is not None
    assert captured["monitor_cfg"].alert_entropy_min == 2.75
    assert captured["monitor_cfg"] != MonitorConfig()  # not the bare-default fallback


# ── RED-TEAM F-3 — the composition root RE-VALIDATES, it does not merely type-check ───────
def test_compose_run_refuses_a_model_copy_the_LOADER_would_reject(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """The twelfth route: a genuine `RunConfig` that never re-ran its cross-field validators.

    `RunConfig.model_copy(update=…)` is the idiomatic pydantic-v2 way to rig a config on a
    copy. The result is the real class, every typed read succeeds, and the strict gate
    accepts it — but the model validators never re-ran, so it can carry a sync cadence the
    run never reaches. Measured before this pin: 20 real training steps, ONE actor sync, and
    a lag threshold of 5000 that a 20-step run can never trip either. That is run3's frozen
    actor plus the "two knobs fail open together" defect the schema-level F-2 bound closed —
    reopened at the COMPOSITION level, with the loader refusing the identical payload.

    So the closure cannot be a type check, and this test asserts that in three parts:

      1. the type gate still ACCEPTS the rigged config — pinning that the type rule is not
         the instrument here, so a future "simplification" that folds the two rules together
         cannot be mistaken for this one;
      2. `compose_run` RAISES, naming the field the loader names;
      3. nothing was built and nothing was driven — the spy `build_run_safety` raises if
         called at all, and the trainer's step is still 0. A re-validation that fired AFTER
         the subsystems were constructed would leave a started pool behind.
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
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
        )

    assert trainer.step == 0, (
        f"the rigged config drove {trainer.step} training steps before being refused; the "
        "frozen actor this closes is a RUN, so refusing it after the run is no refusal"
    )
    assert pool.sync_payloads == [], "no run may have started, so no sync may have happened"


def test_revalidation_does_not_over_reject_a_good_config_or_a_validated_subclass(
    smoke_run_config,
) -> None:
    """The other direction, and it is not optional (LAW-07): a re-validation that rejected
    everything would satisfy the test above while breaking every real run.

    Two arms. A loaded minted config must round-trip to an EQUAL config — not merely a
    non-raising one, because a re-validation that silently dropped or re-defaulted a section
    would pass a truthiness check and quietly change what the run composes from. And a
    validated SUBCLASS must survive as itself: `require_run_config` admits it on LSP grounds,
    so a re-validation that downgraded it to the base class would make the two rules
    disagree about the same object.
    """
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
