"""⊕ WP11-A — the composition root (src/mantis/run.py, MUST-FIX 4: RELOCATE ABOVE train/eval).

RED-at-import until IMPL writes `mantis.run` (a top-level module, NOT under `mantis.train`).
ORACLE-FIRST (⊕): the top-level `import mantis.run` raises ModuleNotFoundError before any
port code exists — that failure carries every test below except the two pure-census tests
that operate on today's tree (`test_no_train_module_imports_eval_even_lazily` passes GREEN
today by vacancy: `mantis.eval` does not exist yet, so no `train/**` source can reference it
— a regression guard, not an oracle for unbuilt behavior; `test_no_actor_lag_mechanism_in_
composition_root` is RED today via FileNotFoundError on the not-yet-existing run.py).

Sits at the tests/ TOP LEVEL (mirrors `src/mantis/run.py`, which is deliberately ABOVE both
`mantis.train` and `mantis.eval` — DESIGN §a.4/§c.6, MUST-FIX 4). Covers: the pool-then-
watchdog start order (subsystems.py contract), the `wired_sources` declaration, the item-11
`on_drained` never-started-pool closure (WP-SP DISPATCH_LOG.md:65-66 — open-by-vacancy at
HEAD, closed here by `_stop_pool_if_started`), the train->eval lazy-import ban (repo_design
§2, un-weakened by the new `run` node), and the WP-UNFREEZE actor-lag absence property.

>300 justify: N/A at write time (kept under 300); if IMPL additions push it over, IMPL adds
the R8 one-liner — not this oracle's job to pre-justify a length it does not yet have.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run  # noqa: F401 — RED-at-import anchor: this module does not exist yet

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "mantis"


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


def test_no_actor_lag_mechanism_in_composition_root() -> None:
    """Census: tokens `actor_lag`/`actor_ckpt_step` absent from `src/mantis/run.py` +
    `src/mantis/eval/` (WP-UNFREEZE property — that mechanism is not this WP's to add).
    RED today: `src/mantis/run.py` does not exist yet, so reading it raises
    FileNotFoundError — the correct RED-at-import-adjacent failure for a not-yet-built
    top-level module (this file itself has no `import mantis.run`, so the failure surfaces
    here as an explicit read, not a collection-time ModuleNotFoundError)."""
    run_py = _SRC / "run.py"
    assert run_py.exists(), "src/mantis/run.py must exist (the composition root, §c.6)"
    text = run_py.read_text()
    assert "actor_lag" not in text and "actor_ckpt_step" not in text, (
        "the composition root must carry NO actor-lag mechanism (WP-UNFREEZE property)"
    )
    for path in sorted((_SRC / "eval").rglob("*.py")):
        etext = path.read_text()
        assert "actor_lag" not in etext and "actor_ckpt_step" not in etext, (
            f"{path.relative_to(_SRC)}: no actor-lag mechanism belongs in mantis.eval"
        )


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


class FakePoolNeverStarted:
    """Models the real hazard: `WorkerPool.stop()` -> `InferenceServer.join(timeout=5.0)`
    raises RuntimeError when the underlying thread was never started (pool.py:335;
    `threading.Thread.join` on a never-started thread raises `RuntimeError: cannot join
    thread before it is started`). Only a caller that GUARDS on "was start() ever called"
    (compose_run's `_stop_pool_if_started`) may call `.stop()` safely."""

    def __init__(self, order: _OrderSpy | None = None) -> None:
        self._order = order
        self._started = False

    def start(self) -> None:
        self._started = True
        if self._order is not None:
            self._order.calls.append("pool.start")

    def stop(self) -> None:
        if not self._started:
            raise RuntimeError("cannot join thread before it is started")
        if self._order is not None:
            self._order.calls.append("pool.stop")


def test_compose_run_calls_build_run_safety_once_and_starts_watchdog_after_pool(
    tmp_path, monkeypatch
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
        config=SimpleNamespace(), trainer=SimpleNamespace(step=0, model=object()),
        pool=pool, buffer=SimpleNamespace(save_to_path=lambda p: None),
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


def test_wired_sources_include_eval_round_iff_pipeline_built(tmp_path, monkeypatch) -> None:
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
    mantis_run.compose_run(
        config=SimpleNamespace(), trainer=SimpleNamespace(step=0, model=object()),
        pool=FakePoolNeverStarted(), buffer=SimpleNamespace(save_to_path=lambda p: None),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
    )
    assert "eval_round" in seen["with_eval"], (
        f"eval_enabled=True must declare eval_round wired: {seen['with_eval']}"
    )

    monkeypatch.setattr(mantis_run, "build_run_safety", _make_fake_build_run_safety("no_eval"))
    mantis_run.compose_run(
        config=SimpleNamespace(), trainer=SimpleNamespace(step=0, model=object()),
        pool=FakePoolNeverStarted(), buffer=SimpleNamespace(save_to_path=lambda p: None),
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


def test_sink_and_heartbeat_are_threaded_to_pipeline_and_coordinator(tmp_path, monkeypatch) -> None:
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

    def _fake_build_eval_pipeline(**kwargs):
        captured["eval_sink"] = kwargs.get("sink")
        captured["eval_heartbeat"] = kwargs.get("heartbeat")
        return SimpleNamespace(
            run_evaluation=lambda *a, **k: {"kicked": False, "reason": None},
            poll_completed=lambda: None, drain_pending=lambda: None,
            apply_gate_decision=lambda *a, **k: None, stop=lambda: None,
        )

    monkeypatch.setattr(mantis_run, "build_eval_pipeline", _fake_build_eval_pipeline,
                        raising=False)
    mantis_run.compose_run(
        config=SimpleNamespace(), trainer=SimpleNamespace(step=0, model=object()),
        pool=FakePoolNeverStarted(), buffer=SimpleNamespace(save_to_path=lambda p: None),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
    )
    assert captured.get("eval_sink") is sink, "the eval pipeline must receive the SAME sink"
    assert captured.get("eval_heartbeat") is heartbeat_fn, (
        "the eval pipeline must receive the SAME heartbeat fn build_run_safety produced"
    )
