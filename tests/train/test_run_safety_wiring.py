"""⊕ O-27 (+ O-25, O-23, the exit-code equality pin) — the run-safety composition + close-out.

RED-at-import until IMPL writes `mantis.monitor.config` + `mantis.train.lifecycle.heartbeat_watchdog`.
ORACLE-FIRST (⊕): the top-level imports raise ModuleNotFoundError before any port code exists.

O-27 (MUST-2, P-27) — the disarm-at-close_out WIRING oracle. A dropped/late disarm turns every
clean finish with a >30 min terminal eval into a false-42 supervisor relaunch STORM. With the
watchdog ARMED + its thread running (tiny deadline), `drain.close_out` entering a fake eval
drain that BLOCKS longer than the staleness deadline must produce ZERO staleness fires; the
heartbeat-file `seq` keeps advancing through it; a persist increment mid-drain STILL fires 43
(persist-fatal is NEVER disarmed). The late-disarm mutant (disarm AFTER the blocked flush) is
shown to false-fire — proving the FIRST-action ordering is load-bearing.

Also: O-25 MonitorConfig live-consumer sweep (dead knobs cannot ship), O-23 L-A census (no
`mantis.eval` top-level import under train/monitor), and the `42 == 42` exit-code equality pin.

>300 justify: ONE seam under test — the run-safety COMPOSITION (`build_run_safety`) and the
close-out it hands to `drain.close_out` — sharing one harness (`_make_watchdog`,
`BlockingPipeline`, the exit/sink spies, `_drain_coord`). The disarm ordering, the wiring
census that proves all three heartbeat sources are reachable (RED-TEAM F3), the result-seam
routing and the config sweeps are all assertions about that same composition; splitting them
would duplicate the harness and let the "what is wired" and "what happens at teardown" halves
drift apart, which is precisely the gap F3 exploited.
"""
from __future__ import annotations

import ast
import dataclasses
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import (
    HEARTBEAT_SOURCES,
    HeartbeatRegistry,
    WATCHDOG_STALL_EXIT_CODE,
    read_heartbeat_file,
)
from mantis.train.coordinator import drain
from mantis.train.lifecycle.heartbeat_watchdog import HeartbeatWatchdog
from mantis.train.lifecycle.watchdog import SELFPLAY_STALL_EXIT_CODE

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"


# ── harness ───────────────────────────────────────────────────────────────────────────
class _ExitSpy:
    def __init__(self) -> None:
        self.codes: list[int] = []

    def __call__(self, code: int) -> None:
        self.codes.append(int(code))


class SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class BlockingPipeline:
    """A fake eval pipeline whose `drain_pending` BLOCKS longer than the staleness deadline —
    the legally-long close-out wait that must not false-fire the watchdog."""

    def __init__(self, *, block_sec: float, counters_box=None) -> None:
        self._block = block_sec
        self._counters_box = counters_box   # if given, bump it at drain start (mid-drain persist)

    def drain_pending(self):
        if self._counters_box is not None:
            self._counters_box[0] += 1      # a persist failure DURING the blocked drain
        threading.Event().wait(timeout=self._block)   # bounded block on the MAIN thread
        return None

    def run_evaluation(self, *a, **k):
        return None


def _make_watchdog(*, sink, exit_fn, counters_fn, hb_file, deadline=0.3):
    reg = HeartbeatRegistry()               # real monotonic clock
    return HeartbeatWatchdog(
        registry=reg,
        deadlines={s: deadline for s in HEARTBEAT_SOURCES},
        sink=sink, counters_fn=counters_fn, heartbeat_file=hb_file,
        file_interval_sec=0.05, poll_interval_sec=0.05, clock=time.monotonic,
        save_snapshot=lambda: None, exit_fn=exit_fn,
    )


def _fake_coord(*, watchdog, pipeline, sink):
    return SimpleNamespace(
        heartbeat_watchdog=watchdog, eval_pipeline=pipeline,
        config=SimpleNamespace(terminal_eval_enabled=False),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        _train_step=1000, _sink=sink, eval_model=object(), full_config={},
        on_eval_round_complete=lambda result: None,
    )


# ── O-27 — disarm-at-close_out (the ⊕ oracle) ─────────────────────────────────────────
def test_close_out_disarms_staleness_no_false_fire(tmp_path) -> None:
    """O-27 / P-27 — `close_out` disarms staleness FIRST, so a blocked >deadline drain produces
    ZERO staleness fires and the heartbeat-file seq keeps advancing through the whole window."""
    sink, exit_spy, hb = SpySink(), _ExitSpy(), tmp_path / "hb.json"
    wd = _make_watchdog(sink=sink, exit_fn=exit_spy, counters_fn=lambda: 0, hb_file=hb)
    pipe = BlockingPipeline(block_sec=0.8)
    coord = _fake_coord(watchdog=wd, pipeline=pipe, sink=sink)
    wd.start()
    try:
        before = read_heartbeat_file(hb)
        seq_before = before.seq if before is not None else 0
        drain.close_out(coord)              # blocks ~0.8 s inside flush_pending_eval
        assert 42 not in exit_spy.codes, (
            "a clean close-out must NOT fire staleness (else every long terminal eval false-42s)"
        )
        state = read_heartbeat_file(hb)
        assert state is not None and state.seq >= 1, (
            "the watchdog thread must keep mirroring a fresh seq through the blocked drain"
        )
        # P-27 registers STRICT increase across the blocked window: a `seq` that merely
        # exists proves nothing — the supervisor keys on PROGRESSION, so a frozen-but-present
        # seq during a long close-out is exactly the stale-kill condition.
        assert state.seq > seq_before, (
            f"seq must STRICTLY increase through the blocked close-out "
            f"({seq_before} → {state.seq})"
        )
    finally:
        wd.stop()


def test_persist_fatal_still_fires_after_close_out_disarm(tmp_path) -> None:
    """O-27 / P-27 — persist-fatal is NEVER disarmed: a persist increment DURING the blocked
    close-out drain still fires 43, while staleness stays silent (42 absent)."""
    sink, exit_spy, hb = SpySink(), _ExitSpy(), tmp_path / "hb.json"
    box = [0]
    wd = _make_watchdog(sink=sink, exit_fn=exit_spy, counters_fn=lambda: box[0], hb_file=hb)
    pipe = BlockingPipeline(block_sec=0.8, counters_box=box)
    coord = _fake_coord(watchdog=wd, pipeline=pipe, sink=sink)
    wd.start()
    try:
        drain.close_out(coord)
        assert 43 in exit_spy.codes, "a persist failure during close-out must still fire 43"
        assert 42 not in exit_spy.codes, "staleness stays disarmed even while persist fires"
    finally:
        wd.stop()


def test_late_disarm_mutant_false_fires(tmp_path) -> None:
    """O-27 / P-27 (mutant self-test) — the late-disarm mutant: if disarm happens AFTER the
    blocked flush (not FIRST), a false 42 lands during the block. Reproducing that ordering here
    MUST fire 42 — proving the disarm-FIRST ordering in the real close_out is load-bearing and
    the oracle above is not vacuous."""
    sink, exit_spy, hb = SpySink(), _ExitSpy(), tmp_path / "hb.json"
    wd = _make_watchdog(sink=sink, exit_fn=exit_spy, counters_fn=lambda: 0, hb_file=hb)
    pipe = BlockingPipeline(block_sec=0.8)
    wd.start()
    try:
        pipe.drain_pending()                # MUTANT: flush BEFORE disarm (staleness still armed)
        wd.disarm_staleness()               # disarm too late
        assert 42 in exit_spy.codes, "a late disarm must false-fire 42 (the oracle detects it)"
    finally:
        wd.stop()


# ── O-25 — MonitorConfig live-consumer sweep ──────────────────────────────────────────
_CONSUMER_SOURCES = (
    _SRC / "monitor" / "rules.py",
    _SRC / "monitor" / "config.py",
    _SRC / "monitor" / "supervise.py",
    _SRC / "train" / "events.py",
    _SRC / "train" / "subsystems.py",
    _SRC / "train" / "lifecycle" / "heartbeat_watchdog.py",
    _SRC / "train" / "coordinator" / "step.py",
)


def test_every_monitor_config_field_has_a_live_consumer() -> None:
    """O-25 / P-25 — every `MonitorConfig` field name is referenced by rules/watchdog/supervisor/
    emission source (LAW-08 applied to the dataclass). A dead knob cannot ship in the schema-debt
    window."""
    blob = "".join(p.read_text() for p in _CONSUMER_SOURCES if p.exists())
    orphans = [f.name for f in dataclasses.fields(MonitorConfig) if f.name not in blob]
    assert orphans == [], f"MonitorConfig fields with no live consumer (dead knobs): {orphans}"


def test_monitor_config_is_frozen_with_no_lenient_from_dict() -> None:
    """O-25 / D1 — MonitorConfig is a FROZEN dataclass (explicit kwargs only); the old lenient
    `from_dict` (silently ignoring unknown keys — an R1 violation in spirit) is DEAD."""
    assert dataclasses.is_dataclass(MonitorConfig)
    cfg = MonitorConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.wr_rolling_threshold = 0.99  # type: ignore[misc]
    assert not hasattr(MonitorConfig, "from_dict"), "the lenient from_dict must not survive"


def test_monitor_config_carries_old_lineage_sealbot_defaults() -> None:
    """O-25 — the sealbot-WR thresholds are the old `MonitoringConfig` values verbatim (0.10/2/
    20000; peak×0.5/25000/3; 0.05/15000). The default DISPOSITION ships WARN-ONLY
    (`wr_hard_abort_enabled=False`, operator G-3): the TRIGGERS are unchanged, only the default
    posture moved — the hard-abort is one field away."""
    cfg = MonitorConfig()
    assert cfg.wr_hard_abort_enabled is False, "operator G-3: ships warn-only, not hard-abort"
    assert MonitorConfig(wr_hard_abort_enabled=True).wr_hard_abort_enabled is True, (
        "the hard-abort remains available via the one-field flip"
    )
    assert cfg.wr_rolling_threshold == 0.10
    assert cfg.wr_rolling_consecutive_evals == 2
    assert cfg.wr_rolling_min_step == 20000
    assert cfg.wr_collapse_from_peak_ratio == 0.5
    assert cfg.wr_collapse_min_step == 25000
    assert cfg.wr_collapse_consecutive_evals == 3
    assert cfg.wr_early_death_threshold == 0.05
    assert cfg.wr_early_death_min_step == 15000


# ── O-23 — L-A census (no in-process eval on train/monitor surfaces) ──────────────────
def _top_level_imports(tree: ast.Module) -> list[str]:
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.append(node.module)
            targets.extend(f"{node.module}.{a.name}" for a in node.names)
    return targets


def test_no_top_level_eval_import_under_train_or_monitor() -> None:
    """O-23 / P-23 — no `mantis.eval` top-level import under `train/**` or `monitor/**`; eval is
    reached only via the injected `EvalPipelineLike`. Co-enforced with WP11-A's subprocess-
    isolation side — nothing here constructs an in-process CUDA eval from the run loop."""
    violations: list[str] = []
    for root in (_SRC / "train", _SRC / "monitor"):
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for target in _top_level_imports(tree):
                if target == "mantis.eval" or target.startswith("mantis.eval."):
                    violations.append(f"{path.relative_to(_SRC)} -> {target}")
    assert violations == [], f"train/monitor must not top-level import mantis.eval: {violations}"


# ── exit-code equality pin ────────────────────────────────────────────────────────────
def test_stall_exit_code_equality_pin() -> None:
    """O-17/O-27 — the WP10 games-progress watchdog and the L-B heartbeat watchdog share ONE
    restart-wrapper key (42); the supervisor keys on exactly this value."""
    assert SELFPLAY_STALL_EXIT_CODE == WATCHDOG_STALL_EXIT_CODE == 42


# ══ RED-TEAM F3 — the composition root must actually WIRE all three sources ═══════════
def test_build_run_safety_wires_the_heartbeat_into_every_declared_source(tmp_path) -> None:
    """RED-TEAM F3 — the wiring the watchdog depends on had NO oracle: `build_run_safety` has
    no in-repo caller, so nothing asserted that the pool and the inference server can even
    receive `registry.beat`. An unwired source then looks exactly like a wedged one (false 42
    ⇒ relaunch storm ⇒ rc 44).

    Asserts, end to end: the composition root exposes THE `HeartbeatFn` (bound to its own
    registry); both self-play collaborators ACCEPT a `heartbeat=` kwarg; the beat lands in the
    registry for each of the three declared sources; and the declaration reaches the watchdog.
    """
    import inspect

    from mantis.selfplay.inference_server import InferenceServer
    from mantis.selfplay.pool import WorkerPool
    from mantis.train.subsystems import build_run_safety

    run_safety = build_run_safety(
        log_dir=tmp_path, run_id="wiring", buffer=None,
        buffer_persist_path=tmp_path / "replay_buffer.bin",
        wired_sources=HEARTBEAT_SOURCES,
    )
    try:
        assert run_safety.heartbeat.__self__ is run_safety.registry, (
            "run_safety.heartbeat must BE the registry's beat, not a detached callable"
        )
        for owner in (WorkerPool, InferenceServer):
            assert "heartbeat" in inspect.signature(owner.__init__).parameters, (
                f"{owner.__name__} must accept a heartbeat= kwarg for the L-B registry"
            )
        for source in HEARTBEAT_SOURCES:
            run_safety.heartbeat(source)
        assert run_safety.registry.beaten_sources() == frozenset(HEARTBEAT_SOURCES)
    finally:
        run_safety.sink.close()


def test_every_declared_heartbeat_source_has_a_live_emitter() -> None:
    """RED-TEAM F3 — each of the three declared sources must be emitted by a REAL producer
    module (the quoted source literal at its beat site), so the registry cannot declare a
    stage nothing feeds. Complements the manifest's `heartbeat.*` event_literal rows."""
    emitters = {
        "train_step": _SRC / "train" / "coordinator" / "step.py",
        "inference_dispatch": _SRC / "selfplay" / "inference_server.py",
        "selfplay_drain": _SRC / "selfplay" / "pool_drain.py",
    }
    assert set(emitters) == set(HEARTBEAT_SOURCES), "the emitter map must cover every source"
    for source, path in emitters.items():
        body = path.read_text()
        assert f'"{source}"' in body, f"{path.name} must carry the quoted literal {source!r}"


def test_build_run_safety_requires_an_explicit_wiring_declaration() -> None:
    """RED-TEAM F3 — `wired_sources` has NO default: the root must STATE what it wired, so a
    forgotten `heartbeat=` kwarg surfaces as a loud `heartbeat_source_unwired` rather than as
    a 42 on a healthy run. Bites a signature that infers the declaration."""
    import inspect

    from mantis.train.subsystems import build_run_safety

    param = inspect.signature(build_run_safety).parameters["wired_sources"]
    assert param.default is inspect.Parameter.empty, "wired_sources must be required"


# ══ RED-TEAM F7 / F8 — the result seam and the disarm must never fail SILENTLY ════════
class _RecordingCoord(SimpleNamespace):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.routed: list = []
        self.on_eval_round_complete = self.routed.append


def _drain_coord(result, sink):
    pipeline = SimpleNamespace(drain_pending=lambda: result,
                               run_evaluation=lambda *a, **k: None)
    return _RecordingCoord(
        heartbeat_watchdog=None, eval_pipeline=pipeline,
        config=SimpleNamespace(terminal_eval_enabled=False),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        _train_step=1000, _sink=sink, eval_model=object(), full_config={},
    )


def test_a_batch_of_eval_rounds_is_routed_not_dropped() -> None:
    """RED-TEAM F7 — a drain returning a LIST of completed rounds (the N-2 handshake shape
    WP11-A may plausibly return) used to be dropped with `routed=0`, no event, no counter and
    no log: the sealbot gate's only feed path going quiet exactly the way F-10 did. Every
    Mapping in the batch must reach the handler, one call per round."""
    sink = SpySink()
    rounds = [{"step": 1, "wr_sealbot": 0.4}, {"step": 2, "wr_sealbot": 0.5}]
    coord = _drain_coord(rounds, sink)
    drain.flush_pending_eval(coord)
    assert coord.routed == rounds, "every completed round in a batch must be routed"


def test_an_unroutable_eval_result_is_recorded_loudly() -> None:
    """RED-TEAM F7 — a shape the seam cannot consume is RECORDED
    (`eval_result_unroutable`), never dropped in silence. It is deliberately not a raise: a
    raise escapes into `close_out` and skips `pool.stop` + the terminal eval (F10)."""
    sink = SpySink()
    coord = _drain_coord("a bare ack string", sink)
    drain.flush_pending_eval(coord)
    assert coord.routed == []
    unroutable = [e for e in sink.events if e.get("event") == "eval_result_unroutable"]
    assert len(unroutable) == 1 and unroutable[0]["result_type"] == "str"


def test_close_out_fails_loud_when_the_watchdog_cannot_disarm() -> None:
    """RED-TEAM F8 — a duck-typed watchdog without `disarm_staleness` used to make `close_out`
    SKIP the disarm silently: the exact false-42 relaunch storm MUST-2 exists to prevent,
    reachable by injecting the wrong object. A wiring bug must fail loud."""
    sink = SpySink()
    coord = _drain_coord(None, sink)
    coord.heartbeat_watchdog = SimpleNamespace(arm=lambda: None)   # no disarm_staleness
    with pytest.raises(TypeError) as ei:
        drain.close_out(coord)
    assert "disarm_staleness" in str(ei.value)
