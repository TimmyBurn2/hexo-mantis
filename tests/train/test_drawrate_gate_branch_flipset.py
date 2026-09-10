""">300 justify (R8): one gate, one branch table. Every drive exercises a different arm of
`_run_hard_abort_gates`/`_sample` against the SAME coordinator harness, and the "which branch has
no input that takes it" claim is only checkable while the drives sit together.

WHAT WENT WRONG: the gate shipped with a `draw and spec is not None` test whose second conjunct
had no flip-set — `_sample` returns False whenever its producer is `None`, and that producer is
`None` exactly when `spec is None`, so the `elif draw:` arm was provably unreachable while its
comment claimed it counted the EXPLICIT-off case.

WHAT THIS ORACLE IS THE SOLE WITNESS FOR: branch reachability MEASURED with `sys.settrace`, so a
dead arm re-introduced anywhere makes the covered-line union fall short; each remaining condition
being independently decisive; the exact LAW-18 accounting pinned as an EQUALITY per gate run,
since an inequality could not tell a single skip from a double; and the NO-OBSERVATION contract,
where insufficient evidence skip-counts through the SAME one site and appends NOTHING.

`_run_hard_abort_gates` is called DIRECTLY, not via `step()`, so ONE gate run is one call. No
monkeypatch of the subject and no synthetic clock: the drives differ only in which of the two real
inputs is present, and the config comes from the production builder.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import CodeType, SimpleNamespace

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

#: A spec whose `min_step=0` puts the live path in reach of a single drive; the FIRE arm
#: additionally needs `consec` consecutive samples at/above threshold. `N_pool_min=10` is
#: deliberately > 1: the NO-OBSERVATION arm needs a bar a drive can sit UNDER.
_LIVE = DrawRateAbortSpec(threshold=0.4, min_step=0, N_pool_min=10, consec=3)
_ZERO = {"checks": 0, "fires": 0, "skips": 0, "warns": 0}
_GATE = "draw_rate_collapse"


class _Pool:
    """The pool surface the draw-rate gate touches, and only that. The producer is
    `pooled_draw_counts() -> (draws, completed)` and takes no bar — the evidence bar is applied at
    the abort decision."""

    games_completed = 0

    def __init__(self, counts: tuple[int, int]) -> None:
        self._counts = (int(counts[0]), int(counts[1]))
        self.counts_calls = 0

    def pooled_draw_counts(self) -> tuple[int, int]:
        self.counts_calls += 1
        return self._counts


class _NoProducerPool(_Pool):
    """A pool whose `pooled_draw_counts` reads as absent — the shape a `getattr(..., None)` sees
    when the producer has not landed. Subclassing keeps every other surface identical, so this
    drive differs from the live one in exactly one input."""

    pooled_draw_counts = None


class _Buffer:
    size, capacity = 1000, 100_000

    def save_to_path(self, path) -> None:
        return None


class _SpySink:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


#: `drain_caps` is a config-authored builder parameter with no default, so it arrives from a
#: MINTED `monitor.drain` block — a literal would be a second authority over `monitor.drain.*`.
_DRAIN_CAPS = resolve_drain_caps(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").monitor)
#: The builder's fourth config-authored parameter, from the same minted config.
_KNOBS = resolve_coordinator_knobs(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train)
#: The builder's FIFTH config-authored parameter — `monitor.gate_interval`, the ARMING cadence.
#: Harnesses that set `log_interval` MIRROR it onto `gate_interval`, the shipped posture.
_GATE_INTERVAL = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").monitor.gate_interval


def _coordinator(*, spec, pool):
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=spec,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        # Gate cadence mirrors narration cadence; this file calls `_run_hard_abort_gates`
        # directly anyway, so neither knob gates its drives.
        log_interval=1, gate_interval=1, eval_interval=1, min_buf_size=1,
        terminal_eval_enabled=False,
    )
    shutdown = ShutdownState()
    coord = StepCoordinator(
        trainer=SimpleNamespace(step=0), buffer=_Buffer(), pretrained_buffer=None,
        recent_buffer=None, pool=pool, eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        full_config={}, train_cfg={}, mixing_cfg={}, sink=_SpySink(),
        heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, config=config)


def _executable_lines(code: CodeType) -> set[int]:
    """Every line number `code` and its nested code objects can execute. `co_lines()` is the
    interpreter's own line table, so this is the REAL executable set, not a source-text guess."""
    lines = {lineno for _, _, lineno in code.co_lines() if lineno is not None}
    for const in code.co_consts:
        if isinstance(const, CodeType):
            lines |= _executable_lines(const)
    return lines


def _run_traced(harness) -> tuple[bool, set[int]]:
    """One `_run_hard_abort_gates` call; returns `(fired, lines of the gate that ran)`."""
    code = StepCoordinator._run_hard_abort_gates.__code__
    wanted = {code} | {c for c in code.co_consts if isinstance(c, CodeType)}
    seen: set[int] = set()

    def _local(frame, event, _arg):
        if event in ("call", "line"):
            seen.add(frame.f_lineno)
        return _local

    def _global(frame, event, arg):
        # The `call` event carries the `def`/`lambda` line, which `co_lines()` reports as
        # executable but which never raises a `line` event, so both events are recorded.
        return _local(frame, event, arg) if frame.f_code in wanted else None

    previous = sys.gettrace()
    sys.settrace(_global)
    try:
        fired = harness.coord._run_hard_abort_gates(harness.config)
    finally:
        sys.settrace(previous)
    return fired, seen


def test_every_branch_of_the_draw_rate_gate_has_an_input_that_takes_it() -> None:
    """The flip-set proof: five drives, one per branch outcome.

    Two are early returns through one disjunct alone (spec absent with a live producer, and the
    reverse), which is why neither disjunct is decoration; one is the full live path with no fire;
    one is the live path's FIRE arm; and one is evidence BELOW `N_pool_min`, the NO-OBSERVATION
    return that makes the `if not self._sample(...)` conjunct non-decorative.

    The union of the five must cover every executable line of the function. That is the assertion
    a resurrected dead arm cannot survive, in whatever shape it comes back.
    """
    b1 = _coordinator(spec=None, pool=_Pool((99, 100)))
    fired1, lines1 = _run_traced(b1)
    b2 = _coordinator(spec=_LIVE, pool=_NoProducerPool((99, 100)))
    fired2, lines2 = _run_traced(b2)
    b3 = _coordinator(spec=_LIVE, pool=_Pool((0, 100)))
    fired3, lines3 = _run_traced(b3)
    # A LIVE producer whose answer is `None`: one game under the bar, every one of them DRAWN. The
    # rate would be 1.0 and would fire instantly if it were observed at all, so a gate that ignored
    # the bar reds here rather than passing quietly.
    b5 = _coordinator(spec=_LIVE, pool=_Pool((_LIVE.N_pool_min - 1, _LIVE.N_pool_min - 1)))
    fired5, lines5 = _run_traced(b5)
    assert (fired1, fired2, fired3, fired5) == (False, False, False, False), (
        "none of B1/B2/B3/B5 may report a fire: two are absent-input early returns, one is a "
        "live sample below threshold, and B5 is a total collapse the gate has INSUFFICIENT "
        "EVIDENCE to report (R92)"
    )

    # The FIRE arm needs `consec` consecutive samples at/above threshold, so ONE coordinator is
    # driven until the rule speaks. `consec` is read off the shipped config, never a literal here.
    b4 = _coordinator(spec=_LIVE, pool=_Pool((90, 100)))
    lines4: set[int] = set()
    fired4 = False
    for _ in range(_LIVE.consec + 2):
        fired4, seen = _run_traced(b4)
        lines4 |= seen
        if fired4:
            break
    assert fired4 is True, (
        "the FIRE arm must be reachable: a gate that can only skip or pass is as dead as "
        f"the arm this oracle exists for (stats: {b4.coord._gate_stats[_GATE]})"
    )
    assert b4.shutdown.running is False, "a fired hard abort must stop the run"

    # ── the flip-sets, stated as the divergence they are ──────────────────────────────
    assert b1.coord._draw_rate_history == [] and b2.coord._draw_rate_history == [], (
        "B1 and B2 must take the early return — no sample may be appended when EITHER "
        "input is absent, or a disarmed gate is feeding the abort history"
    )
    assert b1.pool.counts_calls == 0, (
        "on the disarmed posture the producer must never be CALLED: there is no bar to "
        "judge its answer against (R80/R92)"
    )
    assert b3.coord._draw_rate_history == [0.0], (
        "B3 differs from B1 only in `spec` and from B2 only in the producer, and it must "
        "take the LIVE path. If it did not, neither disjunct would have a flip-set and the "
        "early return would be as dead as the arm it replaced"
    )
    assert b5.pool.counts_calls == 1 and b5.coord._draw_rate_history == [], (
        "B5's flip-set, stated as the divergence it is: the producer WAS called (so this is "
        "not B1 or B2 wearing a different hat) and NOTHING was appended (so this is not B3). "
        f"A total collapse below the bar must leave the history empty; got "
        f"{b5.coord._draw_rate_history}"
    )
    assert b5.coord._gate_stats[_GATE] == {"checks": 1, "fires": 0, "skips": 1, "warns": 0}, (
        "…and it must SKIP-COUNT through `_sample`'s ONE counter (LAW-18), exactly as an "
        f"absent producer does: {b5.coord._gate_stats[_GATE]}"
    )

    # ── reachability: no line of the gate is dead ─────────────────────────────────────
    executable = _executable_lines(StepCoordinator._run_hard_abort_gates.__code__)
    covered = lines1 | lines2 | lines3 | lines4 | lines5
    assert executable - covered == set(), (
        "these lines of `_run_hard_abort_gates` are executed by NO input: "
        f"{sorted(executable - covered)}. R72: a branch with no flip-set is dead code, and "
        "dead code in a hard-abort gate reads as coverage that does not exist (DR-1)"
    )
    assert covered - executable == set(), (
        "the trace picked up lines outside the function's own line table — the "
        "instrumentation is measuring something other than its subject"
    )


def test_a_disarmed_gate_run_records_exactly_one_check_and_one_skip() -> None:
    """A disarmed gate run records exactly one check and one skip, per gate run.

    The numbers are unchanged by the refactor but come from a different site — `_sample`'s skip
    counter alone, rather than that counter plus a dead `elif` arm that never ran — so this is an
    EQUALITY over two consecutive runs. The last arm is the comment correction itself.
    """
    harness = _coordinator(spec=None, pool=_Pool((99, 100)))
    stats = harness.coord._gate_stats[_GATE]
    assert stats == _ZERO, "harness precondition: the gate's counters start at zero"

    assert harness.coord._run_hard_abort_gates(harness.config) is False
    assert stats == {"checks": 1, "fires": 0, "skips": 1, "warns": 0}, (
        "one disarmed gate run must record exactly one check and one skip — the LAW-18 "
        f"reading an operator sees for `train.draw_rate_abort: null`; got {stats}"
    )
    assert harness.coord._run_hard_abort_gates(harness.config) is False
    assert stats == {"checks": 2, "fires": 0, "skips": 2, "warns": 0}, (
        "the accounting must be per gate run and must not double-count: a second skip site "
        f"beside `_sample`'s would show up here as 2 checks / 4 skips; got {stats}"
    )
    assert harness.coord._draw_rate_history == [], (
        "a disarmed gate must append NO sample: a fabricated reading in the abort history "
        "is the class R80's inclusion bar exists to keep out"
    )

    # `_sample` is what SKIP-counts an absent producer — the fact the deleted comment denied.
    # Driven on the real method, against a gate key of its own so the counters above stay pinned.
    before = dict(harness.coord._gate_stats["sealbot_wr_abort"])
    assert harness.coord._sample("sealbot_wr_abort", [], None) is False
    after = harness.coord._gate_stats["sealbot_wr_abort"]
    assert after["checks"] == before["checks"] + 1 and after["skips"] == before["skips"] + 1, (
        "`_sample` itself counts the check AND the skip for an absent producer. The skip arm "
        f"DR-1 deleted claimed to be what counted the EXPLICIT-off case; it never ran. "
        f"before={before} after={after}"
    )


def test_insufficient_evidence_appends_nothing_and_counts_a_skip() -> None:
    """Insufficient evidence appends NOTHING and counts a skip, never a fabricated healthy reading.

    An empty included set once produced a rate of `0.0` the gate APPENDED as a real measurement —
    at its worst precisely when the pool was collapsing, because drawn games are the LONGEST games
    and a collapsing pool banks completed games slower. The drive is maximally adversarial in both
    directions: every completed game is a DRAW, and there is exactly one game less than the bar.
    THE MUTATIONS THAT RED IT: return `0.0` below the bar and the history becomes `[0.0]`; return
    the rate unguarded and it becomes `[1.0]`.
    """
    bar = _LIVE.N_pool_min
    harness = _coordinator(spec=_LIVE, pool=_Pool((bar - 1, bar - 1)))
    assert harness.coord._run_hard_abort_gates(harness.config) is False

    assert harness.coord._draw_rate_history == [], (
        "insufficient evidence must append NOTHING. A `0.0` here is DR-4's fabricated "
        "healthy reading; a `1.0` is a total-collapse abort taken on evidence the operator "
        f"declared insufficient; got {harness.coord._draw_rate_history}"
    )
    assert harness.coord._gate_stats[_GATE] == {"checks": 1, "fires": 0, "skips": 1, "warns": 0}, (
        "…and it must be visible in-run as ONE check and ONE skip (LAW-18), through "
        "`_sample`'s single counter — the same site an absent producer uses, because 'no "
        f"observation' is one fact however it arises; got {harness.coord._gate_stats[_GATE]}"
    )
    assert harness.pool.counts_calls == 1, (
        "harness precondition: the producer WAS called exactly once. A gate that short-"
        "circuited before it would satisfy both assertions above while witnessing nothing"
    )

    # ONE more completed game — the ONLY difference — and the same collapse is observed.
    at_bar = _coordinator(spec=_LIVE, pool=_Pool((bar, bar)))
    assert at_bar.coord._run_hard_abort_gates(at_bar.config) is False, (
        "one observation is not `consec` observations, so this run must not fire yet"
    )
    assert at_bar.coord._draw_rate_history == [1.0], (
        "at the bar the SAME collapse is a real observation of 1.0. Without this arm a gate "
        "that never observed anything would pass the whole test above"
    )
    assert at_bar.coord._gate_stats[_GATE] == {"checks": 1, "fires": 0, "skips": 0, "warns": 0}, (
        "…and it must count a check with NO skip: the two postures are distinguishable in "
        f"the LAW-18 stream, which is what makes the skip counter informative; got "
        f"{at_bar.coord._gate_stats[_GATE]}"
    )
