""">300 justify (R8), stated at this file's MEASURED size of 389 lines. Already over the cap
with NO justify (the pre-existing gap WPMINT Phase K-A flagged); K-B touches it for
`DrawRateAbortSpec.consec`, so it is written now. One gate, one branch table: every drive below
exercises a different arm of `_run_hard_abort_gates`/`_sample` against the SAME coordinator
harness, and the "which branch has no input that takes it" claim is only checkable while the
drives sit together.

⊕ WPMINT DR-FIX ORACLE — `_run_hard_abort_gates` carries no branch without a flip-set
(R72), and its LAW-18 skip accounting is unchanged by the fix (finding DR-1).

WHAT WENT WRONG. WPAX Phase D shipped the gate as::

    draw = self._sample("draw_rate_collapse", self._draw_rate_history,
        (lambda: ...) if rates_fn is not None and spec is not None else None)
    if draw and spec is not None:
        ...
    elif draw:
        self._gate_stats["draw_rate_collapse"]["skips"] += 1

`_sample` returns False whenever its producer is `None`, and that producer is `None`
exactly when `spec is None` — so `draw` IMPLIED `spec is not None`. The `spec is not None`
conjunct therefore had NO flip-set (no input makes the branch outcome depend on it — R72's
conjunct law) and the `elif draw:` arm was provably unreachable. Its comment claimed the arm
"now counts the EXPLICIT-off case"; that was measured false — `_sample` counts it, which is
the last assertion in this file.

WHAT THIS ORACLE IS THE SOLE WITNESS FOR. Three things no other test in the tree covers:

* **branch reachability, measured rather than read**: every executable line of
  `_run_hard_abort_gates` (its own code object AND the producer lambda nested in it) is
  executed by at least one of the five drives below, collected with `sys.settrace`. A dead
  arm re-introduced anywhere in the function makes the union fall short and reds this test.
  `test_drawrate_abort_threading.py`'s O-D8/O-D10 drive the gate but never observe WHICH
  lines ran, so a resurrected unreachable arm stays invisible to them.
* **each remaining condition's flip-set is non-empty**: `spec is None` and `rates_fn is
  None` are asserted to be independently decisive — B1 flips only `spec`, B2 flips only the
  producer, and each on its own diverts the run from the live path B3 takes.
* **the exact LAW-18 accounting**, pinned as an EQUALITY (`checks=1, skips=1, fires=0` per
  gate run on a disarmed run), not the `>= 1` inequalities O-D8 uses. DR-1's fix moved the
  disarmed posture from a two-site accounting (`_sample`'s skip plus a dead `elif`) to a
  one-site accounting, and an inequality could not tell a single skip from a double.
* **the NO-OBSERVATION contract R92 adds** (WPMINT Phase DS): insufficient evidence
  (`Sum(completed) < N_pool_min`) must skip-count through the SAME one site and append
  NOTHING — never the healthy `0.0` that DR-4 measured being recorded as a real reading.
  Drive B5 and `test_insufficient_evidence_appends_nothing_and_counts_a_skip` are its
  witnesses, and the second states the mutation that reds it.

`_run_hard_abort_gates` is called DIRECTLY here (not via `step()`) so that ONE gate run is
one call — the whole point of the accounting pin. No monkeypatch of the subject and no
synthetic clock: the drives differ only in which of the two real inputs is present, and the
coordinator config comes from the production builder (`mantis.run._step_coordinator_config`)
rather than a hand-written census of the ~22 knobs CARD-COORD-KNOBS owns.

R5: the fakes below are local, never imported from a sibling test module. R7 / gate 6:
nothing here writes a file at all.
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
#: additionally needs `consec` consecutive samples at/above threshold (WPMINT K-B moved
#: that term off the coordinator dataclass and into `train.draw_rate_abort.consec`).
#: `N_pool_min=10` is deliberately > 1 (WPMINT Phase DS): the NO-OBSERVATION arm R92 adds
#: needs a bar a drive can sit UNDER, and a bar of 1 would make that arm unreachable.
_LIVE = DrawRateAbortSpec(threshold=0.4, min_step=0, N_pool_min=10, consec=3)
_ZERO = {"checks": 0, "fires": 0, "skips": 0, "warns": 0}
_GATE = "draw_rate_collapse"


# ── local fakes (R5: no cross-test import) ────────────────────────────────────────────
class _Pool:
    """The pool surface the draw-rate gate touches, and only that.

    WPMINT Phase DS (R92): the producer is `pooled_draw_counts() -> (draws, completed)` and
    takes no bar — the evidence bar is applied at the abort decision."""

    games_completed = 0

    def __init__(self, counts: tuple[int, int]) -> None:
        self._counts = (int(counts[0]), int(counts[1]))
        self.counts_calls = 0

    def pooled_draw_counts(self) -> tuple[int, int]:
        self.counts_calls += 1
        return self._counts


class _NoProducerPool(_Pool):
    """A pool whose `pooled_draw_counts` reads as absent — the shape
    `getattr(self.pool, "pooled_draw_counts", None)` sees when the producer has not
    landed. Subclassing keeps every other surface identical, so B2 differs from B3 in
    exactly one input."""

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


#: WPMINT Phase K-A (R93): `drain_caps` is a third config-authored builder parameter with no
#: default, so it arrives from a MINTED `monitor.drain` block — a literal would be a second
#: authority over `monitor.drain.*`, which is the DR-11 defect in miniature.
_DRAIN_CAPS = resolve_drain_caps(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").monitor)
#: WPMINT Phase K-B: the builder's fourth config-authored parameter, from the same minted
#: config — the 19 coordinator knobs are `train.*` keys now, not builder literals.
_KNOBS = resolve_coordinator_knobs(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train)


def _coordinator(*, spec, pool):
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=spec,
                                 drain_caps=_DRAIN_CAPS, knobs=_KNOBS),
        log_interval=1, eval_interval=1, min_buf_size=1, terminal_eval_enabled=False,
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


# ── line-level reachability instrumentation ───────────────────────────────────────────
def _executable_lines(code: CodeType) -> set[int]:
    """Every line number `code` and its nested code objects can execute.

    `co_lines()` is the interpreter's own line table, so this is the function's REAL
    executable set — not a source-text guess about which lines "look like" statements.
    """
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
        # The `call` event carries the `def` / `lambda` line, which `co_lines()` reports as
        # executable (the RESUME instruction) but which never raises a `line` event. Both
        # events are recorded so the two sets are measured on the same footing.
        return _local(frame, event, arg) if frame.f_code in wanted else None

    previous = sys.gettrace()
    sys.settrace(_global)
    try:
        fired = harness.coord._run_hard_abort_gates(harness.config)
    finally:
        sys.settrace(previous)
    return fired, seen


# ── R72 — every branch has an input that takes it, and no line is dead ────────────────
def test_every_branch_of_the_draw_rate_gate_has_an_input_that_takes_it() -> None:
    """The R72 flip-set proof. Four drives, one per branch outcome:

    * **B1** — `spec is None`, producer LIVE: the disarmed posture. Takes the early return
      through the FIRST disjunct alone, which is that disjunct's flip-set against B3.
    * **B2** — spec LIVE, producer absent: the same early return through the SECOND
      disjunct alone. B1 and B2 together are why neither disjunct is decoration.
    * **B3** — both LIVE, the rule returns no message: the full live path, no fire.
    * **B4** — both LIVE, sustained collapse: the live path's FIRE arm.
    * **B5** — both LIVE, evidence BELOW `N_pool_min`: the NO-OBSERVATION return that
      WPMINT Phase DS (R92) adds. This is the drive that makes the new
      `if not self._sample(...)` conjunct non-decorative. DR-1 correctly ruled that
      conjunct out when `_sample` could only return True past the early return; R92's
      `pooled_draw_rate` returns `None` on insufficient evidence, so the False arm has a
      real flip-set now and B5 is it.

    The union of the five must cover every executable line of the function. That is the
    assertion a resurrected dead arm cannot survive: a line no input reaches is exactly the
    `elif draw:` defect this oracle exists for, in whatever shape it comes back.
    """
    b1 = _coordinator(spec=None, pool=_Pool((99, 100)))
    fired1, lines1 = _run_traced(b1)
    b2 = _coordinator(spec=_LIVE, pool=_NoProducerPool((99, 100)))
    fired2, lines2 = _run_traced(b2)
    b3 = _coordinator(spec=_LIVE, pool=_Pool((0, 100)))
    fired3, lines3 = _run_traced(b3)
    # B5 — a LIVE producer whose answer is `None`: `_LIVE.N_pool_min - 1` completed games,
    # every one of them DRAWN. The rate would be 1.0 and would fire instantly if it were
    # observed at all, so a gate that ignored the bar reds here rather than passing quietly.
    b5 = _coordinator(spec=_LIVE, pool=_Pool((_LIVE.N_pool_min - 1, _LIVE.N_pool_min - 1)))
    fired5, lines5 = _run_traced(b5)
    assert (fired1, fired2, fired3, fired5) == (False, False, False, False), (
        "none of B1/B2/B3/B5 may report a fire: two are absent-input early returns, one is a "
        "live sample below threshold, and B5 is a total collapse the gate has INSUFFICIENT "
        "EVIDENCE to report (R92)"
    )

    # B4 — the FIRE arm needs `consec` consecutive samples at/above threshold, so
    # ONE coordinator is driven until the rule speaks. `consec` is read off the shipped
    # config, never written as a literal here (R78 keeps it a coordinator-owned knob).
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


# ── LAW-18 — the disarmed accounting, pinned exactly ──────────────────────────────────
def test_a_disarmed_gate_run_records_exactly_one_check_and_one_skip() -> None:
    """The behavioural pin DR-1's refactor must not move.

    HEAD's accounting for a disarmed run was `checks=1, skips=1, fires=0` per gate run, and
    it still is — but for a different reason (`_sample`'s skip counter alone, rather than
    `_sample`'s counter plus a dead `elif` arm that never ran). An operator reading
    `monitor_gates` must see the same numbers as before the fix, so this is an EQUALITY over
    two consecutive gate runs, not the `>= 1` O-D8 asserts.

    The last arm is the comment correction itself: `_sample` is called with a `None`
    producer, directly, and IS observed to advance both counters. That is the fact the
    deleted in-code comment denied.
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

    # `_sample` is what SKIP-counts an absent producer — the fact DR-1's replaced comment
    # denied. Driven on the real method, against a gate key of its own so the draw-rate
    # counters above stay the ones this test pinned.
    before = dict(harness.coord._gate_stats["sealbot_wr_abort"])
    assert harness.coord._sample("sealbot_wr_abort", [], None) is False
    after = harness.coord._gate_stats["sealbot_wr_abort"]
    assert after["checks"] == before["checks"] + 1 and after["skips"] == before["skips"] + 1, (
        "`_sample` itself counts the check AND the skip for an absent producer. The skip arm "
        f"DR-1 deleted claimed to be what counted the EXPLICIT-off case; it never ran. "
        f"before={before} after={after}"
    )


# ── R92 — insufficient evidence is a SKIP, never a fabricated healthy reading ─────────
def test_insufficient_evidence_appends_nothing_and_counts_a_skip() -> None:
    """WPMINT Phase DS (R92), the answer to RECHECK_D finding DR-4, driven as a behaviour.

    At `d0b3974` an empty included set produced `recent_pool_draw_rate({}) == 0.0` and the
    gate APPENDED it to `_draw_rate_history` as a real measurement — a healthy reading
    fabricated from no evidence, and (DR-4) at its worst precisely when the pool was
    collapsing, because drawn games are the LONGEST games and a collapsing pool banks
    completed games slower.

    R92 makes that unrepresentable by TYPE. The drive below is the maximally adversarial
    input for both directions at once: every completed game is a DRAW (so an unguarded gate
    would read 1.0 and fire) and there is exactly one game less than `N_pool_min` (so the
    guarded gate must report nothing at all).

    THE MUTATION THAT REDS THIS TEST, stated so a later reader can run it: make
    `pooled_draw_rate` return `0.0` instead of `None` below the bar. `_draw_rate_history`
    becomes `[0.0]` — DR-4's exact defect — and the first assertion fails. Make it return the
    rate unguarded instead and the history becomes `[1.0]`; the same assertion fails from the
    other side. Neither mutation touches the disarmed accounting the test above pins, so the
    two are independent subjects rather than one assertion twice.
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
