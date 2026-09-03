# >300 justify (R8). The seven eval-failure routes are ONE claim (each route yields its OWN typed
# reason, its own phase, one emitted event that agrees with the routed result, and a
# traceback where an exception was in flight) driven over ONE harness. The fake-process /
# fake-context / spy-sink rig plus the seven-route driver is the majority of the file and
# every row needs all of it; R5 bars cross-test imports, so a split forks that rig into two
# copies which then drift while both stay green. Executable content is a minority — the
# rest is the per-oracle "what defect is this the only witness to" rationale LAW-07 asks
# each row to carry.
"""⊕ WP12-R Phase O / O-02, O-03, O-04, O-15, O-30, O-31 (R152) — every eval-failure route
produces its OWN typed reason, and the stream says which.

RED-at-import until IMPL writes `EvalBrokenReason` into `src/mantis/eval/errors.py`.
ORACLE-FIRST (⊕): the top-level import raises ImportError before any port code exists.

The defect this file exists to close, stated once: at HEAD every broken round routes a bare
`str` reason and NOTHING in `src/` reads it (`DESIGN_O §a.3` — the routed `error` key has
zero production consumers). A broken round is safe today only because `rounds.py:195` forces
`promoted=False`. So the seven failures are indistinguishable to anything but a human
reading a log line, and one of them — `ladder_persist_failed`, a LAW-14 persistence-fatal
route — has ZERO tests and ZERO doc mentions anywhere in the tree.

The oracles, and the defect each is the ONLY witness to:

- O-02 `test_each_route_yields_its_own_typed_reason` (7 sub-cases) — each censused route
  produces its OWN member, read off the ROUTED RESULT (never the emitted event: O-31 is
  what pins the two agree, and an O-02 that read the event could not tell them apart).
  Sole witness to two routes being cross-wired. MUTATION (M-O2): swap the reasons at
  `pipeline.py:475-480` → both sub-cases red.
- O-03 `test_the_seven_emitted_reasons_are_pairwise_distinct` — 21/21 pairs distinct in the
  EMITTED events. This is the "distinguishable from each other" leg of R84's template, and
  it is taken in the ONE channel because the rc taxonomy is deliberately many-to-one
  (`PREREG_O §0a`). MUTATION (M-O3): collapse two members onto one value → red.
- O-04 `test_the_reason_to_phase_map_is_fixed` — `phase` stays on the payload and is a
  FUNCTION of the reason, never an independent axis that can contradict it. Sole witness:
  nothing else reads `phase` at all. MUTATION (M-O4): emit `phase="drain"` for
  `result_missing` → red naming the pair.
- O-15 `test_a_ladder_persist_failure_is_a_named_broken_round_and_is_never_swallowed` — the
  LAW-14 route with zero coverage at HEAD gains its first oracle. Sole witness that the
  `LadderStateError` catch at `pipeline.py:521` still ROUTES a broken round rather than
  degrading to a log line. MUTATION (M-O15): `except LadderStateError: pass` → red.
- O-30 `test_..._logs_a_traceback` (2 arms) — the two routes that carry an exception
  (`round_completion_error`, `ladder_persist_failed`) must log with `_LOG.exception`, i.e.
  a record with POPULATED `exc_info`. On the round-completion route the in-tree contract is
  "never a swallowed exception, never a bare log line (isolation law 2)"
  (`pipeline.py:438-441`), and `repr(exc)` alone does not say WHERE the exception came from.
  Sole witness: no existing test in the tree carries a `caplog` assertion over either
  route. MUTATIONS (M-O30a / M-O30b): downgrade either `_LOG.exception` to `_LOG.error`.
- O-31 `test_the_emitted_event_reason_equals_the_routed_result_reason` — the event and the
  result are read by DIFFERENT consumers (a supervisor reads the stream; `promote.py` reads
  the mapping), so a divergence between them is invisible to every other oracle here.
  MUTATION (M-O31): derive the event's `reason` from a second local → red on that route.

**What is real here and what is not.** Real: the shipped `EvalPipeline`, its real
`_finalize_round` / `_read_worker_result` / `_broken_result` / `_success_result` chain, a
real `LadderState`, real event emission and a real round-result mapping. Fake: the worker
SUBPROCESS (an injected fake `multiprocessing` context — the house rig from
`tests/eval/test_eval_broken.py` / `test_round_completion_error.py`, kept as a private copy
per those files' own docstrings) and, on two routes, a monkeypatched raise standing in for
a persistence fault and for an uncaught completion crash. Spawning real OS subprocesses for
seven routes would trade determinism for nothing: the subject is the reason assembly, not
the spawn mechanics, which `test_pipeline_isolation.py` already owns.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.errors import EvalBrokenReason, LadderStateError  # RED-at-import anchor
from mantis.eval.ladder import LadderState
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.model import CnnArch, build_net

#: The seven routes, each with the member it must produce and the phase that member forces.
#: Stated here rather than derived from the enum: an oracle that read its expectation out of
#: the object under test would be satisfied by any consistent renaming (R81).
_ROUTE_REASON = {
    "join_timeout": "join_timeout",
    "killed": "killed",
    "exit_nonzero": "exit_nonzero",
    "result_missing": "result_missing",
    "result_invalid": "result_invalid",
    "ladder_persist_failed": "ladder_persist_failed",
    "round_completion_error": "round_completion_error",
}
_ROUTE_PHASE = {
    "join_timeout": "drain",
    "killed": "worker_exit",
    "exit_nonzero": "worker_exit",
    "result_missing": "worker_exit",
    "result_invalid": "worker_exit",
    "ladder_persist_failed": "ladder_persist",
    "round_completion_error": "round_completion",
}
_ROUTES = tuple(_ROUTE_REASON)

#: A worker sidecar result that satisfies `validate_worker_result` — the ONLY way to reach
#: `_success_result`, which is where the ladder-persist route lives.
_VALID_WORKER_RESULT = {
    "step": 1000, "gate": None, "rungs": {}, "skipped_rungs": [],
    "random": {"games": 0, "wr": None}, "worker_pid": 7,
}


# ── harness (private copy; house convention, see this file's docstring) ─────────────────
def _tiny_model():
    arch = CnnArch(board_size=5, in_channels=4, filters=8, res_blocks=1)
    net = build_net(arch)
    net.arch = arch
    return net


def _eval_cfg() -> EvalConfig:
    rungs = [
        LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
                   opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32),
    ]
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    ladder = LadderConfig(
        rungs=rungs, round_games=64, min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=8, bootstrap_resamples=1000,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    return EvalConfig(
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=5.0, worker_kill_grace_sec=0.2, gate=gate, ladder=ladder,
        ply_cap_adjudication=None, strength_floor=None,
    )


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        # `event` is subscripted, not `.get`-ed: a payload without it is a producer defect
        # and must be loud here rather than silently filtered out of every assertion.
        return [e for e in self.events if e["event"] == name]


class _FakeProcess:
    """A spawn-context child stand-in. `terminate()`/`kill()` set the POSIX-signed exit
    code the real `multiprocessing.Process` would report; the drive sets `exitcode`
    directly for the routes whose subject is the exit code itself."""

    def __init__(self, *, target=None, args=(), kwargs=None, daemon=None) -> None:
        self._target = target
        self.args = args
        self.kwargs = kwargs
        self.daemon = daemon
        self.pid = 4242
        self.alive = False
        self.exitcode: int | None = None
        self.terminated = False
        self.killed = False

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -9


class _FakeCtx:
    def __init__(self) -> None:
        self.last_process: _FakeProcess | None = None

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> _FakeProcess:
        proc = _FakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self.last_process = proc
        return proc


class _InjectedCompletionError(RuntimeError):
    """Stands in for the real uncaught exception RED-TEAM's F1 reproduced deep inside
    `_success_result` — any exception class must reach the same catch-all."""


class _Drive:
    """One driven route: the routed result and the events the round actually emitted."""

    def __init__(self, result: dict[str, Any], sink: _SpySink) -> None:
        self.result = result
        self.sink = sink

    def broken_event(self) -> dict[str, Any]:
        events = self.sink.named("eval_broken")
        assert events, "the route emitted NO eval_broken event — a broken round is never silent"
        return events[-1]


def _quiesce_poller(pipeline: Any) -> None:
    """Stop the persistent poller BEFORE the round is driven.

    Not a weakening: the poller's own exception-proofing and its heartbeat are
    `tests/eval/test_round_completion_error.py`'s and `test_eval_heartbeat.py`'s subjects.
    Here it is a RACE — the poller finalizes any round whose process stops looking alive,
    so leaving it running would make which code path assembled the reason nondeterministic,
    and a flaky oracle proves nothing about a taxonomy.
    """
    pipeline._stop_event.set()          # noqa: SLF001 -- deliberate, test-only quiescing
    pipeline._poller.join(5.0)          # noqa: SLF001
    assert not pipeline._poller.is_alive(), (  # noqa: SLF001
        "the poller thread did not stop inside 5 s; the drive below would race it"
    )


def _drive(route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Drive:
    """Drive ONE censused failure route through the real `EvalPipeline` and return what it
    routed and emitted. Every branch below reproduces a real production condition; none of
    them reaches into the reason assembly itself."""
    assert route in _ROUTE_REASON, f"unknown route {route!r}"
    ctx = _FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    sink = _SpySink()
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_eval_pipeline(
        leaf_batch_size=1, amp_dtype="bf16",
        eval_cfg=_eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=2.0, eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=2.0, terminal_eval_hard_cap_sec=2.0,
        ),
        encoding="v6_live2_ls", run_id="oracle_test_run", spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json",
        # F-816-10 D-1: the pipeline resolves the fused-forward memory bound ONCE in the
        # parent and carries it to every `RoundSpec` — the eval child is a SECOND
        # allocator on the same card that no in-process bound can see. `None` is the
        # GRID arm, written out rather than omitted.
        fused_graph_caps=None,
        inference_batching=None,
        promotion=DeployTagHooks(
            anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
            best_model_path=tmp_path / "best_model.pt", run_id="oracle_test_run",
            encoding="v6_live2_ls", save_anchor=lambda *a, **k: None,
            guarded_load=lambda *a, **k: None,
        ),
        sink=sink,
    )
    try:
        _quiesce_poller(pipeline)
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={},
                                      best_model_step=None)
        assert ack["kicked"] is True, "premise: the round was kicked"
        proc = ctx.last_process
        assert proc is not None, "premise: the pipeline spawned a (fake) worker"
        result_path = Path(pipeline._inflight["spec"].result_path)  # noqa: SLF001

        if route == "join_timeout":
            proc.alive = True                       # the child never exits on its own
        elif route == "killed":
            proc.alive, proc.exitcode = False, -9   # POSIX-signed: negative == signal death
        elif route == "exit_nonzero":
            proc.alive, proc.exitcode = False, 3
        elif route == "result_missing":
            proc.alive, proc.exitcode = False, 0    # clean exit, no sidecar written
        elif route == "result_invalid":
            proc.alive, proc.exitcode = False, 0
            result_path.write_text(json.dumps({"step": 1000}))   # contract keys missing
        elif route == "ladder_persist_failed":
            proc.alive, proc.exitcode = False, 0
            result_path.write_text(json.dumps(_VALID_WORKER_RESULT))

            def _persist_boom(self: Any, path: Any) -> None:
                raise LadderStateError(f"simulated persistence fault writing {path}")

            monkeypatch.setattr(LadderState, "save", _persist_boom)
        else:  # round_completion_error
            proc.alive, proc.exitcode = False, 0

            def _completion_boom(inflight: Any, *, exit_code: Any, wall_sec: Any) -> None:
                raise _InjectedCompletionError("simulated round-completion crash (F1 shape)")

            pipeline._read_worker_result = _completion_boom  # noqa: SLF001

        result = pipeline.drain_pending()
        assert isinstance(result, dict), (
            f"premise: route {route!r} must route ONE completed round mapping; got {result!r}"
        )
        return _Drive(result, sink)
    finally:
        pipeline.stop()


# ══ O-02 — each route yields its OWN member, read off the ROUTED RESULT ════════════════
@pytest.mark.parametrize("route", _ROUTES)
def test_each_route_yields_its_own_typed_reason(route, tmp_path, monkeypatch) -> None:
    """O-02. The routed result carries a typed `eval_broken_reason`, and it is the member
    THIS route owns.

    Read off the ROUTED RESULT and not off the emitted event, deliberately: `promote.py` is
    the production consumer of the mapping, and pairing this oracle with the event would
    make M-O31 (an event that disagrees with the result) invisible to the whole file.
    """
    drive = _drive(route, tmp_path, monkeypatch)
    reason = drive.result["eval_broken_reason"]
    assert isinstance(reason, EvalBrokenReason), (
        f"route {route!r} routed a {type(reason).__name__} ({reason!r}) — a bare string is "
        "exactly the second authority R152 deletes; the builder must take the enum"
    )
    assert reason.value == _ROUTE_REASON[route], (
        f"route {route!r} must produce {_ROUTE_REASON[route]!r}; got {reason.value!r}"
    )
    assert drive.result["promoted"] is False, (
        "a broken round never promotes — `promoted` is DERIVED from the reason, and a "
        "broken round that promotes is the defect the derivation exists to make impossible"
    )


# ══ O-03 — 21/21 pairs distinct in the ONE channel ═════════════════════════════════════
def test_the_seven_emitted_reasons_are_pairwise_distinct(tmp_path, monkeypatch) -> None:
    """O-03. R84's "distinguishable from each other" leg, taken where the design says it is
    taken: the EVENT STREAM. The rc taxonomy is many-to-one by decision
    (`PREREG_O §0a`, all seven map to 48), so if the seven emitted `reason` values ever
    collide there is NOTHING left that separates a killed worker from a garbage result.

    Reads the EMITTED values (O-02 reads the routed ones) so a collision introduced on the
    emit side alone is still caught.
    """
    emitted = {}
    for route in _ROUTES:
        drive = _drive(route, tmp_path / route, monkeypatch)
        emitted[route] = drive.broken_event()["reason"]

    values = list(emitted.values())
    collisions = [(a, b) for i, a in enumerate(_ROUTES) for b in _ROUTES[i + 1:]
                  if emitted[a] == emitted[b]]
    assert collisions == [], (
        f"routes sharing one emitted reason: {collisions} (full map: {emitted})"
    )
    assert len(set(values)) == len(_ROUTES) == 7, (
        f"7 routes must emit 7 distinct reasons; got {sorted(set(values))}"
    )


# ══ O-04 — reason → phase is a fixed map, never an independent axis ════════════════════
def test_the_reason_to_phase_map_is_fixed(tmp_path, monkeypatch) -> None:
    """O-04. `phase` stays on the payload (it is not folded into the enum) precisely because
    it is a FUNCTION of the reason. Nothing else in the tree reads `phase`, so without this
    pin a mislabelled phase is invisible: a supervisor triaging `result_missing` under
    `phase=drain` would go looking at the drain budget for a missing sidecar file.
    """
    observed = {}
    for route in _ROUTES:
        drive = _drive(route, tmp_path / route, monkeypatch)
        event = drive.broken_event()
        observed[event["reason"]] = event["phase"]

    expected = {_ROUTE_REASON[route]: _ROUTE_PHASE[route] for route in _ROUTES}
    assert observed == expected, (
        "the reason→phase map moved; a phase that can contradict its reason is a second "
        f"axis nobody reads.\n  expected: {expected}\n  observed: {observed}"
    )


# ══ O-31 — the emitted event and the routed result agree, on every route ═══════════════
def test_the_emitted_event_reason_equals_the_routed_result_reason(tmp_path, monkeypatch) -> None:
    """O-31. Two consumers read two different objects: a supervisor reads the `eval_broken`
    EVENT, `promote.py` reads the round-result MAPPING. Nothing else here compares them, so
    a second local feeding the event (M-O31) would leave O-02 green (the routed result is
    still right) and O-03 green (the values are still 7 and distinct) while the stream told
    an operator the wrong story about which failure happened.
    """
    for route in _ROUTES:
        drive = _drive(route, tmp_path / route, monkeypatch)
        event_reason = drive.broken_event()["reason"]
        result_reason = drive.result["eval_broken_reason"]
        assert event_reason == result_reason, (
            f"route {route!r}: the emitted event says {event_reason!r} and the routed "
            f"result says {result_reason!r} — one emitter, one value (R152)"
        )


# ══ O-15 — the LAW-14 route that had no oracle at all ══════════════════════════════════
def test_a_ladder_persist_failure_is_a_named_broken_round_and_is_never_swallowed(
    tmp_path, monkeypatch
) -> None:
    """O-15. `ladder_persist_failed` (`pipeline.py:530`) is a LAW-14 persistence-fatal route
    with ZERO tests and ZERO doc mentions anywhere in the tree at HEAD — DESIGN_O §a.2's
    census found it and `PREREG_O §0b` armed an escalation in case it turned out to be
    unreachable. It is reachable: `LadderState.save` wraps `OSError` into `LadderStateError`
    and `pipeline.py:521` catches it.

    What must hold: the failure is NAMED (its own member), it ROUTES a broken round, and the
    games already played are NOT reported as a success — the on-disk ladder state did not
    durably advance, so a "success" here silently drifts the in-memory state ahead of the
    persisted one.

    MUTATION THAT REDS IT (M-O15): wrap the save in `except LadderStateError: pass`. The
    round then reports a clean success and LAW-14 is a log line.
    """
    drive = _drive("ladder_persist_failed", tmp_path, monkeypatch)

    assert drive.result["eval_broken_reason"] is EvalBrokenReason.LADDER_PERSIST_FAILED, (
        "a persistence fault must surface as its OWN reason, never as a generic break — "
        f"got {drive.result['eval_broken_reason']!r}"
    )
    assert drive.result["promoted"] is False, (
        "a round whose ladder state never reached disk must not promote off it"
    )
    event = drive.broken_event()
    assert event["phase"] == "ladder_persist", (
        f"the phase names WHERE it broke; got {event['phase']!r}"
    )
    assert drive.sink.named("eval_round_complete"), (
        "the round still completes loudly — LAW-14 is fail-loud, not fail-silent"
    )


# ══ O-30 — the two exception-bearing routes keep their traceback ═══════════════════════
def test_the_round_completion_route_logs_a_traceback_and_the_detail(
    tmp_path, monkeypatch, caplog
) -> None:
    """O-30, arm 1 (round_completion_error).

    `_round_completion_error_result`'s own contract is "never a swallowed exception, NEVER A
    BARE LOG LINE (isolation law 2)" (`pipeline.py:438-441`) — the surface RED-TEAM's F1
    created. `repr(exc)` says WHAT was raised; only the traceback says WHERE, and on a
    catch-all that is the entire diagnostic value. Phase O collapses this route's emitter
    into `_broken_result` (whose own logging is `_LOG.error`, no traceback), so without this
    oracle the collapse silently deletes the stack. Nothing at HEAD stands under it:
    `grep -n caplog tests/eval/test_round_completion_error.py` returns nothing.

    MUTATION THAT REDS IT (M-O30a): downgrade the raising site's `_LOG.exception` to
    `_LOG.error`. Every payload assertion in this file stays green — the mechanism is the
    LOG RECORD and only the log record.
    """
    with caplog.at_level(logging.ERROR, logger="mantis.eval.pipeline"):
        drive = _drive("round_completion_error", tmp_path, monkeypatch)

    records = [r for r in caplog.records if r.name == "mantis.eval.pipeline"]
    with_traceback = [r for r in records if r.exc_info is not None]
    assert with_traceback, (
        "the round-completion catch-all must log WITH the live traceback (`_LOG.exception`); "
        f"captured {len(records)} record(s), none carrying exc_info: "
        f"{[r.getMessage() for r in records]}"
    )
    assert any("_InjectedCompletionError" in repr(r.exc_info) for r in with_traceback), (
        "the captured traceback must be THIS round's exception, not an unrelated live "
        f"context: {[repr(r.exc_info) for r in with_traceback]}"
    )
    assert any("_InjectedCompletionError" in r.getMessage() for r in records), (
        "…and the `repr(exc)` detail must still travel in the message text, so an operator "
        f"grepping the log without a traceback reader still sees the class: "
        f"{[r.getMessage() for r in records]}"
    )
    assert any("eval_broken" in r.getMessage() for r in records), (
        "…AND the emitter's own `eval_broken …` ERROR line is still there: the two records "
        f"are the raising site and the one emitter, not one replacing the other: "
        f"{[r.getMessage() for r in records]}"
    )
    assert drive.result["eval_broken_reason"] is EvalBrokenReason.ROUND_COMPLETION_ERROR


def test_the_ladder_persist_route_logs_a_traceback_and_the_detail(
    tmp_path, monkeypatch, caplog
) -> None:
    """O-30, arm 2 (ladder_persist_failed).

    The sibling exception-bearing route, and the SHAPE Phase O adopts for both: log with
    `_LOG.exception` at the RAISING site, then call the one emitter. Pinned because the
    collapse's whole argument is that the two routes converge — if this arm is allowed to
    drift, "convergence" stops being a claim anybody checks.

    MUTATION THAT REDS IT (M-O30b): `_LOG.exception` → `_LOG.error` at `pipeline.py:528`.
    O-15 stays green (the reason still routes), which is exactly why this arm is separate.
    """
    with caplog.at_level(logging.ERROR, logger="mantis.eval.pipeline"):
        drive = _drive("ladder_persist_failed", tmp_path, monkeypatch)

    records = [r for r in caplog.records if r.name == "mantis.eval.pipeline"]
    with_traceback = [r for r in records if r.exc_info is not None]
    assert with_traceback, (
        "a LAW-14 persistence fault must log its traceback — the OSError chain underneath "
        f"`LadderStateError` is the only thing that says which write failed; captured: "
        f"{[r.getMessage() for r in records]}"
    )
    assert any("LadderStateError" in repr(r.exc_info) for r in with_traceback), (
        f"the traceback must be the persistence fault's: {[repr(r.exc_info) for r in with_traceback]}"
    )
    assert any("eval_broken" in r.getMessage() for r in records), (
        "…and the emitter's own ERROR line is still present beside it: "
        f"{[r.getMessage() for r in records]}"
    )
    assert drive.result["eval_broken_reason"] is EvalBrokenReason.LADDER_PERSIST_FAILED
