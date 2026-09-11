# >300 justify (R8): the PARENT-ONLY half of tools/ci_gates/preflight_mint.py — exactly the leaf
# layers with NO dependency on the tool side: shared vocabulary and exit taxonomy, the (a)/(b)
# evaluators, the report helpers, the child classifier and the segment/verdict/identity leaves.
"""The parent-only half of the mint preflight (CI gate 12).

LOADED ONLY by tools/ci_gates/preflight_mint.py, via `spec_from_file_location` on a
`__file__`-relative path — never sys.path — and it re-exports every public seam by plain
assignment, so the module-path attribute authority stays the tool file the oracles load. This
file must never import the tool (no cycle) and must never define a `MANIFEST` global.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from mantis.config.schema import RunConfig
from mantis.monitor.heartbeat import (
    DISK_SPACE_EXHAUSTED_EXIT_CODE,
    DRAW_RATE_COLLAPSE_EXIT_CODE,
    TERMINAL_EVAL_BROKEN_EXIT_CODE,
)

A_KEYS = ("a1", "a2", "a3", "a4")
B_KEYS = ("b0", "b1", "b2", "b3", "b4a", "b4b", "b4c", "b5a")
A4_PINS = "single-producer / no sink line loss"
B1_SCOPE = "source-mutation detector; vacuous against an unmodified watchdog"
#: The modes `_new_report` may be called with. NO default: falling back would publish some other
#: mode's disclaimer.
REPORT_MODES: tuple[str, ...] = ("audit", "preflight")
#: The `not_run` disclaimer's two halves, selected by WHAT THE RUN DID. See `_not_run_reason`.
NOT_BOOTED_REASON = ("NO boot was spawned and NO burst was attempted, so (a) sync-cadence and "
                     "(b) lag-transport had nothing to measure")
BOOTED_REASON = ("a boot WAS spawned and a burst attempted, but the run did not reach the "
                 "point where (a) and (b) could be evaluated")
#: The draw-rate row key `_burst_floors` gives, named ONCE: two spellings of one row key is how a
#: tier silently stops matching the floor it is derived from.
DRAW_RATE_FLOOR_KEY = "train.draw_rate_abort.min_step"
#: The MINT TIERS: which of `_burst_floors`' rules the ACCEPTED burst cleared — what the run DID.
TIER_NONE = "none"
TIER_SYNC_LAG = "sync_lag"
TIER_FULL = "full"
#: BOTH are required for a mint, and `full` COVERS `sync_lag`, so one green `full` run discharges
#: both. MEASURED: on a production config the short tier is unreachable, because an armed
#: `draw_rate_collapse` row raises the floor past it. Two COVERAGE CLAIMS, not two runs.
MINT_REQUIRED_TIERS: tuple[str, ...] = (TIER_SYNC_LAG, TIER_FULL)
#: What each tier does NOT prove. NO default: falling back would publish ANOTHER tier's disclaimer.
TIER_NOT_PROVEN: dict[str, str] = {
    # Worded to be true on EVERY route that lands here, so it is pinned to the field that records
    # the fact (`tier.burst_steps` is null) rather than to a story about how it got that way.
    TIER_NONE: ("NO burst was accepted — `tier.burst_steps` is null, whether because none was "
                "requested (mode AUDIT) or because the run stopped at or before "
                "`_burst_bound` (rc 10 / 11 / 30 / 31) — so this report proves "
                "nothing about any tier: not (a) sync-cadence, not (b) lag-transport, and not "
                "that the run reaches the step at which train.draw_rate_abort can fire"),
    TIER_SYNC_LAG: ("the accepted burst clears the actor-lag and sync-cadence floors ONLY: it "
                    "proves (a) sync and (b) lag and NOT that the run reaches the draw-rate "
                    "abort's first firing step. On a config that arms " + DRAW_RATE_FLOOR_KEY
                    + " that reachability is the schema's own `min_step < max_train_steps` on "
                    "the minted run length — the burst is a PREFIX of that run, never a "
                    "mutation of it (CARD-STAMP-FLOOR) — and tier `full` needs a burst past "
                    "the floor. On a config that arms no draw-rate row, tier `full` is "
                    "UNAVAILABLE, not merely unrun"),
    TIER_FULL: ("the accepted burst clears " + DRAW_RATE_FLOOR_KEY + ", so a run that "
                "COMPLETES it reaches the first step at which the draw-rate abort can fire. "
                "That is REACHABILITY and nothing else: it does not show the abort firing, it "
                "does not show it firing CORRECTLY, and a healthy run must NOT fire it. The "
                "statistic's correctness is pinned by the coordinator's own oracles, never by "
                "this tool"),
}


RC_CONVENTION = (
    "POSIX Popen.returncode — NEGATIVE on signal death, never 128+N"
)
#: The child's own named outcomes propagate unchanged.
PASS_THROUGH = range(10, 42)
#: Reserved by the run's own machinery — `monitor/heartbeat.py`, `monitor/supervise.py:39`.
WATCHDOG_CODES = (42, 43, 45)
RELAUNCH_BUDGET_CODE = 44
#: The cooperative half of the reserved band. NOT watchdog codes: RETURNED by the child after the
#: run unwound through its own close-out. A code outside `PASS_THROUGH` and outside this set falls
#: through every arm to `PreflightBootFailedError` and COLLAPSES TO 33, destroying the signal.
ARMED_ABORT_CODES = (DRAW_RATE_COLLAPSE_EXIT_CODE, DISK_SPACE_EXHAUSTED_EXIT_CODE,
                     TERMINAL_EVAL_BROKEN_EXIT_CODE)
#: The full 42–47 band the docstring declares, derived so the claim and the tuples cannot drift.
RESERVED_CODES = tuple(sorted({*WATCHDOG_CODES, RELAUNCH_BUDGET_CODE, *ARMED_ABORT_CODES}))


class PreflightError(Exception):
    """Base for every named preflight outcome. `rc` is the process exit code."""

    rc = 1

    def __init__(self, message: str, **detail: object) -> None:
        super().__init__(message)
        self.detail = detail


class PreflightInternalError(PreflightError):
    rc = 1


class PreflightConfigError(PreflightError):
    rc = 10


class PreflightBurstTooShortError(PreflightError):
    rc = 11


class PreflightResumedTrainerError(PreflightError):
    rc = 12


class PreflightOutDirInsideRepoError(PreflightError):
    rc = 13


class PreflightConfigIdentityError(PreflightError):
    """The child's published boot identity does not match the config the parent audited: the
    evidence artifact would describe one config while the burst ran another."""
    rc = 14


class PreflightOutDirReusedError(PreflightError):
    """The out-dir already holds THIS run_id's event segments. `_read_segment` scopes by run_id,
    so a same-run_id reuse of a dirty --out-dir would read a PREVIOUS burst's events as this
    run's evidence. Refused before the boot, never an mtime heuristic."""
    rc = 15


class PreflightMirrorReceiptsError(PreflightError):
    """The burst's bundle and first shard were not receipted off-box within the wait (R349(b)):
    a START HALT decided AFTER the boot, because it needs the boot's own artifacts."""
    rc = 16


class PreflightCudaBuildError(PreflightError):
    """The config declares a cuda device and the installed torch cannot compute on a GPU.
    Conditioned on what the RUN declares, never on sniffing the host: a box may run a `+cpu` wheel
    deliberately, so host presence of a card proves nothing about which torch a run needs."""
    rc = 17


class PreflightArmingAuditError(PreflightError):
    rc = 30


class PreflightManifestError(PreflightError):
    rc = 31


class PreflightTreeDefectError(PreflightError):
    rc = 32


class PreflightBootFailedError(PreflightError):
    rc = 33


class PreflightWatchdogFiredError(PreflightError):
    rc = 34


class PreflightChildSignaledError(PreflightError):
    rc = 35


class PreflightInterruptedError(PreflightError):
    """The run was interrupted before it reached a verdict.

    A `KeyboardInterrupt` or a callee's `SystemExit` is a `BaseException`: it unwinds straight
    through `main`'s two `except` arms into the report-writing `finally`, which then landed
    carrying the SKELETON's `verdict: "pass", rc: 0` beside blocks all saying `not_run`.
    """

    rc = 36


class PreflightVerdictUnreachedError(PreflightError):
    """The report says `pass` but its assertions never reached one. Never raised — it is the
    `failure` NAME `_finalise_verdict` stamps when it downgrades such a report."""

    rc = 37


class PreflightTimeoutError(PreflightError):
    rc = 40


class PreflightReportUnwritableError(PreflightError):
    rc = 41


class PreflightAssertionsFailedError(PreflightError):
    """(a) or (b) failed. The rc and the reported NAME are both lifted from the failing block, so
    the report's `failure` and the process exit code have one authority."""

    def __init__(self, failure_name: str, message: str) -> None:
        super().__init__(message)
        self.failure_name = failure_name
        self.rc = FAILURE_CODES.get(failure_name, PreflightBootFailedError.rc)


class PreflightChildOutcomeError(PreflightError):
    """The child's own named outcome, propagated UNCHANGED."""

    def __init__(self, rc: int, message: str) -> None:
        super().__init__(message)
        self.rc = int(rc)


class PreflightArmedAbortFiredError(PreflightError):
    """A manifest-registered armed abort FIRED, and its authored rc propagates UNCHANGED.

    Distinct from rc 34 on purpose: a watchdog fire is `os._exit` from a thread mid-run, while an
    armed abort is COOPERATIVE — the run unwound, saved, and returned the manifest's number, so a
    supervisor reads the same number either side of this tool.
    """

    def __init__(self, rc: int, message: str) -> None:
        super().__init__(message)
        self.rc = int(rc)


#: assertion-block failure name -> exit code. One authority, so a report's `failure` and the rc
#: can never disagree.
FAILURE_CODES = {
    "PreflightSyncAbsentError": 20,
    "PreflightSyncCadenceError": 21,
    "PreflightBurstIncompleteError": 22,
    "PreflightInversionUndiscriminatedError": 23,
    "PreflightLagUnobservableError": 25,
    "PreflightLagFrozenError": 26,
    "PreflightLagArithmeticError": 27,
    "PreflightLagSourceMismatchError": 28,
    "PreflightLagInvertedError": 29,
}


def _named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


def _step_ground_truth(events: list[dict], samples: list[dict]) -> dict:
    """The step count N, from an independent terminal witness, with the witness NAMED. The
    canonical `training_step` narration is `log_interval`-gated (1000), so a 101-step burst
    carries none; the trainer's per-step diagnostic rows are deliberately not counted either."""
    saves = _named(events, "shutdown_save")
    if saves:
        return {"source": "shutdown_save", "value": int(saves[-1]["step"])}
    terminal = _named(events, "terminal_eval")
    if terminal:
        return {"source": "terminal_eval", "value": int(terminal[-1]["step"])}
    if samples:
        return {"source": "actor_lag_sample",
                "value": max(int(sample["learner_step"]) for sample in samples)}
    return {"source": "absent", "value": 0}


def _evaluate_sync(events: list[dict], *, cadence_steps: int, burst_steps: int,
                   ground: dict) -> dict:
    """(a) — sync presence and cadence-consistency.

    `a2` is ORDERED-LIST equality: a set collapses a duplicate, and the earlier set form PASSED an
    over-firing stream (measured at 102 events). `a1` carries the `extra`/`missed` sub-reason.
    `a4` is NOT a cadence predicate — it pins single-producer / no sink line loss.
    """
    syncs = _named(events, "actor_sync")
    n = int(ground["value"])
    cadence = int(cadence_steps)
    # The `{1} ∪` term is not decoration: `maybe_sync` syncs UNCONDITIONALLY on the first call
    # (`actor_sync.py:63`). Dropping it is invisible at cadence 1, which is every minted config.
    expected = sorted({1} | {k for k in range(1, n + 1) if k % cadence == 0})
    block: dict = {
        "verdict": "pass", "failure": None, "sub_reason": None,
        "cadence": cadence, "N": n,
        "expected_syncs": len(expected), "observed_syncs": len(syncs),
        "a1": None, "a2": None, "a3": None, "a4": None,
        "a4_pins": A4_PINS, "step_ground_truth": ground,
    }
    if n != int(burst_steps):
        block["verdict"] = "fail"
        block["failure"] = "PreflightBurstIncompleteError"
        return block
    if not syncs:
        # A different diagnosis from "synced at the wrong steps", so a different code; the four
        # sub-predicates stay None because nobody measured a cadence.
        block["verdict"] = "fail"
        block["failure"] = "PreflightSyncAbsentError"
        return block
    block["a1"] = len(syncs) == len(expected)
    block["a2"] = [int(event["step"]) for event in syncs] == expected
    block["a3"] = all(int(event.get("cadence_steps", -1)) == cadence for event in syncs)
    block["a4"] = ([int(event.get("sync_count", -1)) for event in syncs]
                   == list(range(1, len(syncs) + 1)))
    if block["a1"] is False:
        block["sub_reason"] = "extra" if len(syncs) > len(expected) else "missed"
    elif block["a2"] is False:
        block["sub_reason"] = "missed"
    elif block["a3"] is False:
        block["sub_reason"] = "cadence"
    elif block["a4"] is False:
        block["sub_reason"] = "counter"
    for key in A_KEYS:  # table order: the reported name must be deterministic
        if block[key] is False:
            block["verdict"] = "fail"
            block["failure"] = "PreflightSyncCadenceError"
            break
    return block


_LAG_FAILURES = {
    "b1": ("PreflightLagArithmeticError", None),
    "b2": ("PreflightLagFrozenError", "learner"),
    "b3": ("PreflightLagFrozenError", "actor"),
    "b4a": ("PreflightLagSourceMismatchError", "foreign"),
    "b4b": ("PreflightLagSourceMismatchError", "regressed"),
    "b4c": ("PreflightLagSourceMismatchError", "stale"),
    "b5a": ("PreflightLagInvertedError", None),
}


def _evaluate_lag(events: list[dict], *, cadence_steps: int,
                  poll_interval_sec: float) -> dict:
    """(b) — the lag TRANSPORT. Every predicate is stated so a CONSTANT fails. b4 is split three
    ways: the old max-equality conjunct was a false-positive generator on a healthy run (sampling
    stops at close-out while syncs continue), so b4c is a `ts`-bounded lower bound instead."""
    samples = _named(events, "actor_lag_sample")
    negatives = _named(events, "actor_lag_negative")
    syncs = _named(events, "actor_sync")
    block: dict = {key: None for key in B_KEYS}
    block.update(verdict="pass", failure=None, sub_reason=None, samples=len(samples),
                 b1_scope=B1_SCOPE, poll_interval_sec=float(poll_interval_sec),
                 inversion_discrimination="unproven", discriminating_samples=0,
                 inversion_reason=None)
    block["b0"] = len(samples) >= 2
    if block["b0"] is False:
        # b0 GATES the rest: reporting b1…b5a True over an absent measurement is a green over
        # nothing, which is the exact shape this gate exists to refuse.
        block["verdict"] = "fail"
        block["failure"] = "PreflightLagUnobservableError"
        return block

    learners = [int(sample["learner_step"]) for sample in samples]
    actors = [int(sample["actor_ckpt_step"]) for sample in samples]
    lags = [int(sample["lag_steps"]) for sample in samples]
    sync_steps = {int(event["step"]) for event in syncs}
    poll = float(poll_interval_sec)

    def _visible_floor(sample: dict) -> int:
        """The highest sync the watchdog MUST already have seen when it read this sample, bounded
        by one full poll interval: the watchdog reads both callables at the top of
        `_check_actor_lag`, so a sync landing in that gap can legitimately look one step behind."""
        cutoff = float(sample.get("ts", 0.0)) - poll
        return max({0} | {int(event["step"]) for event in syncs
                          if float(event.get("ts", 0.0)) <= cutoff})

    block["b1"] = all(lag == learner - actor
                      for lag, learner, actor in zip(lags, learners, actors, strict=True))
    # The second conjunct is PROVABLY redundant for the non-negative learner steps this stream
    # carries; kept as a stated-domain guard against a NEGATIVE learner step ever entering it.
    block["b2"] = max(learners) > min(learners) and max(learners) >= 1
    block["b3"] = max(actors) > min(actors)
    block["b4a"] = set(actors) <= ({0} | sync_steps)
    block["b4b"] = all(first <= second for first, second in zip(actors, actors[1:], strict=False))
    block["b4c"] = all(int(sample["actor_ckpt_step"]) >= _visible_floor(sample)
                       for sample in samples)
    block["b5a"] = (not negatives) and all(lag >= 0 for lag in lags)

    discriminating = sum(1 for learner, actor in zip(learners, actors, strict=True) if learner != actor)
    block["discriminating_samples"] = discriminating
    block["inversion_discrimination"] = "proven" if discriminating >= 1 else "unproven"

    for key in B_KEYS[1:]:  # table order
        if block[key] is False:
            failure, sub_reason = _LAG_FAILURES[key]
            block["verdict"] = "fail"
            block["failure"] = failure
            block["sub_reason"] = sub_reason
            return block

    # b5b — the inversion axis. `unproven` is a NON-GREEN outcome, never rc 0: at cadence 1 a
    # swapped-operand wiring is indistinguishable from a healthy one. At cadence > 1 the learner is
    # STRUCTURALLY ahead between syncs, so zero discriminating samples is frozen, not merely blind.
    if block["inversion_discrimination"] == "unproven":
        block["verdict"] = "fail"
        if int(cadence_steps) == 1:
            block["failure"] = "PreflightInversionUndiscriminatedError"
            block["inversion_reason"] = (
                f"actor_sync_cadence_steps == 1: 0 of {len(samples)} lag samples observed "
                "learner_step != actor_ckpt_step, so a SWAPPED-OPERAND wiring is "
                "indistinguishable from a healthy one on this burst"
            )
        else:
            block["failure"] = "PreflightLagFrozenError"
            block["sub_reason"] = "both"
            block["inversion_reason"] = (
                f"actor_sync_cadence_steps == {int(cadence_steps)}: the learner is "
                "structurally ahead between syncs, so zero discriminating samples means "
                "the reading is frozen on BOTH sides"
            )
    return block


def evaluate_assertions(events: list[dict], *, cadence_steps: int, burst_steps: int,
                        poll_interval_sec: float) -> dict:
    """The two dynamic assertions over one JSONL segment, in the report shape. Pure over the event
    stream, which is what makes (a) and (b) LAW-07-satisfiable while TD-1 blocks the composition:
    the stream comes from the REAL collaborators, and this is what the mutation corpus drives."""
    ground = _step_ground_truth(events, _named(events, "actor_lag_sample"))
    return {
        "a_sync": _evaluate_sync(events, cadence_steps=cadence_steps,
                                 burst_steps=burst_steps, ground=ground),
        "b_lag": _evaluate_lag(events, cadence_steps=cadence_steps,
                               poll_interval_sec=poll_interval_sec),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_block(path: Path, config: RunConfig) -> dict:
    return {"path": str(path), "sha256": _sha256(path), "run_id": config.run_id,
            "encoding": config.identity.encoding,
            "representation": config.identity.representation}


def _coordinator_block(config: RunConfig) -> dict:
    """The RESOLVED step-coordinator config, for the evidence artifact.

    The second half of a COMPARISON: `config.sha256` says what the operator wrote, this says what
    the composition root produced from it. DERIVED, never restated — every value comes from the
    SHIPPED resolvers via `dataclasses.asdict`. A disarmed arm is a NAMED `None`, not an omission,
    since an absent key would be indistinguishable from a block this function forgot to fill. It
    is taken in the PARENT, so it witnesses config -> resolver -> builder and nothing further.
    """
    from mantis.config.resolve.coordinator import resolve_coordinator_knobs
    from mantis.config.resolve.drain import resolve_drain_caps
    from mantis.config.resolve.draw_rate import resolve_draw_rate_abort
    from mantis.config.resolve.run_length import resolve_max_train_steps

    abort = resolve_draw_rate_abort(config.train)
    return {
        "source": "resolved in the preflight PARENT from the config's own resolvers; see"
                  " _coordinator_block for what this does and does not witness",
        "knobs": dataclasses.asdict(resolve_coordinator_knobs(config.train)),
        "drain_caps": dataclasses.asdict(resolve_drain_caps(config.monitor)),
        "stop_step": int(resolve_max_train_steps(config.train)),
        "draw_rate_abort": None if abort is None else dataclasses.asdict(abort),
    }


def _not_run_reason(report: dict) -> str:
    """The `not_run` disclaimer for (a) and (b), derived from what the run DID.

    The discriminator is `report["child"]`, the field that records whether a boot happened; the
    mode is merely NAMED. Keying on `mode` instead published "a boot was spawned and a burst
    attempted" beside `"child": null` on every failure landing before `_run_child`.
    """
    mode = report["mode"]
    if mode not in REPORT_MODES:
        raise PreflightInternalError(
            f"unknown report mode {mode!r} — the not_run disclaimer names the mode and there "
            "is no code-side default (R1). A fallback here would publish ANOTHER mode's "
            "disclaimer into the evidence artifact, which is exactly ADJ-13 F-3."
        )
    child = report.get("child")
    if child is None:
        return f"mode={mode} — {NOT_BOOTED_REASON}; see `failure` for where the run stopped"
    return (f"mode={mode} — {BOOTED_REASON} (child rc {child.get('rc')}); see `child`, "
            "`failure` and `events` for where it stopped")


def _finalise_not_run(report: dict) -> dict:
    """Re-derive every still-`not_run` disclaimer from the report's OWN final state: the report is
    built before the run and written in a `finally` (LAW-14), so construction-time text is a
    PREDICTION and this is where it is replaced by the measurement."""
    for name in ("a_sync", "b_lag"):
        block = report["assertions"][name]
        if block.get("verdict") == "not_run":
            block["reason"] = _not_run_reason(report)
    return report


def _tier_covered(report: dict) -> list[str]:
    """Which mint tiers this run actually COVERED — from the report's own verdicts.

    A tier is *requested* by the burst and *covered* by the outcome, and the two are not the same
    fact. `full` covers `sync_lag` because it clears every floor `sync_lag` clears and one more.
    """
    block = report.get("tier")
    if block is None:
        return []
    assertions = report.get("assertions") or {}
    verdicts = {(assertions.get(name) or {}).get("verdict") for name in ("a_sync", "b_lag")}
    if verdicts != {"pass"}:
        return []
    tier = block["tier"]
    if tier == TIER_FULL:
        return [TIER_SYNC_LAG, TIER_FULL]
    if tier == TIER_SYNC_LAG:
        return [TIER_SYNC_LAG]
    return []


def _tier_disclaimer(report: dict) -> str:
    """WHICH TIER RAN and WHAT IT DOES NOT PROVE, computed from the report's own fields so the
    disclaimer and the blocks beside it cannot disagree. An unknown tier is a NAMED internal
    failure: a fallback would publish one tier's disclaimer under another tier's name."""
    block = report.get("tier")
    tier = TIER_NONE if block is None else block["tier"]
    if tier not in TIER_NOT_PROVEN:
        raise PreflightInternalError(
            f"unknown mint tier {tier!r} — the tier disclaimer names the tier and there is no "
            "code-side default (R1). A fallback here would publish ANOTHER tier's 'what this "
            "does not prove' into the evidence artifact, which is exactly ADJ-13 F-3."
        )
    covered = _tier_covered(report)
    owed = [name for name in MINT_REQUIRED_TIERS if name not in covered]
    reached = ("The run REACHED a verdict on (a) sync-cadence and (b) lag-transport"
               if covered else
               "The run did NOT reach a verdict on (a) or (b), so NOTHING in this tier is "
               "demonstrated by this report")
    return (f"tier={tier} — {TIER_NOT_PROVEN[tier]}. {reached}. mint tiers still OWED: "
            f"{', '.join(owed) if owed else '(none)'}")


def _finalise_tier(report: dict) -> dict:
    """Re-derive `covered`, `owed` and the disclaimer from the report's OWN final state — the
    twin of `_finalise_not_run`, for the same reason: construction-time text is a PREDICTION."""
    block = report.get("tier")
    if block is None:
        return report
    covered = _tier_covered(report)
    block["covered"] = covered
    block["owed"] = [name for name in MINT_REQUIRED_TIERS if name not in covered]
    block["does_not_prove"] = _tier_disclaimer(report)
    return report


#: Which assertion blocks a mode's PASS verdict is DERIVED from. Audit mode never boots, so (a)
#: and (b) are `not_run` by construction. NO default: it would derive from another mode's rules.
MODE_REQUIRED_ASSERTIONS: dict[str, tuple[str, ...]] = {
    "audit": ("c_arming",),
    "preflight": ("a_sync", "b_lag", "c_arming"),
}


def _finalise_verdict(report: dict) -> dict:
    """Derive the top-level verdict from the assertion blocks, at write time.

    `_new_report` constructs the report already saying `verdict: "pass", rc: 0` and nothing on the
    success path ever SETS it, so a `BaseException` reaches the `finally` and writes `pass` while
    every assertion says `not_run`. DOWNGRADE ONLY: a recorded `failure` is left as the raising
    arm wrote it, and no report is ever promoted to `pass` here.
    """
    if report.get("verdict") != "pass":
        return report
    mode = str(report.get("mode"))
    required = MODE_REQUIRED_ASSERTIONS.get(mode)
    if required is None:
        raise PreflightInternalError(
            f"unknown report mode {mode!r} — the top-level verdict is DERIVED from this "
            f"mode's assertion blocks and there is no code-side default (R1). Known modes: "
            f"{sorted(MODE_REQUIRED_ASSERTIONS)}"
        )
    blocks = report.get("assertions") or {}
    unreached = [name for name in required
                 if (blocks.get(name) or {}).get("verdict") != "pass"]
    if not unreached:
        return report
    report["verdict"] = "not_reached"
    report["rc"] = PreflightVerdictUnreachedError.rc
    report["failure"] = PreflightVerdictUnreachedError.__name__
    report["verdict_unreached"] = unreached
    return report


def _report_name(report: dict) -> str:
    run_id = (report.get("config") or {}).get("run_id") or "unknown"
    stamp = report["ts_utc"].replace("-", "").replace(":", "")
    return f"preflight_{run_id}_{stamp}.json"


def _write_report(out_dir: Path, report: dict) -> None:
    """Write the evidence report — in a `finally`, ALWAYS (LAW-14).

    The one case a `finally` cannot cover is the write itself failing, and that is rc 41. The
    finalisers run HERE, not at the call site, so their invariants hold for every write path
    there will ever be, and the tier disclaimer is PRINTED from the finalised report so the
    terminal sentence is byte-identical to the one on disk.
    """
    _finalise_not_run(report)
    _finalise_tier(report)
    _finalise_verdict(report)
    tier_block = report.get("tier")
    if tier_block is not None:
        print(f"preflight: {tier_block['does_not_prove']}")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / _report_name(report)).write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PreflightReportUnwritableError(
            f"the evidence report could not be written to {out_dir}: {exc}"
        ) from exc


def _watchdog_reason(child: dict) -> str:
    """The parenthetical in rc-34's message, DERIVED from what the run actually READ.

    The old `or 'reason not found in the segment'` fallback named a search that in one reachable
    posture never happened — the same report's `events` block said `segments: [], lines: 0`. It is
    now derived from `child["segments_scanned"]`: `None` means no scan is recorded, `[]` means the
    scan ran and read nothing.
    """
    reason = child.get("fired_reason")
    if reason:
        return str(reason)
    scanned = child.get("segments_scanned")
    if scanned is None:
        return ("no segment scan is recorded on this child block, so no reason was read — "
                "see `events`")
    if not scanned:
        return ("NO segment was read for this run, so no reason could be found — see "
                "`events` and the child's own `stderr_tail`")
    return (f"{len(scanned)} segment(s) were read and none of them carries a "
            "`heartbeat_watchdog_fired` reason")


def _classify_child(child: dict) -> None:
    """The child's exit status is evaluated BEFORE any predicate — the anti-evasion rule. The
    first matching arm wins, and NO arm may be skipped because the event stream looks plausible."""
    rc = int(child["rc"])
    if child["timed_out"]:
        raise PreflightTimeoutError(
            f"the child exceeded --timeout-sec and was killed; observed state at kill time "
            f"is in the report. rc={rc}"
        )
    if rc < 0:
        raise PreflightChildSignaledError(
            f"the child died on signal {-rc} ({child.get('signal_name')}) — "
            f"{RC_CONVENTION}"
        )
    if rc in WATCHDOG_CODES:
        raise PreflightWatchdogFiredError(
            f"the run's own watchdog fired: child rc {rc} ({_watchdog_reason(child)})"
        )
    if rc == RELAUNCH_BUDGET_CODE:
        raise PreflightBootFailedError(
            f"child rc {rc} is the supervisor's RELAUNCH_BUDGET_EXIT_CODE and cannot "
            "legitimately be raised by a preflight child"
        )
    if rc in ARMED_ABORT_CODES:
        # BEFORE the rc-0 arm and the generic tail sniff, for the same anti-evasion reason arm 4
        # sits before arm 5: this rc is the child's own authored outcome, and it cannot reach the
        # [10, 41] pass-through, so without this arm it fell to rc 33.
        raise PreflightArmedAbortFiredError(
            rc, f"the run's own ARMED ABORT fired and stopped the run cooperatively: child "
                f"rc {rc}, the exit code `mantis.config.armed_aborts.MANIFEST` authors for "
                f"the rule that fired. The run unwound through its own close-out, so this is "
                f"a COMPLETED abort, not a crashed boot"
        )
    if rc == 0:
        return
    tail = str(child.get("stderr_tail") or "")
    if rc in PASS_THROUGH:
        # BEFORE the stderr sniff on purpose: the loader wrappers append the underlying exception
        # text, so a pydantic/yaml `'X' object has no attribute 'y'` lands in the tail of a child
        # that exited with its OWN named code, which sniffing first turned into rc 32.
        raise PreflightChildOutcomeError(
            rc, f"child exited {rc} with its own named outcome:\n{tail}")
    if "object has no attribute" in tail:
        raise PreflightTreeDefectError(
            "the child hit a TREE DEFECT — a real collaborator is missing a method the "
            "composition root calls. Nothing was supplied in its place (R64); the wall is "
            f"reported and carded. stderr tail:\n{tail}",
        )
    raise PreflightBootFailedError(f"child exited {rc}:\n{tail}")


def _read_segment(log_dir: Path, *, run_id: str) -> tuple[list[Path], list[dict]]:
    """The run's OWN segments, in filename order, and every event in them. The glob is scoped by
    `run_id` because `JsonlEventSink` writes `events_<run_id>_seg<NNNN>.jsonl`; an unscoped glob
    read a stale or foreign segment as THIS run's evidence."""
    segments = sorted(log_dir.glob(f"events_{run_id}_*.jsonl"))
    events: list[dict] = []
    for segment in segments:
        for line in segment.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return segments, events


def _events_block(segments: list[Path], events: list[dict]) -> dict:
    """The report's evidence-integrity block. `sha256` used to hash only the last segment while
    `lines` counted ALL of them — an integrity claim broader than what it hashed. It now hashes
    exactly the bytes that were read and NAMES every segment that went into it."""
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.read_bytes())
    return {"segments": [str(segment) for segment in segments],
            "segment": str(segments[-1]) if segments else None,
            "lines": len(events),
            "sha256": digest.hexdigest() if segments else None}


def _verdict_exit(blocks: dict) -> None:
    """The verdict -> exit-code seam, extracted so it has a producer at all.

    As three inline lines at the tail of `_run_preflight`, no test reached them — every preflight
    child dies at TD-4 first. Table order is (a) then (b), matching the report's own key order.
    """
    for block in (blocks["a_sync"], blocks["b_lag"]):
        if block["verdict"] != "pass":
            failure = str(block["failure"])
            raise PreflightAssertionsFailedError(
                failure,
                f"{failure} sub_reason={block.get('sub_reason')!r} "
                f"{block.get('inversion_reason') or ''}".strip())


def child_config_identity(events: list, *, parent_sha: str) -> tuple:
    """The child-side half of the identity witness: `(child_sha, verdict)` from scanned events.
    Three verdicts, honestly distinct — "match", "mismatch" (the caller raises rc 14) and
    "unwitnessed" (no `run_boot_identity` event; disclosed, never silently equal)."""
    identity = [e for e in events if e.get("event") == "run_boot_identity"]
    if not identity:
        return None, "unwitnessed"
    child_sha = str(identity[-1].get("config_sha256"))
    return child_sha, ("match" if child_sha == parent_sha else "mismatch")
