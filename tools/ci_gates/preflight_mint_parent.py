# >300 justify (R8): this file is the PARENT-ONLY half of tools/ci_gates/preflight_mint.py,
# split out at WPBOX Phase Q (CARD-PREFLIGHT-SPLIT-PARENT-HALF, R110-approved plan in
# GROUND_PFC.md). Producer for the figure: every line here was extracted VERBATIM from
# preflight_mint.py (S-3 hunk hashes in wp/WPBOX/DISPATCH_LOG.md); the prose-heavy style is
# that file's, deliberately unchanged. What lives here is exactly the leaf layers with NO
# dependency on the tool side: the shared vocabulary + exit taxonomy (one authority for both
# halves), the (a)/(b) assertion evaluators, the report helpers, the child classifier and the
# segment/verdict/identity leaves. What does NOT live here, and why, is stated in
# preflight_mint.py's own R8 header: the parser and child boot are pinned in-file by the
# byte-frozen oracle's AST/token census; the audit half stays beside the MANIFEST module
# globals the frozen ring-2 monkeypatch seam requires; the tier arithmetic reads
# `_burst_floors` (child closure); orchestration weaves both sides.
"""The parent-only half of the mint preflight (CI gate 12).

LOADED ONLY by tools/ci_gates/preflight_mint.py, via `spec_from_file_location` on a
`__file__`-relative path (never sys.path — the frozen O-3 census bans it). The tool
re-exports every public seam by plain assignment, so the module-path attribute authority
(`TOOL.evaluate_assertions`, `TOOL.PreflightManifestError`, ...) remains the tool file the
oracles load. This file must never import the tool (no cycle) and must never define a
`MANIFEST` global (the audit read path is the tool's, where the frozen monkeypatch seam
lives). The O-2/O-3 token bans are extended over this file by
tests/tools/test_preflight_parent_census.py — the census-extension arm GROUND_PFC §2.2
requires so a line that moved out of the frozen sweep does not leave the discipline.
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
#: The modes `_new_report` may be called with. DATA, and there is NO default (R1): a mode with
#: no entry is a named internal failure, because falling back would publish some other mode's
#: disclaimer, which is ADJ-13 F-3 itself.
REPORT_MODES: tuple[str, ...] = ("audit", "preflight")
#: The `not_run` disclaimer's two halves, selected by WHAT THE RUN DID rather than by what it
#: intended. See `_not_run_reason`.
NOT_BOOTED_REASON = ("NO boot was spawned and NO burst was attempted, so (a) sync-cadence and "
                     "(b) lag-transport had nothing to measure")
BOOTED_REASON = ("a boot WAS spawned and a burst attempted, but the run did not reach the "
                 "point where (a) and (b) could be evaluated")
#: The row key `_burst_floors` gives the draw-rate rule, named ONCE so `_burst_tier` can select
#: it without re-typing the dotted path. Two spellings of one row key is how a tier silently
#: stops matching the floor it is derived from.
DRAW_RATE_FLOOR_KEY = "train.draw_rate_abort.min_step"
#: The MINT TIERS (WPMINT Phase B / CARD-D-BURST-FLOOR). A tier names WHICH of `_burst_floors`'
#: rules the burst the validators ACCEPTED actually cleared — so, like `_not_run_reason`, it is
#: derived from what the run DID and never from what it intended. `--burst-steps` is a request;
#: only a burst that survives `_apply_burst_override` is a tier.
TIER_NONE = "none"
TIER_SYNC_LAG = "sync_lag"
TIER_FULL = "full"
#: BOTH are required for a mint, and `full` COVERS `sync_lag` (it clears every floor `sync_lag`
#: clears and one more), so a single green `full` run discharges both. That is not the card's
#: presumptive shape — two SEPARATE runs — and the deviation is MEASURED, not argued: on a
#: production config the short tier is unreachable. `PRODUCTION_CONFIGS` rows must arm
#: `draw_rate_collapse` (assertion (c), rc 30 otherwise), an armed row puts `min_step + 1` into
#: `_burst_floors`, and `_apply_burst_override` refuses anything under the max at rc 11. So the
#: only way to run a production config in tier `sync_lag` is to disarm the row the mint exists
#: to arm — a change to a run5 armed value (R82/R85, hard stop) and a faked axis (R64). The two
#: tiers are therefore two COVERAGE CLAIMS, not two runs.
MINT_REQUIRED_TIERS: tuple[str, ...] = (TIER_SYNC_LAG, TIER_FULL)
#: What each tier does NOT prove. DATA, and there is NO default (R1) — the same discipline as
#: `REPORT_MODES`, for the same reason: a tier with no entry is a named internal failure,
#: because falling back would publish ANOTHER tier's disclaimer into the evidence artifact,
#: which is ADJ-13 F-3 itself.
TIER_NOT_PROVEN: dict[str, str] = {
    # Worded to be true on EVERY route that lands here, which is not the same as worded for
    # the route it was written for. Drafting this as "no burst SURVIVED the cross-field
    # validators" was measured false in mode AUDIT on the first drive — AUDIT requests no
    # burst, so nothing was refused — and that is ADJ-13 F-3 committed inside the fix for it.
    # The claim is now pinned to the field that records the fact (`tier.burst_steps` is null)
    # rather than to a story about how it got that way.
    TIER_NONE: ("NO burst was accepted — `tier.burst_steps` is null, whether because none was "
                "requested (mode AUDIT) or because the run stopped at or before "
                "`_apply_burst_override` (rc 10 / 11 / 30 / 31) — so this report proves "
                "nothing about any tier: not (a) sync-cadence, not (b) lag-transport, and not "
                "that the run reaches the step at which train.draw_rate_abort can fire"),
    TIER_SYNC_LAG: ("the accepted burst clears the actor-lag and sync-cadence floors ONLY. "
                    "This config declares no " + DRAW_RATE_FLOOR_KEY + " floor, so NO burst "
                    "length on it can show the run reaching the draw-rate abort's first "
                    "firing step — tier `full` is UNAVAILABLE on this config, not merely "
                    "unrun, and a config that cannot reach tier `full` is not a config this "
                    "tool can preflight for a mint"),
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
#: The child's own named outcomes propagate unchanged (§6.3a arm 4).
PASS_THROUGH = range(10, 42)
#: Reserved by the run's own machinery — `monitor/heartbeat.py`, `monitor/supervise.py:39`.
WATCHDOG_CODES = (42, 43, 45)
RELAUNCH_BUDGET_CODE = 44
#: The cooperative half of the reserved band (WPMINT Phase X / R84). NOT a watchdog code: it is
#: not delivered by `os._exit` from the fire path, it is RETURNED by the child after the run
#: unwound through its own close-out. Read from the ONE authority rather than re-typed as 46 —
#: a literal here would be a third place the number is written (the constant, the manifest row,
#: and this), and the whole point of the rule-name carrier is that there is no such third place.
#: 47 joins it at WPMAIN RT-2/R132 (the disk-guard abort), by the same rule and for the same
#: reason: read from the ONE authority, never re-typed here. 48 joins at WP12-R Phase O
#: (R152, the broken terminal eval) — and it is exactly the hole 46 sat in before Phase X:
#: outside `PASS_THROUGH` and outside the reserved set, a child exiting 48 would fall through
#: every arm to `PreflightBootFailedError` and COLLAPSE TO 33, so the tool meant to surface
#: the authored signal would destroy it, on the very run an operator most needs the number
#: from. `RESERVED_CODES` below is derived and moves with this tuple.
ARMED_ABORT_CODES = (DRAW_RATE_COLLAPSE_EXIT_CODE, DISK_SPACE_EXHAUSTED_EXIT_CODE,
                     TERMINAL_EVAL_BROKEN_EXIT_CODE)
#: The full 42–47 band the docstring declares. Derived, so the docstring's claim and the three
#: tuples above cannot drift apart.
RESERVED_CODES = tuple(sorted({*WATCHDOG_CODES, RELAUNCH_BUDGET_CODE, *ARMED_ABORT_CODES}))


# ── named outcomes (§6.3) ─────────────────────────────────────────────────────────────
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
    """The child's published boot identity does not match the config the parent audited
    (F-B1 closure, WPCLEAN Phase RES): the evidence artifact would describe one config
    while the burst ran another. A NAMED failure, never a footnote in a green report."""
    rc = 14


class PreflightOutDirReusedError(PreflightError):
    """The out-dir already holds THIS run_id's event segments
    (CARD-PREFLIGHT-OUTDIR-REUSE, executed WPCLEAN Phase PFC): `_read_segment` scopes by
    run_id, so a same-run_id reuse of a dirty --out-dir would read a PREVIOUS burst's
    events as this run's evidence. Refused before the boot, a NAMED rc — never an mtime
    heuristic quietly picking which evidence to believe."""
    rc = 15


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
    """The run was interrupted before it reached a verdict (AUDIT-1 F-03).

    A `KeyboardInterrupt` during a long burst, or a callee's `SystemExit`, is a
    `BaseException`: it unwinds straight through `main`'s two `except` arms into the
    `finally` that writes the report. The artifact then landed carrying the SKELETON's
    `verdict: "pass", rc: 0` beside assertion blocks that all say `not_run` — a mint sign-off
    reads that file. rc 36 sits in the parent-side band, not the 42-47 band the run's own
    machinery reserves.
    """

    rc = 36


class PreflightVerdictUnreachedError(PreflightError):
    """The report says `pass` but its assertions never reached one (AUDIT-1 F-03).

    Never raised — it is the `failure` NAME `_finalise_verdict` stamps when it downgrades a
    report whose skeleton verdict outlived the run that was supposed to replace it. Carries
    an rc so the name and the code come from one place.
    """

    rc = 37


class PreflightTimeoutError(PreflightError):
    rc = 40


class PreflightReportUnwritableError(PreflightError):
    rc = 41


class PreflightAssertionsFailedError(PreflightError):
    """(a) or (b) failed. The rc and the reported NAME are both lifted from the failing
    block, so the report's `failure` and the process exit code have one authority."""

    def __init__(self, failure_name: str, message: str) -> None:
        super().__init__(message)
        self.failure_name = failure_name
        self.rc = FAILURE_CODES.get(failure_name, PreflightBootFailedError.rc)


class PreflightChildOutcomeError(PreflightError):
    """§6.3a arm 4 — the child's own named outcome, propagated UNCHANGED."""

    def __init__(self, rc: int, message: str) -> None:
        super().__init__(message)
        self.rc = int(rc)


class PreflightArmedAbortFiredError(PreflightError):
    """A manifest-registered armed abort FIRED, and its authored rc propagates UNCHANGED.

    Distinct from `PreflightWatchdogFiredError` (rc 34) on purpose, and the distinction is
    the card's: a watchdog fire is `os._exit` from a thread mid-run, so the parent reports
    its OWN rc 34 and the child's number is evidence inside the report. An armed abort is
    COOPERATIVE — the run decided to stop, unwound, saved, and returned the number the
    manifest authored for it. Rewriting that number to 34 (or to 33) would discard exactly
    the signal R84 opened this card to create, so the parent exits with the child's rc and a
    supervisor reads the same number either side of this tool.
    """

    def __init__(self, rc: int, message: str) -> None:
        super().__init__(message)
        self.rc = int(rc)


#: assertion-block failure name -> exit code (§6.3). One authority, so a report's `failure`
#: and the process rc can never disagree.
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


# ── the predicate surface (§7) ────────────────────────────────────────────────────────
def _named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


def _step_ground_truth(events: list[dict], samples: list[dict]) -> dict:
    """§7.3. The canonical `training_step` narration is `log_interval`-gated (1000), so a
    101-step burst carries none — and this reader deliberately does NOT count the trainer's
    per-step `trainer_step` diagnostic rows (delivered since F-R-P2B-2's sink threading):
    N comes from an independent terminal witness, and the report NAMES which one spoke, so
    a reader never has to guess."""
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
    """(a) — sync presence and cadence-consistency (§7.1, MF-4).

    `a2` is ORDERED-LIST equality, not set equality: a set collapses a duplicate, and the
    design's earlier set form PASSED an over-firing stream (measured at 102 events). `a1`
    exists separately only to carry the `extra` / `missed` sub-reason. `a4` is NOT a cadence
    predicate — a contiguous `sync_count` holds for any sequence one `ActorSync` produces;
    what it pins is single-producer / no sink line loss, and the report says so.
    """
    syncs = _named(events, "actor_sync")
    n = int(ground["value"])
    cadence = int(cadence_steps)
    # The `{1} ∪` term is not decoration: `maybe_sync` syncs UNCONDITIONALLY on the first
    # call (`actor_sync.py:63`), which is why "a burst too short to produce a sync" does not
    # exist. Dropping it is invisible at cadence 1, which is every minted config.
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
        # A different diagnosis from "synced at the wrong steps", so a different code: the
        # four sub-predicates stay None because nobody measured a cadence.
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
    for key in A_KEYS:  # table order — F-3: the reported name must be deterministic
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
    """(b) — the lag TRANSPORT (§7.4, MF-5). Every predicate is stated so a CONSTANT fails.

    b4 is split three ways. The old `max(l.actor_ckpt_step) == max(s.actor_ckpt_step)`
    conjunct was a false-positive generator on a healthy run — sampling stops at close-out
    while syncs continue, so the last reading is always behind the last sync. b4c replaces
    it with a `ts`-bounded lower bound that keeps b4's transport cross-check (a stale mirror
    still dies) without failing a healthy tail.
    """
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
        # b0 GATES the rest: reporting b1…b5a True over an absent measurement is a green
        # over nothing, which is the exact shape this whole gate exists to refuse.
        block["verdict"] = "fail"
        block["failure"] = "PreflightLagUnobservableError"
        return block

    learners = [int(sample["learner_step"]) for sample in samples]
    actors = [int(sample["actor_ckpt_step"]) for sample in samples]
    lags = [int(sample["lag_steps"]) for sample in samples]
    sync_steps = {int(event["step"]) for event in syncs}
    poll = float(poll_interval_sec)

    def _visible_floor(sample: dict) -> int:
        """The highest sync the watchdog MUST already have seen when it read this sample.

        Bounded by one full poll interval: the watchdog reads both callables at the top of
        `_check_actor_lag` and emits microseconds later, so a sync landing in that gap can
        legitimately make the sample look one step behind. Anything OLDER than `P` was
        necessarily visible at the read.
        """
        cutoff = float(sample.get("ts", 0.0)) - poll
        return max({0} | {int(event["step"]) for event in syncs
                          if float(event.get("ts", 0.0)) <= cutoff})

    block["b1"] = all(lag == learner - actor
                      for lag, learner, actor in zip(lags, learners, actors, strict=True))
    # CARD-B2-POSITIVITY-CONJUNCT (WPCLEAN Phase PFC, the card's justify arm): the second
    # conjunct is PROVABLY redundant for the non-negative learner steps this stream carries
    # (R72 row L5; recheck VERIFIED the proof) — `max > min >= 0` already forces `max >= 1`.
    # Kept as a stated-domain guard: it is load-bearing only if a NEGATIVE learner step ever
    # enters the stream, and deleting shipped predicate behaviour needs its own oracle event.
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

    for key in B_KEYS[1:]:  # table order — F-3
        if block[key] is False:
            failure, sub_reason = _LAG_FAILURES[key]
            block["verdict"] = "fail"
            block["failure"] = failure
            block["sub_reason"] = sub_reason
            return block

    # b5b — the inversion axis (MF-3). `unproven` is a NON-GREEN outcome, never rc 0: with
    # the lag operands exchanged, a cadence-1 config produces no negative lag and no
    # discriminating sample, and the design's earlier ruling returned rc 0 on a genuinely
    # inverted wiring. At cadence > 1 the learner is STRUCTURALLY ahead between syncs, so a
    # reading that never shows it is frozen (26), not merely undiscriminated (23).
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
    """The two dynamic assertions over one JSONL segment, in DESIGN §9.1's report shape.

    Pure over the event stream, which is what makes (a) and (b) LAW-07-satisfiable while
    TD-1 blocks the composition: the stream is produced by the REAL `ActorSync`, the REAL
    `HeartbeatWatchdog` and the REAL `JsonlEventSink`, and this function is what the
    mutation corpus drives.
    """
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
    """The RESOLVED step-coordinator config, for the evidence artifact (WPMINT Phase K-B).

    R78'S FIRST DESIGN QUESTION, ANSWERED YES. R78 made the preflight's JSON dump of the
    resolved coordinator config "where CARD-COORD-KNOBS starts — i.e. make the unauthored
    values visible in the mint record before deciding which become config". Phase K's census
    measured the answer at HEAD as NO: the report's key set was
    schema/tool_sha256/ts_utc/mode/verdict/rc/failure/config/override/manifest/assertions/
    child/events/tier, the tool built a coordinator config and read exactly `.capacity` from
    it, and the ~30 knobs that decide a run's shape were invisible in the artifact a mint
    sign-off reads.

    Two things changed with K-B, and together they turn the rider into a real instrument:

    * the values are AUTHORED now, so this block is no longer "here are some code constants".
      It is the second half of a COMPARISON — `config.sha256` says what the operator wrote,
      this says what the composition root actually produced from it. O-D1's named RED is
      exactly that gap ("the config says 0.25 while the runtime uses something else, so the
      audit reads the config, goes green, and the run is disarmed"), and until now nothing in
      the evidence artifact could witness it for anything but the two armed-abort rows;
    * it is DERIVED, never restated. Every value comes from the SHIPPED resolvers via
      `dataclasses.asdict`, and the key set comes from the dataclasses themselves — no
      literal key list, no hand-typed number. `_not_run_reason`'s doctrine names the opposite
      as the defect class: "a shipped constant asserting something the run measured
      otherwise, in the evidence artifact a mint sign-off reads". A hand-written census here
      would be that, one block over.

    THE DISARMED ARM IS A NAMED `None`, not an omission: `train.draw_rate_abort: null` is a
    real posture four of the five committed configs carry, and `draw_rate_abort: null` in the
    report is that posture stated. An absent KEY would be indistinguishable from a block this
    function forgot to fill.

    WHAT IT DOES NOT PROVE, stated because the tier block's own discipline demands it: this
    is the value the RESOLVERS produce from the config, taken in the PARENT. It is not a
    read-back from the booted child, so it witnesses the config -> resolver -> builder seam
    and NOT a child that was handed something else. That would need the child to publish its
    own coordinator config as an event, which is a bigger contract than this card owns.
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


# ── the evidence report (§9.1) ────────────────────────────────────────────────────────
def _not_run_reason(report: dict) -> str:
    """The `not_run` disclaimer for (a) and (b), derived from **what the run DID**.

    ADJ-13 F-3, and its CORRECTIVE PASS. **Class: a shipped constant asserting something the
    run measured otherwise, in the evidence artifact a mint sign-off reads.** The first fix
    keyed the disclaimer on `mode`, which is the run's INTENT, not its history — so every mode
    PREFLIGHT failure landing before `_run_child` (rc 10 config-path, rc 11 burst-below-floor,
    rc 30 arming, rc 31 manifest) published "a boot was spawned and a burst attempted" beside
    `"child": null`, and the sentence's own pointer ("see `child` … and `events`") aimed at two
    null fields. That is the SAME class the fix was closing, inverted: the pre-fix string was
    false about the mode and true about the boot; the post-fix string was true about the mode
    and false about the boot. Measured at rc 11 by the recheck, on the fix's own producer drive.

    So the discriminator is `report["child"]` — the field that records whether a boot happened —
    and the mode is merely NAMED, never used to assert a fact. The report is the single source:
    the sentence and the `child` block cannot disagree, because one is computed from the other.

    Called twice by design: once by `_new_report` (where `child` is `None`, and "no boot" is
    true at that instant) and once by `_finalise_not_run` immediately before the report is
    written, when `child` records what actually happened.
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
    """Re-derive every still-`not_run` disclaimer from the report's OWN final state.

    The report is built before the run and written in a `finally` (LAW-14), so any sentence
    stamped at construction time is a PREDICTION. This is where the prediction is replaced by
    the measurement. Blocks that reached a real verdict are left alone — only a block still
    saying `not_run` has a disclaimer to correct.
    """
    for name in ("a_sync", "b_lag"):
        block = report["assertions"][name]
        if block.get("verdict") == "not_run":
            block["reason"] = _not_run_reason(report)
    return report


def _tier_covered(report: dict) -> list[str]:
    """Which mint tiers this run actually COVERED — from the report's own verdicts.

    A tier is *requested* by the burst and *covered* by the outcome, and the two are not the
    same fact. At HEAD they are never the same fact: every mode-PREFLIGHT child dies at TD-4
    (CARD-POOL-ENCODING-BRIDGE) with (a) and (b) still `not_run`, so `covered` is `[]` and
    BOTH tiers stay owed. That is the answer the card wants published — a tier that cannot be
    run today says so, instead of a `not_run` an operator can read as optional.

    `full` covers `sync_lag` because it clears every floor `sync_lag` clears and one more; the
    subsumption is stated here, in the one place that computes coverage, rather than left for a
    reader of `MINT_REQUIRED_TIERS` to infer.
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
    """WHICH TIER RAN and WHAT IT DOES NOT PROVE — the card's explicit requirement.

    Same discipline as `_not_run_reason`, and deliberately the same shape: the sentence is
    computed from the report's own fields (`tier`, then the (a)/(b) verdicts), so the
    disclaimer and the blocks beside it cannot disagree. An unknown tier is a NAMED internal
    failure and never a fallback — a `.get(tier, <some tier>)` here would publish one tier's
    disclaimer under another tier's name, which is ADJ-13 F-3 with the mode swapped for the
    tier.
    """
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
    """Re-derive `covered`, `owed` and the disclaimer from the report's OWN final state.

    The twin of `_finalise_not_run`, and it runs beside it in `_write_report` for the same
    reason: everything stamped at construction time is a PREDICTION, and this is where the
    prediction is replaced by the measurement.
    """
    block = report.get("tier")
    if block is None:
        return report
    covered = _tier_covered(report)
    block["covered"] = covered
    block["owed"] = [name for name in MINT_REQUIRED_TIERS if name not in covered]
    block["does_not_prove"] = _tier_disclaimer(report)
    return report


#: Which assertion blocks a mode's PASS verdict is DERIVED from. Audit mode never boots, so
#: (a) and (b) are `not_run` by construction and are not its subject; preflight mode owns all
#: three. A mode with no entry is a named internal failure, on the same grounds as
#: `_not_run_reason`'s unknown-mode arm: falling back would derive the verdict from ANOTHER
#: mode's requirements, which is the class this table exists to close.
MODE_REQUIRED_ASSERTIONS: dict[str, tuple[str, ...]] = {
    "audit": ("c_arming",),
    "preflight": ("a_sync", "b_lag", "c_arming"),
}


def _finalise_verdict(report: dict) -> dict:
    """Derive the top-level verdict from the assertion blocks, at write time.

    AUDIT-1 F-03. `_new_report` constructs the report already saying `verdict: "pass", rc: 0`,
    and nothing on the success path ever SETS that verdict — the two `except` arms in `main`
    only overwrite it on failure. So a `BaseException` (a `KeyboardInterrupt` during a long
    burst, a callee's `SystemExit`) unwinds through both arms, reaches the `finally`, and
    writes an artifact whose top-level verdict is `pass` while every assertion says `not_run`.

    Contract #10 already says "a verdict that was REACHED is never overwritten". This is its
    CONVERSE, which did not exist: a verdict that was never reached is never PUBLISHED. The
    twin of `_finalise_not_run` and `_finalise_tier`, running beside them in `_write_report`
    for the same reason — everything stamped at construction time is a prediction.

    DOWNGRADE ONLY. A recorded `failure` is left exactly as the raising arm wrote it, and no
    report is ever promoted to `pass` here; the only edit this makes is turning an unearned
    `pass` into `not_reached` with the blocks that did not get there named in `failure`.
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
    """LAW-14: written in a `finally`, ALWAYS. The one case a `finally` cannot cover is the
    write itself failing, and that is rc 41 — loud and fatal, never a silent except.

    `_finalise_not_run` runs HERE rather than at the call site so that the invariant — no
    report on disk claims a boot its own `child` block does not record — holds for every write
    path there will ever be, and cannot be lost by a second caller forgetting the step.
    `_finalise_tier` rides the same rule for the same reason, and `_finalise_verdict` — the
    converse of contract #10 — joins them under AUDIT-1 F-03.

    The tier disclaimer is PRINTED from the finalised report rather than composed for stdout,
    so the sentence on the terminal is byte-identical to the one on disk. A second composition
    site is a second thing that can disagree with the artifact, which is the F-3 class this
    file keeps closing.
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

    ADJ-13 F-3's species, found by the R72 CLOSING PASS in the same report. **Class: a
    shipped constant asserting something the run measured otherwise, in the evidence
    artifact a mint sign-off reads.** This line was
    `child.get('fired_reason') or 'reason not found in the segment'`, and the fallback names
    a search that in one reachable posture NEVER HAPPENED. Measured on the shipped
    `_run_preflight`, three postures, one watchdog rc each:

        segment carries the event + a reason -> "(actor_lag_exceeded)"
        segment read, event NOT in it        -> "(reason not found in the segment)"   true
        `out_dir/logs` never written         -> "(reason not found in the segment)"   FALSE

    In the third the `events` block of the very same report says
    `segments: [], lines: 0, sha256: null` — nothing was read, so nothing could be "not found
    in" it, and the two indistinguishable sentences send an operator to look for a segment
    that does not exist. Exactly F-3: the report's own measurement contradicts the report's
    own sentence.

    So the sentence is derived from `child["segments_scanned"]`, the list `_run_preflight`
    publishes of the segments it actually read. `None` means no scan is recorded on this
    block at all (a child classified without a post-child scan, e.g. driven directly); `[]`
    means the scan ran and read nothing. The three are distinguishable in the message because
    they are distinguishable in the run, and each sentence is computed from the field beside
    it, so the two cannot disagree.
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
    """§6.3a's total order. The first matching arm wins, and NO arm may be skipped because
    the event stream looks plausible: the child's exit status is evaluated BEFORE any
    predicate, which is the anti-evasion rule."""
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
        # BEFORE the rc-0 arm and before the generic tail sniff, for the same anti-evasion
        # reason arm 4 sits before arm 5: this rc is the child's own authored outcome and a
        # later arm would rewrite it. It cannot reach the [10, 41] pass-through — 46 is
        # outside that range — so without this arm it fell to `PreflightBootFailedError`.
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
        # §6.3a arm 4, and it is BEFORE arm 5 on purpose. MF-I1: the reverse order re-creates
        # MF-6 — `_load` wraps any loader exception as `…raised: {exc}` and
        # `_apply_burst_override` appends `Validator said: {exc}`, so an underlying
        # `'X' object has no attribute 'y'` from pydantic or yaml lands in the tail of a
        # child that exited with its OWN named code. Testing the tail first made that child
        # exit 32 and lose its name, which is the exact defect arm 4 exists to prevent. A
        # child that named its own outcome is believed about it; the stderr sniff is the
        # fallback for children that did NOT.
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
    """The run's OWN segments, in filename order, and every event in them.

    SF-I2, half one. The glob is scoped by `run_id` because `JsonlEventSink` writes
    `events_<run_id>_seg<NNNN>.jsonl` (`monitor/sink.py:41`) and this function used to glob
    `events_*.jsonl` with no scope at all: a stale segment left in a reused `--out-dir`, or
    any other run's segment, was read as THIS run's evidence. Demonstrated by REVIEW-impl
    against a planted directory — a foreign `actor_sync` was consumed by the predicates.
    A same-`run_id` re-use of a dirty out-dir is still readable and is
    CARD-PREFLIGHT-OUTDIR-REUSE; the foreign-run case is closed here.
    """
    segments = sorted(log_dir.glob(f"events_{run_id}_*.jsonl"))
    events: list[dict] = []
    for segment in segments:
        for line in segment.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return segments, events


def _events_block(segments: list[Path], events: list[dict]) -> dict:
    """The report's evidence-integrity block.

    SF-I2, half two. `sha256` used to hash only `segments[-1]` while `lines` counted ALL of
    them, so the report's own integrity field did not cover the events it evaluated — an
    integrity claim strictly broader than what it hashed. It now hashes the concatenation of
    exactly the bytes that were read, and NAMES every segment that went into it.
    """
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.read_bytes())
    return {"segments": [str(segment) for segment in segments],
            "segment": str(segments[-1]) if segments else None,
            "lines": len(events),
            "sha256": digest.hexdigest() if segments else None}


def _verdict_exit(blocks: dict) -> None:
    """The verdict -> exit-code seam (§6.3), extracted so it has a producer at all.

    MF-I6: this was three inline lines at the tail of `_run_preflight`, and NO test in the
    repo reached them — every preflight child dies at TD-4 first, so routes RR-10 (`never
    raises`) and RR-32 (`FAILURE_CODES = {}`) both stayed green at full tier. It is a pure
    function of `blocks`, so the only thing that made it unproduced was where it lived.
    MF-3 and MF-7 both reduce to "rc 0 must mean what it says"; this is the code that makes
    the rc track the verdict, and it is now driven directly the way ORACLE drives
    `evaluate_assertions`.

    Table order is (a) then (b), matching the report's own key order.
    """
    for block in (blocks["a_sync"], blocks["b_lag"]):
        if block["verdict"] != "pass":
            failure = str(block["failure"])
            raise PreflightAssertionsFailedError(
                failure,
                f"{failure} sub_reason={block.get('sub_reason')!r} "
                f"{block.get('inversion_reason') or ''}".strip())


def child_config_identity(events: list, *, parent_sha: str) -> tuple:
    """The child-side half of F-B1's witness: `(child_sha, verdict)` from the scanned events.

    Three verdicts, honestly distinct: "match" (the child published an identity and it is
    the parent's), "mismatch" (published and DIFFERENT — the caller raises, rc 14), and
    "unwitnessed" (no `run_boot_identity` event in the scanned segments — a child that died
    before its sink existed, or a pre-closure segment; disclosed, never silently equal).
    The LAST event wins when several exist: segments are run_id-scoped and appended in boot
    order, so the last is the burst under audit.
    """
    identity = [e for e in events if e.get("event") == "run_boot_identity"]
    if not identity:
        return None, "unwitnessed"
    child_sha = str(identity[-1].get("config_sha256"))
    return child_sha, ("match" if child_sha == parent_sha else "mismatch")
