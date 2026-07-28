#!/usr/bin/env python3
# >300 justify (R8), stated at the file's MEASURED size rather than at the size it was
# written for. Producer for both figures: an AST transitive closure from `_boot_main` over
# this file's own top-level functions — 1284 lines total, 126 in the child closure, 809 in
# parent-only function bodies. Re-measured at ADJ-13's R72 CLOSING PASS (was 1234/126/764 at
# the corrective pass, 1156/126/691 before it, 1090/126/643 before that): this pass added 50
# lines — `_watchdog_reason` (F-3's species a second time: the rc-34 parenthetical derived
# from the segments the run actually READ instead of a constant that named a search which in
# one reachable posture never happened) and the post-child scan's `segments_scanned` record —
# every one of them on the PARENT side, and the child closure is STILL 126 lines and STILL
# exactly the same six functions, which is the justification's load-bearing claim. Two
# reasons, because the first covers only 126 of the 1284
# and saying so is the point (SF-7's own ruling: a justification that is not true is worse
# than none):
#
#  (1) CHILD SIDE (126 lines: _boot_main, _build_buffer, _load, _apply_burst_override,
#      _minimum_legal_burst, _resolve_config_path). The parent re-execs ITSELF as the boot
#      child by os.path.abspath(__file__) (DESIGN_P §6.2) — one file IS the containment
#      mechanism, not a packaging preference. Splitting parent from child replaces that
#      self-exec with a sibling-path exec, which is a second thing that can be wrong (a
#      moved, renamed or unshipped sibling) in the one code path whose whole job is to
#      survive a child that dies without unwinding.
#
#  (2) PARENT SIDE (809 lines: the two predicate evaluators, the classifier, the exit
#      taxonomy, the audit path, the report writer). Reason (1) does NOT reach these — they
#      could move to a sibling module with zero effect on the self-exec, and REVIEW-impl
#      measured exactly that (SF-I1: 462 parent-only lines before this pass added to them).
#      They stay for a different and narrower reason: `evaluate_assertions` and
#      `verify_source_pins` are the seams the
#      BYTE-FROZEN oracles load off this path (tests/tools/test_preflight_mint.py:100,120
#      and tests/config/test_armed_abort_manifest.py:70,84 both spec_from_file_location THIS
#      file), so relocating them is an R43 oracle event, not a refactor. When the freeze
#      lifts, the parent half should be split out and this clause deleted — carded as
#      CARD-PREFLIGHT-SPLIT-PARENT-HALF.
#
# Roughly half the file is comment carrying the "what defect does this line exist for"
# rationale LAW-07 wants; deleting it is what makes the next reader re-derive MF-5.
"""CI gate 12 (R61) — the mint preflight: one tool, two modes, one manifest.

Mode AUDIT (`--audit-only`): no boot, no burst, no GPU. Reads the committed production
configs through the REAL loader and audits assertion (c) — every `required` row of
`mantis.config.armed_aborts.MANIFEST` must be ARMED — plus manifest integrity and the R56
source-pin tamper scan. This is the per-commit CI gate. **rc 0 in this mode covers
assertion (c) ONLY**; assertions (a) and (b) are reported `not_run`, in the report and on
stdout, on every run including a green one.

Mode PREFLIGHT (`--config --burst-steps --out-dir --timeout-sec --device`): everything
AUDIT does, then the REAL `compose_run` boot in production posture, a bounded burst, a
timeout-bounded join, and assertions (a) sync-cadence and (b) lag-transport over the run's
own JSONL segment. **This is the MANUAL mint gate — no CI step invokes it.**

R64 posture, which is the whole point: this tool contains NO stand-in for a production
object. It constructs the real `Trainer` through the real `init_trainer`, the real
`WorkerPool`, the real buffer selected off `config.identity.representation` (never sniffed,
never defaulted — LAW-11), and hands them to the real `compose_run`, which builds
`build_run_safety`, `StepCoordinatorConfig` and `ActorSync` for itself. When a collaborator
is missing a method the tool does NOT supply one: the `AttributeError` reaches the process
boundary and is reported as `PreflightTreeDefectError` naming the attribute and the card.

At HEAD mode PREFLIGHT terminates on **CARD-POOL-ENCODING-BRIDGE** (TD-4): `WorkerPool`
construction calls `resolve_pool_encoding` -> `resolve_from_config`, which raises
`MissingEncodingError` because `RunConfig.model_dump()` carries `identity.encoding` and no
flat top-level `encoding` key. Parent rc **33**, child rc 1, ~1.4 s, with a real `Trainer`
already built through the real `init_trainer`. That is a useful preflight result: it is the
answer to "can run5 launch?", and the answer is no, for a nameable reason.

This docstring previously named CARD-TRAINSTEP-ADAPTER (`train/coordinator/step.py:573`,
rc 32) as the terminal wall, copying DESIGN_P §3.4. That was **measured false**
by IMPL and re-produced independently by REVIEW-impl: TD-4 fires FIRST, before `compose_run`
is ever called, so TD-1 is BEHIND TD-4, not in front of it. Corrected here rather than carded
— a gate whose own docstring states a measured-false fact is the first thing the next reader
believes about it. DESIGN_P's copy of the same sentence is CARD-DESIGN-P-3.4-ORDERING.

Containment is a SUBPROCESS, not a thread: `build_run_safety`'s `exit_fn` is `os._exit` and
`compose_run` does not override it, so an in-process boot that trips exit 42/43/45 dies
without unwinding and the evidence report is never written. A tool whose failure mode is
"no report" cannot report failure.

Exit codes — every outcome NAMED (DESIGN_P §6.3). A child rc in [10, 41] PROPAGATES
UNCHANGED, so a child exiting 12 exits the parent 12 rather than collapsing to 33. 42 / 43 /
44 / 45 are RESERVED by the run's own machinery and are never emitted by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mantis.config.armed_aborts import (
    EXEMPT_CONFIGS,
    MANIFEST,
    PRODUCTION_CONFIGS,
    ArmedAbort,
    Status,
    audit_arming,
)
from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import RunConfig

#: SF-4: every repo-root resolution lives HERE, never in the shipped package.
REPO_ROOT = Path(os.path.abspath(__file__)).resolve().parents[2]

#: O-4 / §5.4: the burst override writes exactly ONE dotted key and reads nothing, so
#: `stop_step` keeps exactly one source (`train.max_train_steps` -> `resolve_max_train_steps`
#: -> `run.py:167-170` -> `step.py:233`). The report's `override.keys` is emitted from this
#: same constant, so the two cannot disagree. A second entry here would make the preflight a
#: second run-length (or arming) authority, which is the R1 breach §5.4 discriminates against.
OVERRIDE_KEYS: tuple[str, ...] = ("train.max_train_steps",)

REPORT_SCHEMA = "preflight-mint-v1"
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
#: Recheck R-9: printed at the TOP of `_run_audit`, i.e. before `_audit_manifest_and_configs`
#: can raise, so it appears on rc-30 and rc-31 runs too. Made CONDITIONAL rather than moved:
#: it is the first line a CI log reader sees and it must be true on a red run as well as a
#: green one. The pinned substring `rc 0 covers assertion (c) ONLY` is preserved verbatim —
#: `tests/tools/test_preflight_mint.py::test_audit_only_is_green_on_the_real_tree` (BYTE-FROZEN,
#: `bd8e65e682c6a2dc`) asserts it, and rewording past it is an R43 event. Measured: the first
#: attempt at this nit dropped the substring and turned the frozen oracle red.
AUDIT_STDOUT_LINE = (
    "preflight: mode=AUDIT — assertions (a) sync and (b) lag were NOT RUN (no boot, no "
    "burst). If this run is green, rc 0 covers assertion (c) ONLY."
)
RC_CONVENTION = (
    "POSIX Popen.returncode — NEGATIVE on signal death, never 128+N"
)
#: The child's own named outcomes propagate unchanged (§6.3a arm 4).
PASS_THROUGH = range(10, 42)
#: Reserved by the run's own machinery — `monitor/heartbeat.py:39-46`, `monitor/supervise.py:39`.
WATCHDOG_CODES = (42, 43, 45)
RELAUNCH_BUDGET_CODE = 44


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


# ── the manifest's repo-root half (SF-4) ──────────────────────────────────────────────
def verify_source_pins(
    rows: "tuple[ArmedAbort, ...]", *, repo_root: "str | Path"
) -> "tuple[ArmedAbort, ...]":
    """R56 tamper-evidence: every `source_pin`'s exact text must still be in its file.

    Returns the BROKEN rows. The asymmetry is the load-bearing part
    (`silent_encoding_gate.py:338-344`): a pin that matches NOTHING — including a pinned
    file that no longer exists — is a HARD failure, never a quiet pass. That is what makes
    Phase D's deletion of `draw_rate_threshold: float = 0.0` impossible to forget.
    """
    root = Path(repo_root)
    broken: list[ArmedAbort] = []
    for row in rows:
        if row.source_pin is None:
            continue
        rel, text = row.source_pin
        pinned = root / rel
        if not pinned.is_file():
            broken.append(row)
            continue
        if text not in pinned.read_text(encoding="utf-8", errors="replace"):
            broken.append(row)
    return tuple(broken)


def _resolve_production_configs() -> "list[Path]":
    """PRODUCTION_CONFIGS holds repo-relative STRINGS (data); resolving them is ours."""
    return [REPO_ROOT / rel for rel in PRODUCTION_CONFIGS]


#: SF-4 again: the manifest module may make no filesystem call, so DISCOVERY lives here.
CONFIG_DIR_REL = "configs"


def _discovered_configs() -> "list[str]":
    """Every config actually on disk, repo-relative. The scope check's left-hand side.

    ADJ-13 F-1. This used to be its own flat `glob("*.yaml")` — a SECOND answer to "what is a
    config", beside gate 7's `**/*.yaml` + `**/*.yml`. Measured consequence, both halves:

    * a disarmed `configs/run6.yml` or `configs/prod/run6.yaml` validated under gate 7 (`OK
      configs/run6.yml`) and was **never audited** by gate 12 — rc 0, no UNDECLARED line, with
      the actor-lag abort off. MF-7's fix had been fitted to the reviewer's `run6.yaml` rather
      than to the class, so two of the three ways to add a config still walked through.
    * the INVERSE, which is worse than a scope miss: a subdirectory config could not be legally
      DECLARED either. Named in `PRODUCTION_CONFIGS` and present on disk, `configs/prod/run6.yaml`
      was reported STALE — "declared, absent from disk" — a false statement about a file the
      tool was looking straight at.

    Discovery is now `mantis.config.loader.discover_configs`, the same call gate 7 makes, and
    the path is emitted relative to the repo root INCLUDING subdirectory components, so a
    declaration can name what discovery finds. R71: one authority, and widening it widens both
    gates together.

    **Corrective pass (recheck R-2), as re-ruled by R75.** Unifying the two globs was not the
    class. The class was that discovery filtered by EXTENSION while `load_config` filtered by
    CONTENT, so the launchable set stayed strictly larger than the discovered set and the next
    suffix walked through: `configs/run6.txt` and `configs/run6.YAML` were schema-valid,
    DISARMED on the required row, and rc 0 from gate 7 AND gate 12. R75 declined closing that
    from the loader's side; the protection is the **shared-authority invariant** (loader accepts
    => audit sees), so `discover_configs` is name-agnostic and this function's left-hand side is
    complete against everything a run can be launched from under `configs/`, whatever it is
    called.
    """
    return sorted(path.relative_to(REPO_ROOT).as_posix()
                  for path in discover_configs(REPO_ROOT / CONFIG_DIR_REL))


def _config_declaration_drift() -> "tuple[list[str], list[str], list[str]]":
    """MF-I7 (i). The two tuples must PARTITION `discover_configs(configs/)` — EXACTLY the set
    gate 7 validates, which after ADJ-13 F-1 is one authority rather than two globs.

    Returns (undeclared, stale, overlapping) — all three empty is the only legal state:

    * **undeclared** — a config on disk named by neither tuple. This is the escape REVIEW-impl
      demonstrated: `sed 's/actor_lag_abort_enabled: true/…: false/' configs/run5.yaml >
      configs/run6.yaml` then `--audit-only` → **rc 0**, because a config that is not listed is
      never audited. Nothing pinned `discover_configs(configs/) ⊆ declared`.
    * **stale** — a tuple naming a config that is not on disk. R65's Phase D re-mints run5; if
      the re-mint lands under a new filename, an unchecked tuple goes on auditing a file
      nobody will run. This is `silent_encoding_gate.py:338-344`'s stale-`KNOWN_DEBT` rule.
    * **overlapping** — a config in BOTH tuples, i.e. two answers to one question.

    Why "audit every config in `configs/`" is NOT what this does, and why that matters: a
    bare discovery rule would silently start binding a config nobody classified, and the
    operator would learn about it from a red gate on an unrelated commit with no statement of
    intent anywhere. The partition keeps ONE authority for *which configs the law binds*
    (`PRODUCTION_CONFIGS`) and makes its COMPLETENESS machine-checked, so adding a config to
    `configs/` forces a one-line declaration on one side or the other. "Exempt" and
    "forgotten" stop being the same observable, which is the whole defect.

    **The scope of that claim, stated exactly** (ADJ-13 F-1 falsified the earlier, wider
    wording; the recheck falsified the replacement; R75 rules the third): "a config" here means
    ANY path under `configs/` that is not a real directory — precisely the set gate 7 validates,
    and a superset of everything `load_config` will read, which is the **shared-authority
    invariant**. There is no excluded class left to be a hole: a name the loader would read is a
    name this partition binds. Pinned by, in `tests/tools/test_preflight_mint_process.py`,
    `test_a_config_shaped_file_at_an_UNRECOGNISED_suffix_is_DISCOVERED_and_AUDITED`
    and
    `test_gate_12_is_RED_on_every_escape_that_ever_walked_through`,
    and at the loader level by `tests/config/test_config_discovery_authority.py`'s invariant
    row — so a later narrowing of discovery cannot silently re-open the gap.
    """
    present = set(_discovered_configs())
    production = set(PRODUCTION_CONFIGS)
    exempt = {rel for rel, _reason in EXEMPT_CONFIGS}
    return (sorted(present - production - exempt),
            sorted((production | exempt) - present),
            sorted(production & exempt))


def _audit_paths(named: "Path | None") -> "list[Path]":
    """The configs assertion (c) binds — ONE rule, used by BOTH modes (MF-I7 (ii)).

    `_run_audit` used to REPLACE the production set when `--config` was given while
    `_run_preflight` UNIONED it, so `--audit-only --config X` and the full preflight returned
    rc 0 and rc 30 on the same tree. Two authorities for one law, in one tool. Union is the
    safe direction: naming a config can only ever ADD scrutiny, never remove it.

    **Recheck R-6 — F-2's class at a site F-2's own census missed.** Set membership IS a
    path-identity comparison, and the two sides normalised differently: `named` arrives
    `.resolve()`d from `_resolve_config_path`, while `_resolve_production_configs()` returns a
    plain `REPO_ROOT / rel`. Under a symlinked `configs/run5.yaml` the union held both spellings
    of ONE config, which was then audited twice and published twice in `audited_configs`. The
    direction was fail-safe, which is why it survived; the SCOPING is the lesson — F-2 was
    censused over `os.path.abspath` call SITES rather than over path COMPARISONS. Both sides
    now normalise through the same call, so the set is a set of configs rather than of spellings.
    """
    paths = {path.resolve() for path in _resolve_production_configs()}
    if named is not None:
        paths.add(Path(named).resolve())
    return sorted(paths)


# ── the predicate surface (§7) ────────────────────────────────────────────────────────
def _named(events: "list[dict]", name: str) -> "list[dict]":
    return [event for event in events if event.get("event") == name]


def _step_ground_truth(events: "list[dict]", samples: "list[dict]") -> dict:
    """§7.3. There is no per-step event in a 101-step burst (`log_interval=1000`,
    `run.py:90`), so N comes from an independent witness — and the report NAMES which one
    spoke, so a reader never has to guess."""
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


def _evaluate_sync(events: "list[dict]", *, cadence_steps: int, burst_steps: int,
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


def _evaluate_lag(events: "list[dict]", *, cadence_steps: int,
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
                      for lag, learner, actor in zip(lags, learners, actors))
    block["b2"] = max(learners) > min(learners) and max(learners) >= 1
    block["b3"] = max(actors) > min(actors)
    block["b4a"] = set(actors) <= ({0} | sync_steps)
    block["b4b"] = all(first <= second for first, second in zip(actors, actors[1:]))
    block["b4c"] = all(int(sample["actor_ckpt_step"]) >= _visible_floor(sample)
                       for sample in samples)
    block["b5a"] = (not negatives) and all(lag >= 0 for lag in lags)

    discriminating = sum(1 for learner, actor in zip(learners, actors) if learner != actor)
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


def evaluate_assertions(events: "list[dict]", *, cadence_steps: int, burst_steps: int,
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


# ── config handling ───────────────────────────────────────────────────────────────────
def _resolve_config_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    fallback = REPO_ROOT / raw
    if fallback.is_file():
        return fallback.resolve()
    raise PreflightConfigError(f"config path does not exist: {raw!r}")


def _load(path: Path) -> RunConfig:
    """The ONE loader — `yaml.load` -> `RunConfig.model_validate` and nothing else."""
    try:
        return load_config(path)
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightConfigError(f"load_config({str(path)!r}) raised: {exc}") from exc


def _minimum_legal_burst(config: RunConfig) -> int:
    """The floor the two cross-field validators impose (`config/schema/core.py:280,307,314`)."""
    return max(int(config.monitor.actor_lag_threshold_steps),
               int(config.train.actor_sync_cadence_steps)) + 1


def _apply_burst_override(config: RunConfig, burst_steps: int) -> RunConfig:
    """`dump -> mutate ONE key -> model_validate` — byte-for-byte the loader's own final
    step (`loader.py:39-44`), so every cross-field validator re-runs. NOT F-3's route:
    F-3's route SKIPS the validators; this one IS the validator."""
    raw = config.model_dump()
    for dotted in OVERRIDE_KEYS:
        section, key = dotted.split(".")
        raw[section][key] = int(burst_steps)
    try:
        return RunConfig.model_validate(raw)
    except Exception as exc:
        minimum = _minimum_legal_burst(config)
        raise PreflightBurstTooShortError(
            f"--burst-steps {int(burst_steps)} does not survive the config's own cross-field "
            f"validators. The MINIMUM legal burst for this config is {minimum}.\n"
            f"  The binding rule: monitor.actor_lag_threshold_steps "
            f"({int(config.monitor.actor_lag_threshold_steps)}) and "
            f"train.actor_sync_cadence_steps "
            f"({int(config.train.actor_sync_cadence_steps)}) must BOTH be < "
            f"train.max_train_steps — 'a threshold the run never reaches is an invariant "
            f"that can never fire' (config/schema/core.py:307,314).\n"
            f"  Re-run with --burst-steps {minimum} or more.\n"
            f"  Validator said: {exc}",
            minimum=minimum, requested=int(burst_steps),
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_block(path: Path, config: RunConfig) -> dict:
    return {"path": str(path), "sha256": _sha256(path), "run_id": config.run_id,
            "encoding": config.identity.encoding,
            "representation": config.identity.representation}


# ── assertion (c) + the manifest (§8) ─────────────────────────────────────────────────
def _print_deferred_rows() -> None:
    """R56's loud print, on EVERY run including a green one: registered debt that stops
    being visible stops being debt and starts being the status quo
    (`silent_encoding_gate.py:346-351`, copied in shape, not reinvented)."""
    deferred = [row for row in MANIFEST if row.status is Status.DEFERRED]
    if not deferred:
        return
    print(f"preflight: {len(deferred)} DEFERRED abort row(s) — NOT audited, owned, not yet "
          "closed:")
    for row in deferred:
        print(f"  {row.name}  owner={row.owner}")
        print(f"    arming surface DOES NOT EXIST yet: {row.config_path}")
        if row.source_pin is not None:
            rel, text = row.source_pin
            print(f"    pinned to {rel}: {text!r}")
        # SF-I4 / LAW-08: `note` is where "WHY is this row deferred" lives, and it had no
        # live consumer at all — read only by the oracle's flip simulation. A deferred row
        # whose reason is invisible is a row nobody can re-adjudicate, which is the same
        # rot R56's loud print exists to prevent. Printed, not dropped.
        print(f"    why: {row.note}")


def _audit_manifest_and_configs(paths: "list[Path]") -> dict:
    """Assertion (c) plus manifest integrity. Raises the named outcome; returns the report
    block on success."""
    required = [row for row in MANIFEST if row.status is Status.REQUIRED]
    if not required or not PRODUCTION_CONFIGS:
        raise PreflightManifestError(
            "the armed-abort manifest is vacuous: an empty required set audits every "
            "config green, and an empty PRODUCTION_CONFIGS binds no config at all"
        )
    undeclared, stale, overlapping = _config_declaration_drift()
    if undeclared or stale or overlapping:
        reasons = dict(EXEMPT_CONFIGS)
        raise PreflightManifestError(
            "the config declaration no longer partitions the configs/ tree, so assertion "
            "(c)'s SCOPE is not knowable (MF-7; scope widened to gate 7's own discovery by "
            "ADJ-13 F-1 and made name-agnostic by R75 — every path at any depth under "
            "configs/ that is not a directory):\n"
            f"  UNDECLARED (on disk, in neither tuple — NEVER audited): {undeclared}\n"
            "    -> add each to PRODUCTION_CONFIGS (it gets audited) or to EXEMPT_CONFIGS "
            "with a written reason (R59 permits deliberate disarming off the production "
            "set). A config nobody declared is not exempt; it is forgotten.\n"
            f"  STALE (declared, absent from disk): {stale}\n"
            "    -> the declaration is auditing a file nobody will run; re-point or drop it.\n"
            f"  IN BOTH TUPLES: {overlapping}\n"
            f"  exemption reasons on record: {reasons}",
            undeclared=undeclared, stale=stale, overlapping=overlapping,
        )
    # MF-I3: the R56 scan's RESULT is what the report publishes. `source_pins_ok` used to be
    # the literal `True`, so deleting this call left the report claiming a scan that never
    # ran — measured: RR-08 removed the call with the whole default tier green. Both report
    # fields below are now derived from `broken` / `scanned`, so a deleted call is a
    # NameError, not a quiet green. §8.4 calls this scan "the forcing function that makes
    # Phase D's flip unforgettable"; a forcing function with no producer is LAW-07's phantom
    # gate input, which is what it was.
    scanned = [row.name for row in MANIFEST if row.source_pin is not None]
    broken = verify_source_pins(MANIFEST, repo_root=REPO_ROOT)
    if broken:
        raise PreflightManifestError(
            "source pin(s) no longer match their file — re-adjudicate the row rather than "
            f"editing the pin (R56): {[row.name for row in broken]}"
        )
    disarmed: list[str] = []
    audit = None
    for path in paths:
        audit = audit_arming(_load(path))
        for row in audit.disarmed:
            disarmed.append(f"{path.name}: {row.name} ({row.config_path})")
    if disarmed:
        raise PreflightArmingAuditError(
            "a REQUIRED armed-abort row is DISARMED on a production config — minting this "
            f"config re-enables the failure the abort exists to catch: {disarmed}"
        )
    return {
        "module_sha256": _sha256(REPO_ROOT / "src" / "mantis" / "config" / "armed_aborts.py"),
        "required": [row.name for row in (audit.required if audit else ())],
        # SF-I4 / LAW-08 again: `ArmedAbort.exit_code` was read by nothing but the oracle.
        # It is the code the abort FIRES with (45 for actor_lag), which is precisely what a
        # reader of a run's exit status needs to map that number back to a manifest row.
        "required_rows": [{"name": row.name, "config_path": row.config_path,
                           "exit_code": row.exit_code}
                          for row in (audit.required if audit else ())],
        # …and `AuditResult.deferred` — the tool used to re-derive this list from MANIFEST
        # directly (RR-07: the field could return `()` with the whole tier green). It is now
        # read from the audit's own result, so the published block and the audit cannot
        # disagree about which rows are deferred.
        "deferred": [{"name": row.name, "owner": row.owner, "config_path": row.config_path,
                      "source_pin": list(row.source_pin) if row.source_pin else None,
                      "note": row.note}
                     for row in (audit.deferred if audit else ())],
        "disarmed": [],
        "source_pins_ok": not broken,
        "source_pins_scanned": scanned,
        "audited_configs": [str(path) for path in paths],
        "exempt_configs": [rel for rel, _reason in EXEMPT_CONFIGS],
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


def _new_report(mode: str) -> dict:
    """The report skeleton. The `not_run` reason is a PREDICTION here (no boot has happened
    yet, and `child` is None, so "no boot was spawned" is true at this instant);
    `_finalise_not_run` re-derives it from the run's own history before the write."""
    not_run_reason = _not_run_reason({"mode": mode, "child": None})
    return {
        "schema": REPORT_SCHEMA,
        "tool_sha256": _sha256(Path(os.path.abspath(__file__))),
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode, "verdict": "pass", "rc": 0, "failure": None,
        "config": None, "override": None, "manifest": None,
        "assertions": {
            "a_sync": {"verdict": "not_run", "reason": not_run_reason},
            "b_lag": {"verdict": "not_run", "reason": not_run_reason},
            "c_arming": {"verdict": "not_run", "reason": "the audit did not complete"},
        },
        "child": None, "events": None,
    }


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
    """
    _finalise_not_run(report)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / _report_name(report)).write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PreflightReportUnwritableError(
            f"the evidence report could not be written to {out_dir}: {exc}"
        ) from exc


# ── CLI ───────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    """R1 posture at the CLI boundary: NO `default=` anywhere. Requiredness is per-MODE and
    is enforced below, because argparse cannot express "required in mode PREFLIGHT only"
    and gate 12 invokes `--audit-only` alone."""
    parser = argparse.ArgumentParser(
        prog="preflight_mint",
        description="CI gate 12 / the mint preflight (R61). --audit-only is the gate; the "
                    "full preflight is the MANUAL mint gate.",
    )
    parser.add_argument("--audit-only", action="store_true",
                        help="mode AUDIT: assertion (c) + manifest integrity, no boot")
    parser.add_argument("--config", help="path to the config to preflight")
    parser.add_argument("--burst-steps", type=int,
                        help="the burst length; overrides train.max_train_steps ONLY")
    parser.add_argument("--out-dir", help="evidence + run artifacts; must be OUTSIDE the repo")
    parser.add_argument("--timeout-sec", type=float, help="hard bound on the child boot")
    parser.add_argument("--device", help="torch device string for the trainer")
    parser.add_argument("--_boot", action="store_true", help=argparse.SUPPRESS)
    return parser


def _require_preflight_args(parser: argparse.ArgumentParser, args) -> None:
    missing = [name for name, value in (("--config", args.config),
                                        ("--burst-steps", args.burst_steps),
                                        ("--out-dir", args.out_dir),
                                        ("--timeout-sec", args.timeout_sec),
                                        ("--device", args.device)) if value is None]
    if missing:
        parser.error(
            "mode PREFLIGHT requires " + ", ".join(missing) +
            " — R1: there is no code-side default for any of them"
        )


def _git_toplevel() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, check=False)
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()
    return REPO_ROOT


def _checked_out_dir(raw: str) -> Path:
    """§9.2. The child's `log_dir` is a real `JsonlEventSink` writing `*.jsonl`, and gate 6
    rejects stray `*.jsonl` outside `tests/fixtures/`. A gate that can dirty the tree it
    gates is a gate that will — so the refusal lands BEFORE anything is created.

    ADJ-13 F-2. Both sides of the comparison must resolve symlinks or the comparison is
    between two different naming schemes. This line was `Path(os.path.abspath(resolved))`,
    which normalises `..` and makes absolute but does NOT follow symlinks, while
    `_git_toplevel()` returns a `.resolve()`d path — so a symlink whose target was inside the
    working tree compared unequal, the refusal never fired, and the tool wrote its evidence
    report into the repo (measured: `?? reports_redteam_probe/`, rc 33 from the boot wall
    rather than rc 13 from this guard). The `..`-relative and plain-absolute-inside forms were
    refused because `abspath` handles those TEXTUALLY; only the one form that needs the
    filesystem escaped, which is the signature of this class. `.resolve()` is non-strict, so a
    symlink to a path that does not exist yet still resolves to its target — which is exactly
    the case that matters, because the guard runs before anything is created.

    Recheck R-10: the line was `resolved = out_dir if out_dir.is_absolute() else (Path.cwd() /
    out_dir)` followed by `.resolve()`. Once `abspath` became `.resolve()` that cwd-join was
    DEAD — `Path("rel").resolve()` already resolves against the cwd — and the recheck's own
    mutation of it was a proven no-op at full tier. A conjunct that cannot change an outcome
    cannot have a flip row, so it is deleted rather than covered.
    """
    resolved = Path(raw).expanduser().resolve()
    toplevel = _git_toplevel()
    if resolved == toplevel or toplevel in resolved.parents:
        raise PreflightOutDirInsideRepoError(
            f"--out-dir {resolved} resolves INSIDE the repo working tree ({toplevel}). The "
            "preflight writes JSONL segments and checkpoints; gate 6 rejects those, so the "
            "gate would manufacture the violation it exists to guard against. Refused "
            "before anything was created."
        )
    return resolved


# ── mode PREFLIGHT: the child (§4) ────────────────────────────────────────────────────
def _build_buffer(config: RunConfig, capacity: int):
    """Selected off `config.identity.representation` — never sniffed off a live module,
    never defaulted (LAW-11). An unknown representation RAISES.

    MF-I4: this raise had NO producer. O-9 asserts only that the TOKENS `HexgBuffer`,
    `ReplayBuffer`, `identity`, `representation` appear in the source, and all four survive
    replacing the raise with a silent `ReplayBuffer` default — measured green at full tier
    (RR-12). Gate 11 cannot catch it either: `silent_encoding_gate.py:63` is
    `SCAN_ROOTS = ("src", "crates")`, so `tools/` is outside the scan.

    **Ruling on widening gate 11's SCAN_ROOTS to cover `tools/`: NOT taken here, and the
    reason is measured rather than argued.** Driving `scan()` with
    `SCAN_ROOTS = ("src","crates","tools")` scans 231 files and returns **2 violations**:
    `tools/ci_gates/silent_encoding_gate.py:126` (the gate's own `KNOWN_DEBT` table, a
    self-flag) and `tools/mint_opening_book.py:23` (`_MINT_ENCODING = "gnn_axis_v1"`). So the
    widening is not a one-line scope change — it needs a self-exemption and an adjudication of
    `mint_opening_book`, both of which are gate 11's owner's (WP12-R per its own KNOWN_DEBT
    row), not this phase's. Filed as CARD-GATE11-SCAN-TOOLS. The defect itself is closed the
    stronger way instead: `tests/tools/test_preflight_mint_process.py` drives all three arms
    of this function, so the raise has a real producer rather than a token census.
    """
    representation = config.identity.representation
    if representation == "graph":
        from mantis._engine import HexgBuffer

        return HexgBuffer(capacity, config.identity.encoding)
    if representation == "grid":
        from mantis._engine import ReplayBuffer

        return ReplayBuffer(capacity, config.identity.encoding)
    raise PreflightConfigError(
        f"identity.representation {representation!r} selects no buffer — an absent or "
        "unknown representation is an ERROR, never a dense default (LAW-11)"
    )


def _boot_main(args) -> int:
    """The `--_boot` child: the REAL production posture, with nothing routed around.

    Every collaborator here is the production object. `compose_run` builds `build_run_safety`,
    `StepCoordinatorConfig`, `ActorSync` and the eval pipeline for itself — this function
    passes none of them and patches none of them (R64 / O-9).
    """
    import torch

    from mantis.run import _default_step_coordinator_config, compose_run
    from mantis.selfplay.pool import WorkerPool
    from mantis.train.determinism import seed_everything
    from mantis.train.orchestrator import init_trainer

    config = _load(_resolve_config_path(args.config))
    booted = _apply_burst_override(config, args.burst_steps)
    seed_everything(booted.seed)

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    checkpoint_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_trainer(config=booted.model_dump(), checkpoint_dir=str(checkpoint_dir),
                           device=torch.device(args.device), sink=None)
    # §4.2: a run RESUMED past its ceiling terminates having performed zero syncs, which
    # looks EXACTLY like the frozen actor this preflight exists to find. The tool never
    # passes `checkpoint_path`, and a nonzero step here is a named refusal, not a warning.
    if int(trainer.step) != 0:
        raise PreflightResumedTrainerError(
            f"the freshly-built trainer reports step {int(trainer.step)}, not 0: a preflight "
            "over a resumed trainer measures nothing while looking like the defect it exists "
            "to find (§4.2)"
        )
    buffer = _build_buffer(booted, int(_default_step_coordinator_config().capacity))
    pool = WorkerPool(model=trainer.model, config=booted.model_dump(),
                      device=torch.device(args.device), replay_buffer=buffer,
                      arch=trainer.arch, sink=None, heartbeat=None)
    compose_run(config=booted, trainer=trainer, pool=pool, buffer=buffer,
                log_dir=log_dir, checkpoint_dir=checkpoint_dir,
                # ADJ-11: there is no `eval_enabled` config key — it is a `compose_run`
                # parameter with a code-side default True (`run.py:107`), filed as
                # CARD-EVAL-ENABLED-KEY. R64 BANS False as an escape, so the literal is
                # unconditional: not a flag, not a config read, not an expression.
                eval_enabled=True, run_id=booted.run_id)
    return 0


def _child_argv(args) -> "list[str]":
    return [sys.executable, os.path.abspath(__file__), "--_boot",
            "--config", str(args.config), "--burst-steps", str(int(args.burst_steps)),
            "--out-dir", str(args.out_dir), "--timeout-sec", str(float(args.timeout_sec)),
            "--device", str(args.device)]


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


def _run_child(args, report: dict) -> dict:
    started = time.monotonic()
    proc = subprocess.Popen(_child_argv(args), start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=float(args.timeout_sec))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=15.0)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
    rc = int(proc.returncode)
    child = {"rc": rc, "rc_convention": RC_CONVENTION,
             "raised_by": "child" if rc in PASS_THROUGH else "parent",
             "wall_clock_sec": round(time.monotonic() - started, 3), "timed_out": timed_out,
             "stdout_tail": (stdout or "")[-4000:], "stderr_tail": (stderr or "")[-4000:]}
    if rc < 0:
        child["signal"] = -rc
        child["signal_name"] = signal.Signals(-rc).name
    report["child"] = child
    return child


def _read_segment(log_dir: Path, *, run_id: str) -> "tuple[list[Path], list[dict]]":
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


def _events_block(segments: "list[Path]", events: "list[dict]") -> dict:
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


def _run_preflight(args, report: dict, out_dir: Path) -> None:
    path = _resolve_config_path(args.config)
    config = _load(path)
    report["config"] = _config_block(path, config)
    report["manifest"] = _audit_manifest_and_configs(_audit_paths(path))
    report["assertions"]["c_arming"] = {"verdict": "pass", "disarmed": [],
                                        "required_armed": report["manifest"]["required"]}
    booted = _apply_burst_override(config, args.burst_steps)
    report["override"] = {"keys": list(OVERRIDE_KEYS),
                          "from": int(config.train.max_train_steps),
                          "to": int(args.burst_steps),
                          "booted_config_sha256": hashlib.sha256(
                              json.dumps(booted.model_dump(), sort_keys=True,
                                         default=str).encode()).hexdigest()}
    child = _run_child(args, report)
    log_dir = out_dir / "logs"
    segments, events = (_read_segment(log_dir, run_id=booted.run_id)
                        if log_dir.is_dir() else ([], []))
    # R72 CLOSING PASS. The scan PUBLISHES what it read, so `_watchdog_reason` can say which
    # of "read and not found" / "nothing was read" actually happened instead of asserting the
    # first unconditionally. `[]` and "no scan recorded" are different facts and the child
    # block now distinguishes them.
    child["segments_scanned"] = [str(segment) for segment in segments]
    fired = [event for event in events if event.get("event") == "heartbeat_watchdog_fired"]
    if fired:
        child["fired_reason"] = fired[-1].get("reason")
    report["events"] = _events_block(segments, events)
    _classify_child(child)  # the child's status is evaluated BEFORE the predicates
    blocks = evaluate_assertions(events,
                                 cadence_steps=int(booted.train.actor_sync_cadence_steps),
                                 burst_steps=int(args.burst_steps),
                                 poll_interval_sec=float(
                                     booted.monitor.heartbeat_poll_interval_sec))
    report["assertions"]["a_sync"] = blocks["a_sync"]
    report["assertions"]["b_lag"] = blocks["b_lag"]
    _verdict_exit(blocks)


def _run_audit(args, report: dict) -> None:
    print(AUDIT_STDOUT_LINE)
    named = _resolve_config_path(args.config) if args.config else None
    paths = _audit_paths(named)
    # SF-I9: the manifest audit runs BEFORE anything indexes `paths`. It carries the vacuity
    # guard, so an empty PRODUCTION_CONFIGS is rc 31 by name rather than an `IndexError`
    # collapsing through the generic handler into an unnamed rc 1.
    report["manifest"] = _audit_manifest_and_configs(paths)
    subject = named if named is not None else paths[0]
    report["config"] = _config_block(subject, _load(subject))
    report["assertions"]["c_arming"] = {"verdict": "pass", "disarmed": [],
                                        "required_armed": report["manifest"]["required"]}


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "_boot"):
        try:
            return _boot_main(args)
        except PreflightError as exc:
            # §6.3a arm 4: the child exits with its OWN named code so the parent can
            # propagate it unchanged. A tree defect is deliberately NOT caught — its
            # traceback is what the parent classifies into rc 32.
            print(f"preflight child: rc {exc.rc} — {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return int(exc.rc)
    if not args.audit_only:
        _require_preflight_args(parser, args)

    report = _new_report("audit" if args.audit_only else "preflight")
    out_dir: Path | None = None
    rc = 0
    try:
        _print_deferred_rows()  # loud on EVERY run, including a green one (R56)
        if args.out_dir is not None:
            out_dir = _checked_out_dir(args.out_dir)
        if args.audit_only:
            _run_audit(args, report)
        else:
            assert out_dir is not None
            _run_preflight(args, report, out_dir)
    except PreflightError as exc:
        rc = int(exc.rc)
        name = getattr(exc, "failure_name", type(exc).__name__)
        report.update(verdict="fail", rc=rc, failure=name)
        print(f"PREFLIGHT NOT GREEN: rc {rc} — {name}: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — the tool's own failure is NAMED, never bare
        rc = PreflightInternalError.rc
        report.update(verdict="fail", rc=rc, failure="PreflightInternalError")
        print(f"PREFLIGHT NOT GREEN: rc {rc} — PreflightInternalError: {exc!r}",
              file=sys.stderr)
    finally:
        if out_dir is not None:
            try:
                _write_report(out_dir, report)
            except PreflightReportUnwritableError as exc:
                print(f"PREFLIGHT NOT GREEN: rc 41 — PreflightReportUnwritableError: {exc}",
                      file=sys.stderr)
                rc = PreflightReportUnwritableError.rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
