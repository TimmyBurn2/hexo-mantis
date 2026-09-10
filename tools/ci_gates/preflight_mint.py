#!/usr/bin/env python3
# >300 justify (R8): the parent re-execs ITSELF as the boot child by os.path.abspath(__file__),
# so ONE FILE is the containment mechanism rather than a packaging preference; the frozen token
# censuses sweep this file, and the audit read path must see THIS module's globals at call time
# for the `TOOL.MANIFEST` monkeypatch seam. The parent-only leaves live in
# `preflight_mint_parent.py` and are re-exported by plain assignment, so every oracle binding
# `TOOL.<name>` still binds one object.
"""CI gate 12 — the mint preflight: one tool, two modes, one manifest.

Mode AUDIT (`--audit-only`): no boot, no burst, no GPU. Reads the committed production configs
through the REAL loader and audits assertion (c) — every `required` manifest row must be ARMED
and still ABLE TO FIRE inside its own run — plus manifest integrity and the source-pin tamper
scan. rc 0 in this mode covers assertion (c) ONLY; (a) and (b) report `not_run` on every run.

Assertion (c)'s second half exists because `Mechanism.is_armed` reads a threshold while whether
the machinery that reads it ever RUNS is a different question: `gate_interval: 1000000000` on a
40-step run produced zero gate boundaries and audited green. `audit_cadence` computes each armed
row's earliest possible fire from that config's own cadence keys and refuses any row past
`EARLIEST_FIRE_FRACTION * train.max_train_steps` at rc 30; the one sanctioned disarm is an
explicit deferred row with an owner and a source pin, never a large interval. Every row is judged
IN ITS OWN SAMPLE CLOCK, and a clock whose period cannot be derived is rc 31 rather than a silent
fall back to the step clock — an axis judged in a clock it does not tick in audits GREEN on
exactly the configs it exists to refuse. `_cadence_self_test` proves the trigger fires in both
directions before any verdict is published.

Mode PREFLIGHT (`--config --burst-steps --out-dir --timeout-sec`): everything AUDIT does, then
the REAL `compose_run` boot in production posture, a bounded burst, a timeout-bounded join, and
assertions (a) sync-cadence and (b) lag-transport over the run's own JSONL segment. **This is the
MANUAL mint gate — no CI step invokes it.**

This tool contains NO stand-in for a production object and constructs none: the child calls the
same builder/composer pair `launch_run` calls, and when a collaborator is missing a method the
failure reaches the process boundary uncaught. rc 32 is a FALLBACK SNIFF — the literal
`"object has no attribute"` in the child's stderr tail — never a wall registry; anything else
lands rc 33 with its traceback in the tail.

CARD-POOL-ENCODING-BRIDGE has landed: `resolve_from_config` reads `identity.encoding` in its
nested shape, so the `MissingEncodingError` that used to stop `WorkerPool` construction is gone.
This docstring previously named `train/coordinator/step.py` as the terminal wall; that was
measured false — the encoding wall fired FIRST, before `compose_run` was ever called.

Where the boot stops now is a config fact. A `train.device: cuda` config dies in `init_trainer`
on a non-CUDA box (rc 33), which is what stops a cpu preflight false-clearing a cuda run's GPU
wall; and production configs mint `inference.fused_graph_caps` placeholders that are schema-VALID
so the repo ships a complete config and runtime-REFUSED so an uncalibrated config cannot
construct its graph inference server — `UncalibratedFusedGraphCapsError` at the `WorkerPool`
seam. Calibrating restores clean-boot evidence, and `fused_graph_caps_calibrated` is a DEFERRED
manifest row so every run says so out loud.

Containment is a SUBPROCESS, not a thread: `build_run_safety`'s `exit_fn` is `os._exit`, so an
in-process boot that trips exit 42/43/45 dies without unwinding and no report is written.

Exit codes. Every outcome this tool DIAGNOSES is NAMED; rc 1 `PreflightInternalError` is the
catch-all for one it does not, so a NEW rc 1 is a finding. A child rc in [10, 41] PROPAGATES
UNCHANGED, and 42-48 are RESERVED by the run's own machinery and never an assertion outcome here.
46 sits outside the pass-through band, so without its own arm a child exiting 46 would collapse
to rc 33 and destroy the authored abort signal; `_boot_main` resolves the number from
`RunHandles.shutdown.abort_rule` through `exit_code_for_abort` and nowhere else, and a rule that
fired with NO authored code is rc 33 naming the rule.

MINT TIERS. The report says what the burst it ran does and does not prove, in a `tier` block
derived from `_burst_floors`: `none`, `sync_lag` and `full`, where `full` COVERS `sync_lag`. On a
production config `sync_lag` is UNREACHABLE, since arming `draw_rate_collapse` is assertion (c)
and an armed row puts `min_step + 1` in the floor set: measured, run5's floor is 25001 and the
other four `configs/` entries' floor is 101. What `full` costs is a LOWER BOUND never measured
here — at the recorded 41.66 ms/train-step, 25001 steps is >= 1041.5 s of pure train-step
compute, and the coordinator is GAME-BOUND, so it also needs 25001 completed games.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from mantis.config.armed_aborts import (
    EARLIEST_FIRE_FRACTION,
    EXEMPT_CONFIGS,
    MANIFEST,
    PRODUCTION_CONFIGS,
    RUN_LENGTH_PATH,
    ArmedAbort,
    ArmingSurfaceMissingError,
    Cadence,
    SampleClock,
    SampleClockNotDerivableError,
    Status,
    audit_arming,
    audit_cadence,
    exit_code_for_abort,
)
from mantis.config.loader import config_identity_sha256, discover_configs, load_config
from mantis.config.schema import RunConfig
from mantis.diagnostics.workspace_durability import WorkspaceNotDurableError, assert_durable

#: Every repo-root resolution lives HERE, never in the shipped package.
REPO_ROOT = Path(os.path.abspath(__file__)).resolve().parents[2]


#: The parent-only half loads off THIS file's own directory — never sys.path — and is re-exported
#: by PLAIN assignment so every oracle binding `TOOL.<name>` binds one object. The sys.modules
#: guard keys on the sibling's resolved path, so two trees each get their own sibling.
_PARENT_HALF_PATH = Path(__file__).resolve().with_name("preflight_mint_parent.py")
_PARENT_HALF_MODULE = "_preflight_mint_parent"


def _load_parent_half():
    import importlib.util

    cached = sys.modules.get(_PARENT_HALF_MODULE)
    # `cached is not None` is provably redundant; kept as a stated-domain guard.
    if cached is not None and getattr(cached, "__file__", None) == str(_PARENT_HALF_PATH):
        return cached
    spec = importlib.util.spec_from_file_location(_PARENT_HALF_MODULE, _PARENT_HALF_PATH)
    # Both leaves are pyright-load-bearing (the Optional contract must be narrowed before
    # `spec.loader.exec_module`) and unproducible at runtime for a file that exists.
    if spec is None or spec.loader is None:
        raise ImportError(f"the preflight parent half is missing beside the tool: "
                          f"{_PARENT_HALF_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PARENT_HALF_MODULE] = module
    spec.loader.exec_module(module)
    return module


_parent_half = _load_parent_half()

# Shared vocabulary — ONE authority (the sibling), these names re-published unchanged.
A_KEYS = _parent_half.A_KEYS
B_KEYS = _parent_half.B_KEYS
A4_PINS = _parent_half.A4_PINS
B1_SCOPE = _parent_half.B1_SCOPE
REPORT_MODES = _parent_half.REPORT_MODES
NOT_BOOTED_REASON = _parent_half.NOT_BOOTED_REASON
BOOTED_REASON = _parent_half.BOOTED_REASON
DRAW_RATE_FLOOR_KEY = _parent_half.DRAW_RATE_FLOOR_KEY
TIER_NONE = _parent_half.TIER_NONE
TIER_SYNC_LAG = _parent_half.TIER_SYNC_LAG
TIER_FULL = _parent_half.TIER_FULL
MINT_REQUIRED_TIERS = _parent_half.MINT_REQUIRED_TIERS
TIER_NOT_PROVEN = _parent_half.TIER_NOT_PROVEN
RC_CONVENTION = _parent_half.RC_CONVENTION
PASS_THROUGH = _parent_half.PASS_THROUGH
WATCHDOG_CODES = _parent_half.WATCHDOG_CODES
RELAUNCH_BUDGET_CODE = _parent_half.RELAUNCH_BUDGET_CODE
ARMED_ABORT_CODES = _parent_half.ARMED_ABORT_CODES
RESERVED_CODES = _parent_half.RESERVED_CODES
#: Re-published THROUGH the sibling, which imports it from the one authority
#: (mantis.monitor.heartbeat); a direct re-import here would be a second chain for one number.
DRAW_RATE_COLLAPSE_EXIT_CODE = _parent_half.DRAW_RATE_COLLAPSE_EXIT_CODE

# Exit taxonomy — the same single object set on both sides of the split.
PreflightError = _parent_half.PreflightError
PreflightInternalError = _parent_half.PreflightInternalError
PreflightConfigError = _parent_half.PreflightConfigError
PreflightBurstTooShortError = _parent_half.PreflightBurstTooShortError
PreflightResumedTrainerError = _parent_half.PreflightResumedTrainerError
PreflightOutDirInsideRepoError = _parent_half.PreflightOutDirInsideRepoError
PreflightConfigIdentityError = _parent_half.PreflightConfigIdentityError
PreflightOutDirReusedError = _parent_half.PreflightOutDirReusedError
PreflightWorkspaceNotDurableError = _parent_half.PreflightWorkspaceNotDurableError
PreflightCudaBuildError = _parent_half.PreflightCudaBuildError
PreflightArmingAuditError = _parent_half.PreflightArmingAuditError
PreflightManifestError = _parent_half.PreflightManifestError
PreflightTreeDefectError = _parent_half.PreflightTreeDefectError
PreflightBootFailedError = _parent_half.PreflightBootFailedError
PreflightWatchdogFiredError = _parent_half.PreflightWatchdogFiredError
PreflightChildSignaledError = _parent_half.PreflightChildSignaledError
PreflightTimeoutError = _parent_half.PreflightTimeoutError
PreflightInterruptedError = _parent_half.PreflightInterruptedError
PreflightVerdictUnreachedError = _parent_half.PreflightVerdictUnreachedError
PreflightReportUnwritableError = _parent_half.PreflightReportUnwritableError
PreflightAssertionsFailedError = _parent_half.PreflightAssertionsFailedError
PreflightChildOutcomeError = _parent_half.PreflightChildOutcomeError
PreflightArmedAbortFiredError = _parent_half.PreflightArmedAbortFiredError
FAILURE_CODES = _parent_half.FAILURE_CODES

# The parent-half functions the oracles and the orchestration bind off this module path.
_named = _parent_half._named
_step_ground_truth = _parent_half._step_ground_truth
_evaluate_sync = _parent_half._evaluate_sync
_LAG_FAILURES = _parent_half._LAG_FAILURES
_evaluate_lag = _parent_half._evaluate_lag
evaluate_assertions = _parent_half.evaluate_assertions
_sha256 = _parent_half._sha256
_config_block = _parent_half._config_block
_coordinator_block = _parent_half._coordinator_block
_not_run_reason = _parent_half._not_run_reason
_finalise_not_run = _parent_half._finalise_not_run
_tier_covered = _parent_half._tier_covered
_tier_disclaimer = _parent_half._tier_disclaimer
_finalise_tier = _parent_half._finalise_tier
_finalise_verdict = _parent_half._finalise_verdict
MODE_REQUIRED_ASSERTIONS = _parent_half.MODE_REQUIRED_ASSERTIONS
_report_name = _parent_half._report_name
_write_report = _parent_half._write_report
_watchdog_reason = _parent_half._watchdog_reason
_classify_child = _parent_half._classify_child
_read_segment = _parent_half._read_segment
_events_block = _parent_half._events_block
_verdict_exit = _parent_half._verdict_exit
child_config_identity = _parent_half.child_config_identity


#: The burst override writes exactly ONE dotted key and reads nothing, so `stop_step` keeps
#: one source. The report's `override.keys` is emitted from this same constant. A second entry
#: would make the preflight a second run-length authority.
OVERRIDE_KEYS: tuple[str, ...] = ("train.max_train_steps",)

REPORT_SCHEMA = "preflight-mint-v1"
#: Printed at the TOP of `_run_audit`, before `_audit_manifest_and_configs` can raise, so it
#: appears on rc-30 and rc-31 runs too. The pinned substring `rc 0 covers assertion (c) ONLY` is
#: preserved verbatim: a BYTE-FROZEN oracle asserts it and rewording past it turned that red.
AUDIT_STDOUT_LINE = (
    "preflight: mode=AUDIT — assertions (a) sync and (b) lag were NOT RUN (no boot, no "
    "burst). If this run is green, rc 0 covers assertion (c) ONLY."
)


def verify_source_pins(
    rows: tuple[ArmedAbort, ...], *, repo_root: str | Path
) -> tuple[ArmedAbort, ...]:
    """Tamper-evidence: every `source_pin`'s exact text must still be in its file, returning the
    BROKEN rows. The asymmetry is load-bearing — a pin that matches NOTHING, a missing pinned
    file included, is a HARD failure. It does NOT prove the right VALUE flows."""
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


def _resolve_production_configs() -> list[Path]:
    """PRODUCTION_CONFIGS holds repo-relative STRINGS (data); resolving them is ours."""
    return [REPO_ROOT / rel for rel in PRODUCTION_CONFIGS]


#: The manifest module may make no filesystem call, so DISCOVERY lives here.
CONFIG_DIR_REL = "configs"


def _discovered_configs() -> list[str]:
    """Every config actually on disk, repo-relative — the scope check's left-hand side.

    Discovery is `discover_configs`, the same call gate 7 makes, emitted relative to the repo root
    INCLUDING subdirectory components: a second flat glob here let a disarmed config validate
    under gate 7 and never be audited. It is also name-agnostic, because filtering by EXTENSION
    while `load_config` filters by CONTENT leaves the launchable set strictly larger.
    """
    return sorted(path.relative_to(REPO_ROOT).as_posix()
                  for path in discover_configs(REPO_ROOT / CONFIG_DIR_REL))


def _config_declaration_drift() -> tuple[list[str], list[str], list[str]]:
    """The two tuples must PARTITION `discover_configs(configs/)` — exactly the set gate 7
    validates. Returns (undeclared, stale, overlapping); all three empty is the only legal state.

    Deliberately not "audit every config in `configs/`": a bare discovery rule silently starts
    binding a config nobody classified, whereas the partition keeps ONE authority for which
    configs the law binds and makes its COMPLETENESS machine-checked, so "exempt" and "forgotten"
    stop being the same observable. An UNDECLARED config is never audited — copying run5 with the
    actor-lag abort flipped off audited rc 0 — a STALE one audits a file nobody will run, and an
    overlapping one is two answers to one question.
    """
    present = set(_discovered_configs())
    production = set(PRODUCTION_CONFIGS)
    exempt = {rel for rel, _reason in EXEMPT_CONFIGS}
    return (sorted(present - production - exempt),
            sorted((production | exempt) - present),
            sorted(production & exempt))


def _audit_paths(named: Path | None) -> list[Path]:
    """The configs assertion (c) binds — ONE rule, used by BOTH modes.

    Union, not replacement: naming a config can only ever ADD scrutiny. Both sides normalise
    through the same `.resolve()` call, so the set is a set of configs rather than of spellings —
    under a symlinked config the union once held both spellings and audited it twice.
    """
    paths = {path.resolve() for path in _resolve_production_configs()}
    if named is not None:
        paths.add(Path(named).resolve())
    return sorted(paths)


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


def _burst_floors(config: RunConfig) -> list[tuple[str, int, int]]:
    """Every cross-field rule that binds the burst from below: (key, its value, its floor).
    Enumerated rather than folded into one number, because an operator told only the maximum
    cannot see WHICH rule moved their floor from 101 to 25001."""
    floors = [("monitor.actor_lag_threshold_steps",
               int(config.monitor.actor_lag_threshold_steps),
               int(config.monitor.actor_lag_threshold_steps) + 1),
              ("train.actor_sync_cadence_steps",
               int(config.train.actor_sync_cadence_steps),
               int(config.train.actor_sync_cadence_steps) + 1)]
    block = config.train.draw_rate_abort
    if block is not None:
        floors.append((DRAW_RATE_FLOOR_KEY, int(block.min_step), int(block.min_step) + 1))
    return floors


def _minimum_legal_burst(config: RunConfig) -> int:
    """The floor the cross-field validators impose (`config/schema/core.py:280,307,314,321`)."""
    return max(floor for _key, _value, floor in _burst_floors(config))


def _burst_tier(config: RunConfig, burst_steps: int) -> str:
    """Which mint tier a burst of this length on this config IS, read off `_burst_floors` so the
    tier a report claims and the floor arithmetic an operator was shown cannot drift apart.

    The `sync_lag` arm is why the tier is not just "cleared the max": on
    `train.draw_rate_abort: null` the max floor is 101 and clearing it says nothing whatever
    about draw-rate reachability.
    """
    floors = _burst_floors(config)
    if int(burst_steps) < max(floor for _key, _value, floor in floors):
        return TIER_NONE
    draw = [floor for key, _value, floor in floors if key == DRAW_RATE_FLOOR_KEY]
    return TIER_FULL if draw else TIER_SYNC_LAG


def _apply_burst_override(config: RunConfig, burst_steps: int) -> RunConfig:
    """`dump -> mutate ONE key -> model_validate` — byte-for-byte the loader's own final step,
    so every cross-field validator re-runs rather than being skipped."""
    raw = config.model_dump()
    for dotted in OVERRIDE_KEYS:
        section, key = dotted.split(".")
        raw[section][key] = int(burst_steps)
    try:
        return RunConfig.model_validate(raw)
    except Exception as exc:
        minimum = _minimum_legal_burst(config)
        rules = "".join(
            f"    {key} ({value}) must be < train.max_train_steps, so its floor is {floor}\n"
            for key, value, floor in _burst_floors(config)
        )
        raise PreflightBurstTooShortError(
            f"--burst-steps {int(burst_steps)} does not survive the config's own cross-field "
            f"validators. The MINIMUM legal burst for this config is {minimum}.\n"
            f"  The binding rules — 'a threshold the run never reaches is an invariant that "
            f"can never fire' (config/schema/core.py:307,314,321):\n"
            f"{rules}"
            f"  Re-run with --burst-steps {minimum} or more.\n"
            f"  Validator said: {exc}",
            minimum=minimum, requested=int(burst_steps),
        ) from exc


def _print_deferred_rows(*, manifest: tuple[ArmedAbort, ...] = MANIFEST) -> None:
    """Print registered debt on EVERY run including a green one: debt that stops being visible
    stops being debt. `manifest` is a keyword because the SHIPPED manifest can hold zero deferred
    rows, and keeping one deferred to keep an assertion true would shape it to suit a test."""
    deferred = [row for row in manifest if row.status is Status.DEFERRED]
    if not deferred:
        return
    print(f"preflight: {len(deferred)} DEFERRED abort row(s) — NOT audited, owned, not yet "
          "closed:")
    for row in deferred:
        print(f"  {row.name}  owner={row.owner}")
        # A row can be deferred because its arming surface is missing OR because nobody
        # pre-registered a value; `surface` is hoisted out of the f-string because a replacement
        # field spanning a line break is 3.12-only syntax, a SyntaxError on the 3.11 floor.
        surface = "present" if row.ceiling_path is None else f"present, ceiling {row.ceiling_path}"
        print(f"    arming surface: {row.config_path} "
              f"({surface}) — NOT audited, so a mint does not gate on it")
        # A deferred row's `cadence` is declared so the flip to REQUIRED stays a one-field
        # data edit, which would leave it a field nothing reads until that flip. Printed.
        cadence = "NOT DECLARED" if row.cadence is None else (
            f"{row.cadence.value} over {list(row.cadence_paths)}")
        print(f"    earliest-fire cadence: {cadence} — judged only once this row is REQUIRED")
        # The CLOCK says which key the row's evidence arrives on; `period_path` is `None` on
        # the two clocks with no config period, and the print says WHICH of the two.
        if row.cadence is not None:
            clock = row.cadence.sample_clock
            tick = (f"1 tick = {clock.period_path}" if clock.period_path is not None
                    else ("1 tick = 1 training step, definitional"
                          if clock.is_step_clocked else "no train-step tick at all"))
            print(f"    sample clock: {clock.value} ({tick})")
        if row.source_pin is not None:
            rel, text = row.source_pin
            print(f"    pinned to {rel}: {text!r}")
        # `note` is where "why is this row deferred" lives; a deferred row whose reason is
        # invisible is a row nobody can re-adjudicate.
        print(f"    why: {row.note}")


#: Synthetic operands and PERIODS for `_cadence_self_test`, and the run length they are judged
#: against: HEALTHY is run5's shape, VACUOUS a sampling period three orders of magnitude past the
#: whole run. None is read from any config, and the period is separate from the operand tuple
#: because a period belongs to the axis's SAMPLE CLOCK, not to a row.
_SELF_TEST_RUN_LENGTH = 1_000_000
_SELF_TEST_HEALTHY_PERIOD = 1_000
_SELF_TEST_VACUOUS_PERIOD = 1_000_000_000
#: (consec, min_step) on the gate-boundary clock — run5's draw-rate shape.
_SELF_TEST_HEALTHY = (3, 25_000)
#: (collapse_consec, early_death_min_step, collapse_min_step, rolling_consec,
#: rolling_min_step) on the eval-round clock — run5's own sealbot-WR shape.
_SELF_TEST_WR = (3, 15_000, 25_000, 2, 20_000)


def _cadence_self_test() -> list[str]:
    """Prove the cadence trigger CAN fire, in BOTH directions, before any verdict is trusted.

    This check's failure mode is SILENCE: a fraction read once and discarded, or an
    `earliest_fire_step` collapsed to a constant, leaves gate 12 green on exactly the configs it
    was built to refuse. Deliberately PURE ARITHMETIC over synthetic operands.
    """
    failures: list[str] = []
    if not (0.0 < EARLIEST_FIRE_FRACTION <= 1.0):
        failures.append(
            f"    the bound arm: EARLIEST_FIRE_FRACTION is {EARLIEST_FIRE_FRACTION!r}, which "
            "is not a fraction of a run — <= 0 fails every armed row and > 1 can fail none"
        )
        return failures
    bound = EARLIEST_FIRE_FRACTION * _SELF_TEST_RUN_LENGTH
    healthy = Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
        _SELF_TEST_HEALTHY, period_steps=_SELF_TEST_HEALTHY_PERIOD)
    vacuous = Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
        _SELF_TEST_HEALTHY, period_steps=_SELF_TEST_VACUOUS_PERIOD)
    lag = Cadence.STEP_LAG_THRESHOLD.earliest_fire_step((100,), period_steps=1)
    wr_healthy = Cadence.EVAL_ROUND_CONSEC.earliest_fire_step(
        _SELF_TEST_WR, period_steps=_SELF_TEST_HEALTHY_PERIOD)
    wr_vacuous = Cadence.EVAL_ROUND_CONSEC.earliest_fire_step(
        _SELF_TEST_WR, period_steps=_SELF_TEST_VACUOUS_PERIOD)
    if healthy is None or healthy > bound:
        failures.append(
            f"    arm A: operands {_SELF_TEST_HEALTHY} at period "
            f"{_SELF_TEST_HEALTHY_PERIOD} computed {healthy!r}, which does not clear the "
            f"bound {bound} — the check would refuse a healthy production config"
        )
    if vacuous is None or vacuous <= bound:
        failures.append(
            f"    arm B: operands {_SELF_TEST_HEALTHY} at period "
            f"{_SELF_TEST_VACUOUS_PERIOD} computed {vacuous!r}, which the bound {bound} "
            "ACCEPTS — the ADJ-D22 config would audit ARMED all over again"
        )
    if lag is None or lag != 101.0:
        failures.append(
            f"    arm C: the lag cadence computed {lag!r} for a threshold of 100, not 101 — "
            "member dispatch has collapsed and every row is being judged by one arithmetic"
        )
    if (wr_healthy is None or wr_healthy > bound
            or wr_vacuous is None or wr_vacuous <= bound):
        failures.append(
            f"    arm D: the EVAL-ROUND cadence computed {wr_healthy!r} at period "
            f"{_SELF_TEST_HEALTHY_PERIOD} and {wr_vacuous!r} at period "
            f"{_SELF_TEST_VACUOUS_PERIOD} against the bound {bound} — the sealbot-WR axis "
            "must clear a healthy eval cadence and FAIL one that outruns the run, or gate 12 "
            "is back to having no opinion about the axis at all (R265 / ADJ-D38)"
        )
    periods = {clock: clock.period_path for clock in SampleClock
               if clock.period_path is not None}
    if len(set(periods.values())) != len(periods):
        failures.append(
            f"    arm E: the sample clocks name overlapping period keys {periods} — two axes "
            "sharing one cadence key IS the D38 defect: one of them is being judged in a "
            "clock it does not tick in, with every other arm green"
        )
    refused_the_fallback = False
    try:
        answered = Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
            _SELF_TEST_HEALTHY, period_steps=None)
    except SampleClockNotDerivableError:
        refused_the_fallback = True
        answered = None
    if not refused_the_fallback:
        failures.append(
            f"    arm F: a step-clocked cadence handed NO period answered {answered!r} "
            "instead of raising — the one-tick-is-one-training-step FALLBACK is back, and "
            "every axis is auditable in the step clock again (R265 / ADJ-D38)"
        )
    return failures


def _audit_manifest_and_configs(paths: list[Path]) -> dict:
    """Assertion (c) plus manifest integrity. Raises the named outcome; returns the report
    block on success."""
    broken_trigger = _cadence_self_test()
    if broken_trigger:
        raise PreflightManifestError(
            "the cadence check's SELF-TEST failed, so no cadence verdict on this run can be "
            "trusted (R251 / LAW-07 — a gate publishes no verdict from an instrument it has "
            "not just watched work):\n" + "\n".join(broken_trigger)
        )
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
    # The scan's RESULT is what the report publishes. `source_pins_ok` used to be the literal
    # `True`, so deleting this call left the report claiming a scan that never ran, with the
    # whole default tier green. Both report fields are derived from `broken` / `scanned`.
    scanned = [row.name for row in MANIFEST if row.source_pin is not None]
    broken = verify_source_pins(MANIFEST, repo_root=REPO_ROOT)
    if broken:
        raise PreflightManifestError(
            "source pin(s) no longer match their file — re-adjudicate the row rather than "
            f"editing the pin (R56): {[row.name for row in broken]}"
        )
    disarmed: list[str] = []
    cadence_rows: list[dict] = []
    cadence_disarmed: list[str] = []
    audit = None
    for path in paths:
        try:
            config = _load(path)
            audit = audit_arming(config)
            # The SECOND half of assertion (c), on the same loaded config so the two answers
            # cannot be about different bytes. `fraction` is passed EXPLICITLY because the report
            # block and the rc-30 message interpolate it as read HERE.
            verdicts = audit_cadence(config, fraction=EARLIEST_FIRE_FRACTION)
        except ArmingSurfaceMissingError as exc:
            # The shipped module raises the NAMED error and the tool maps it onto its own
            # manifest code; uncaught it became rc 1 "the tool broke" for a one-line defect.
            raise PreflightManifestError(
                f"an armed-abort row's arming surface does not resolve on {path.name}: {exc}"
            ) from exc
        except SampleClockNotDerivableError as exc:
            # An underivable sample clock must reach the operator as rc 31, never as rc 1 and
            # never as a quiet fall back to the training-step clock.
            raise PreflightManifestError(
                f"an armed-abort row's SAMPLE CLOCK does not resolve on {path.name}: {exc}"
            ) from exc
        for row in audit.disarmed:
            disarmed.append(f"{path.name}: {row.name} ({row.config_path})")
        for verdict in verdicts:
            cadence_rows.append({
                "config": path.name, "name": verdict.row.name,
                "cadence": None if verdict.row.cadence is None else verdict.row.cadence.value,
                "cadence_paths": list(verdict.row.cadence_paths),
                # "Judged in the wrong clock" and "judged" were the same observable before
                # the clock, its live period key and the period itself were published here.
                "sample_clock": verdict.clock.value,
                "clock_period_path": verdict.clock.period_path,
                "clock_period_steps": verdict.period_steps,
                "earliest_fire_samples": verdict.earliest_samples,
                "bound_samples": verdict.bound_samples,
                "earliest_fire_step": verdict.earliest_step,
                "bound": verdict.bound, "within": verdict.within,
                "detail": verdict.detail,
            })
            if not verdict.within:
                cadence_disarmed.append(
                    f"{path.name}: {verdict.row.name} — earliest possible fire step "
                    f"{verdict.earliest_step}, bound {verdict.bound} "
                    f"({EARLIEST_FIRE_FRACTION} x {RUN_LENGTH_PATH}); {verdict.detail}"
                )
    if disarmed:
        raise PreflightArmingAuditError(
            "a REQUIRED armed-abort row is DISARMED on a production config — minting this "
            f"config re-enables the failure the abort exists to catch: {disarmed}"
        )
    if cadence_disarmed:
        # Ordered AFTER the arming check: a row that is simply off is the plainer diagnosis, and
        # the rc is the same because an abort that cannot fire is not armed.
        raise PreflightArmingAuditError(
            "a REQUIRED armed-abort row is ARMED but CADENCE-DISARMED on a production config "
            "— its own cadence keys, in its own SAMPLE CLOCK (R265: the clock its evidence "
            "actually arrives in, named per row and derived from a live key), put its "
            f"earliest possible fire step outside {EARLIEST_FIRE_FRACTION} of "
            f"{RUN_LENGTH_PATH}, so minting this config ships an "
            "abort that cannot fire in time to catch what it exists to catch (R251/ADJ-D22). "
            "A large interval is NEVER a sanctioned disarm: the one sanctioned spelling is "
            f"the explicit R56-style deferred row with an owner and a source pin. "
            f"{cadence_disarmed}"
        )
    return {
        "module_sha256": _sha256(REPO_ROOT / "src" / "mantis" / "config" / "armed_aborts.py"),
        "required": [row.name for row in (audit.required if audit else ())],
        # `exit_code` is the code the abort FIRES with — what a reader of a run's exit
        # status needs to map that number back to a manifest row.
        "required_rows": [{"name": row.name, "config_path": row.config_path,
                           "exit_code": row.exit_code}
                          for row in (audit.required if audit else ())],
        # Read from the audit's own result rather than re-derived from MANIFEST, so the
        # published block and the audit cannot disagree about which rows are deferred.
        "deferred": [{"name": row.name, "owner": row.owner, "config_path": row.config_path,
                      "source_pin": list(row.source_pin) if row.source_pin else None,
                      "note": row.note}
                     for row in (audit.deferred if audit else ())],
        "disarmed": [],
        # The cadence half is PUBLISHED on the green path too, per config per row: a check
        # whose only visible output is its own failure is one nobody can audit for vacuity.
        "cadence_fraction": EARLIEST_FIRE_FRACTION,
        "cadence_bound_path": RUN_LENGTH_PATH,
        "cadence": cadence_rows,
        "cadence_disarmed": [],
        "source_pins_ok": not broken,
        "source_pins_scanned": scanned,
        "audited_configs": [str(path) for path in paths],
        "exempt_configs": [rel for rel, _reason in EXEMPT_CONFIGS],
    }


def _tier_skeleton() -> dict:
    """The tier block before any burst has been accepted. `none` is TRUE at this instant."""
    return {"tier": TIER_NONE, "burst_steps": None, "floors": None,
            "required_for_mint": list(MINT_REQUIRED_TIERS),
            "covered": [], "owed": list(MINT_REQUIRED_TIERS), "does_not_prove": None}


def _tier_block(config: RunConfig, burst_steps: int) -> dict:
    """The tier block for a burst the cross-field validators ACCEPTED, built AFTER the override
    returns: a burst that was refused is not a tier that ran. `floors` carries every row with its
    own `cleared` flag so the operator can see WHICH rule made the tier what it is."""
    return {"tier": _burst_tier(config, burst_steps),
            "burst_steps": int(burst_steps),
            "floors": [{"key": key, "value": value, "floor": floor,
                        "cleared": int(burst_steps) >= floor}
                       for key, value, floor in _burst_floors(config)],
            "required_for_mint": list(MINT_REQUIRED_TIERS),
            "covered": [], "owed": list(MINT_REQUIRED_TIERS), "does_not_prove": None}


def _new_report(mode: str) -> dict:
    """The report skeleton. The `not_run` reason and the `tier` block are PREDICTIONS here —
    no boot has happened yet — and `_finalise_not_run` / `_finalise_tier` re-derive them from
    the run's own history before the write."""
    not_run_reason = _not_run_reason({"mode": mode, "child": None})
    return _finalise_tier({
        "schema": REPORT_SCHEMA,
        "tool_sha256": _sha256(Path(os.path.abspath(__file__))),
        "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode, "verdict": "pass", "rc": 0, "failure": None,
        "config": None, "coordinator": None, "override": None, "manifest": None,
        "assertions": {
            "a_sync": {"verdict": "not_run", "reason": not_run_reason},
            "b_lag": {"verdict": "not_run", "reason": not_run_reason},
            "c_arming": {"verdict": "not_run", "reason": "the audit did not complete"},
        },
        "child": None, "events": None, "tier": _tier_skeleton(),
    })


def _build_parser() -> argparse.ArgumentParser:
    """No `default=` anywhere: requiredness is per-MODE and enforced below, because argparse
    cannot express "required in mode PREFLIGHT only" and gate 12 invokes `--audit-only`
    alone."""
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
    parser.add_argument("--_boot", action="store_true", help=argparse.SUPPRESS)
    return parser


def _require_preflight_args(parser: argparse.ArgumentParser, args) -> None:
    missing = [name for name, value in (("--config", args.config),
                                        ("--burst-steps", args.burst_steps),
                                        ("--out-dir", args.out_dir),
                                        ("--timeout-sec", args.timeout_sec)) if value is None]
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
    """Resolve `--out-dir` and refuse one inside the repo working tree, before anything is
    created: the child writes `*.jsonl` and gate 6 rejects stray ones. Both sides must
    `.resolve()`, since `abspath` normalises textually without following symlinks and a symlink
    into the tree once let the tool write its report into the repo."""
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


def _boot_main(args) -> int:
    """The `--_boot` child: the REAL production posture, through the ONE composition authority.

    Every build step lives at `mantis.run.build_run_collaborators`, which `launch_run` calls too.
    What survives here is the CONTAINMENT mechanism and two instruments that wrap AROUND the
    composer: `_apply_burst_override`, a CONFIG-level transform before the boot, and the
    resumed-trainer refusal, a READ-ONLY check between builder and composer — which is why the
    authority is a PAIR of functions rather than one opaque `boot()`. Nothing may be assigned onto
    `collab`. The DEVICE and EVAL posture are the config's own.

    The child is spawned with `start_new_session=True` so the parent's timeout `killpg` reaches
    every grandchild, which also makes it unreachable by any signal aimed at the parent: MEASURED
    2026-08-18, one was found at PPID 1, 4 h 06 m old at 682% CPU against its own
    `--timeout-sec 45.0`, because its parent had been killed. The first thing it does is ask the
    KERNEL to end it when its parent dies.
    """
    from mantis.train.lifecycle.signals import arm_parent_death_signal

    arm_parent_death_signal()

    config = _load(_resolve_config_path(args.config))
    booted = _apply_burst_override(config, args.burst_steps)
    from mantis.run import build_run_collaborators, compose_run

    collab = build_run_collaborators(config=booted, out_dir=args.out_dir)
    # A run RESUMED past its ceiling terminates having performed zero syncs, which looks
    # EXACTLY like the frozen actor this preflight exists to find. The builder never passes
    # `checkpoint_path`, and a nonzero step here is a named refusal, not a warning.
    if int(collab.trainer.step) != 0:
        raise PreflightResumedTrainerError(
            f"the freshly-built trainer reports step {int(collab.trainer.step)}, not 0: a "
            "preflight over a resumed trainer measures nothing while looking like the defect "
            "it exists to find (§4.2)"
        )
    handles = compose_run(config=booted, trainer=collab.trainer, pool=collab.pool,
                          buffer=collab.buffer, log_dir=collab.log_dir,
                          checkpoint_dir=collab.checkpoint_dir,
                          resume_state=collab.resume_state)
    return _abort_rc(handles.shutdown.abort_rule)


def _abort_rc(rule: str | None) -> int:
    """The child's rc, decided by WHETHER AN ABORT FIRED and by nothing else.

    No rule fired -> 0, `abort_rule is None` being the ONLY thing that means a clean run; a rule
    fired WITH an authored code -> that code, from the manifest row and never written here; a rule
    fired with NO authored code -> a NAMED failure, never 0 and never an invented number.
    """
    if rule is None:
        return 0
    code = exit_code_for_abort(rule)
    if code is None:
        raise PreflightBootFailedError(
            f"the run's hard-abort rule {rule!r} FIRED and stopped the run, but "
            "`mantis.config.armed_aborts.MANIFEST` authors no exit code for it. Reported as "
            "a failed boot rather than as rc 0: an aborted run is not a clean one. No code is "
            "invented here — R84 declined to author one for a rule nobody pre-registered, and "
            "doing it in this tool would be that same class one layer down"
        )
    return int(code)


def _child_argv(args) -> list[str]:
    return [sys.executable, os.path.abspath(__file__), "--_boot",
            "--config", str(args.config), "--burst-steps", str(int(args.burst_steps)),
            "--out-dir", str(args.out_dir), "--timeout-sec", str(float(args.timeout_sec))]


def _run_child(args, report: dict) -> dict:
    started = time.monotonic()
    proc = subprocess.Popen(_child_argv(args), start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # `report["child"]` is assigned BEFORE the blocking `communicate` — the ordinary place for an
    # interrupt to land — because `_not_run_reason` discriminates on `child is None` and would
    # otherwise publish "NO boot was spawned" for a child that is running.
    child: dict = {"rc": None, "rc_convention": RC_CONVENTION, "raised_by": "parent",
                   "spawned": True, "pid": int(proc.pid), "wall_clock_sec": None,
                   "timed_out": False, "outcome": "in_flight"}
    report["child"] = child
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
    # The 4000-char tails were an invented budget AND the classifier's input, so a truncated
    # traceback silently downgraded a tree defect from 32 to 33. The FULL streams spool beside the
    # report; the classifier keeps reading the TAIL, where a wall's `AttributeError` line lands.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # a child may die before creating it
    stdout_spool = out_dir / "child_stdout.log"
    stderr_spool = out_dir / "child_stderr.log"
    stdout_spool.write_text(stdout or "", encoding="utf-8")
    stderr_spool.write_text(stderr or "", encoding="utf-8")
    child.update({"rc": rc, "raised_by": "child" if rc in PASS_THROUGH else "parent",
                  "wall_clock_sec": round(time.monotonic() - started, 3),
                  "timed_out": timed_out, "outcome": "exited",
                  "stdout_tail": (stdout or "")[-4000:],
                  "stderr_tail": (stderr or "")[-4000:],
                  "stdout_spool": str(stdout_spool), "stderr_spool": str(stderr_spool)})
    if rc < 0:
        child["signal"] = -rc
        child["signal_name"] = signal.Signals(-rc).name
    return child


def _assert_start_halts(booted: RunConfig, out_dir: Path, report: dict) -> None:
    """The two START pre-flight HALTs, both decided before the boot.

    Args:
        booted: the config the child will run.
        out_dir: the run directory the child will write into.
        report: the preflight report; each halt records its own evidence block.

    Raises:
        PreflightWorkspaceNotDurableError: the run directory would not survive the machine.
        PreflightCudaBuildError: the config declares a cuda device and the installed torch is
            not a CUDA build that computes correctly.
    """
    try:
        report["workspace"] = assert_durable(out_dir)
    except WorkspaceNotDurableError as exc:
        raise PreflightWorkspaceNotDurableError(str(exc)) from exc
    devices = {booted.train.device, booted.eval.worker_device}
    if "cuda" not in devices:
        report["cuda_build"] = {"verdict": "not_run", "reason": f"no cuda device declared: "
                                f"train.device={booted.train.device!r}, "
                                f"eval.worker_device={booted.eval.worker_device!r}"}
        return
    # Imported here and not at module scope: AUDIT mode is the per-commit CI gate and must
    # not pay torch's import to audit YAML.
    from mantis.diagnostics.cuda_build_guard import CudaBuildRefusal, check

    try:
        report["cuda_build"] = check()
    except CudaBuildRefusal as exc:
        raise PreflightCudaBuildError(
            f"{booted.run_id} declares train.device={booted.train.device!r} and "
            f"eval.worker_device={booted.eval.worker_device!r}, and the installed torch "
            f"cannot compute on a GPU: {exc}"
        ) from exc


def _run_preflight(args, report: dict, out_dir: Path) -> None:
    path = _resolve_config_path(args.config)
    config = _load(path)
    report["config"] = _config_block(path, config)
    report["coordinator"] = _coordinator_block(config)
    report["manifest"] = _audit_manifest_and_configs(_audit_paths(path))
    report["assertions"]["c_arming"] = {"verdict": "pass", "disarmed": [],
                                        "required_armed": report["manifest"]["required"]}
    booted = _apply_burst_override(config, args.burst_steps)
    # Stamped only once the validators have ACCEPTED the burst, so a rc-11 refusal leaves
    # `tier: none` — the truth, not a placeholder.
    report["tier"] = _tier_block(config, int(args.burst_steps))
    report["override"] = {"keys": list(OVERRIDE_KEYS),
                          "from": int(config.train.max_train_steps),
                          "to": int(args.burst_steps),
                          # THE one identity authority: the same function the child's
                          # compose_run hashes its own loaded config with.
                          "booted_config_sha256": config_identity_sha256(booted)}
    log_dir = out_dir / "logs"
    # Refuse a dirty out-dir BEFORE the boot, scoped to pre-existing segments under THIS
    # run_id — the ones `_read_segment`'s scope would believe.
    stale = sorted(log_dir.glob(f"events_{booted.run_id}_*.jsonl")) if log_dir.is_dir() else []
    if stale:
        raise PreflightOutDirReusedError(
            f"--out-dir {out_dir} already holds {len(stale)} event segment(s) for run_id "
            f"{booted.run_id!r} (first: {stale[0].name}): a same-run_id reuse would read a "
            "previous burst's events as this run's evidence. Use a fresh out-dir"
        )
    _assert_start_halts(booted, out_dir, report)
    child = _run_child(args, report)
    segments, events = (_read_segment(log_dir, run_id=booted.run_id)
                        if log_dir.is_dir() else ([], []))
    # The scan PUBLISHES what it read, so `_watchdog_reason` can say which of "read and not
    # found" / "nothing was read" happened. `[]` and "no scan recorded" are different facts.
    child["segments_scanned"] = [str(segment) for segment in segments]
    fired = [event for event in events if event.get("event") == "heartbeat_watchdog_fired"]
    if fired:
        child["fired_reason"] = fired[-1].get("reason")
    report["events"] = _events_block(segments, events)
    _classify_child(child)  # the child's status is evaluated BEFORE the predicates
    # F-B1 closure: copy the child's OWN published boot identity into the child block and
    # Ordered AFTER _classify_child (a dead child is a child-status failure, not an identity
    # one) and BEFORE the predicates (a burst on the wrong config proves nothing).
    child["booted_config_sha256"], child["config_identity"] = child_config_identity(
        events, parent_sha=str(report["override"]["booted_config_sha256"]))
    if child["config_identity"] == "mismatch":
        raise PreflightConfigIdentityError(
            f"the child's run_boot_identity sha ({child['booted_config_sha256']}) does not "
            f"match the config the parent audited "
            f"({report['override']['booted_config_sha256']}): parent and child read "
            "different configs, and every other block of this report would be evidence "
            "about the wrong run (F-B1)"
        )
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

    def _publish(subject: Path) -> None:
        """The subject's own blocks: WHICH config this report is about, and what the
        composition root resolves from it.

        Hoisted into a closure so a NAMED subject can be published BEFORE the manifest audit:
        a red audit used to write a report whose `config` was still `null`.
        """
        subject_config = _load(subject)
        report["config"] = _config_block(subject, subject_config)
        report["coordinator"] = _coordinator_block(subject_config)

    if named is not None:
        _publish(named)
    # The manifest audit runs BEFORE anything indexes `paths`: it carries the vacuity guard,
    # so an empty PRODUCTION_CONFIGS is rc 31 by name rather than an `IndexError` collapsing
    # into an unnamed rc 1.
    report["manifest"] = _audit_manifest_and_configs(paths)
    if named is None:
        _publish(paths[0])
    report["assertions"]["c_arming"] = {"verdict": "pass", "disarmed": [],
                                        "required_armed": report["manifest"]["required"]}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args._boot:
        try:
            return _boot_main(args)
        except PreflightError as exc:
            # The child exits with its OWN named code so the parent can propagate it
            # unchanged. A tree defect is deliberately NOT caught — its traceback is what the
            # parent classifies into rc 32.
            print(f"preflight child: rc {exc.rc} — {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return int(exc.rc)
    if not args.audit_only:
        _require_preflight_args(parser, args)

    report = _new_report("audit" if args.audit_only else "preflight")
    out_dir: Path | None = None
    rc = 0
    try:
        _print_deferred_rows()  # loud on EVERY run, including a green one
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
    except Exception as exc:  # noqa: BLE001 — the UNDIAGNOSED outcome, named as such
        rc = PreflightInternalError.rc
        report.update(verdict="fail", rc=rc, failure="PreflightInternalError")
        print(f"PREFLIGHT NOT GREEN: rc {rc} — PreflightInternalError: {exc!r}",
              file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 — AUDIT-1 F-03: stamp, then RE-RAISE
        # A `KeyboardInterrupt` during a long burst, or a callee's `SystemExit`, used to
        # unwind past both arms above and land an artifact still carrying the skeleton's
        # `verdict: "pass", rc: 0`. Stamped here and RE-RAISED: this arm changes what the
        # report says, never what the process does.
        rc = PreflightInterruptedError.rc
        report.update(verdict="fail", rc=rc,
                      failure=PreflightInterruptedError.__name__,
                      interrupted_by=type(exc).__name__)
        print(f"PREFLIGHT NOT GREEN: rc {rc} — PreflightInterruptedError: "
              f"{type(exc).__name__}", file=sys.stderr)
        if out_dir is not None:
            _write_report(out_dir, report)
            out_dir = None  # written; the `finally` must not write it twice
        raise
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
