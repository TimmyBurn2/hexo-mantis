#!/usr/bin/env python3
# >300 justify (R8). The old clause (2) — "when
# the freeze lifts, the parent half should be split out and this clause deleted" — is
# DISCHARGED: the leaf parent layers (shared vocabulary, exit taxonomy, the two
# predicate evaluators, report helpers, the classifier, the segment/verdict/identity
# leaves) moved VERBATIM to `preflight_mint_parent.py` (S-3 hunk hashes in
# wp/WPBOX/DISPATCH_LOG.md) and are re-exported by plain assignment, so every oracle that
# binds `TOOL.<name>` off this module path keeps binding one object.
# Two reasons the file is still >300, and saying exactly what stays is the point:
#
#  (1) CHILD SIDE (_boot_main, _abort_rc, _load, _apply_burst_override,
#      _minimum_legal_burst, _burst_floors, _resolve_config_path) — SHRUNK by WPMAIN
#      (CARD-RUN-MAIN, R121(a)): the boot itself now lives at `mantis.run`
#      (`build_run_collaborators` + `compose_run`), and `_build_buffer` moved with it as
#      `mantis.run._select_buffer`. What is left on this side is config surgery and the rc
#      instrument — the two things that are the TOOL's and not the run's.
#      `_abort_rc` belongs on this side by the same rule as the rest:
#      it runs IN the child, after `compose_run` returns, and it is what turns the run's own
#      `abort_rule` into the process rc a supervisor reads. The parent re-execs ITSELF as
#      the boot child by os.path.abspath(__file__) (DESIGN_P §6.2) — one file IS the
#      containment mechanism, not a packaging preference. The frozen O-9/O-10/O-5 token
#      census and the O-1 parser census sweep THIS file, so the boot and the parser cannot
#      leave it without an R43 event.
#
#  (2) PARENT SIDE THAT STAYS (585 lines of function bodies), each piece pinned by a NAMED
#      seam rather than by inertia: `verify_source_pins` (SF-4 — the tamper scan lives in
#      the TOOL, recorded at tests/config/test_armed_abort_manifest.py:5) and the whole
#      audit half beside it, because the frozen ring-2 monkeypatch seam
#      (`TOOL.MANIFEST = bad -> main(--audit-only) == 31`,
#      test_drawrate_arming_surface_named_failure.py) requires the audit read path to see
#      THIS module's globals at call time; `_burst_tier`/`_tier_block`/`_tier_skeleton`,
#      which read `_burst_floors` (child closure); `_new_report`, whose `tool_sha256`
#      hashes THIS file; `_child_argv`/`_run_child` (the self-exec + spool); and the
#      orchestration (`_run_preflight`/`_run_audit`/`main`), which weaves the audit half,
#      the child closure and the sibling's leaves. The sibling must never import this file
#      and never define a `MANIFEST` global — pinned, with the O-2/O-3 census extension,
#      by tests/tools/test_preflight_parent_census.py.
#
# Roughly half the file is comment carrying the "what defect does this line exist for"
# rationale LAW-07 wants; deleting it is what makes the next reader re-derive MF-5.
"""CI gate 12 (R61) — the mint preflight: one tool, two modes, one manifest.

Mode AUDIT (`--audit-only`): no boot, no burst, no GPU. Reads the committed production
configs through the REAL loader and audits assertion (c) — every `required` row of
`mantis.config.armed_aborts.MANIFEST` must be ARMED, and every armed row must still be
ABLE TO FIRE inside its own run — plus manifest integrity and the R56 source-pin tamper
scan. This is the per-commit CI gate. **rc 0 in this mode covers assertion (c) ONLY**;
assertions (a) and (b) are reported `not_run`, in the report and on stdout, on every run
including a green one.

Assertion (c)'s SECOND half is R251 / ADJ-D22 and it is not a refinement of the first: the
audit never read `monitor.gate_interval`, so `gate_interval: 1000000000` on a 40-step run
produced zero gate boundaries, left the draw-rate threshold armed in the config and unread
in the run, and audited GREEN. `Mechanism.is_armed` reads a threshold; whether the machinery
that reads that threshold ever RUNS is a different question, and `ge=1` on the interval bans
one spelling of "never gate" while permitting every larger one. So `audit_cadence` computes
each armed row's earliest possible fire step from that config's own cadence keys and refuses
any row whose value exceeds `EARLIEST_FIRE_FRACTION * train.max_train_steps` — rc 30, the
same code, because an abort that cannot fire is not armed in any sense that protects the
run. A large interval is NEVER a sanctioned disarm; the one sanctioned spelling stays the
explicit R56-style deferred row with an owner and a source pin. `_cadence_self_test` proves
the trigger fires in both directions before any verdict of it is published.

R265 / ADJ-D38 GENERALISES THAT HALF: every row is judged IN ITS OWN SAMPLE CLOCK. R251
computed every earliest-fire in TRAINING STEPS, which is the right clock for the draw-rate
gate (it ticks on `monitor.gate_interval`) and the wrong one for the sealbot-WR trajectory,
whose evidence arrives once per EVAL ROUND (`train.eval_interval`). A row's cadence now names
its `SampleClock`, the CLOCK derives its period from a live key — no row may declare the key
its own axis is sampled by — and the comparison happens in ticks of that clock. A clock whose
period cannot be derived is rc 31 by name (`SampleClockNotDerivableError`), never a silent
fall back to the step clock: an axis judged in a clock it does not tick in audits GREEN on
exactly the configs it exists to refuse. `sealbot_wr_abort` joins the manifest in the same
ruling as a DEFERRED row — the axis had no row at all, so gate 12 could not compute even a
false affirmative for it — and is printed, with its clock, on every run.

Mode PREFLIGHT (`--config --burst-steps --out-dir --timeout-sec`): everything
AUDIT does, then the REAL `compose_run` boot in production posture, a bounded burst, a
timeout-bounded join, and assertions (a) sync-cadence and (b) lag-transport over the run's
own JSONL segment. **This is the MANUAL mint gate — no CI step invokes it.**

R64 posture, which is the whole point: this tool contains NO stand-in for a production
object — and since WPMAIN it constructs no production object at all. The child calls
`mantis.run.build_run_collaborators` and `mantis.run.compose_run`, the SAME pair
`mantis.run.launch_run` calls, so the real trainer, the real self-play pool, the real buffer
(selected off `config.identity.representation` — never sniffed, never defaulted, LAW-11),
the run-safety triple, the step-coordinator config and the sync engine are all built by the
composition root and by nothing here. A CI gate that builds its own collaborators is the
one-authority violation CARD-RUN-MAIN ended. When a collaborator is missing a method the
tool does NOT supply one: the failure reaches the process boundary uncaught. Classification honesty (CARD-PREFLIGHT-WALL-CLASSIFIER, resolved at WPCLEAN Phase
PFC to this WORDING rather than to a wall table): rc 32 is a FALLBACK SNIFF — the literal
`"object has no attribute"` in the child's stderr tail — not a wall registry. A wall that
surfaces any other way lands rc 33 with its traceback in the tail, and the register of
known walls stays the TD cards; an exception-type→card table in this tool would be a second
authority beside the register (the card's own named risk), so none exists.

**CARD-POOL-ENCODING-BRIDGE (TD-4) HAS LANDED (WPBRIDGE Phase T).** Mode PREFLIGHT used to
terminate there: `WorkerPool` construction calls `resolve_pool_encoding` ->
`resolve_from_config`, which raised `MissingEncodingError` because `RunConfig.model_dump()`
carries `identity.encoding` and no flat top-level `encoding` key. Parent rc **33**, child
rc 1, ~1.4 s. `resolve_from_config` now reads the nested shape as one of its declared forms
— one authority, no caller-side injection — and that wall is GONE.

Where the boot stops NOW, measured on the WPBRIDGE dev-box rehearsal (2026-07-29, CPU,
`--burst-steps 25001 --timeout-sec 300`): **nowhere, inside the rehearsal window.** The child
boots clean through `init_trainer`, `WorkerPool`, `compose_run`, `build_run_safety` and
`run_training_loop`, arms both watchdogs (`heartbeat_watchdog_armed`,
`selfplay_stall_watchdog_armed`), streams `actor_lag_sample` and `system_stats`, and is still
running healthily when `--timeout-sec` kills it: parent rc **40** `PreflightTimeoutError`,
child rc -15, EMPTY stderr. `buffer_size` is **0** for the whole window — one CPU self-play
worker at run5's settings finishes no games, so the coordinator never leaves its warmup arm
(`_run_loop` O4, `buffer.size < cfg.min_buf_size`) and never takes a training step.

CORRECTION, measured 2026-07-30 (WPMAIN / R126 + R130). The paragraph above is preserved as
the WPBRIDGE record; two of its facts have EXPIRED and a reader must not take them for the
current tree. (i) `configs/run5.yaml` no longer boots on a CPU box AT ALL: the device is a
CONFIG FACT now (`train.device`, R126 — the `--device` flag is DELETED from this tool), run5
mints `cuda`, and the child dies in `init_trainer` with torch's own "Torch not compiled with
CUDA enabled" -> parent rc **33**. That refusal is the POINT (it is what stops a cpu preflight
false-clearing a cuda run's GPU wall) and is pinned by
`tests/tools/test_preflight_mint_process.py::test_booting_run5_on_a_non_CUDA_box_fails_LOUD_in_init_trainer`.
(ii) On the minted CPU twin the rest of the paragraph still holds — clean boot, both watchdogs
armed, `buffer_size` 0, killed at the timeout, parent rc **40** — but the CHILD rc is now
**0, not -15**, with non-empty stderr: the timeout SIGTERMs the child's process group, and
WPMAIN installed LAW-16's handlers (dead in every composed run before it), so the child
save-then-exits instead of dying on the signal.

SECOND CORRECTION, measured 2026-08-17 (F-816-10 / R276(f)), and it EXPIRES the (ii) above
for as long as the production configs carry the R119 placeholder. `configs/run5.yaml` — and
therefore its minted CPU twin, which differs from it in `run_id` and `train.device` alone —
now mints `inference.fused_graph_caps: {max_fused_edges: null, max_fused_nodes: null}`. That
is the fused graph inference forward's memory bound, and `null` is NOT an off state: it is the
placeholder, schema-VALID so the repo ships a complete config and gate 7 stays green, and
runtime-REFUSED so an uncalibrated production config CANNOT CONSTRUCT ITS GRAPH INFERENCE
SERVER. The twin's boot therefore stops at the `WorkerPool` composition seam with
`UncalibratedFusedGraphCapsError` naming the member, the calibration entry point
(`python -m mantis.diagnostics.fusion_calibrate`) and the `--set` line that fixes it: parent
rc **33** `PreflightBootFailedError`, child rc 1, seconds.

**THIS IS THE DESIGNED BEHAVIOUR AND IT IS ALSO A REAL LOSS, and both halves belong here.**
It is designed: an unbounded fused inference forward is the defect F-816-10 exists to make
unconstructible, and a run that boots on a cap nobody measured is exactly the silently-usable
guess the packet forbids. It is a loss: the clean-boot / both-watchdogs-armed / rc-40 evidence
recorded in (ii) is UNAVAILABLE from this tool until the operator calibrates at the box and
mints the pair. The armed-abort manifest carries `fused_graph_caps_calibrated` as a DEFERRED
row so every gate-12 run says so out loud, and closing that row is what restores this
paragraph. Until then, a reader must not take (ii) for the current tree.

**CARD-TRAINSTEP-ADAPTER (TD-1) IS DEAD (WPTS, R102).** The straight self-play arm no longer
calls a `train_step` that does not exist: `step.py::_run_training_step` routes through the
DECLARED dispatcher (`coordinator/dispatch.py::run_declared_train_step`), keyed on the
resolved `identity.representation` — graph → `sample_graph_batch` → collate →
`trainer.train_step_from_graph_batch`; grid → `sample_batch_with_pos` →
`trainer.train_step_from_tensors`; anything else raises. `TrainerLike` declares both typed
entry points and the seam is conformance-gated
(`tests/train/test_trainer_seam_conformance.py`). A real graph gradient step from the
coordinator path is pinned by `tests/train/test_train_step_dispatch.py`. What a CPU
rehearsal still cannot prove is THROUGHPUT: one CPU worker at run5's settings may not leave
warmup inside the window, and that remains a box fact, not a tree defect.

This docstring previously named CARD-TRAINSTEP-ADAPTER (`train/coordinator/step.py:573`,
rc 32) as the terminal wall, copying DESIGN_P §3.4. That was **measured false**
by IMPL and re-produced independently by REVIEW-impl: TD-4 fired FIRST, before `compose_run`
was ever called, so TD-1 was BEHIND TD-4, not in front of it. Corrected here rather than
carded — a gate whose own docstring states a measured-false fact is the first thing the next
reader believes about it. DESIGN_P's copy of the same sentence is CARD-DESIGN-P-3.4-ORDERING.
With TD-4 landed, TD-1 is once again the frontier — but by clearing the wall in front of it,
not by the ordering DESIGN_P asserted.

Containment is a SUBPROCESS, not a thread: `build_run_safety`'s `exit_fn` is `os._exit` and
`compose_run` does not override it, so an in-process boot that trips exit 42/43/45 dies
without unwinding and the evidence report is never written. A tool whose failure mode is
"no report" cannot report failure.

Exit codes (DESIGN_P §6.3). Every outcome this tool DIAGNOSES is NAMED, and rc 1
`PreflightInternalError` is the catch-all for a failure it does not: RED-TEAM_P's F-4
measured that an armed-abort row whose `config_path` does not resolve fell into it, so
"the tool broke" was reported for a one-line manifest defect. That class is now named at
rc 31 (`ArmingSurfaceMissingError` -> `PreflightManifestError`, WPAX Phase D); rc 1
remains what it always was — the outcome nobody diagnosed — and a NEW rc-1 is a finding,
not a routine failure mode. A child rc in [10, 41] PROPAGATES
UNCHANGED, so a child exiting 12 exits the parent 12 rather than collapsing to 33. **42–48 are
RESERVED by the run's own machinery**: 42 stall/livelock, 43 persist-fatal, 44 the supervisor's
relaunch budget, 45 actor-lag, 46 the cooperative armed-abort code
(`monitor/heartbeat.py::DRAW_RATE_COLLAPSE_EXIT_CODE`), 47 the second cooperative member —
the disk-guard abort (`monitor/heartbeat.py::DISK_SPACE_EXHAUSTED_EXIT_CODE`, WPMAIN
RT-2/R132) — and 48 the third, a terminal eval round that produced no promotion decision
(`monitor/heartbeat.py::TERMINAL_EVAL_BROKEN_EXIT_CODE`, WP12-R Phase O / R152). None of the
seven is ever an assertion outcome of this tool.

46 joined that band at WPMINT Phase X (CARD-ABORT-EXIT / R84) and the taxonomy had to move with
it, in the same change: 46 is outside the [10, 41] pass-through and was outside the reserved
set, so a child exiting 46 would have fallen through to `PreflightBootFailedError` and
**collapsed to rc 33** — the tool meant to surface the authored abort signal would have
destroyed it. 46 also differs from 42/43/45 in one respect this tool must not paper over: the
child EMITS it deliberately. `_boot_main` reads `RunHandles.shutdown.abort_rule` — the rule
name the coordinator's own `_fire_hard_abort` recorded — and resolves it through
`mantis.config.armed_aborts.exit_code_for_abort`, so the number the parent propagates comes
from the manifest row and from nowhere else. A rule that fired with NO authored code (the
resolver returns `None` — `grad_norm_hard_abort` and `sealbot_wr_abort` are not
pre-registered) is rc 33 `PreflightBootFailedError` naming the rule: an abort that stopped the
run is never reported as a clean boot, and no exit code is invented for a rule nobody
registered.

MINT TIERS (WPMINT Phase B / CARD-D-BURST-FLOOR). `configs/run5.yaml`'s minimum legal burst is
**25001**, because arming `train.draw_rate_abort` at `min_step: 25000` puts a third row in
`_burst_floors` (measured: `max(100+1, 1+1, 25000+1)`). The floor cannot be shrunk — `min_step`
is a run5 armed value, mint-prereg-only (R82/R85) — and a shorter burst that pretended to cover
the draw-rate axis is out under R64. So the report says what the burst it ran DOES and DOES NOT
prove, in a `tier` block derived from `_burst_floors` and finalised by `_finalise_tier`:
`none` (no burst accepted), `sync_lag` (actor-lag + sync-cadence floors only, on a config with
no draw-rate row), `full` (a draw-rate `min_step` floor cleared too). BOTH `sync_lag` and
`full` are required for a mint; `full` COVERS `sync_lag`, so one green `full` run discharges
both.

That is a deviation from the card's presumptive shape — two SEPARATE preflight runs — and the
grounds are measured, not argued: `PRODUCTION_CONFIGS` rows must arm `draw_rate_collapse`
(assertion (c), rc 30 otherwise), an armed row puts `min_step + 1` in the floor set, and
`_apply_burst_override` refuses anything below the max at rc 11 (measured on the real tool:
`--burst-steps 101` on run5 -> rc 11, `child: null`, no boot). **On a production config tier
`sync_lag` is therefore unreachable**, and the only way to reach it is to disarm the row the
mint exists to arm. Measured on the committed tree: run5's floor is 25001 and the other four
`configs/` entries' floor is 101 — because only run5 arms the abort. The short tier is not a
shorter run of run5; it is what a config WITHOUT the armed row already gets.

What the `full` tier costs is an ESTIMATE and a LOWER BOUND, and this tool cannot measure it:
TD-4 stops the child before `compose_run`, so no burst of any length has ever run here. Basis
— WP10's recorded bench floor, median 41.66 ms/train-step (IQR 0.76, n=200, gnn_axis_v1/graph/
bf16, CPU 1-thread, `b29f0bc`) -> 25001 steps is **>= 1041.5 s** of pure train-step compute. The
MISSING TERM, named rather than guessed: the coordinator is GAME-BOUND. With
`training_steps_per_game=1.0` and `max_train_burst=1` (`configs/run5.yaml:120-121`), `_steps_budget`
returns exactly 1, and a round with no new game sleeps instead of training
(`train/coordinator/step.py:260-266`) — so 25001 train steps needs >= 25001 COMPLETED self-play
games, generated on run5's `selfplay.n_workers: 1`. That generation time is not in 41.66 ms and
is not estimated here.
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

#: SF-4: every repo-root resolution lives HERE, never in the shipped package.
REPO_ROOT = Path(os.path.abspath(__file__)).resolve().parents[2]


# ── the parent half (CARD-PREFLIGHT-SPLIT-PARENT-HALF, WPBOX Phase Q) ────────────────
#: The parent-only half lives in a sibling module, loaded off THIS file's own directory —
#: never sys.path (frozen O-3) — and re-exported by PLAIN assignment so every oracle that
#: binds `TOOL.<name>` off this module path keeps binding the one object. The sys.modules
#: guard keys on the sibling's resolved path: two different trees (the process suite's
#: byte-copy mini-tree rig) each get THEIR OWN sibling, while repeated loads of one tree
#: share one, keeping the exception taxonomy a single object set per tree.
_PARENT_HALF_PATH = Path(__file__).resolve().with_name("preflight_mint_parent.py")
_PARENT_HALF_MODULE = "_preflight_mint_parent"


def _load_parent_half():
    import importlib.util

    cached = sys.modules.get(_PARENT_HALF_MODULE)
    # `cached is not None` is provably redundant (`getattr(None, …, None)` is safe and
    # never equals a path string) — kept as a stated-domain guard, same justify-arm as the
    # b2 conjunct; the R72 row for it is UNCOVERED by proof, not by oversight.
    if cached is not None and getattr(cached, "__file__", None) == str(_PARENT_HALF_PATH):
        return cached
    spec = importlib.util.spec_from_file_location(_PARENT_HALF_MODULE, _PARENT_HALF_PATH)
    # Both leaves are pyright-load-bearing (gate 14 basic: the Optional contract must be
    # narrowed before `spec.loader.exec_module`), unproducible at runtime for a file that
    # exists — justify-arm; their R72 rows are UNCOVERED by the API's contract, not oversight.
    # A MISSING sibling fails loudly at exec_module (FileNotFoundError), before any re-export.
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
#: Re-published THROUGH the sibling (which imports it from the one authority,
#: mantis.monitor.heartbeat) — the process suite binds it as a TOOL attribute; a direct
#: re-import here would be a second import chain for the same number.
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
PreflightArmingAuditError = _parent_half.PreflightArmingAuditError
PreflightManifestError = _parent_half.PreflightManifestError
PreflightTreeDefectError = _parent_half.PreflightTreeDefectError
PreflightBootFailedError = _parent_half.PreflightBootFailedError
PreflightWatchdogFiredError = _parent_half.PreflightWatchdogFiredError
PreflightChildSignaledError = _parent_half.PreflightChildSignaledError
PreflightTimeoutError = _parent_half.PreflightTimeoutError
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
_report_name = _parent_half._report_name
_write_report = _parent_half._write_report
_watchdog_reason = _parent_half._watchdog_reason
_classify_child = _parent_half._classify_child
_read_segment = _parent_half._read_segment
_events_block = _parent_half._events_block
_verdict_exit = _parent_half._verdict_exit
child_config_identity = _parent_half.child_config_identity


#: O-4 / §5.4: the burst override writes exactly ONE dotted key and reads nothing, so
#: `stop_step` keeps exactly one source (`train.max_train_steps` -> `resolve_max_train_steps`
#: -> `run.py:167-170` -> `step.py:233`). The report's `override.keys` is emitted from this
#: same constant, so the two cannot disagree. A second entry here would make the preflight a
#: second run-length (or arming) authority, which is the R1 breach §5.4 discriminates against.
OVERRIDE_KEYS: tuple[str, ...] = ("train.max_train_steps",)

REPORT_SCHEMA = "preflight-mint-v1"
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


# ── the manifest's repo-root half (SF-4) ──────────────────────────────────────────────
def verify_source_pins(
    rows: tuple[ArmedAbort, ...], *, repo_root: str | Path
) -> tuple[ArmedAbort, ...]:
    """R56 tamper-evidence: every `source_pin`'s exact text must still be in its file.

    Returns the BROKEN rows. The asymmetry is the load-bearing part
    (`silent_encoding_gate.py:338-344`): a pin that matches NOTHING — including a pinned
    file that no longer exists — is a HARD failure, never a quiet pass.

    R73 name-truth, WPMINT DR-10: this used to say the asymmetry "makes Phase D's deletion
    of `draw_rate_threshold: float = 0.0` impossible to forget". Phase D landed and that
    literal is gone. The manifest's one pin now binds `run.py`'s `resolve_draw_rate_abort`
    threading, so what the scan makes impossible to forget today is deleting, renaming or
    reordering THAT call. What it does NOT prove is that the right VALUE flows — it is a
    whole-file substring scan (SF-2's correction); `tests/train/test_drawrate_abort_
    threading.py`'s O-D2 is the sole witness for "pinned text present, wrong value flowing".
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


def _resolve_production_configs() -> list[Path]:
    """PRODUCTION_CONFIGS holds repo-relative STRINGS (data); resolving them is ours."""
    return [REPO_ROOT / rel for rel in PRODUCTION_CONFIGS]


#: SF-4 again: the manifest module may make no filesystem call, so DISCOVERY lives here.
CONFIG_DIR_REL = "configs"


def _discovered_configs() -> list[str]:
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


def _config_declaration_drift() -> tuple[list[str], list[str], list[str]]:
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


def _audit_paths(named: Path | None) -> list[Path]:
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


def _burst_floors(config: RunConfig) -> list[tuple[str, int, int]]:
    """Every cross-field rule that binds the burst from below: (key, its value, its floor).

    Enumerated rather than folded into one number, because the number alone stopped being
    enough at WPAX Phase D: a THIRD rule joined the two (`train.draw_rate_abort.min_step`
    must be inside the run, the twin of the actor-lag rule), and an operator told only the
    maximum cannot see WHICH rule moved their floor from 101 to 25001. Each row states its
    own arithmetic, and `_minimum_legal_burst` is their max.
    """
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
    """Which mint tier a burst of this length on this config IS — read off `_burst_floors`.

    Derived from the same three rows the refusal message prints, so the tier a report claims
    and the floor arithmetic an operator was shown cannot drift apart. Three outcomes:

    * below the max floor -> `none`. No burst was accepted, so no tier was run. Defensive
      rather than dead: `_run_preflight` calls this only AFTER `_apply_burst_override`
      returned, but a caller that computed a tier from a REQUESTED burst would publish a tier
      for a run that never happened, which is the ADJ-13 F-3 class one field over;
    * clears every floor AND the config declares a draw-rate floor -> `full`;
    * clears every floor and the config declares NO draw-rate floor -> `sync_lag`.

    The third arm is why the tier is not just "cleared the max": on a config with
    `train.draw_rate_abort: null` the max floor is 101 and clearing it says nothing whatever
    about draw-rate reachability. Calling that `full` would be the overclaim this block exists
    to prevent — it is the WHOLE of the card's "what does it NOT prove".
    """
    floors = _burst_floors(config)
    if int(burst_steps) < max(floor for _key, _value, floor in floors):
        return TIER_NONE
    draw = [floor for key, _value, floor in floors if key == DRAW_RATE_FLOOR_KEY]
    return TIER_FULL if draw else TIER_SYNC_LAG


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


# ── assertion (c) + the manifest (§8) ─────────────────────────────────────────────────
def _print_deferred_rows(*, manifest: tuple[ArmedAbort, ...] = MANIFEST) -> None:
    """R56's loud print, on EVERY run including a green one: registered debt that stops
    being visible stops being debt and starts being the status quo
    (`silent_encoding_gate.py:346-351`, copied in shape, not reinvented).

    `manifest` is a keyword for the same reason `audit_arming` has one: after WPAX Phase D
    flipped the draw-rate row the SHIPPED manifest holds ZERO deferred rows, so the only way
    to drive this mechanism — which survives because CARD-COORD-KNOBS (R78/R80) will feed it
    rows — is on a synthetic one. Keeping a row deferred so an assertion stayed true was
    REJECTED by R81: it would shape the shipped manifest to suit a test."""
    deferred = [row for row in manifest if row.status is Status.DEFERRED]
    if not deferred:
        return
    print(f"preflight: {len(deferred)} DEFERRED abort row(s) — NOT audited, owned, not yet "
          "closed:")
    for row in deferred:
        print(f"  {row.name}  owner={row.owner}")
        # R73 name-truth (WPMINT Phase K-B): this line read "arming surface DOES NOT EXIST
        # yet", which was true of the only deferred row that had ever existed and is FALSE of
        # the first one to actually reach it — `grad_norm_hard_abort` names
        # `train.hard_gn_threshold`, a key this same phase authored, and the row is deferred
        # because nobody has PRE-REGISTERED a value, not because the surface is missing. The
        # print says which of the two it is instead of asserting one.
        # Hoisted out of the f-string: the replacement field spanned a line break, which is
        # 3.12-only syntax — a SyntaxError on the pinned 3.11 CI floor (WPCLEAN Phase LT).
        surface = "present" if row.ceiling_path is None else f"present, ceiling {row.ceiling_path}"
        print(f"    arming surface: {row.config_path} "
              f"({surface}) — NOT audited, so a mint does not gate on it")
        # R251 / LAW-08: a deferred row's `cadence` is declared so the flip to REQUIRED stays
        # a one-field data edit (§8.5), which would leave it a field nothing reads until that
        # flip. Printed instead — the same rule that put `note` on this path.
        cadence = "NOT DECLARED" if row.cadence is None else (
            f"{row.cadence.value} over {list(row.cadence_paths)}")
        print(f"    earliest-fire cadence: {cadence} — judged only once this row is REQUIRED")
        # R265 / ADJ-D38: the CLOCK is printed beside the cadence for the same reason the
        # cadence is printed at all — it is the field that says which key the row's evidence
        # arrives on, and a deferred row whose axis nobody can see is the state ADJ-D38
        # measured on the WR axis. `period_path` is `None` on the two clocks that have no
        # config period, and the print says WHICH of the two rather than eliding it.
        if row.cadence is not None:
            clock = row.cadence.sample_clock
            tick = (f"1 tick = {clock.period_path}" if clock.period_path is not None
                    else ("1 tick = 1 training step, definitional"
                          if clock.is_step_clocked else "no train-step tick at all"))
            print(f"    sample clock: {clock.value} ({tick})")
        if row.source_pin is not None:
            rel, text = row.source_pin
            print(f"    pinned to {rel}: {text!r}")
        # SF-I4 / LAW-08: `note` is where "WHY is this row deferred" lives, and it had no
        # live consumer at all — read only by the oracle's flip simulation. A deferred row
        # whose reason is invisible is a row nobody can re-adjudicate, which is the same
        # rot R56's loud print exists to prevent. Printed, not dropped.
        print(f"    why: {row.note}")


#: Synthetic operands and PERIODS for `_cadence_self_test`, and the run length they are
#: judged against. HEALTHY is run5's own shape; VACUOUS is ADJ-D22's measured defect — a
#: sampling period three orders of magnitude past the whole run — and none of it is read from
#: any config, so the self-test keeps working on a tree whose configs have all moved.
#:
#: R265 / ADJ-D38 splits the period OUT of the operand tuple, because that is exactly what the
#: ruling did to the rows: a period is a property of the axis's SAMPLE CLOCK, not an operand a
#: row supplies. The same two periods then drive both clocks, which is what makes arm D a
#: statement about the WR axis rather than a second copy of arm B.
_SELF_TEST_RUN_LENGTH = 1_000_000
_SELF_TEST_HEALTHY_PERIOD = 1_000
_SELF_TEST_VACUOUS_PERIOD = 1_000_000_000
#: (consec, min_step) on the gate-boundary clock — run5's draw-rate shape.
_SELF_TEST_HEALTHY = (3, 25_000)
#: (collapse_consec, early_death_min_step, collapse_min_step, rolling_consec,
#: rolling_min_step) on the eval-round clock — run5's own sealbot-WR shape.
_SELF_TEST_WR = (3, 15_000, 25_000, 2, 20_000)


def _cadence_self_test() -> list[str]:
    """R251 + LAW-07: prove the cadence trigger CAN fire, in BOTH directions, before any
    verdict of it is trusted. The pattern is gates 8/14/15's — `r8_header_gate.py::self_test`
    runs on every invocation and refuses to publish a verdict from an instrument it has not
    just watched work.

    It is worth having because this check's failure mode is SILENCE: a fraction read once and
    discarded, or an `earliest_fire_step` that collapsed to a constant, leaves gate 12 green
    on exactly the configs it was built to refuse — which is the state of the tree ADJ-D22
    measured. Both mutations are caught here: an infinite (or absent) bound loses arm B, and a
    constant computation cannot satisfy A and B at once.

    R265 / ADJ-D38 adds three arms, and they cover the failure mode ADJ-D38 measured rather
    than ADJ-D22's. Arm D drives the WR axis in the EVAL-ROUND clock in both directions — the
    axis that had no arm at all before, on the gate LAW-15/F-30 says actually kills runs. Arm
    E refuses a period table that has collapsed onto one key, which is the mutation that would
    re-create "every axis audited in one clock" with arms A-D still green. Arm F refuses the
    step-clock FALLBACK by name: a step-clocked member handed no period must RAISE, because an
    answer there is the D38 defect in one call.

    Deliberately PURE ARITHMETIC over synthetic operands. It builds no config and no
    collaborator: O-2 bans a stand-in for a production object inside this tool, and the audit's
    WIRING to real configs is driven end-to-end elsewhere (the mini-tree rig), not simulated
    here. Arm E and arm F respect that too — they read the clock TABLE and the members' own
    refusals, never a config object.
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
    cadence_rows: list[dict] = []
    cadence_disarmed: list[str] = []
    audit = None
    for path in paths:
        try:
            config = _load(path)
            audit = audit_arming(config)
            # R251 / ADJ-D22: the SECOND half of assertion (c), on the same loaded config so
            # the two answers cannot be about different bytes. `audit_cadence` judges only the
            # rows `audit_arming` found ARMED — a disarmed row is the `disarmed` list's, and
            # naming it twice would send the operator chasing a cadence question about an
            # abort that is simply off.
            # `fraction` is passed EXPLICITLY rather than left to the callee's default, and
            # that is not a style choice: the report block and the rc-30 message below both
            # interpolate `EARLIEST_FIRE_FRACTION` as read HERE, so a call that let the
            # default supply it would publish one number while comparing against another the
            # moment the two could differ. It also makes the tool-level name the ONE the
            # audit sees, so neutering it reaches the comparison itself and not only the
            # self-test — the pin-3 mutation would otherwise stop half-way.
            verdicts = audit_cadence(config, fraction=EARLIEST_FIRE_FRACTION)
        except ArmingSurfaceMissingError as exc:
            # F-4, fixed to its class (R71). The shipped module raises the NAMED error; the
            # tool maps it onto its own already-defined manifest code. Without this the
            # AttributeError fell through to main's bare `except Exception` and became rc 1
            # PreflightInternalError — "the tool broke" — for what is a manifest defect the
            # operator can fix in one line. The message is carried through verbatim so the
            # row, the full dotted path and the failing segment survive the mapping.
            raise PreflightManifestError(
                f"an armed-abort row's arming surface does not resolve on {path.name}: {exc}"
            ) from exc
        except SampleClockNotDerivableError as exc:
            # R265 / ADJ-D38, mapped to its class exactly as F-4's failure is one line up.
            # An underivable sample clock must reach the operator as a NAMED manifest defect
            # (rc 31) and never as the tool's own rc 1 internal error — and never, ever as a
            # quiet fall back to the training-step clock, which is the whole ruling.
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
                # R265 / ADJ-D38: the CLOCK the row was judged in, the live key its period
                # came from and the period itself are published beside the step answer. The
                # vacuity argument the whole block already makes, one axis further: a reader
                # must be able to see WHICH clock a row cleared its bound in, because "judged
                # in the wrong clock" and "judged" were the same observable before this.
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
        # Ordered AFTER the arming check: a row that is simply off is the plainer diagnosis,
        # and an operator reading "cannot fire in time" about an abort that is not armed at
        # all would be chasing the wrong key. Same rc, because it is the same assertion —
        # an abort that cannot fire is not armed in any sense that protects the run (R251).
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
        # R251: the cadence half is PUBLISHED on the green path too, per config per row, with
        # the computed step and the bound it cleared. A check whose only visible output is its
        # own failure is a check nobody can audit for vacuity — the shape MF-3's
        # `source_pins_ok: True` literal took.
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
    """The tier block for a burst the cross-field validators ACCEPTED.

    Built in `_run_preflight` AFTER `_apply_burst_override` returns, never before: a burst that
    was refused (rc 11) is not a tier that ran, and stamping one would publish a coverage claim
    for a run that never started. `floors` carries every row with its own `cleared` flag so the
    operator can see WHICH rule made the tier what it is rather than re-deriving it.
    """
    return {"tier": _burst_tier(config, burst_steps),
            "burst_steps": int(burst_steps),
            "floors": [{"key": key, "value": value, "floor": floor,
                        "cleared": int(burst_steps) >= floor}
                       for key, value, floor in _burst_floors(config)],
            "required_for_mint": list(MINT_REQUIRED_TIERS),
            "covered": [], "owed": list(MINT_REQUIRED_TIERS), "does_not_prove": None}


def _new_report(mode: str) -> dict:
    """The report skeleton. The `not_run` reason is a PREDICTION here (no boot has happened
    yet, and `child` is None, so "no boot was spawned" is true at this instant);
    `_finalise_not_run` re-derives it from the run's own history before the write. The `tier`
    block is a prediction in exactly the same sense, and `_finalise_tier` is its half."""
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
def _boot_main(args) -> int:
    """The `--_boot` child: the REAL production posture, through the ONE composition
    authority, with nothing routed around.

    WPMAIN (CARD-RUN-MAIN, R121(a)) inverted this function. It used to BUILD the run for
    itself — seed, out-dirs, trainer, buffer, pool — which made a CI gate the owner of the
    only real collaborator build in the tree, and made "the preflight boots what run5 boots"
    a claim with no producer on either side. Every one of those steps now lives at
    `mantis.run.build_run_collaborators`, and `mantis.run.launch_run` calls the same pair.
    What survives here is the CONTAINMENT mechanism (the parent re-execs this file with
    `--_boot`, §6.2) and the tool's own two sanctioned instruments, which wrap AROUND the
    composer rather than reaching inside it:

      1. `_apply_burst_override` — a CONFIG-level transform, BEFORE the boot;
      2. the §4.2 resumed-trainer refusal — a READ-ONLY check, BETWEEN the builder and the
         composer, which is why the authority is a PAIR of functions and not one opaque
         `boot()`: a single call would have needed a preflight hook smuggled inside it,
         which is the divergence seam the card exists to close.

    Nothing else may sit between the two calls, and nothing may be assigned onto `collab` —
    a collaborator swapped after the build is a boot the composer never agreed to
    (`tests/test_run_one_authority.py::test_the_preflight_child_boots_through_one_builder_and_one_composer_only`).

    The DEVICE is the config's own (`train.device`, R126): there is no `--device` flag on
    this tool any more, so preflighting run5 boots run5's minted device and a `--device cpu`
    invocation can no longer false-clear a cuda-minted run's memory wall (the WPBOX 16 GiB
    OOM; LAW-03's instrument-that-cannot-false-clear corollary). The EVAL posture is the
    config's own too (`eval_enabled`, R120): the child passes nothing and CAN pass nothing.

    F-816-14 (R284(f)): this child is spawned with `start_new_session=True`, which is what lets
    the parent's timeout `killpg` reach it AND every grandchild in one act — and is also what
    makes it unreachable by any signal aimed at the parent. Its death therefore depends entirely
    on the parent living long enough to run its `except TimeoutExpired` block. MEASURED on the
    local host 2026-08-18: one of these was found at PPID 1, **4 h 06 m old at 682% CPU against
    its own `--timeout-sec 45.0`**, because its parent had been killed. The first thing the child
    now does is ask the KERNEL to end it when its parent dies.
    """
    from mantis.train.lifecycle.signals import arm_parent_death_signal

    arm_parent_death_signal()

    config = _load(_resolve_config_path(args.config))
    booted = _apply_burst_override(config, args.burst_steps)
    from mantis.run import build_run_collaborators, compose_run

    collab = build_run_collaborators(config=booted, out_dir=args.out_dir)
    # §4.2: a run RESUMED past its ceiling terminates having performed zero syncs, which
    # looks EXACTLY like the frozen actor this preflight exists to find. The builder never
    # passes `checkpoint_path`, and a nonzero step here is a named refusal, not a warning.
    if int(collab.trainer.step) != 0:
        raise PreflightResumedTrainerError(
            f"the freshly-built trainer reports step {int(collab.trainer.step)}, not 0: a "
            "preflight over a resumed trainer measures nothing while looking like the defect "
            "it exists to find (§4.2)"
        )
    handles = compose_run(config=booted, trainer=collab.trainer, pool=collab.pool,
                          buffer=collab.buffer, log_dir=collab.log_dir,
                          checkpoint_dir=collab.checkpoint_dir)
    return _abort_rc(handles.shutdown.abort_rule)


def _abort_rc(rule: str | None) -> int:
    """The child's rc, decided by WHETHER AN ABORT FIRED and by nothing else (R84).

    This is the CHILD's process boundary. It is no longer the only one in the repo: WPMAIN
    landed `mantis.run.main()` as a real launcher, and it reads THIS SAME resolver rather
    than re-deriving the mapping — which is exactly what the OWED paragraph below asked for,
    and what `repo_design.md`'s own OWED clause is discharged by. Two boundaries, one
    resolver, and the numbers still come from the manifest row and from nowhere else.

    Three outcomes, and the middle one is the one that must not be quietly rounded off:

    * no rule fired -> 0. `abort_rule is None` is the ONLY thing that means a clean run;
    * a rule fired WITH an authored code -> that code, resolved from the manifest row. The
      number is never written here;
    * a rule fired with NO authored code -> a NAMED failure, never 0 and never an invented
      number. `grad_norm_hard_abort` and `sealbot_wr_abort` share `_fire_hard_abort` and are
      not pre-registered; R84 refused to invent codes for them and this refuses again rather
      than reporting an aborted run as a clean boot.

    DISCHARGED (WPMAIN): the production launcher landed and reads this same resolver —
    `mantis.run.main`, whose unregistered-rule arm raises `UnregisteredAbortExitError` with
    this function's own three-outcome doctrine. The clause is kept as the record of what the
    obligation was, not as an open one.
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
    # CARD-PREFLIGHT-CHILD-STDERR-BUDGET (WPCLEAN Phase PFC, the card's spool arm): the
    # 4000-char tails were an invented budget AND the classifier's input — a truncated
    # traceback silently downgrades a tree defect from 32 to 33 (REVIEW_IMPL_P's widening).
    # The FULL streams now spool beside the report; the tails stay (report readability +
    # the process tests' carriage semantics), and the classifier keeps reading the tail —
    # a wall's `AttributeError` line is the traceback's LAST line, which a tail keeps and
    # a head would lose. Spool failure is a run-fatal report defect, not a swallow (LAW-14):
    # the spool exists precisely for the bytes the tail dropped.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # a child may die before creating it
    stdout_spool = out_dir / "child_stdout.log"
    stderr_spool = out_dir / "child_stderr.log"
    stdout_spool.write_text(stdout or "", encoding="utf-8")
    stderr_spool.write_text(stderr or "", encoding="utf-8")
    child = {"rc": rc, "rc_convention": RC_CONVENTION,
             "raised_by": "child" if rc in PASS_THROUGH else "parent",
             "wall_clock_sec": round(time.monotonic() - started, 3), "timed_out": timed_out,
             "stdout_tail": (stdout or "")[-4000:], "stderr_tail": (stderr or "")[-4000:],
             "stdout_spool": str(stdout_spool), "stderr_spool": str(stderr_spool)}
    if rc < 0:
        child["signal"] = -rc
        child["signal_name"] = signal.Signals(-rc).name
    report["child"] = child
    return child


def _run_preflight(args, report: dict, out_dir: Path) -> None:
    path = _resolve_config_path(args.config)
    config = _load(path)
    report["config"] = _config_block(path, config)
    report["coordinator"] = _coordinator_block(config)
    report["manifest"] = _audit_manifest_and_configs(_audit_paths(path))
    report["assertions"]["c_arming"] = {"verdict": "pass", "disarmed": [],
                                        "required_armed": report["manifest"]["required"]}
    booted = _apply_burst_override(config, args.burst_steps)
    # WPMINT Phase B: the tier is stamped only once the validators have ACCEPTED the burst, so
    # a rc-11 refusal leaves `tier: none` — which is the truth, not a placeholder.
    report["tier"] = _tier_block(config, int(args.burst_steps))
    report["override"] = {"keys": list(OVERRIDE_KEYS),
                          "from": int(config.train.max_train_steps),
                          "to": int(args.burst_steps),
                          # THE one identity authority (F-B1): the same function the child's
                          # compose_run hashes its own loaded config with.
                          "booted_config_sha256": config_identity_sha256(booted)}
    log_dir = out_dir / "logs"
    # CARD-PREFLIGHT-OUTDIR-REUSE: refuse a dirty out-dir BEFORE the boot. Scoped exactly
    # to the hole the card measured — pre-existing segments under THIS run_id, the ones
    # `_read_segment`'s scope would believe — so a fresh dir, a foreign run_id's litter and
    # every audit-mode drive are untouched.
    stale = sorted(log_dir.glob(f"events_{booted.run_id}_*.jsonl")) if log_dir.is_dir() else []
    if stale:
        raise PreflightOutDirReusedError(
            f"--out-dir {out_dir} already holds {len(stale)} event segment(s) for run_id "
            f"{booted.run_id!r} (first: {stale[0].name}): a same-run_id reuse would read a "
            "previous burst's events as this run's evidence. Use a fresh out-dir"
        )
    child = _run_child(args, report)
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
    # F-B1 closure: copy the child's OWN published boot identity into the child block and
    # compare it to the parent's. Ordered AFTER _classify_child (a dead child is a
    # child-status failure, not an identity one) and BEFORE the predicates (a burst on the
    # wrong config proves nothing about sync or lag).
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

        WPMINT Phase K-B (R78's first design question) adds the coordinator block and
        publishes both in AUDIT mode — the per-commit gate is the report a reader sees most,
        and the block costs no boot. It is hoisted into a closure so a NAMED subject can be
        published BEFORE the manifest audit: a red audit used to write a report whose
        `config` was still `null`, so the artifact could not say which config failed.
        """
        subject_config = _load(subject)
        report["config"] = _config_block(subject, subject_config)
        report["coordinator"] = _coordinator_block(subject_config)

    if named is not None:
        _publish(named)
    # SF-I9: the manifest audit runs BEFORE anything indexes `paths`. It carries the vacuity
    # guard, so an empty PRODUCTION_CONFIGS is rc 31 by name rather than an `IndexError`
    # collapsing through the generic handler into an unnamed rc 1. The hoist above touches
    # `named` only, which is never an index, so that guard is untouched.
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
    except Exception as exc:  # noqa: BLE001 — the UNDIAGNOSED outcome, named as such
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
