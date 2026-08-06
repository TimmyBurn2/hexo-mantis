# R8 >300 justify: the manifest ROWS are data and their reason text IS the row —
# `note` is a live consumer's field, printed by gate 12 on every run, not a comment that
# can be trimmed. The two walkers below (`_dotted`, `audit_arming`) carry the F-4 named-arm
# and disarmed-short-circuit rationale; splitting them from the
# rows they walk would put "which aborts must arm" and the predicate that reads it on
# opposite sides of an import, which is the drift this module exists to prevent. Phase X adds
# `exit_code_for_abort` for the same reason: "which code does a
# fired abort exit with" is answered BY the rows, and a resolver living anywhere else becomes
# a second authority for that answer the first time a row's `exit_code` moves. Phase K-B adds
# a third row and the `ceiling_path` mechanism it needs: the row
# is where "why is grad-norm deferred, and what closes it" is written, and gate 12 prints that
# text on every run. WPMAIN RT-2/R132 adds a FOURTH row plus the
# exported `DISK_SPACE_ABORT_RULE` the composition root imports: the disk-guard abort's rule
# name has two readers — this row and `mantis.run` — because `mantis.train` may not import this
# module, so a bare literal at each would be the duplicated-authority shape R1 exists to kill.
# The row's own `note` carries why it is REQUIRED, what drift it exists to catch and the
# operator-SIGTERM residual R132 did not close, which is data gate 12 prints, not a comment.
# WP12-R Phase O (R152) adds a FIFTH row plus the exported
# `TERMINAL_EVAL_BROKEN_ABORT_RULE`, for the reason the disk-guard row above exists in this
# shape: the broken-terminal-eval rule name has two readers (this row and `mantis.run`), and
# `mantis.train` may not import this module. Its `note` carries the one-code-for-seven-reasons
# decision, the two disclosed residuals and the cooperative-delivery argument — gate-12-printed
# data, not commentary. The same phase deletes the last exit-code LITERAL in the row set
# (`actor_lag`'s bare 45 became `ACTOR_LAG_EXIT_CODE`): an AST census now forbids an integer
# literal in any `exit_code=`, because an identity check cannot witness one — CPython interns
# the small ints, so a typed `48` IS the imported constant under `is`.
"""The armed-abort manifest — WHICH aborts a production config MUST arm (R61, DESIGN_P §8).

ONE authority, and it is DATA. A markdown register would need a parser, and the parser's
grammar becomes a second authority with its own failure modes (a row that parses to nothing
reads as "no such requirement"). A typed frozen dataclass is read by `import`, carries its
invariant in `__post_init__`, and cannot drift from a doc twin — so this module ships no doc
twin of the rows. The precedent is `config/resolve/composition.py:10-13`: "the rule is a
config-layer fact, so it lives in the config layer", and "which aborts a production config
must arm" is a config-layer fact of exactly that species.

SF-4 — THE LAYER BOUNDARY. This module makes ZERO filesystem calls. `PRODUCTION_CONFIGS`
holds repo-relative STRINGS (data); resolving them against a repo root, and reading a
`source_pin`'s pinned file, both live in `tools/ci_gates/preflight_mint.py`, where
`REPO_ROOT = Path(__file__).resolve().parents[2]` is structurally sound. A shipped package
that resolved `parents[3]` to the repo root would be depending on an editable install.
Pinned by `tests/config/test_armed_abort_manifest.py`,
`test_the_manifest_module_makes_no_filesystem_call`.

`wr_hard_abort_enabled` is ABSENT BY DECISION, not by oversight: it is the sealbot win-rate
abort, which ships WARN-ONLY by operator ruling G-3, and STATE §6 names the mint-blocking
pair as "draw-rate + actor-lag". A later reader must not "fix" it in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# The exit-code AUTHORITY (WPMINT Phase X, CARD-ABORT-EXIT / R84). 42/43/45 already live in
# `mantis.monitor.heartbeat` and `monitor/supervise.py` imports two of them across the
# supervisor seam, so 46 joins them there rather than opening a fourth site for one family.
# Imported rather than re-typed: the row below is the manifest's copy of the number, and a
# literal here would be the second place "which code does the draw-rate abort use" is
# written. `mantis.monitor.heartbeat` imports nothing from `mantis` (stdlib only), so this is
# a leaf edge in the same direction `config/resolve/monitor.py` already takes (gate 9).
from mantis.monitor.heartbeat import (
    ACTOR_LAG_EXIT_CODE,
    DISK_SPACE_EXHAUSTED_EXIT_CODE,
    DRAW_RATE_COLLAPSE_EXIT_CODE,
    TERMINAL_EVAL_BROKEN_EXIT_CODE,
)

#: The disk-guard abort's RULE NAME — one spelling, exported (WPMAIN RT-2 / R132).
#:
#: Every other rule name in this file is a string literal typed once, because its producer
#: (`StepCoordinator._fire_hard_abort`) receives the name from the gate that fires and never
#: has to agree with a row. The disk-guard rule is different: `mantis.train` may not import
#: this module (the rule-name carrier's whole point), so the guard publishes only the FACT
#: that it fired and `mantis.run.compose_run` — which already imports `exit_code_for_abort`
#: for its own rc resolution — names the rule. That name therefore has TWO readers, the row
#: below and the root, and a bare literal at each would be exactly the duplicated-authority
#: shape R1 exists to kill: rename the row and the root goes on recording a rule the resolver
#: answers `None` for, which is `UnregisteredAbortExitError` at every disk-full event.
DISK_SPACE_ABORT_RULE: str = "disk_space_exhausted"

#: The broken-terminal-eval RULE NAME — one spelling, exported (WP12-R Phase O, R152).
#:
#: Same shape and same grounds as `DISK_SPACE_ABORT_RULE` one line above: `mantis.train`
#: may not import this module, so the coordinator's terminal-eval latch carries only the
#: FACT (the round's own reason string) and `mantis.run.compose_run` — which already
#: imports `exit_code_for_abort` for its own rc resolution — names the rule. Two readers,
#: the row below and the root, so a bare literal at each is the duplicated-authority shape
#: R1 exists to kill: rename the row and the root goes on recording a rule the resolver
#: answers `None` for, i.e. `UnregisteredAbortExitError` on every broken terminal battery.
TERMINAL_EVAL_BROKEN_ABORT_RULE: str = "terminal_eval_broken"


def _is_real_number(value: Any) -> bool:
    """True for a finite `int`/`float` that is not a `bool` (WPMINT Phase K-B).

    Extracted from `Mechanism.is_armed`'s own guard so the ceiling and the value are judged
    by ONE rule. `bool` is excluded because `isinstance(True, int)` is True and `True`
    arriving on a threshold path is a type confusion, not a threshold. Non-finite is excluded
    because `inf` compares greater than every ceiling and `nan` compares False against all of
    them — both would decide an arming question by accident.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


class Status(StrEnum):
    """REQUIRED rows are audited and gate; DEFERRED rows are printed loudly and do not."""

    REQUIRED = "required"
    DEFERRED = "deferred"


class Mechanism(StrEnum):
    """The predicate that decides "armed" for a row's value. DATA, not a branch on `name`.

    `audit_arming` never branches on a row's identity: `status` selects the list and
    `mechanism` selects the predicate, which is what makes Phase D's DEFERRED→REQUIRED flip
    a one-field data edit (§8.5, proven by O-7 rather than asserted).
    """

    CONFIG_BOOL = "config_bool"
    CONFIG_THRESHOLD_GT_ZERO = "config_threshold_gt_zero"
    #: WPMINT Phase K-B (adjudication call K-c). An UPPER-bounded threshold: armed iff the
    #: value is a real, finite, positive number that is ALSO no greater than a ceiling read
    #: off a second config path (`ArmedAbort.ceiling_path`).
    #:
    #: Why a second member rather than a tighter `CONFIG_THRESHOLD_GT_ZERO`: `> 0` is the
    #: correct and complete predicate for `train.draw_rate_abort.threshold`, whose schema
    #: already closes the high end at `le=1`. It is the WRONG predicate for
    #: `train.hard_gn_threshold`, whose range is genuinely unbounded above and whose shipped
    #: `1e9` is finite, positive and unreachable by any real gradient norm — `> 0` reads that
    #: as ARMED, which is the "armed in the config, absent in effect" defect this manifest
    #: exists to make visible.
    #:
    #: Why the ceiling is DATA on the row and not a number here: a literal ceiling in this
    #: enum would be a policy value nobody pre-registered, which is the class R84 refused when
    #: it ratified `exit_code=None` rather than fabricating a `46`. The row names a config
    #: path instead, so the ceiling is a value the operator already minted for the same
    #: quantity, it moves when the config moves, and `is_armed` stays a pure predicate.
    CONFIG_THRESHOLD_BELOW_CEILING = "config_threshold_below_ceiling"

    def is_armed(self, value: Any, *, ceiling: Any = None) -> bool:
        """True iff `value` arms the abort. A real predicate in BOTH directions — a
        constant here would silently arm or disarm every row at once.

        `ceiling` is consumed ONLY by `CONFIG_THRESHOLD_BELOW_CEILING` and is resolved by
        `audit_arming` from the row's own `ceiling_path`; the other two mechanisms ignore it,
        which is why it is keyword-only with a `None` that means "no ceiling was named". A
        `CONFIG_THRESHOLD_BELOW_CEILING` row with no usable ceiling reports DISARMED rather
        than ARMED: an unjudgeable row must fail toward visibility, never toward silence.

        `CONFIG_THRESHOLD_GT_ZERO`'s arms are BYTE-UNCHANGED, deliberately: this phase authors
        knobs and changes no verdict. `_is_real_number`'s finiteness test is applied only on
        the new branch, so `is_armed(inf)` still answers True for the `> 0` mechanism (a value
        `train.draw_rate_abort.threshold`'s `le=1` cannot produce anyway).
        """
        if self is Mechanism.CONFIG_BOOL:
            return value is True
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if self is Mechanism.CONFIG_THRESHOLD_BELOW_CEILING:
            if not _is_real_number(value) or not _is_real_number(ceiling):
                return False
            return 0.0 < float(value) <= float(ceiling)
        return float(value) > 0.0


@dataclass(frozen=True)
class ArmedAbort:
    """One row: an abort, the config surface that arms it, and its ownership posture.

    `owner` and `source_pin` are REQUIRED on a DEFERRED row; `owner` is FORBIDDEN on a
    REQUIRED one and `source_pin` is UNCONSTRAINED there. Each of the three rules
    `__post_init__` enforces is a way for a row to go invisible: an owner-less deferred row
    has nobody to chase, a pin-less one is not tamper-evident, and a required row carrying
    an owner reads as already-excused.

    N-1 (WPAX Phase D, R73): this sentence used to say `source_pin` was FORBIDDEN on a
    REQUIRED row. That was FALSE — `__post_init__` never constrained it — and acting on it
    would have dropped the draw-rate pin at Phase D's flip, leaving the newly-REQUIRED row
    with no tamper-evidence exactly as it started gating a production mint, and silently
    emptying the two pin-scan tests that stand on "no pinned row means this test has no
    subject". A REQUIRED row MAY keep a pin, and this one does.
    """

    name: str
    config_path: str
    mechanism: Mechanism
    status: Status
    exit_code: int | None
    owner: str | None
    source_pin: tuple[str, str] | None
    note: str
    #: WPMINT Phase K-B: the SECOND config path a `CONFIG_THRESHOLD_BELOW_CEILING` row needs
    #: — where its upper bound is minted. It is the one field here that carries a default,
    #: and the default is safe for the reason the other three rules are enforced rather than
    #: documented: `__post_init__` REQUIRES it on the mechanism that consumes it and FORBIDS
    #: it on the two that do not, in both directions, so `None` can neither arm a row nor
    #: excuse one. A no-default field would instead have forced nine frozen-oracle
    #: construction sites to type `ceiling_path=None`, which buys nothing the predicate below
    #: does not already guarantee.
    ceiling_path: str | None = None

    def __post_init__(self) -> None:
        if self.status is Status.DEFERRED and not self.owner:
            raise ValueError(
                f"armed-abort row {self.name!r} is DEFERRED and carries no `owner`: "
                "deferred debt with no owner is debt nobody is chasing (R56)"
            )
        if self.status is Status.DEFERRED and not self.source_pin:
            raise ValueError(
                f"armed-abort row {self.name!r} is DEFERRED and carries no `source_pin`: "
                "a deferred row that is not tamper-evident rots into the status quo (§8.4)"
            )
        if self.status is Status.REQUIRED and self.owner:
            raise ValueError(
                f"armed-abort row {self.name!r} is REQUIRED and carries an `owner`: an "
                "owner on a required row reads as already-excused; drop the owner or "
                "declare the row DEFERRED"
            )
        needs_ceiling = self.mechanism is Mechanism.CONFIG_THRESHOLD_BELOW_CEILING
        if needs_ceiling and not self.ceiling_path:
            raise ValueError(
                f"armed-abort row {self.name!r} uses {self.mechanism.value} and names no "
                "`ceiling_path`: that predicate is DISARMED without a ceiling, so the row "
                "would read disarmed forever for a reason nobody could see in the row"
            )
        if not needs_ceiling and self.ceiling_path:
            raise ValueError(
                f"armed-abort row {self.name!r} names a `ceiling_path` "
                f"({self.ceiling_path!r}) but its mechanism {self.mechanism.value} ignores "
                "it: a config path the predicate never reads is a claim the audit does not "
                "make (LAW-07's phantom-input class)"
            )


@dataclass(frozen=True)
class AuditResult:
    """What `audit_arming` publishes. `disarmed` is the only field that gates."""

    required: tuple[ArmedAbort, ...]
    deferred: tuple[ArmedAbort, ...]
    disarmed: tuple[ArmedAbort, ...]


#: The rows. R61 fixes the set, ADJ-08's census supplies the values, R65 fixes the statuses.
MANIFEST: tuple[ArmedAbort, ...] = (
    ArmedAbort(
        name="actor_lag",
        config_path="monitor.actor_lag_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL,
        status=Status.REQUIRED,
        exit_code=ACTOR_LAG_EXIT_CODE,
        owner=None,
        source_pin=None,
        note=(
            "The frozen-actor hard abort (exit 45). Armed on configs/run5.yaml since the "
            "R59 flip; disarming it on a production config is the run3 failure mode "
            "re-enabled."
        ),
    ),
    ArmedAbort(
        name="draw_rate_collapse",
        config_path="train.draw_rate_abort.threshold",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO,
        status=Status.REQUIRED,
        exit_code=DRAW_RATE_COLLAPSE_EXIT_CODE,
        owner=None,
        source_pin=(
            "src/mantis/run.py",
            "draw_rate_abort=resolve_draw_rate_abort(config.train)",
        ),
        note=(
            "The self-play draw-rate collapse hard abort. Armed on configs/run5.yaml at "
            "threshold 0.25 (R82) with min_step 25000 and N_pool_min 50 (R92's guards; the "
            "per-worker min_samples bar was DELETED with the filtered-mean statistic it "
            "guarded), all three pre-registered at mint prereg. The gated statistic is the "
            "pooled count-weighted rate Sum(draws)/Sum(completed) over the union of worker "
            "windows; below N_pool_min completed games the gate makes NO OBSERVATION "
            "(skip-counted), never a healthy 0.0. NOTE for the mint record (WPMINT DS-VERIFY, "
            "correcting the WITHDRAWN DR-8): consec=3 counts consecutive CHECKS at a stride of "
            "log_interval train steps, so at the shipped log_interval 1000 the three samples "
            "span 2000 steps. The earliest possible fire is nonetheless step 25000, because "
            "SAMPLING IS NOT GATED BY min_step — the history accumulates from step 1000, holds "
            "25 samples by step 25000, and min_step gates only the FIRE. DR-8's contrary claim "
            "(earliest fire 27000) was MEASURED FALSE and withdrawn. The pin binds to the THREADING at the "
            "construction site, so deleting it, renaming the resolver or reordering the "
            "call past it all break the R56 scan. exit_code is 46 "
            "(monitor.heartbeat.DRAW_RATE_COLLAPSE_EXIT_CODE) since WPMINT Phase X discharged "
            "CARD-ABORT-EXIT (R84). This row's exit_code was None until then — truthfully, "
            "because the gate stops the run COOPERATIVELY and no distinct process exit code "
            "existed. Delivery is STILL cooperative and that is deliberate: the gate sets "
            "shutdown.running = False and returns, so the loop unwinds through close_out, the "
            "terminal-eval drain and the shutdown checkpoint, which an os._exit(46) would "
            "discard (LAW-16 save-then-exit). Family parity is taken in this registry and in "
            "the supervisor's READING of the rc, not in the delivery mechanism. What makes a "
            "fired abort distinguishable from a clean run is ShutdownState.abort_rule, which "
            "_fire_hard_abort sets to the rule NAME beside the stop; a process boundary maps "
            "it here through exit_code_for_abort. The three clean stops (stop(), O2 "
            "iteration limit, O3 shutdown-save) leave the field None."
        ),
    ),
    ArmedAbort(
        name=DISK_SPACE_ABORT_RULE,
        config_path="monitor.disk_guard.fail_gb",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO,
        status=Status.REQUIRED,
        exit_code=DISK_SPACE_EXHAUSTED_EXIT_CODE,
        owner=None,
        source_pin=(
            "src/mantis/run.py",
            "shutdown.record_abort(DISK_SPACE_ABORT_RULE)",
        ),
        note=(
            "LAW-16 leg 3, the disk guard (exit 47). WPMAIN constructed the guard for the "
            "first time in any run (R121(b)/R122) and WPMAIN's RED-TEAM then measured what "
            "that armed: the critical arm SIGTERMs its own pid, the handler sets "
            "shutdown_save/running and NEVER abort_rule, and mantis.run.main read "
            "`abort_rule is None` and returned 0. A run the disk guard killed reported "
            "SUCCESS — R44's class, on this WP's own new subsystem — and a supervisor reading "
            "only the rc relaunches into the same full volume. R132 closes it the way R84 "
            "closed the draw-rate leg: a registered code, resolved through this manifest, "
            "never a second literal. "
            "WHY REQUIRED AND NOT DEFERRED, since gate 12 audits every required row against "
            "every production config: nothing has to be invented for this row and nothing is "
            "owed, which is the exact test the grad-norm row below FAILS. The arming surface "
            "monitor.disk_guard.fail_gb is a minted operator value on all six committed "
            "configs (5.0), its schema carries gt=0, and the block is a REQUIRED field of "
            "MonitorConfig — so a validated RunConfig arms this row by construction and a "
            "DEFERRED status would demand an `owner` for debt that does not exist. What the "
            "row is FOR, then, is the drift it makes loud: `_dotted` short-circuits a "
            "mid-walk None to DISARMED, so the day someone makes the disk-guard block "
            "optional or nullable — the posture that let the guard sit unconstructed with "
            "dead 60/10/5 literals for the whole migration — gate 12 goes RED on run5 "
            "instead of the guard quietly disappearing again. "
            "DELIVERY IS COOPERATIVE, like 46 and for the same reason: the SIGTERM is "
            "save-then-exit, so the run unwinds through close_out, the terminal-eval drain "
            "and the shutdown checkpoint. An os._exit(47) from the guard thread would discard "
            "the very save the guard fires to protect. What carries the signal is NOT a "
            "cross-thread write: the guard latches `critical_fired` (RT-2b — the unlatched "
            "arm re-fired every interval_sec and supplied LAW-16's second press itself, "
            "sys.exit(1) mid-save against 14400 s drain caps), and compose_run's teardown "
            "reads that latch AFTER disk_guard.stop() has joined the thread, then records "
            "this rule on the ShutdownState it owns. "
            "The pin binds THAT recording line, which is the whole mechanism: delete it, "
            "rename the rule constant or drop the transfer past the guard's stop() and the "
            "R56 scan breaks rather than the rc silently returning to 0. "
            "RESIDUAL, disclosed (RT-2's wider claim, NOT closed here): an OPERATOR's SIGTERM "
            "and a supervisor's own stop still resolve to rc 0, because ShutdownState carries "
            "no rule for them and R132's scope is the guard. A deliberate operator stop is "
            "arguably a clean stop; that judgement is not taken here."
        ),
    ),
    ArmedAbort(
        name=TERMINAL_EVAL_BROKEN_ABORT_RULE,
        config_path="train.terminal_eval_enabled",
        mechanism=Mechanism.CONFIG_BOOL,
        status=Status.REQUIRED,
        exit_code=TERMINAL_EVAL_BROKEN_EXIT_CODE,
        owner=None,
        source_pin=(
            "src/mantis/run.py",
            "shutdown.record_abort(TERMINAL_EVAL_BROKEN_ABORT_RULE)",
        ),
        note=(
            "The broken-terminal-eval outcome (exit 48), WP12-R Phase O / R152, closing "
            "R133's measured caveat 'rc 0 does not certify eval health'. At HEAD the "
            "terminal round's result — reason included — was computed, emitted, routed and "
            "then THROWN AWAY one frame below ShutdownState (drain.close_out discarded "
            "run_terminal_eval's return), and promote.py's refusal to promote a broken "
            "round was the ONLY production consumer of broken-ness anywhere in src/. So a "
            "run whose terminal battery was killed, whose worker returned garbage or whose "
            "ladder state never reached disk exited 0 and the supervisor above recorded a "
            "clean finish — LAW-15's 'no promotion decision = deliverable incomplete', "
            "invisible at the process boundary. "
            "ONE code for SEVEN reason classes, on the record: the family is one number per "
            "OUTCOME with the CAUSE in the payload (rc 45 covers every actor-lag fire), and "
            "the seven causes stay pairwise-distinguishable in the ONE channel through "
            "mantis.eval.errors.EvalBrokenReason — on the eval_broken event's reason and on "
            "the round result's eval_broken_reason. A supervisor reading only the rc sees "
            "'terminal eval degraded' and not WHICH break; that is stated, not hidden. "
            "WHY REQUIRED AND NOT DEFERRED, the exact test the grad-norm row below fails: "
            "nothing has to be invented and nothing is owed. train.terminal_eval_enabled is "
            "a REQUIRED typed bool (config/schema/train.py) minted true on all six "
            "committed configs, so gate 12 is green the moment this row lands and NO armed "
            "value moves. What the row is FOR is the drift it makes loud: the day someone "
            "mints a production config with the terminal eval off, gate 12 goes RED instead "
            "of the run quietly shipping with no terminal promotion decision at all. "
            "RESIDUAL, disclosed: the rc is reachable only if BOTH eval_enabled and "
            "train.terminal_eval_enabled are true, and a row carries ONE config_path. The "
            "nearer condition is armed here (it gates the terminal round specifically, "
            "where eval_enabled gates all eval); the other half is held by "
            "tests/config/test_minted_config_remint.py::"
            "test_a_minted_config_carries_the_identity_and_eval_leaves, which asserts "
            "leaves['eval_enabled'] is True over all six committed configs — a real "
            "per-config assertion, not a live-consumer pin, which is why disclosure is "
            "sufficient rather than merely tolerable. "
            "SECOND RESIDUAL: target_integrity_defects, the sibling Phase-T counter this "
            "phase lands in the event stream, reads 0 in EVERY run that survives to emit an "
            "iteration_complete — its latch is run-fatal — so that permanent zero is the "
            "LAW-18 'an idle lever stays VISIBLE at 0' posture and must not be misread as "
            "an unproduced field. "
            "DELIVERY IS COOPERATIVE, like 46 and 47, and it is the cleanest of the three: "
            "46/47 stay cooperative because an os._exit would discard a save still in "
            "flight, while the terminal eval is the LAST action of close_out — the loop is "
            "over, the buffer is saved, and delivery is main returning the number. "
            "drain.run_terminal_eval latches the routed result's own eval_broken_reason on "
            "the coordinator (set-once, one writer, reachable only from the one function "
            "that passes ignore_stride=True), and compose_run's teardown re-parses that "
            "string through EvalBrokenReason — an unregistered spelling is a loud "
            "ValueError, never a silent rc 0 — before recording this rule. The read sits "
            "AFTER the disk-guard read so first-fire-wins keeps the ROOT CAUSE: a disk-full "
            "run whose terminal eval then breaks reports 47, not 48. The pin binds that "
            "recording line, so deleting it, renaming the rule constant or reordering it "
            "past the disk-guard read all break the R56 scan rather than the rc silently "
            "returning to 0."
        ),
    ),
    ArmedAbort(
        name="grad_norm_hard_abort",
        config_path="train.hard_gn_threshold",
        ceiling_path="monitor.alert_grad_norm_max",
        mechanism=Mechanism.CONFIG_THRESHOLD_BELOW_CEILING,
        status=Status.DEFERRED,
        exit_code=None,
        owner="CARD-COORD-KNOBS follow-up — the operator, at run5 mint prereg",
        source_pin=(
            "src/mantis/train/coordinator/step.py",
            "if math.isfinite(step_gn) and step_gn > cfg.hard_gn_threshold:",
        ),
        note=(
            "The optimizer-instability hard abort (`grad_norm_hard_abort`, coordinator/step.py "
            "D3): fire when grad_norm exceeds train.hard_gn_threshold for "
            "train.hard_gn_min_steps consecutive training steps. It has a real gate, a real "
            "`_gate_stats` counter and a real `_fire_hard_abort` path, and it had NO manifest "
            "row at all until WPMINT Phase K-B — while its threshold sat at the unauthored "
            "code-side literal 1e9, which no finite gradient norm reaches. So the run shipped a "
            "hard abort that could not fire and nothing said so. "
            "WHY DEFERRED AND NOT REQUIRED (adjudication call K-c): flipping it REQUIRED would "
            "gate run5's mint on a grad-norm threshold nobody has pre-registered, and the tool "
            "would then be demanding a number this repo would have to invent — the class R84 "
            "refused when it ratified exit_code=None rather than fabricating a 46. A DEFERRED "
            "row prints loudly on every gate-12 run and gates nothing, which is exactly the "
            "posture for a live gate whose value is owed. "
            "WHY THE MECHANISM IS NEW: CONFIG_THRESHOLD_GT_ZERO would read 1e9 as ARMED, which "
            "is 'armed in the config, absent in effect' — the defect the manifest exists to "
            "surface. CONFIG_THRESHOLD_BELOW_CEILING reads the ceiling off `ceiling_path`, "
            "monitor.alert_grad_norm_max: the value the operator ALREADY minted as 'this grad "
            "norm is worth warning about' (10.0 on every committed config). A hard abort set "
            "orders of magnitude above the line the run already WARNS at is not a hard abort. "
            "That ceiling is derived from the config, never from this file, so no number is "
            "invented here either. "
            "TO CLOSE THIS ROW: pre-register a threshold at mint prereg, mint it into "
            "train.hard_gn_threshold, and flip status to REQUIRED — a one-field data edit, the "
            "same shape Phase D's flip took. Until then run5 mints with this abort disarmed, "
            "knowingly and in writing. exit_code is None, truthfully: `_fire_hard_abort` stops "
            "the run cooperatively and R84 authored a code for the draw-rate family only; "
            "inventing one here would be that same refused class one layer down "
            "(`exit_code_for_abort`'s docstring says so by name). The pin binds to the gate's "
            "own comparison, so deleting the gate, renaming the field or inverting the test all "
            "break the R56 scan."
        ),
    ),
)

#: WHICH configs the law binds — one authority. Repo-relative strings only; resolving them
#: is the tool's (SF-4).
#:
#: ADJ-13 N-1: `configs/run5.yaml`'s membership here is not decoration and is not free to
#: move. Nothing pinned it, so moving run5 to `EXEMPT_CONFIGS` with a written reason and
#: promoting an armed smoke config in its place kept gate 12 at **rc 0 with run5 disarmed** —
#: the partition stayed a partition and every exemption still carried a reason, so every
#: existing check was satisfied by the swap. The producer is
#: `test_run5_is_bound_BY_NAME_and_is_not_freely_exemptable`
#: — the run the operator is about to mint is audited BY NAME, and
#: exempting it is a red gate rather than a bookkeeping edit. (Recheck R-8: this citation
#: named a test that does not exist — a LAW-07 producer citation that greps to nothing is
#: LAW-07's own failure mode. Every citation in this pass was `grep`-verified.)
PRODUCTION_CONFIGS: tuple[str, ...] = ("configs/run5.yaml",)

#: The OTHER half of the same authority (MF-7). R59's "deliberate disarming remains legal for
#: smoke configs" used to be expressed by ABSENCE from `PRODUCTION_CONFIGS` — which made
#: "deliberately exempt" and "nobody remembered to list it" the SAME observable, and a
#: disarmed `configs/run6.yaml` dropped into the tree audited GREEN (measured: rc 0).
#:
#: So the exemption is now WRITTEN, and the two tuples must PARTITION the config set EXACTLY.
#: The tool hard-fails (rc 31) on either kind of drift: a config present on disk and
#: named by neither tuple, and a tuple naming a config that is not on disk. That is gate 11's
#: `KNOWN_DEBT` shape (`silent_encoding_gate.py:126,338-344`) applied to the config set —
#: registered debt whose staleness is itself a failure.
#:
#: **What "the config set" means, and why the earlier wording was FALSE** (ADJ-13 F-1). This
#: comment used to say `configs/*.yaml`, and the tool implemented exactly that — a flat
#: `*.yaml` glob — while gate 7 validated `**/*.yaml` + `**/*.yml`. So `configs/run6.yml` and
#: `configs/prod/run6.yaml` passed gate 7 and were never audited by gate 12: the claim below
#: was measured FALSE for two of the three ways to add a config, because MF-7's fix had been
#: fitted to the reviewer's `run6.yaml` rather than to the class. Discovery is now
#: `mantis.config.loader.discover_configs` — ONE authority, consumed by both gates (R71) —
#: and the declaration accepts subdirectory-relative paths, which it previously reported STALE
#: while the file sat on disk.
#:
#: **That was still not enough, and the recheck measured why** (R-2). Widening discovery by one
#: more extension leaves the boundary one extension further out: `configs/run6.txt` and
#: `configs/run6.YAML` were schema-valid, DISARMED on the required row, mintable, launchable —
#: and rc 0 from both gates. The asymmetry was never between two globs; it was that DISCOVERY
#: answered "is this a config" by extension while the LOADER answered it by CONTENT, so the
#: complement of every enumeration stayed launchable and invisible.
#:
#: **R75 rules which side closes it.** Not the loader — narrowing its accept-set was DECLINED,
#: and a run may be launched from a path of any shape. The protection is the **shared-authority
#: invariant**: whatever the loader accepts, the audit must see. `discover_configs` is therefore
#: name-agnostic — every path under `configs/` except a real directory, which `read_text`
#: refuses by type. With that, and only with that:
#:
#: Adding ANY file to `configs/` — at any name, at any depth — now FORCES a one-line declaration
#: here or in `PRODUCTION_CONFIGS`; it can no longer be forgotten into exemption, and there is
#: no longer a class of file the gates are silent about. The cost is deliberate: `configs/` may
#: hold only complete configs, so a stray note or an editor backup is a red gate rather than a
#: quiet resident of the audit root. The one limit still standing, stated rather than implied:
#: this binds `configs/`, and a loadable config OUTSIDE that directory is reachable by
#: `python -m mantis.run` without being discovered (CARD-CONFIG-DISCOVERY-ROOT) — `--config`
#: does audit it, shape-agnostically, which is what covers the mint path.
#:
#: `(repo-relative path, why it is exempt)`. The reason is data, printed by the tool on the
#: failure path, so an exemption cannot be a bare path nobody can justify later.
EXEMPT_CONFIGS: tuple[tuple[str, str], ...] = (
    (
        "configs/dev_example.yaml",
        "developer template, never minted for a run; disarmed at `:200` by design (R59). "
        "ADJ-13 N-3: this file is ALSO the mutation corpus's M1 row (the one real committed "
        "config that demonstrates the audit going red), and exempting it moved gate 12's "
        "red-capability on the real `configs/` tree onto `--config` — which had no producer "
        "in that direction until F-5's. Both halves are now pinned: the declared-production "
        "half by `test_naming_a_config_ADDS_scrutiny_and_never_replaces_the_production_set`'s "
        "bare drive, the `--config` half by "
        "`test_naming_a_DISARMED_config_is_AUDITED_and_never_ignored`.",
    ),
    (
        "configs/smoke_gnn.yaml",
        "smoke config — bounded local drive, not a production run (R59).",
    ),
    (
        "configs/smoke_preflight_armed.yaml",
        "armed preflight-rehearsal smoke config (WPTS Phase F, R103): NOT a production run, "
        "but unlike the R59 smokes it ARMS both required rows at burst-scale guard values so "
        "mode PREFLIGHT can run a completed bounded burst off-run5. Exempt from the "
        "every-production-config gate-12 sweep for the same reason the other smokes are; "
        "`--config` still unions it into the audit set, and its live consumer is the burst "
        "oracle in tests/tools/test_preflight_armed_smoke.py (LAW-08).",
    ),
    (
        "configs/smoke_radius_curriculum.yaml",
        "smoke config — bounded local drive, not a production run (R59).",
    ),
    (
        "configs/sustained_kcluster.yaml",
        "not currently a production run config. WPAX Phase P wrote this row from the tree's "
        "own state (it is absent from PRODUCTION_CONFIGS at HEAD), NOT from an operator "
        "ruling — see CARD-EXEMPT-CONFIGS-OPERATOR-CONFIRM. If it is minted, its row moves "
        "to PRODUCTION_CONFIGS.",
    ),
)


class ArmingSurfaceMissingError(AttributeError):
    """A row's `config_path` does not resolve on a real `RunConfig` (RED-TEAM_P F-4).

    Subclasses `AttributeError` deliberately: `_dotted`'s failure has always been one, so
    every existing caller keeps its behaviour and nothing that catches `AttributeError`
    today starts leaking this one. What changes is that the failure is NAMED and carries
    the three things an operator needs — WHICH ROW is broken, WHAT PATH it declared, and
    WHICH SEGMENT of that path does not exist. Pydantic's `BaseModel.__getattr__` supplies
    only the last, and `preflight_mint.main`'s handler chain then lost even that, collapsing
    the whole class into rc 1 `PreflightInternalError` — the one outcome that tool's own
    docstring says cannot exist. The tool maps this to `PreflightManifestError`, rc 31.

    Written to the CLASS, not to one row (R71): the same route swallows a typo in ANY row's
    `config_path`.
    """


def _dotted(obj: Any, path: str, *, row: str = "<unnamed row>") -> Any:
    """Walk a dotted path into a validated config object.

    Two arms beyond the plain walk, and each is load-bearing:

    * a MISSING attribute raises `ArmingSurfaceMissingError` naming the row, the full path
      and the failing segment. `try/except AttributeError` PER SEGMENT rather than a
      `hasattr` pre-check, because only the former can say which segment failed and because
      the AttributeError comes from pydantic's `BaseModel.__getattr__`, not a plain lookup;
    * a `None` met MID-WALK is an EXPLICITLY DISARMED block and short-circuits to `None`.
      Without it a legitimately disarmed config — `train.draw_rate_abort: null`, the posture
      R59 permits for smoke configs and four of the five committed configs carry — would
      raise `'NoneType' object has no attribute 'threshold'` and fail gate 12 at rc 31
      instead of being reported DISARMED.

    DISCLOSED RESIDUAL: a typo AFTER a legitimately-`None` segment reports "disarmed"
    rather than raising, because the walk short-circuits before reaching it. It is caught
    where it gates — `PRODUCTION_CONFIGS` is run5 and run5 is ARMED, so the walk reaches the
    leaf and the typo raises. Both arms are pinned by
    `tests/tools/test_drawrate_arming_surface_named_failure.py`.
    """
    for part in path.split("."):
        if obj is None:
            return None
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ArmingSurfaceMissingError(
                f"armed-abort row {row!r} declares config_path {path!r}, but segment "
                f"{part!r} is absent on {type(obj).__name__}: a manifest row whose arming "
                "surface does not exist on a real RunConfig is a phantom gate input "
                "(R4 / LAW-07), and it must be a NAMED failure rather than the tool's own "
                "unnamed internal error"
            ) from exc
    return obj


def audit_arming(config: Any, *, manifest: tuple[ArmedAbort, ...] = MANIFEST) -> AuditResult:
    """Assertion (c): every REQUIRED row must be armed in `config`.

    Never branches on a row's `name` and never special-cases draw-rate: `status` selects
    the list, `mechanism` selects the predicate, and both are data. `manifest` is a keyword
    so O-7 can drive an in-memory copy with the deferred row flipped (§8.5).
    """
    required = tuple(row for row in manifest if row.status is Status.REQUIRED)
    deferred = tuple(row for row in manifest if row.status is Status.DEFERRED)
    disarmed = tuple(
        row for row in required
        if not row.mechanism.is_armed(
            _dotted(config, row.config_path, row=row.name),
            # WPMINT Phase K-B: resolved through the SAME walker as the value, so a typo in a
            # `ceiling_path` raises `ArmingSurfaceMissingError` naming the row exactly as a
            # typo in a `config_path` does. Still no branch on a row's identity — `mechanism`
            # selects the predicate and the row supplies both operands.
            ceiling=(None if row.ceiling_path is None
                     else _dotted(config, row.ceiling_path, row=row.name)),
        )
    )
    return AuditResult(required=required, deferred=deferred, disarmed=disarmed)


def exit_code_for_abort(
    rule: str, *, manifest: tuple[ArmedAbort, ...] = MANIFEST
) -> int | None:
    """The process exit code a FIRED abort rule maps to, or `None` if none is authored.

    The other half of CARD-ABORT-EXIT (R84). `StepCoordinator._fire_hard_abort` records the
    rule NAME on `ShutdownState.abort_rule`; this resolves that name to the number a
    supervisor reads off the process, at the boundary where the manifest already lives.

    It NEVER branches on a rule's identity — it looks the row up and returns whatever
    `exit_code` that row carries, exactly as `Mechanism.is_armed` selects a predicate from
    data rather than from a name. So the manifest stays the ONE authority: flipping a row's
    `exit_code` moves this function's answer with no code change here, and no second literal
    exists to disagree with the row.

    `None` has two distinct and equally truthful sources, and neither is an error:

    * a rule with NO manifest row at all — `grad_norm_hard_abort` and `sealbot_wr_abort`
      share `_fire_hard_abort` and neither is pre-registered. R84 refused to invent a code
      for an abort nobody registered, and inventing one HERE would be that same class one
      layer down;
    * a row that is registered but carries `exit_code=None` — the posture the draw-rate row
      itself held until Phase X.

    A caller must therefore treat `None` as "this abort has no authored exit code", never as
    "no abort fired" — `ShutdownState.abort_rule is None` is the only thing that means the
    latter.
    """
    for row in manifest:
        if row.name == rule:
            return row.exit_code
    return None


__all__ = [
    "DISK_SPACE_ABORT_RULE",
    "EXEMPT_CONFIGS",
    "MANIFEST",
    "PRODUCTION_CONFIGS",
    "TERMINAL_EVAL_BROKEN_ABORT_RULE",
    "ArmedAbort",
    "ArmingSurfaceMissingError",
    "AuditResult",
    "Mechanism",
    "Status",
    "audit_arming",
    "exit_code_for_abort",
]
