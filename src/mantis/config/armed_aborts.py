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
# R251 / ADJ-D22 adds the CADENCE axis — `Cadence`, `EARLIEST_FIRE_FRACTION` and
# `audit_cadence` — and it belongs beside the rows for the same reason the rows and their
# walkers do: "which aborts must arm" and "when can an armed one still fire" are one
# question the moment a large `monitor.gate_interval` can leave a threshold armed in the
# config and unread in the run. Splitting them would put the row set and the bound its rows
# are judged against on opposite sides of an import, and the fraction is deliberately here
# rather than in a schema model so no minted config can relax its own audit.
# R265 / ADJ-D38 adds `SampleClock` beside `Cadence`, and it is the same argument once more:
# "when can an armed row fire" is unanswerable without "in WHICH CLOCK does that row's
# evidence arrive", because the two are different keys on different axes — the draw-rate
# gate ticks on `monitor.gate_interval`, the sealbot-WR trajectory on `train.eval_interval`.
# The period belongs to the CLOCK rather than to the row precisely so a row cannot name its
# own: a row that supplies its own period is a row that can be audited in a clock it never
# ticks in, which is the D38 defect stated as a shape.
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

`wr_hard_abort_enabled` is a DEFERRED row since R265 / ADJ-D38, and the sentence this
paragraph used to carry ("ABSENT BY DECISION, not by oversight … a later reader must not
'fix' it in") was HALF right and is corrected here rather than deleted. The half that
stands: the sealbot win-rate abort ships WARN-ONLY by operator ruling G-3, STATE §6 names
the mint-blocking pair as "draw-rate + actor-lag", and NOTHING here may flip it REQUIRED —
that would gate every production mint on `monitor.wr_hard_abort_enabled: true`, a value the
operator deliberately mints false, and moving it is a ruling and not a manifest edit. The
half that was WRONG: absence from the manifest was read as "the axis needs no row", and
ADJ-D38 measured what that bought — gate 12's cadence audit could not compute even a FALSE
affirmative for the WR axis, so the axis sat entirely outside the R251 machinery while its
own consec knobs were free to go unfireable. A DEFERRED row prints loudly on every gate-12
run, gates nothing, and makes the flip a one-field data edit (§8.5); that is the posture for
a live gate whose DISPOSITION is owed, and it is what the row below carries.
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


#: The FRACTION of a run's own `train.max_train_steps` inside which an ARMED abort must be
#: able to fire (R251 / ADJ-D22). ONE authority, consumed by `audit_cadence` below.
#:
#: THE DEFECT IT CLOSES, measured. `monitor.gate_interval: 1000000000` with a 40-step run
#: produces ZERO gate boundaries, so an ARMED `train.draw_rate_abort` is never evaluated —
#: and gate 12 audited that config green, because `Mechanism.is_armed` reads a THRESHOLD and
#: a threshold that is never READ is armed in the config and absent in effect. The interval's
#: `ge=1` bans exactly one spelling of "never gate" and permits every larger one, so the
#: arming predicate alone cannot see the class. A LARGE INTERVAL IS NEVER A SANCTIONED
#: DISARM: the one sanctioned spelling stays the explicit R56-style pin the grad-norm row
#: below carries, which is a written, owned, tamper-evident row and not a number nobody read.
#:
#: WHY IT IS NOT A PER-CONFIG KEY. A config that could set its own audit fraction could relax
#: its own audit, and the disarm this constant exists to refuse would be re-spellable as
#: `earliest_fire_fraction: 1.0` — ADJ-D20's gate-3c self-comparison class relocated one
#: layer down. It lives HERE, beside the rows, because "which aborts must arm" already lives
#: here: the row set and the bound its rows are judged against then cannot drift apart across
#: an import, and `RunConfig`'s `extra="forbid"` makes minting the name a loud refusal rather
#: than a quiet override (driven, not asserted, by
#: `tests/config/test_armed_abort_cadence.py::
#: test_the_fraction_is_a_named_constant_and_a_config_can_never_set_it`).
#:
#: WHY 0.25, and the grounds are that IT BINDS NO MINTED VALUE TODAY. It is a bound, not a
#: target: an armed abort that cannot possibly fire in the first quarter of a run has missed
#: the early-run regime it exists to catch, and every armed row on every committed config
#: clears it with margin (measured on the tree: run5's draw-rate row is earliest-fire 25000
#: of 1000000, a 10x margin; its actor-lag row 101; the armed preflight smoke's draw-rate row
#: 30 of 200). A TIGHTER fraction would begin authoring `train.draw_rate_abort.min_step`
#: policy from a CI gate — the class call K-c/R84 refused when it declined to invent a
#: grad-norm threshold — and a looser one buys nothing, because the class this refuses (an
#: interval that outruns the run by orders of magnitude) sits nowhere near the boundary.
#: Moving this number is a ruling, not a tuning.
EARLIEST_FIRE_FRACTION: float = 0.25

#: The config path the bound is taken FROM. Named once and walked through the SAME `_dotted`
#: every row's own paths go through, so renaming the run-length key is one loud
#: `ArmingSurfaceMissingError` rather than a silent bound of zero.
RUN_LENGTH_PATH: str = "train.max_train_steps"


class SampleClockNotDerivableError(ValueError):
    """A row's SAMPLE CLOCK period could not be derived from the config (R265 / ADJ-D38).

    The audit RAISES here instead of falling back to the training-step clock, and that is the
    whole ruling in one branch: reading an axis in a clock it does not tick in is exactly the
    defect ADJ-D38 measured. Gate 12 computed every row's fire step in TRAINING STEPS while
    the sealbot-WR axis's evidence arrives per EVAL ROUND, so a WR row judged that way would
    be judged against a cadence key it never reads — and it would audit GREEN. A silent
    fallback would make "one tick is one training step" and "nobody could derive this row's
    tick" the same observable, which is MF-7's class relocated onto the cadence axis.

    `preflight_mint.py` maps it onto `PreflightManifestError`, rc 31: a manifest defect an
    operator can fix in one line, never the tool's own unnamed internal error (F-4's class).
    """


class SampleClock(StrEnum):
    """WHICH CLOCK an axis's samples arrive in, and how long ONE tick of it is in TRAINING
    STEPS. DATA, and the period is a live config PATH — never a number written here.

    R265 / ADJ-D38. `Cadence` used to answer "when can this row first fire" in the
    training-step clock for every row, and got away with it because every row it held either
    ticked in that clock or declared its own period as an operand. The sealbot-WR axis breaks
    both halves: its samples arrive once per COMPLETED EVAL ROUND
    (`train/coordinator/step.py::on_eval_round_complete` appends exactly one `(step, wr)` per
    routed result), so the step clock is the wrong denominator — and nothing would have
    stopped a WR row DECLARING `monitor.gate_interval` as its interval operand, which is how
    an axis ends up audited in another axis's clock with every check green.

    So the period is a property of the CLOCK, not of the row. Three consequences, all of them
    the point: every row on an axis reads the SAME live key; a row cannot supply its own
    period; and a second row added to an existing axis inherits the right cadence by
    construction rather than by its author remembering.

    An axis whose period cannot be derived RAISES (`SampleClockNotDerivableError`). There is
    no default tick anywhere in this class.
    """

    #: One sample per TRAINING STEP. The period is 1 by the DEFINITION of the clock, not by a
    #: number this file chose — which is why this member names no config path and why a
    #: `period_path` of `None` here does not mean "underivable". The grad-norm gate (evaluated
    #: inside the burst, per step) and the actor-lag invariant tick here.
    TRAIN_STEP = "train_step"

    #: One sample per hard-abort GATE BOUNDARY. `train/coordinator/step.py::
    #: _run_gate_interval` runs the live hard-abort gates only when
    #: `self._train_step % cfg.gate_interval == 0`, and `monitor.gate_interval` is the minted
    #: key `mantis.run.compose_run` threads into `StepCoordinatorConfig.gate_interval`.
    GATE_BOUNDARY = "gate_boundary"

    #: One sample per completed EVAL ROUND. `train/coordinator/step.py::_maybe_kick_eval`
    #: kicks a round only at `self._train_step % cfg.eval_interval == 0` and never at round 0,
    #: so round `r` lands at training step `r * eval_interval` with `r >= 1`; every routed
    #: result then appends exactly one WR sample. `train.eval_interval` is the minted key
    #: `resolve_coordinator_knobs` reads into `StepCoordinatorConfig.eval_interval`.
    #:
    #: DISCLOSED, because this clock has a second switch the audit cannot see from one path:
    #: `eval_enabled` false builds NO eval pipeline, so the axis ticks zero times however
    #: small the interval is. A row carries ONE `config_path` and the WR row spends it on the
    #: disposition flag; the eval-enabled half is held by
    #: `tests/config/test_minted_config_remint.py::
    #: test_a_minted_config_carries_the_identity_and_eval_leaves`, which asserts it True over
    #: all six committed configs — the same disposition the `terminal_eval_broken` row takes
    #: for the same reason.
    EVAL_ROUND = "eval_round"

    #: NOT step-clocked at all — a wall-clock poll (the disk guard) or a close-out rule (the
    #: terminal eval). Asking such a row for a period is a category error, and `period_steps`
    #: RAISES rather than answering 1: answering 1 is precisely the step-clock fallback this
    #: class exists to make impossible. Such a row is judged by the STEP FLOOR its rule
    #: imposes instead (`Cadence.step_floor`), which is a derived answer and not an exemption.
    NO_STEP_CLOCK = "no_step_clock"

    @property
    def period_path(self) -> str | None:
        """The live config key ONE tick of this clock is measured by.

        `None` on the two members that have no config period, and the two mean different
        things — `TRAIN_STEP`'s tick is definitional, `NO_STEP_CLOCK` has no step tick at all
        — which is why `period_steps` branches on the MEMBER and not on this being `None`.
        """
        return {
            SampleClock.TRAIN_STEP: None,
            SampleClock.GATE_BOUNDARY: "monitor.gate_interval",
            SampleClock.EVAL_ROUND: "train.eval_interval",
            SampleClock.NO_STEP_CLOCK: None,
        }[self]

    @property
    def is_step_clocked(self) -> bool:
        """True iff a training step is a meaningful denominator for this axis at all."""
        return self is not SampleClock.NO_STEP_CLOCK

    def period_steps(self, config: Any, *, row: str) -> float:
        """TRAINING STEPS per tick of this clock, DERIVED from `config`. Never a fallback.

        A value that is not a real, finite, non-`bool` number is UNDERIVABLE and raises:
        `_dotted` short-circuits an explicitly-disarmed block to `None`, and a `None` period
        read as "1 step" would audit an axis that never ticks as one that ticks every step.
        A real number BELOW 1 is a different thing — it is derivable and degenerate — and it
        is passed through so `Cadence.earliest_fire_samples` can answer `math.inf` for it,
        the fail-toward-visibility currency that axis already speaks (a schema `ge=1` on both
        period keys makes it unreachable from a validated config either way).
        """
        if self is SampleClock.TRAIN_STEP:
            return 1.0
        path = self.period_path
        if path is None:
            raise SampleClockNotDerivableError(
                f"armed-abort row {row!r} ticks in {self.value}, which is not a training-step "
                "clock at all: its rule has no per-step sampling cadence, so asking for one "
                "is a category error. Judge it by `Cadence.step_floor` instead — forging a "
                "period here is the step-clock fallback R265 forbids"
            )
        value = _dotted(config, path, row=row)
        if not _is_real_number(value):
            raise SampleClockNotDerivableError(
                f"armed-abort row {row!r} ticks in {self.value}, whose period is minted at "
                f"{path!r} — and that resolves to {value!r}, which is not a number of "
                "training steps. The audit REFUSES to fall back to the training-step clock: "
                "an axis judged in a clock it does not tick in audits GREEN on exactly the "
                "configs it exists to refuse (R265 / ADJ-D38)"
            )
        return float(value)


def _evals_to_first_fire(consec: float, min_step: float, period: float) -> float:
    """EVAL ROUNDS before ONE sealbot-WR trigger can first fire (R265 / ADJ-D38).

    Derived from `monitor/rules.py::sealbot_wr_trajectory_alert`, which every trigger routes
    through, and from nothing else:

    * `len(history) >= n_consec` needs `consec` samples — but `if not wr_history: return
      None` needs at least ONE regardless, so a `consec` of 0 (the schema's `ge=0` admits it)
      does NOT let a trigger fire before the first round. `max(consec, 1)`, and that floor is
      the arithmetic half of the `history[-0:]` hair-trigger ADJ-D38 records as a separate,
      unruled operator question — 0 arms a weaker-evidence variant, it does not disable a
      rule;
    * `current_step > min_step` is STRICT and round `r` lands at `r * period`, so the first
      round past the floor is `floor(min_step / period) + 1`.

    OPTIMISTIC by construction, exactly as `GATE_INTERVAL_CONSEC`'s BUG-1 note is: it assumes
    every round yields a WR sample (a round whose result carries no `wr_sealbot` is
    skip-counted and appends nothing) and, for trigger B, that a positive peak exists. An
    EARLIEST-POSSIBLE-fire bound wants the optimistic case — this is a reachability floor,
    not a prediction of when a run would actually abort.
    """
    if period < 1.0:
        return math.inf
    return max(max(consec, 1.0), float(math.floor(min_step / period)) + 1.0)


class Cadence(StrEnum):
    """WHEN a row's abort can FIRST fire, in TRAINING STEPS. DATA, not a branch on `name`.

    The twin of `Mechanism`, and for the same reason: `audit_cadence` never asks which row it
    is holding — `cadence` selects the arithmetic and the row supplies the operands through
    `cadence_paths`. `mechanism` answers "is this abort armed"; `cadence` answers "can the
    armed thing still fire inside the run", and ADJ-D22 measured that the first answer alone
    is not enough.

    Every member is DERIVED FROM THE CODE THAT EVALUATES THE ROW, cited at the member — never
    from a key name someone typed here. No operand is baked in: the row names its own paths,
    so the arithmetic runs over values the operator minted and this file invents no number.
    That is the rule `CONFIG_THRESHOLD_BELOW_CEILING`'s `ceiling_path` already follows.

    `earliest_fire_step` answers in three currencies and the difference between the last two
    is load-bearing: a finite float is a step, `math.inf` is "these operands can never fire"
    (the ADJ-D22 outcome), and `None` is "NO STEP CADENCE GOVERNS THIS ROW AT ALL" — which is
    a truthful answer for a wall-clock or close-out rule and must never be forged into a
    number, the class R84 refused when it declined to fabricate an exit code.

    R265 / ADJ-D38 SPLITS THAT ANSWER IN TWO, and the split is the ruling. `earliest_fire_
    samples` answers in the row's OWN sample clock — gate boundaries for the draw-rate gate,
    EVAL ROUNDS for the sealbot-WR trajectory — and `earliest_fire_step` is that count times
    the clock's period, which `SampleClock` derives from a live key. No member holds its own
    period any more: the interval `GATE_INTERVAL_CONSEC` used to take as operand 0 is now
    read off `SampleClock.GATE_BOUNDARY`, so a row cannot name a cadence key its axis does
    not tick on, and the audit compares the count against the bound converted into that same
    clock. For a period of exactly what the row used to declare the published STEP is
    unchanged to the bit — the D38 change is which rows CAN be judged, not what the judged
    ones answer.
    """

    #: `train/coordinator/step.py::_run_gate_interval` runs the gate only when
    #: `self._train_step % cfg.gate_interval == 0` — so this member's SAMPLE CLOCK is
    #: `SampleClock.GATE_BOUNDARY` and the interval is read there, NOT declared here (R265).
    #: `monitor/rules.py::check_draw_rate_collapse` then refuses on `len(history) < consec`
    #: and on `current_step < min_step`, so the earliest fire is the first gate BOUNDARY that
    #: is both the `consec`-th observation and at or past `min_step`. Operands, in order:
    #: (consec path, min-step path).
    #:
    #: BUG-1: a boundary that yields no observation neither advances NOR resets `consec`
    #: (`_sample`), so `consec * interval` bounds a real fire from BELOW. That is exactly what
    #: an EARLIEST-POSSIBLE-fire bound wants — the optimistic case — and it is why this is a
    #: reachability floor and not a prediction of when a run would actually abort.
    #:
    #: FORMER DISCLOSED RESIDUAL, CLOSED by ADJ-D36 — and closed by DERIVATION, not by an
    #: import or a copy. This member used to publish a FALSE AFFIRMATIVE for any `consec`
    #: above the coordinator's ring depth: `step.py::_sample` trimmed the gate history to a
    #: literal `_GATE_HISTORY_DEPTH = 32` while `rules.py::check_draw_rate_collapse` refuses
    #: on `len(history) < consec`, so `consec >= 33` was PERMANENTLY unfireable while this
    #: arithmetic computed a finite step and published it as though the run could deliver
    #: it. The literal is DELETED: `_run_hard_abort_gates`'s draw-rate arm now trims its
    #: ring to `spec.consec` — the SAME minted value this member reads as its `consec` operand
    #: (`train.draw_rate_abort.consec`) — so a finite answer here is deliverable by
    #: construction for every schema-legal `consec`. ONE authority on both sides of the
    #: audit, and NO new DAG edge in either direction: `mantis.config` still imports
    #: nothing from `mantis.train`, because there is no longer a constant to import, and
    #: nothing was re-typed here for a first divergence to falsify. The tie between this
    #: member's published number and the machine's actual fire is DRIVEN, not asserted, by
    #: `tests/train/test_drawrate_gate_capacity.py::
    #: test_the_published_earliest_fire_step_is_deliverable_above_the_old_depth`, which
    #: fires a REAL coordinator at the exact observation count this arithmetic publishes
    #: for a `consec` the old code could never satisfy.
    GATE_INTERVAL_CONSEC = "gate_interval_consec"

    #: R265 / ADJ-D38 — the sealbot-WR trajectory abort, in the EVAL-ROUND clock.
    #:
    #: `train/coordinator/step.py::on_eval_round_complete` appends ONE `(step, wr)` sample per
    #: routed eval-round result, and `monitor/rules.py::sealbot_wr_trajectory_alert` fires on
    #: whichever of its three triggers is first satisfiable — C early-death, B collapse-from-
    #: peak, A rolling — each of which needs `len(history) >= its consec` AND
    #: `current_step > its min_step`. The earliest possible fire is therefore the MINIMUM over
    #: the three, in ROUNDS, and `SampleClock.EVAL_ROUND` converts a round to training steps
    #: through `train.eval_interval`. Operands, in the order `_evals_to_first_fire` pairs
    #: them: (collapse-consec path, early-death-min-step path, collapse-min-step path,
    #: rolling-consec path, rolling-min-step path) — B and C SHARE
    #: `monitor.wr_collapse_consecutive_evals`, which is why five paths cover three triggers.
    #:
    #: WHY THIS MEMBER EXISTS AT ALL, which is the ruling: judged in the TRAINING-STEP clock
    #: this axis reads healthy on a config that can never deliver it. An `eval_interval` three
    #: orders of magnitude past the run produces zero rounds and therefore zero WR samples,
    #: while `monitor.gate_interval` — the key a step-clock audit would reach for — says
    #: nothing whatever about it. ADJ-D22's defect, on the axis LAW-15/F-30 names as the one
    #: that actually kills runs.
    EVAL_ROUND_CONSEC = "eval_round_consec"

    #: `train/coordinator/step.py` D3: the grad-norm gate is evaluated PER TRAINING STEP
    #: inside the burst (its producer is the trainer's own loss dict) and fires when
    #: `self._consec_high_gn >= cfg.hard_gn_min_steps`. The counter advances once per step, so
    #: the earliest fire is the `hard_gn_min_steps`-th step. Operands: (min-steps path,).
    CONSEC_TRAIN_STEPS = "consec_train_steps"

    #: `train/lifecycle/heartbeat_watchdog.py::ActorLagSpec` — `learner_step_fn() -
    #: actor_ckpt_step_fn() > threshold_steps`. The QUANTITIES are step-clock; the SAMPLING
    #: rides the watchdog's seconds poll, which adds no STEP floor (the spec's own docstring
    #: says so). With a frozen actor the strict `>` is first satisfiable one step past the
    #: threshold. Operands: (threshold-steps path,).
    STEP_LAG_THRESHOLD = "step_lag_threshold"

    #: `train/lifecycle/disk_guard.py` — a daemon thread on `while not
    #: self._stop_event.wait(timeout=self._interval)`. NO train-step boundary gates it, so the
    #: earliest TRAIN STEP at which it can fire is 0. That is a derived answer and not an
    #: exemption: the row stays inside the comparison, and a guard that ever acquired a step
    #: gate would need its own member. Operands: none.
    #:
    #: DISCLOSED RESIDUAL, and it is the same class one axis over: a wall-clock rule can be
    #: cadence-disarmed in SECONDS (`monitor.disk_guard.interval_sec` set past the run's wall
    #: time) and this fraction cannot see it, because a config carries no wall-clock run
    #: length to take a fraction OF. Stated, not papered over.
    WALL_CLOCK_POLL = "wall_clock_poll"

    #: `train/coordinator/drain.py::close_out` -> `run_terminal_eval` — the LAST action of the
    #: run, reached on every termination including an aborted one. It has no in-run step
    #: cadence, so it answers `None` and the fraction rule does not apply to it: asking when a
    #: close-out rule can fire "early" is a category error, and answering `max_train_steps`
    #: would fail every such row forever for a reason that is not a defect.
    CLOSE_OUT_TERMINAL = "close_out_terminal"

    #: F-816-10 (R276(f)). The rule is evaluated when the object it guards is CONSTRUCTED —
    #: `mantis.config.resolve.fused_graph_caps.resolve_fused_graph_caps`, called eagerly from
    #: the graph branch of `InferenceServer.__init__`, which refuses an uncalibrated cap before
    #: a single training step or self-play game exists. It has no in-run step cadence to speak
    #: of, so it ticks in `SampleClock.NO_STEP_CLOCK` and consumes no operands.
    #:
    #: WHY IT IS A NEW MEMBER AND NOT `CLOSE_OUT_TERMINAL` OR `WALL_CLOCK_POLL`, both of which
    #: also carry no step clock: their `step_floor` answers say different things, and the
    #: difference is the whole content of this axis. `CLOSE_OUT_TERMINAL` answers `None` —
    #: "asking when this fires early is a category error" — which is FALSE here; this rule
    #: fires at the earliest moment there is. `WALL_CLOCK_POLL` answers `0.0` for a daemon
    #: thread that could fire at any wall-clock instant, which is the RIGHT number for the
    #: wrong reason. This member answers `0.0` because the rule is evaluated BEFORE step 0 and
    #: cannot be reached later at all: an uncalibrated config never gets a run to fire during.
    #: Reusing either sibling would put a true number under a false mechanism, which is the
    #: `Mechanism` lesson (a predicate that reads right for the wrong reason) on the sibling
    #: axis.
    CONSTRUCTION_TIME = "construction_time"

    @property
    def sample_clock(self) -> SampleClock:
        """WHICH clock this member's evidence arrives in (R265 / ADJ-D38). DATA, like
        `arity` — `audit_cadence` reads it off the member and never asks which row it holds.

        This is what makes "no axis is auditable in a clock it doesn't tick in" a structural
        property rather than an authoring convention: the member that knows the arithmetic
        also names the clock, and the clock (not the row) owns the period key.
        """
        return {
            Cadence.GATE_INTERVAL_CONSEC: SampleClock.GATE_BOUNDARY,
            Cadence.EVAL_ROUND_CONSEC: SampleClock.EVAL_ROUND,
            Cadence.CONSEC_TRAIN_STEPS: SampleClock.TRAIN_STEP,
            Cadence.STEP_LAG_THRESHOLD: SampleClock.TRAIN_STEP,
            Cadence.WALL_CLOCK_POLL: SampleClock.NO_STEP_CLOCK,
            Cadence.CLOSE_OUT_TERMINAL: SampleClock.NO_STEP_CLOCK,
            Cadence.CONSTRUCTION_TIME: SampleClock.NO_STEP_CLOCK,
        }[self]

    @property
    def arity(self) -> int:
        """How many `cadence_paths` this member CONSUMES. Enforced by `ArmedAbort` in both
        directions, so a path the arithmetic never reads cannot sit on a row pretending to be
        an input (LAW-07's phantom-input class, the rule `ceiling_path` already gets).

        `GATE_INTERVAL_CONSEC` counts 2 and not 3 since R265: its interval moved off the row
        and onto `SampleClock.GATE_BOUNDARY`, so a row can no longer declare the key its own
        axis is sampled by — and the arity rule now REFUSES the row that tries.
        """
        return {
            Cadence.GATE_INTERVAL_CONSEC: 2,
            Cadence.EVAL_ROUND_CONSEC: 5,
            Cadence.CONSEC_TRAIN_STEPS: 1,
            Cadence.STEP_LAG_THRESHOLD: 1,
            Cadence.WALL_CLOCK_POLL: 0,
            Cadence.CLOSE_OUT_TERMINAL: 0,
            Cadence.CONSTRUCTION_TIME: 0,
        }[self]

    def step_floor(self) -> float | None:
        """The earliest TRAINING STEP a NOT-STEP-CLOCKED member's rule imposes (R265).

        Defined only for the two `NO_STEP_CLOCK` members, and it RAISES on the others: a
        step-clocked row that took an answer from here would be skipping its own clock, which
        is the fallback this split exists to make unreachable. The two answers are the ones
        R251 already derived — `0.0` for a wall-clock poll (a daemon thread on a `wait`
        timeout; no train-step boundary gates it, so the floor is genuinely zero) and `None`
        for a close-out rule (asking when a close-out rule fires "early" is a category error,
        and `max_train_steps` would fail every such row forever for a non-defect).

        F-816-10 adds a THIRD, and it takes the `0.0` answer on its own grounds rather than by
        falling into the else-branch: a `CONSTRUCTION_TIME` rule is evaluated before training
        step 0 exists, so `0.0` is not "could fire any time" (the wall-clock reading) but "has
        already fired or will never get the chance". Same number, different statement, and the
        member docstring is where the difference is recorded so a reader is not left to infer
        it from a shared branch.
        """
        if self.sample_clock.is_step_clocked:
            raise SampleClockNotDerivableError(
                f"cadence {self.value} ticks in {self.sample_clock.value}, a real sample "
                "clock: its earliest fire is a COUNT OF TICKS times a derived period, never "
                "a bare step floor. Asking for a floor here would answer a step-clock "
                "question about an axis that has its own clock (R265 / ADJ-D38)"
            )
        return None if self is Cadence.CLOSE_OUT_TERMINAL else 0.0

    def earliest_fire_samples(
        self, values: tuple[Any, ...], *, period_steps: float
    ) -> float | None:
        """How many TICKS OF THIS ROW'S OWN SAMPLE CLOCK before it can first fire (R265).

        The unit is the axis's own: gate BOUNDARIES for the draw-rate gate, EVAL ROUNDS for
        the sealbot-WR trajectory, TRAINING STEPS for the two per-step rules. `period_steps`
        is how many training steps one of those ticks takes, and it arrives from
        `SampleClock.period_steps` — never from an operand, so no row can denominate itself.

        A real function of `values` in every member — a constant here would pass or fail every
        row at once, the warning `is_armed` carries on the sibling axis.

        An operand that is not a real, finite, non-`bool` number answers `math.inf`: an
        unjudgeable rule must fail toward VISIBILITY, never toward silence, and reading a gate
        that never runs as a gate that fires at tick 0 is the exact inversion. A degenerate
        period (below one training step) answers `math.inf` for the same reason, which is the
        answer R251 gave a sub-1 `gate_interval` before the split and gives it still.
        """
        if not self.sample_clock.is_step_clocked:
            raise SampleClockNotDerivableError(
                f"cadence {self.value} ticks in {self.sample_clock.value}, so it has no "
                "sample count at all: its rule is not sampled on any train-step clock. Ask "
                "`step_floor` instead — counting ticks of a clock that does not exist is the "
                "fabricated-number class R84 refused (R265 / ADJ-D38)"
            )
        if not _is_real_number(period_steps):
            raise SampleClockNotDerivableError(
                f"cadence {self.value} was handed period {period_steps!r}, which is not a "
                "number of training steps per tick. The period is DERIVED by "
                "`SampleClock.period_steps` from a live key and a missing one is a loud "
                "refusal, never a silent 1 (R265 / ADJ-D38)"
            )
        period = float(period_steps)
        if not all(_is_real_number(value) for value in values):
            return math.inf
        if self is Cadence.STEP_LAG_THRESHOLD:
            return float(values[0]) + 1.0
        if self is Cadence.CONSEC_TRAIN_STEPS:
            return float(values[0])
        if self is Cadence.EVAL_ROUND_CONSEC:
            collapse_consec, early_min, collapse_min, rolling_consec, rolling_min = (
                float(value) for value in values
            )
            return min(
                _evals_to_first_fire(collapse_consec, early_min, period),    # trigger C
                _evals_to_first_fire(collapse_consec, collapse_min, period),  # trigger B
                _evals_to_first_fire(rolling_consec, rolling_min, period),   # trigger A
            )
        consec, min_step = (float(value) for value in values)
        if period < 1.0 or consec < 1.0:
            return math.inf
        return max(consec, float(math.ceil(min_step / period)))

    def earliest_fire_step(
        self, values: tuple[Any, ...], *, period_steps: float | None
    ) -> float | None:
        """The earliest TRAIN STEP at which this cadence can fire — the SAME three currencies
        R251 published (a finite step, `math.inf` for "these operands can never fire", `None`
        for "no step cadence governs this row at all"), now COMPOSED from the split above.

        For a step-clocked member it is `earliest_fire_samples * period_steps`, and
        `period_steps` is REQUIRED: passing `None` there raises rather than assuming one
        training step per tick, because that assumption IS the D38 defect. For a
        `NO_STEP_CLOCK` member it is `step_floor()`, and `period_steps` must be `None` for the
        mirror-image reason — a period on an axis with no step clock is an operand nobody can
        have derived.
        """
        if not self.sample_clock.is_step_clocked:
            if period_steps is not None:
                raise SampleClockNotDerivableError(
                    f"cadence {self.value} ticks in {self.sample_clock.value} and was handed "
                    f"a period of {period_steps!r}: no config key measures a tick of a clock "
                    "this rule does not run on, so that number came from somewhere it could "
                    "not have been derived (R265 / ADJ-D38)"
                )
            return self.step_floor()
        if period_steps is None:
            raise SampleClockNotDerivableError(
                f"cadence {self.value} ticks in {self.sample_clock.value} and was handed no "
                "period: answering in training steps anyway would be the one-tick-is-one-step "
                "FALLBACK R265 forbids — the audit fails loud instead (ADJ-D38)"
            )
        samples = self.earliest_fire_samples(values, period_steps=period_steps)
        if samples is None or not math.isfinite(samples):
            # `math.inf * 0.0` is `nan`, and a nan step would compare False against every
            # bound and read as WITHIN — the unfireable row auditing green, one multiply late.
            return samples
        return samples * float(period_steps)


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
    #: R251 / ADJ-D22: WHEN this row's abort can first fire, and the paths its arithmetic
    #: reads. Both carry defaults for exactly the reason `ceiling_path` does — a no-default
    #: field would force every synthetic-row construction site in the suite to type
    #: `cadence=None`, buying nothing — and the defaults are safe because they are not silent:
    #: `audit_cadence` reports a REQUIRED row with NO cadence as OUT OF BOUND by name, so an
    #: undeclared cadence gates rather than passes. `__post_init__` enforces the arity pairing
    #: in both directions, so a path the arithmetic never reads cannot sit on a row.
    cadence: Cadence | None = None
    cadence_paths: tuple[str, ...] = ()

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
        wanted = 0 if self.cadence is None else self.cadence.arity
        if len(self.cadence_paths) != wanted:
            declared = "no cadence" if self.cadence is None else self.cadence.value
            raise ValueError(
                f"armed-abort row {self.name!r} declares {len(self.cadence_paths)} "
                f"cadence_paths ({list(self.cadence_paths)}) but {declared} consumes "
                f"{wanted}: an operand the arithmetic never reads is a phantom input, and a "
                "missing one would be computed from a value nobody minted (R251 / LAW-07)"
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
        cadence=Cadence.STEP_LAG_THRESHOLD,
        cadence_paths=("monitor.actor_lag_threshold_steps",),
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
        cadence=Cadence.GATE_INTERVAL_CONSEC,
        # R265 / ADJ-D38: `monitor.gate_interval` is NO LONGER an operand here. It is the
        # PERIOD of this row's sample clock, read off `SampleClock.GATE_BOUNDARY` — the same
        # key, the same value, one authority for every row on the axis, and no row able to
        # name a cadence key its own axis is not sampled by.
        cadence_paths=("train.draw_rate_abort.consec", "train.draw_rate_abort.min_step"),
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
            "correcting the WITHDRAWN DR-8; RE-POINTED by R242/ADJ-D12; the CADENCE CLAIM "
            "CORRECTED after R242 shipped it FALSE): consec=3 counts consecutive "
            "OBSERVATIONS, and an observation is ATTEMPTED once per monitor.gate_interval "
            "train steps — the knob was train.log_interval until R242 split arming from "
            "narration, and the arithmetic below did not move with that split because every "
            "committed config mints gate_interval EQUAL to its own log_interval in the same "
            "bundle. A boundary that yields NO observation (absent producer, or fewer than "
            "N_pool_min completed games) neither advances NOR RESETS the counter, so every "
            "span below is a LOWER BOUND and not an equality: at run5's gate_interval 1000 "
            "three consecutive observations span AT LEAST 2000 steps, and the history holds "
            "AT MOST 25 samples by step 25000 — exactly 25 only if every boundary from step "
            "1000 on cleared the evidence bar, which is precisely what the early-run regime "
            "R242 exists to instrument does not guarantee. The earliest possible fire is "
            "nonetheless step 25000, because SAMPLING IS NOT GATED BY min_step — the history "
            "accumulates from step 1000 and min_step gates only the FIRE. DR-8's "
            "contrary claim (earliest fire 27000) was MEASURED FALSE and withdrawn. The "
            "re-scaled gate_interval an operator actually wants, and the consec re-derived in "
            "those units, are MINT PREREG ROWS: R242 authored the mechanism and deliberately "
            "moved no armed value. The pin binds to the THREADING at the "
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
        cadence=Cadence.WALL_CLOCK_POLL,
        cadence_paths=(),
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
        cadence=Cadence.CLOSE_OUT_TERMINAL,
        cadence_paths=(),
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
        # Declared even though a DEFERRED row is not audited, so the flip to REQUIRED stays
        # the ONE-FIELD data edit §8.5 claims it is. It has a live consumer meanwhile:
        # `preflight_mint.py::_print_deferred_rows` prints it on every gate run.
        cadence=Cadence.CONSEC_TRAIN_STEPS,
        cadence_paths=("train.hard_gn_min_steps",),
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
    ArmedAbort(
        name="sealbot_wr_abort",
        config_path="monitor.wr_hard_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL,
        # Declared for the reason the grad-norm row above declares one, and for a second
        # reason that is this row's whole point: the WR axis had NO row at all, so gate 12's
        # cadence audit could not compute even a FALSE affirmative for it (ADJ-D38). Its live
        # consumer meanwhile is `preflight_mint.py::_print_deferred_rows`, which prints the
        # cadence AND the clock it ticks in on every gate run.
        cadence=Cadence.EVAL_ROUND_CONSEC,
        cadence_paths=("monitor.wr_collapse_consecutive_evals",
                       "monitor.wr_early_death_min_step",
                       "monitor.wr_collapse_min_step",
                       "monitor.wr_rolling_consecutive_evals",
                       "monitor.wr_rolling_min_step"),
        status=Status.DEFERRED,
        exit_code=None,
        owner="operator ruling G-3 — the warn-vs-abort DISPOSITION, at run5 mint prereg",
        source_pin=(
            "src/mantis/train/coordinator/step.py",
            'self._fire_hard_abort("sealbot_wr_abort", hard, step=step)',
        ),
        note=(
            "The sealbot win-rate trajectory abort (monitor/rules.py's triggers A/B/C, fired "
            "from coordinator/step.py::on_eval_round_complete). R265 / ADJ-D38 authors this "
            "row; before it the axis was OUTSIDE the manifest entirely. "
            "WHY DEFERRED AND NOT REQUIRED, and this row's answer is the STRONGEST of the "
            "three deferred cases on record: flipping it REQUIRED would gate every "
            "production mint on monitor.wr_hard_abort_enabled being true, and every "
            "committed config mints it FALSE by operator ruling G-3 (warn-only; STATE §6 "
            "names the mint-blocking pair as draw-rate + actor-lag). So REQUIRED here does "
            "not demand a value nobody pre-registered — it OVERRULES one the operator "
            "pre-registered, from a CI gate, which is worse than the class R84 refused. The "
            "disposition is a ruling; this row is the instrument, not the ruling. "
            "WHAT THE ROW BUYS WHILE DEFERRED, since a deferred row gates nothing: the axis "
            "is now VISIBLE. _print_deferred_rows names its arming surface, its cadence "
            "member, its five operands and the EVAL-ROUND clock they are denominated in, on "
            "every gate-12 run; and the flip to REQUIRED stays the one-field data edit §8.5 "
            "claims, so the day G-3 is revisited the audit is already wired. Before this row "
            "the honest description was that gate 12 had no opinion about the WR axis at "
            "all — not a wrong one, none. "
            "WHY THE CADENCE IS EVAL-ROUND AND NOT GATE-INTERVAL, which is R265 itself: WR "
            "evidence arrives once per COMPLETED EVAL ROUND, so this row's earliest possible "
            "fire is a count of ROUNDS times train.eval_interval. Judged in the training-step "
            "clock — the clock every row was judged in before D38 — an eval_interval that "
            "outruns the run reads perfectly healthy, because monitor.gate_interval says "
            "nothing whatever about when this rule is evaluated. That is ADJ-D22's defect on "
            "the axis LAW-15/F-30 names as the one that actually kills runs. "
            "THE RING BEHIND IT (ADJ-D38's mechanism half, landed with this row): "
            "step.py::on_eval_round_complete used to trim the WR ring to a literal depth of "
            "5 while all three triggers refuse on len(history) >= their consec, so every "
            "schema-legal wr_collapse_consecutive_evals or wr_rolling_consecutive_evals >= 6 "
            "was armed-in-the-config and permanently unfireable. The literal is DELETED: the "
            "capacity now derives from the minted consec keys and from rule B's own peak "
            "window (monitor/rules.py::WR_PEAK_WINDOW_EVALS), which is the ONE thing the "
            "depth was ALSO a semantic constant of and is therefore preserved exactly rather "
            "than widened. Driven by tests/train/test_wr_gate_capacity.py; bit-identical for "
            "every consec <= the old depth, which every committed config mints (2 and 3). "
            "exit_code is None, truthfully and for the grad-norm row's reason: "
            "_fire_hard_abort stops the run COOPERATIVELY and R84 authored a code for the "
            "draw-rate family only. Inventing one for a warn-only rule would be that refused "
            "class twice over. exit_code_for_abort therefore still answers None for "
            "'sealbot_wr_abort' — now from the SECOND of its two truthful sources (a "
            "registered row carrying None) rather than the first (no row at all). "
            "RESIDUAL, disclosed: the EVAL_ROUND clock has a second switch a single "
            "config_path cannot cover — eval_enabled false builds no eval pipeline, so the "
            "axis ticks zero times whatever the interval is. Held by "
            "test_minted_config_remint.py's per-config eval-leaf assertion, the same "
            "disposition the terminal_eval_broken row takes for the same reason. "
            "SECOND RESIDUAL, and it is UNRULED rather than closed: both consec knobs carry "
            "ge=0, and consec 0 does NOT disable a trigger — history[-0:] is the WHOLE ring "
            "in Python, so 0 arms a weaker-evidence variant that fires on however many evals "
            "the ring holds (minimum 1, via the empty-history guard). ADJ-D38 raises 'a rule "
            "that needs zero observations is not a rule' as an OPERATOR question and R265 "
            "does not rule it, so no bound was moved here; _evals_to_first_fire's max(consec, "
            "1) floor is the arithmetic stating the same fact. "
            "The pin binds the fire site, so deleting the gate, renaming the rule or "
            "reordering the disposition past it all break the R56 scan."
        ),
    ),
    ArmedAbort(
        name="fused_graph_caps_calibrated",
        config_path="inference.fused_graph_caps.max_fused_edges",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO,
        # Declared even though a DEFERRED row is not audited, so the flip to REQUIRED stays
        # the ONE-FIELD data edit §8.5 claims it is — and it has a live consumer meanwhile:
        # `preflight_mint.py::_print_deferred_rows` prints it on every gate run. The rule runs
        # at CONSTRUCTION (the resolver is called eagerly from the graph branch of
        # `InferenceServer.__init__`), so it consumes no operands: there is no threshold to
        # accumulate and no window to fill, only a value that is present or is `null`.
        cadence=Cadence.CONSTRUCTION_TIME,
        cadence_paths=(),
        status=Status.REQUIRED,
        exit_code=None,
        owner=None,
        source_pin=(
            "src/mantis/config/resolve/fused_graph_caps.py",
            "raise UncalibratedFusedGraphCapsError(",
        ),
        note=(
            "FLIPPED DEFERRED -> REQUIRED at the F-816-10/-12 box sitting, 2026-08-18, in the "
            "SAME COMMIT as the minted pair (R282(b)'s pre-registered acceptance; R283). The "
            "row was deferred for exactly one reason -- the value was a measurement nobody had "
            "taken -- and that reason is discharged: both production configs now carry "
            "`{max_fused_edges: 1708894, max_fused_nodes: 77781}`, fitted by "
            "`python -m mantis.diagnostics.fusion_calibrate` against a budget whose four terms "
            "were each measured at 24ae93e and tagged with sha + regime (R281(d)(ii)). `owner` "
            "is DROPPED because the dataclass forbids it on a REQUIRED row -- a required row "
            "carrying an owner reads as already-excused; the `source_pin` STAYS, which a "
            "REQUIRED row may do and which is this row's tamper-evidence. Auditing the flip's "
            "cost: it is TWO field edits (status, and the removal of `owner`), not the ONE the "
            "note below predicted. "
            "EVERY PRODUCTION CONFIG'S `inference.fused_graph_caps` IS VALUED (NOT NULL). The "
            "graph inference forward's memory bound (F-816-10, R276(f)) is the training cap's "
            "partner over one card: `inference_batch_size` bounds the number of GRAPHS in a "
            "fused pop and bounds neither quantity that drives memory, so before this block "
            "the fuse had no bound at all — and `train.microbatch_caps` was fitted against a "
            "self-play term measured when the inference forward carried ONE graph. "
            "WHY IT WAS DEFERRED UNTIL 2026-08-18 -- kept verbatim, because this reasoning is what "
            "the flip had to discharge, and deleting it would delete the standard: the "
            "VALUE is a MEASUREMENT the operator takes at "
            "the box with `python -m mantis.diagnostics.fusion_calibrate`, and R119 makes it "
            "their act. Flipping this REQUIRED THEN would have gated run5's mint on a number this "
            "repo would have to invent — the class R84 refused when it ratified "
            "exit_code=None rather than fabricating a 46, and the same class the grad-norm "
            "row above is deferred for. A DEFERRED row prints loudly on every gate-12 run and "
            "gates nothing, which is exactly the posture for a live refusal whose value is "
            "owed. "
            "WHY THE ROW EXISTS AT ALL, given the refusal is already run-fatal: the refusal "
            "fires when a graph run STARTS, and gate 12 runs on every push. The row is what "
            "makes an uncalibrated production config AUDIBLE in CI instead of discovered by a "
            "boot three weeks later. `CONFIG_THRESHOLD_GT_ZERO` reads the R119 `null` "
            "placeholder as DISARMED, which is the truth: `_is_real_number` rejects None, and "
            "the schema's `ge=1` closes the low end so any minted value arms. Only the edges "
            "member is named because the two are minted in ONE act from one fit against one "
            "budget — a half-minted block is a state the calibration cannot produce, and a "
            "second row would be a second authority over one byte budget. "
            "exit_code is None, truthfully: this is a CONSTRUCTION-TIME refusal on the "
            "`MissingEncodingError` shape, not a `_fire_hard_abort` rule, so it exits through "
            "whatever the composer does with a raise and R84's draw-rate codes do not apply. "
            "TO CLOSE THIS ROW: run the calibration at the box, mint what it reports into "
            "`configs/run5.yaml` and `configs/shakedown_20260807.yaml`, and flip status to "
            "REQUIRED — the one-field data edit §8.5 claims it is. The pin binds the "
            "resolver's own refusal, so deleting it, renaming the error or softening the null "
            "check to a default all break the R56 scan rather than the cap silently becoming "
            "absent-and-unbounded while still reporting as present."
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
#: F-P2B (R259 shakedown, MAIN-adopted fix-forward): `configs/shakedown_20260807.yaml` is a
#: PRODUCTION config, not an exempt smoke — it ARMS both required rows (actor-lag bool true,
#: draw-rate triple {0.25, 25000, 50, consec 3}) and soaks the eval/promotion machinery, so
#: gate 12 must audit it BY NAME exactly as it audits run5. Declaring it EXEMPT would have
#: spelled "armed production run" and "deliberately disarmed smoke" with the same observable,
#: which is the confusion MF-7 exists to kill.
PRODUCTION_CONFIGS: tuple[str, ...] = (
    "configs/run5.yaml",
    "configs/shakedown_20260807.yaml",
)

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


@dataclass(frozen=True)
class CadenceVerdict:
    """One armed row judged on the cadence axis. `within` is the only field that gates.

    `earliest_step` carries the three currencies `Cadence.earliest_fire_step` answers in —
    a step, `math.inf` for "these operands can never fire", and `None` for "no step cadence
    governs this row" — because collapsing them here would destroy exactly the distinction
    the operator needs to tell a DEFECT from a rule that is not step-cadenced at all.

    R265 / ADJ-D38 adds the row's OWN CLOCK beside the step answer, and publishes both rather
    than replacing one with the other. `earliest_samples` / `bound_samples` are the pair the
    verdict is actually DECIDED on — a count of the axis's own ticks against the bound
    converted into the same ticks — while `earliest_step` / `bound` stay the currency an
    operator reads a run in. `period_steps` is what ties them, derived from `clock`'s live
    key; it is `None` exactly when `clock` is `NO_STEP_CLOCK`, where a period would be a
    number nobody could have derived and `earliest_step` carries the rule's step FLOOR
    instead. Publishing both is the anti-vacuity posture the whole cadence block already
    takes: a reader can see WHICH clock a row was judged in, not just that it passed.
    """

    row: ArmedAbort
    clock: SampleClock
    period_steps: float | None
    earliest_samples: float | None
    earliest_step: float | None
    bound: float
    bound_samples: float | None
    within: bool
    detail: str


def audit_cadence(
    config: Any,
    *,
    manifest: tuple[ArmedAbort, ...] = MANIFEST,
    fraction: float = EARLIEST_FIRE_FRACTION,
) -> tuple[CadenceVerdict, ...]:
    """R251: can every ARMED required row still FIRE inside this config's own run?

    The second half of assertion (c). `audit_arming` asks whether the arming surface is set;
    this asks whether the machinery that reads it ever runs, which ADJ-D22 measured are
    different questions: `monitor.gate_interval: 1000000000` leaves a threshold armed in the
    config and unread in the run, and every check that existed before this one read it green.

    SCOPE, deliberately narrow in three directions:

    * REQUIRED rows only — a DEFERRED row is printed loudly and audited by nothing (R56);
    * ARMED rows only — a disarmed row is already rc 30 from `audit_arming`, and reporting it
      again under a second name sends the operator chasing a cadence question about an abort
      that is simply off;
    * a row whose cadence answers `None` is REPORTED and never failed. Its rule has no in-run
      step cadence (close-out), so a fraction of the run is a category error for it, and
      forging a number would be the class R84 refused.

    A REQUIRED row that declares NO cadence is OUT OF BOUND by name. That is the same rule
    `Mechanism.is_armed` applies to a missing ceiling: an unjudgeable armed row must fail
    toward visibility, or "nobody declared it" and "it is fine" become one observable.

    R265 / ADJ-D38 — EACH ROW IS JUDGED IN ITS OWN SAMPLE CLOCK. The row's cadence names the
    clock, the clock derives its period from a live key, and the comparison happens in ticks
    of that clock: `earliest_fire_samples <= bound / period`. For a fixed period that is the
    same verdict the step-clock comparison gave (multiply both sides), which is why no
    committed config changes colour; what changes is that an axis sampled on a DIFFERENT key
    can now be judged at all, instead of being judged against a key it never reads or — the
    state ADJ-D38 measured on the WR axis — not being judged at all. A clock whose period
    cannot be derived RAISES `SampleClockNotDerivableError`; there is no step-clock fallback
    anywhere on this path.

    `manifest` and `fraction` are keywords for the same reason `audit_arming`'s `manifest`
    is: the trigger has to be drivable in both directions, and `fraction` is what the audit's
    own mutation pin neuters to prove the comparison is live rather than decorative.
    """
    armed = audit_arming(config, manifest=manifest)
    disarmed = {row.name for row in armed.disarmed}
    total = _dotted(config, RUN_LENGTH_PATH, row="<cadence bound>")
    # An unreadable run length yields a bound of `-inf`, so every judged row reports OUT of
    # bound: the audit must never quietly widen its own bound to "anything goes" because the
    # key it takes the bound from went missing.
    bound = float(fraction) * float(total) if _is_real_number(total) else -math.inf
    verdicts: list[CadenceVerdict] = []
    for row in armed.required:
        if row.name in disarmed:
            continue
        if row.cadence is None:
            verdicts.append(CadenceVerdict(
                row=row, clock=SampleClock.NO_STEP_CLOCK, period_steps=None,
                earliest_samples=None, earliest_step=math.inf, bound=bound,
                bound_samples=None, within=False,
                detail=("declares no cadence, so the audit cannot compute when it could "
                        "first fire — an unjudgeable armed row gates rather than passes"),
            ))
            continue
        clock = row.cadence.sample_clock
        values = tuple(_dotted(config, path, row=row.name) for path in row.cadence_paths)
        operands = ", ".join(f"{path}={value!r}"
                             for path, value in zip(row.cadence_paths, values, strict=True))
        if clock.is_step_clocked:
            period = clock.period_steps(config, row=row.name)
            samples = row.cadence.earliest_fire_samples(values, period_steps=period)
            earliest = row.cadence.earliest_fire_step(values, period_steps=period)
            # A degenerate period cannot divide the bound, and it must not be read as a
            # generous one: `-inf` ticks refuses every row whose clock does not advance. Its
            # `earliest_fire_samples` is already `inf`, so the two agree by construction.
            bound_samples = bound / period if period >= 1.0 else -math.inf
            within = samples is None or samples <= bound_samples
            tick = f"{clock.period_path}={period}"
            detail = (f"cadence {row.cadence.value} sampled on the {clock.value} clock "
                      f"(1 tick = {tick} training steps)"
                      + (f" over {operands}" if operands else "")
                      + f"; earliest fire {samples} tick(s) = training step {earliest}, "
                        f"bound {bound_samples} tick(s)")
        else:
            period = None
            samples = None
            bound_samples = None
            earliest = row.cadence.earliest_fire_step(values, period_steps=None)
            within = earliest is None or earliest <= bound
            detail = (f"cadence {row.cadence.value} on the {clock.value} clock"
                      + (f" over {operands}" if operands else "")
                      + ("; not step-cadenced, so the fraction rule does not bind it"
                         if earliest is None else f"; earliest fire step {earliest}"))
        verdicts.append(CadenceVerdict(
            row=row, clock=clock, period_steps=period, earliest_samples=samples,
            earliest_step=earliest, bound=bound, bound_samples=bound_samples,
            within=within, detail=detail,
        ))
    return tuple(verdicts)


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

    * a rule with NO manifest row at all — R84 refused to invent a code for an abort nobody
      registered, and inventing one HERE would be that same class one layer down. CORRECTED
      at R265: this bullet used to name `grad_norm_hard_abort` and `sealbot_wr_abort` as its
      examples, and BOTH have since gained rows (K-B and ADJ-D38 respectively). They answer
      `None` through the SECOND source below now, not this one; the class stands and its
      examples were transcribed, which is why they went stale;
    * a row that is registered but carries `exit_code=None` — the posture the draw-rate row
      itself held until Phase X, and the posture both DEFERRED rows hold today.

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
    "EARLIEST_FIRE_FRACTION",
    "EXEMPT_CONFIGS",
    "MANIFEST",
    "PRODUCTION_CONFIGS",
    "RUN_LENGTH_PATH",
    "TERMINAL_EVAL_BROKEN_ABORT_RULE",
    "ArmedAbort",
    "ArmingSurfaceMissingError",
    "AuditResult",
    "Cadence",
    "CadenceVerdict",
    "Mechanism",
    "SampleClock",
    "SampleClockNotDerivableError",
    "Status",
    "audit_arming",
    "audit_cadence",
    "exit_code_for_abort",
]
