# R8 >300 justify: the manifest ROWS are data and their `note` text IS the row — a field
# gate 12 prints on every run, not a comment. The walkers (`_dotted`, `audit_arming`,
# `audit_cadence`), the exit-code resolver and the cadence/clock axes each answer a question
# the rows themselves hold the operands for; splitting any of them out would put "which
# aborts must arm" and the predicates that read it on opposite sides of an import.
"""The armed-abort manifest — WHICH aborts a production config MUST arm.

ONE authority, and it is DATA: a typed frozen dataclass read by `import`, carrying its
invariant in `__post_init__`, with no doc twin to drift from.

THE LAYER BOUNDARY. This module makes ZERO filesystem calls. `PRODUCTION_CONFIGS` holds
repo-relative STRINGS; resolving them against a repo root lives in
`tools/ci_gates/preflight_mint.py`, because a shipped package that resolved one would be
depending on an editable install. Pinned by `tests/config/test_armed_abort_manifest.py`.

`wr_hard_abort_enabled` is a DEFERRED row: the sealbot win-rate abort ships WARN-ONLY by
operator ruling and nothing here may flip it REQUIRED, because that would gate every
production mint on a value the operator deliberately mints false. A DEFERRED row prints
loudly on every gate-12 run, gates nothing, and makes the flip a one-field data edit.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# The exit-code AUTHORITY. Imported rather than re-typed: a literal here would be a second
# place "which code does this abort use" is written.
from mantis.monitor.heartbeat import (
    ACTOR_LAG_EXIT_CODE,
    DISK_SPACE_EXHAUSTED_EXIT_CODE,
    DRAW_RATE_COLLAPSE_EXIT_CODE,
    TERMINAL_EVAL_BROKEN_EXIT_CODE,
)

#: The disk-guard abort's RULE NAME — one spelling, exported. It has TWO readers, the row
#: below and `mantis.run.compose_run`, because `mantis.train` may not import this module; a
#: bare literal at each would let a rename leave the resolver answering `None`.
DISK_SPACE_ABORT_RULE: str = "disk_space_exhausted"

#: The broken-terminal-eval RULE NAME — one spelling, exported, same shape and grounds as
#: `DISK_SPACE_ABORT_RULE`: the row below and `mantis.run.compose_run` are its two readers.
TERMINAL_EVAL_BROKEN_ABORT_RULE: str = "terminal_eval_broken"

#: The disk guard's LIVENESS PROBE NAME. Two readers — the row below and
#: `mantis.run.compose_run`, which owns the running guard — and a rename fails LOUDLY:
#: `audit_arming_live` raises `ProducerProbeMissingError` naming both sides.
DISK_GUARD_LIVENESS_PROBE: str = "disk_guard_checks_completed"


def _is_real_number(value: Any) -> bool:
    """True for a finite `int`/`float` that is not a `bool`.

    `bool` is excluded because `isinstance(True, int)` is True and a `True` on a threshold
    path is a type confusion; non-finite is excluded because `inf` beats every ceiling and
    `nan` compares False against all of them, so either would decide arming by accident.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


class Status(StrEnum):
    """REQUIRED rows are audited and gate; DEFERRED rows are printed loudly and do not."""

    REQUIRED = "required"
    DEFERRED = "deferred"


class Mechanism(StrEnum):
    """The predicate that decides "armed" for a row's value. DATA, not a branch on `name`."""

    CONFIG_BOOL = "config_bool"
    CONFIG_THRESHOLD_GT_ZERO = "config_threshold_gt_zero"
    #: An UPPER-bounded threshold: armed iff the value is a real, finite, positive number that
    #: is ALSO no greater than a ceiling read off the row's `ceiling_path`. `> 0` alone reads
    #: an unreachable `1e9` as ARMED, which is "armed in the config, absent in effect".
    CONFIG_THRESHOLD_BELOW_CEILING = "config_threshold_below_ceiling"
    #: A TOKEN, not a number: armed iff the value is a non-empty string. Its subject is
    #: `allocator_posture`, whose value is a regime member or the `null` placeholder — a
    #: numeric predicate would report a correctly minted posture DISARMED forever.
    CONFIG_ENUM_VALUED = "config_enum_valued"
    #: `CONFIG_THRESHOLD_GT_ZERO` with a SECOND operand: the probe named by the row's
    #: `producer_probe` must also answer True, but ONLY when an answer is supplied, so the
    #: pure-config audit gate 12 runs is unchanged. A threshold nobody reads is otherwise
    #: indistinguishable from one being read.
    CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER = "config_threshold_gt_zero_with_live_producer"

    def is_armed(self, value: Any, *, ceiling: Any = None,
                 producer_live: bool | None = None) -> bool:
        """True iff `value` arms the abort. A real predicate in BOTH directions.

        `ceiling` is consumed only by `CONFIG_THRESHOLD_BELOW_CEILING` and is resolved by
        `audit_arming` from the row's own `ceiling_path`; a row with no usable ceiling reports
        DISARMED, because an unjudgeable row must fail toward visibility, never toward silence.
        """
        if self is Mechanism.CONFIG_BOOL:
            return value is True
        if self is Mechanism.CONFIG_ENUM_VALUED:
            # `""` is DISARMED: an empty token is the placeholder wearing another spelling.
            return isinstance(value, str) and value != ""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if self is Mechanism.CONFIG_THRESHOLD_BELOW_CEILING:
            if not _is_real_number(value) or not _is_real_number(ceiling):
                return False
            return 0.0 < float(value) <= float(ceiling)
        if self is Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER:
            # `producer_live is None` means NO ANSWER WAS SUPPLIED, not "the producer is
            # dead": collapsing the two would red the pure-config audit on every commit.
            return float(value) > 0.0 and producer_live is not False
        return float(value) > 0.0


#: The FRACTION of a run's own `train.max_train_steps` inside which an ARMED abort must be able
#: to fire. ONE authority, consumed by `audit_cadence`. Deliberately NOT a config key: a config
#: that could set its own fraction could relax its own audit. 0.25 is a bound, not a target —
#: every armed row on every committed config clears it with margin (run5's draw-rate row is
#: earliest-fire 25000 of 1000000, its actor-lag row 101, the armed smoke's draw-rate row 30 of
#: 200). Moving this number is a ruling, not a tuning.
EARLIEST_FIRE_FRACTION: float = 0.25

#: The config path the bound is taken FROM, walked through the SAME `_dotted` as every row's
#: paths, so renaming it is one loud `ArmingSurfaceMissingError` and not a silent bound of zero.
RUN_LENGTH_PATH: str = "train.max_train_steps"


class SampleClockNotDerivableError(ValueError):
    """A row's SAMPLE CLOCK period could not be derived from the config.

    The audit RAISES instead of falling back to the training-step clock: an axis judged in a
    clock it does not tick in audits GREEN on exactly the configs this exists to refuse.
    `preflight_mint.py` maps it onto `PreflightManifestError`, rc 31.
    """


class SampleClock(StrEnum):
    """WHICH CLOCK an axis's samples arrive in, and how long ONE tick is in TRAINING STEPS.

    The period is a property of the CLOCK and is a live config PATH, never a number written
    here — a row that supplied its own period could be audited in a clock it never ticks in.
    An axis whose period cannot be derived RAISES; there is no default tick in this class.
    """

    #: One sample per TRAINING STEP. The period is 1 by DEFINITION of the clock, which is why
    #: this member names no config path and why its `None` does not mean "underivable".
    TRAIN_STEP = "train_step"

    #: One sample per hard-abort GATE BOUNDARY: `step.py::_run_gate_interval` runs the live
    #: gates only when `self._train_step % cfg.gate_interval == 0`.
    GATE_BOUNDARY = "gate_boundary"

    #: One sample per completed EVAL ROUND: round `r` lands at training step
    #: `r * train.eval_interval` with `r >= 1`, and each routed result appends one WR sample.
    #: DISCLOSED: `eval_enabled` false builds NO eval pipeline, so the axis ticks zero times
    #: however small the interval is; that half is held by `test_minted_config_remint.py`.
    EVAL_ROUND = "eval_round"

    #: NOT step-clocked at all — a wall-clock poll or a close-out rule. `period_steps` RAISES
    #: rather than answering 1, which would be the step-clock fallback this class forbids; such
    #: a row is judged by the STEP FLOOR its rule imposes instead.
    NO_STEP_CLOCK = "no_step_clock"

    @property
    def period_path(self) -> str | None:
        """The live config key ONE tick of this clock is measured by.

        `None` on the two members with no config period, and they mean different things —
        `TRAIN_STEP`'s tick is definitional, `NO_STEP_CLOCK` has no step tick at all — which is
        why `period_steps` branches on the MEMBER and not on this being `None`.
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

        A value that is not a real, finite, non-`bool` number is UNDERIVABLE and raises. A real
        number BELOW 1 is derivable and degenerate, and passes through so
        `Cadence.earliest_fire_samples` can answer `math.inf` for it.
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
    """EVAL ROUNDS before ONE sealbot-WR trigger can first fire.

    Derived from `monitor/rules.py::sealbot_wr_trajectory_alert`: `len(history) >= n_consec`
    needs `consec` samples, but the empty-history guard needs at least ONE regardless, hence
    the `max(consec, 1)` floor; `current_step > min_step` is STRICT and round `r` lands at
    `r * period`. OPTIMISTIC by construction — a reachability floor, not a prediction.
    """
    if period < 1.0:
        return math.inf
    return max(max(consec, 1.0), float(math.floor(min_step / period)) + 1.0)


class Cadence(StrEnum):
    """WHEN a row's abort can FIRST fire, in TRAINING STEPS. DATA, not a branch on `name`.

    The twin of `Mechanism`: `cadence` selects the arithmetic and the row supplies the operands
    through `cadence_paths`. Every member is derived from the code that evaluates the row,
    cited at the member; no operand is baked in.

    `earliest_fire_step` answers in three currencies and the difference between the last two is
    load-bearing: a finite float is a step, `math.inf` is "these operands can never fire", and
    `None` is "no step cadence governs this row at all" — which must never be forged into a
    number. It is `earliest_fire_samples`, in the row's OWN clock, times that clock's period.
    """

    #: The draw-rate gate on `SampleClock.GATE_BOUNDARY`: `check_draw_rate_collapse` refuses on
    #: `len(history) < consec` and on `current_step < min_step`, so the earliest fire is the
    #: first boundary that is both the `consec`-th observation and at or past `min_step`. A
    #: boundary yielding no observation neither advances NOR resets `consec`, so this bounds a
    #: real fire from BELOW. Operands: (consec path, min-step path).
    GATE_INTERVAL_CONSEC = "gate_interval_consec"

    #: The sealbot-WR trajectory abort in the EVAL-ROUND clock: the earliest fire is the
    #: MINIMUM over its three triggers, in ROUNDS, converted through `train.eval_interval`.
    #: Operands: (collapse-consec, early-death-min-step, collapse-min-step, rolling-consec,
    #: rolling-min-step) — B and C SHARE `monitor.wr_collapse_consecutive_evals`, which is why
    #: five paths cover three triggers.
    EVAL_ROUND_CONSEC = "eval_round_consec"

    #: The grad-norm gate is evaluated PER TRAINING STEP inside the burst and fires when
    #: `self._consec_high_gn >= cfg.hard_gn_min_steps`. Operands: (min-steps path,).
    CONSEC_TRAIN_STEPS = "consec_train_steps"

    #: `ActorLagSpec` — `learner_step_fn() - actor_ckpt_step_fn() > threshold_steps`. The
    #: seconds poll adds no STEP floor, so a frozen actor first satisfies the strict `>` one
    #: step past the threshold. Operands: (threshold-steps path,).
    STEP_LAG_THRESHOLD = "step_lag_threshold"

    #: The disk guard's daemon thread. NO train-step boundary gates it, so the earliest TRAIN
    #: STEP is 0 — a derived answer, not an exemption. Operands: none. DISCLOSED: a wall-clock
    #: rule can still be cadence-disarmed in SECONDS (`monitor.disk_guard.interval_sec` past
    #: the run's wall time), which this fraction cannot see.
    WALL_CLOCK_POLL = "wall_clock_poll"

    #: `drain.close_out` -> `run_terminal_eval`, the LAST action of the run. It has no in-run
    #: step cadence, so it answers `None` and the fraction rule does not bind it: asking when a
    #: close-out rule fires "early" is a category error.
    CLOSE_OUT_TERMINAL = "close_out_terminal"

    #: A rule evaluated when the object it guards is CONSTRUCTED (`resolve_fused_graph_caps`,
    #: called eagerly from the graph branch of `InferenceServer.__init__`). Its floor is 0.0
    #: because it fires before step 0 and cannot be reached later — NOT `WALL_CLOCK_POLL`'s
    #: "could fire at any instant" 0.0, and not `CLOSE_OUT_TERMINAL`'s `None`. No operands.
    CONSTRUCTION_TIME = "construction_time"

    @property
    def sample_clock(self) -> SampleClock:
        """WHICH clock this member's evidence arrives in. DATA, like `arity`."""
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
        """How many `cadence_paths` this member CONSUMES; `ArmedAbort` enforces it both ways,
        so a path the arithmetic never reads cannot sit on a row pretending to be an input."""
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
        """The earliest TRAINING STEP a NOT-STEP-CLOCKED member's rule imposes.

        Defined only for the `NO_STEP_CLOCK` members and RAISES on the others. `0.0` for a
        wall-clock poll (no train-step boundary gates it) and for a construction-time rule (it
        has already fired or never will); `None` for a close-out rule, where asking when it
        fires "early" is a category error and `max_train_steps` would fail it forever.
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
        """How many TICKS OF THIS ROW'S OWN SAMPLE CLOCK before it can first fire.

        `period_steps` arrives from `SampleClock.period_steps`, never from an operand, so no
        row can denominate itself. An operand that is not a real, finite, non-`bool` number
        answers `math.inf`, as does a degenerate period below one training step: an unjudgeable
        rule must fail toward VISIBILITY, never toward silence.
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
        """The earliest TRAIN STEP at which this cadence can fire, in the three currencies
        above, COMPOSED from `earliest_fire_samples` and `step_floor`.

        `period_steps` is REQUIRED for a step-clocked member — assuming one step per tick IS
        the defect this split exists to close — and must be `None` for a `NO_STEP_CLOCK` one,
        where a period is an operand nobody can have derived.
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
            # `math.inf * 0.0` is `nan`, and a nan step compares False against every bound and
            # reads as WITHIN — the unfireable row auditing green, one multiply late.
            return samples
        return samples * float(period_steps)


@dataclass(frozen=True)
class ArmedAbort:
    """One row: an abort, the config surface that arms it, and its ownership posture.

    `owner` and `source_pin` are REQUIRED on a DEFERRED row; `owner` is FORBIDDEN on a REQUIRED
    one and `source_pin` is UNCONSTRAINED there, so a REQUIRED row MAY keep its pin. Each rule
    closes a way for a row to go invisible: an owner-less deferred row has nobody to chase, a
    pin-less one is not tamper-evident, and a required row carrying an owner reads as excused.
    """

    name: str
    config_path: str
    mechanism: Mechanism
    status: Status
    exit_code: int | None
    owner: str | None
    source_pin: tuple[str, str] | None
    note: str
    #: The SECOND config path a `CONFIG_THRESHOLD_BELOW_CEILING` row needs — where its upper
    #: bound is minted. The default is safe because `__post_init__` REQUIRES it on the
    #: mechanism that consumes it and FORBIDS it on the others, in both directions, so `None`
    #: can neither arm a row nor excuse one.
    ceiling_path: str | None = None
    #: WHEN this row's abort can first fire, and the paths its arithmetic reads. The defaults
    #: are safe because they are not silent: `audit_cadence` reports a REQUIRED row with NO
    #: cadence as OUT OF BOUND by name, and `__post_init__` enforces the arity pairing both
    #: ways, so a path the arithmetic never reads cannot sit on a row.
    cadence: Cadence | None = None
    cadence_paths: tuple[str, ...] = ()
    #: The NAME of the liveness probe a `CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER` row is
    #: judged by — a name, never a callable, because this module holds import-time data and
    #: makes no filesystem call, so the caller that OWNS the subsystem supplies the answer.
    #: `__post_init__` requires it on the one mechanism that reads it and forbids it elsewhere.
    producer_probe: str | None = None

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
        needs_probe = (
            self.mechanism is Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER
        )
        if needs_probe and not self.producer_probe:
            raise ValueError(
                f"armed-abort row {self.name!r} uses {self.mechanism.value} and names no "
                "`producer_probe`: the liveness operand would have no subject, so the row "
                "would fall back to a bare threshold while claiming to check a producer — "
                "the phantom-input class this mechanism exists to close (LAW-07)"
            )
        if not needs_probe and self.producer_probe:
            raise ValueError(
                f"armed-abort row {self.name!r} names a `producer_probe` "
                f"({self.producer_probe!r}) but its mechanism {self.mechanism.value} ignores "
                "it: a probe the predicate never reads is a claim the audit does not make "
                "(LAW-07's phantom-input class)"
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


#: The rows — the ONE authority for which aborts a production config must arm.
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
        # `monitor.gate_interval` is NOT an operand here: it is the PERIOD of this row's
        # sample clock, read off `SampleClock.GATE_BOUNDARY`, so one authority serves the axis.
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
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER,
        producer_probe=DISK_GUARD_LIVENESS_PROBE,
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
        # Declared even though a DEFERRED row is not audited, so the flip to REQUIRED stays a
        # one-field data edit; `preflight_mint.py::_print_deferred_rows` prints it meanwhile.
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
        # Declared on a DEFERRED row so the flip to REQUIRED stays a one-field data edit. With
        # no row at all the cadence audit could not compute even a FALSE answer for this axis;
        # `preflight_mint.py::_print_deferred_rows` prints the cadence and its clock meanwhile.
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
        # Declared on a DEFERRED row so the flip to REQUIRED stays a one-field data edit. The
        # rule runs at CONSTRUCTION (the resolver is called eagerly from
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
    ArmedAbort(
        name="allocator_posture_minted",
        config_path="allocator_posture",
        mechanism=Mechanism.CONFIG_ENUM_VALUED,
        # The rule runs at CONSTRUCTION — in the run process's builder and in the eval child's
        # first statement — so it consumes no operands: a value is minted or it is `null`.
        cadence=Cadence.CONSTRUCTION_TIME,
        cadence_paths=(),
        status=Status.REQUIRED,
        exit_code=None,
        # `owner` is NOT dropped on the flip, it is set to None: the dataclass takes it
        # positionally, so removing the keyword is a TypeError at import.
        owner=None,
        source_pin=(
            "src/mantis/config/resolve/allocator_posture.py",
            "raise UncalibratedAllocatorPostureError(",
        ),
        note=(
            "THE ALLOCATOR REGIME THE CAPS ARE FITTED UNDER. `PYTORCH_CUDA_ALLOC_CONF` "
            "changes the reserved/allocated ratio the whole memory partition divides by: the "
            "2026-08-22 re-calibration sitting measured 14.98 GiB of card high-water under "
            "DEFAULT against 11.36 under `expandable_segments:True`, same config, same host, "
            "same duration. It kept DEFAULT anyway -- and the reason was governance, not the "
            "measurement: a cap fitted under the better posture would have depended on an "
            "environment variable that no config minted, no gate checked and NO ROW HERE "
            "COVERED. This row is the third of those three. "
            "DEFERRED, and the reason is the one this manifest keeps meeting: the VALUE is a "
            "MEASUREMENT nobody has taken. R308(g)(i) reserves it for the re-calibration "
            "sitting under R282(b), so flipping this REQUIRED now would gate every push on a "
            "regime this repo would have to invent -- the class R84 refused when it ratified "
            "exit_code=None rather than fabricating a 46. A DEFERRED row prints loudly on "
            "every gate-12 run and gates nothing, which is exactly the posture for a live "
            "refusal whose value is owed. "
            "WHY THE ROW EXISTS AT ALL, given the refusal is already run-fatal: the refusal "
            "fires when a cuda process starts, and gate 12 runs on every push. The row is "
            "what makes an unminted production posture AUDIBLE in CI instead of discovered by "
            "a boot at the box. `CONFIG_ENUM_VALUED` reads the R119 `null` placeholder as "
            "DISARMED, which is the truth, and any member of the closed regime set as ARMED. "
            "exit_code is None, truthfully: this is a CONSTRUCTION-TIME refusal on the "
            "`UncalibratedFusedGraphCapsError` shape, not a `_fire_hard_abort` rule. "
            "CLOSED 2026-08-31 by RECAL-SITTING-5's mint under R326. The posture was decided "
            "at the box -- `expandable_segments`, the regime the frag probe measured "
            "1.2487863035511424 under -- and minted into all seven configs IN THE SAME ACT as "
            "the caps fitted under it, because the posture and the pair are one regime and a "
            "config carrying one without the other describes a machine state nobody measured. "
            "Status is REQUIRED and `owner` is None -- NOT dropped: the dataclass takes it "
            "positionally and removing the keyword is a TypeError at import (F-RESIT-5). The pin "
            "binds the resolver's own placeholder refusal, so deleting it, renaming the error "
            "or softening the null check to a default all break the R56 scan."
        ),
    ),
)

#: WHICH configs the law binds — one authority. Repo-relative strings only; resolving them is
#: the tool's job. Membership is audited BY NAME, so exempting the run an operator is about to
#: mint is a red gate rather than a bookkeeping edit.
PRODUCTION_CONFIGS: tuple[str, ...] = ("configs/run6.yaml",)

#: The OTHER half of the same authority: the two tuples must PARTITION the config set EXACTLY,
#: and the tool hard-fails (rc 31) on either kind of drift — a config on disk named by neither
#: tuple, or a tuple naming a config that is not on disk. Exemption by ABSENCE made
#: "deliberately exempt" and "nobody listed it" the same observable.
#:
#: Discovery is `mantis.config.loader.discover_configs`, shared by gates 7 and 12 and
#: NAME-AGNOSTIC, so adding ANY file under `configs/` forces a declaration here or in
#: `PRODUCTION_CONFIGS`; the deliberate cost is that `configs/` may hold only complete configs.
#: A loadable config OUTSIDE `configs/` is still reachable without being discovered.
#:
#: `(repo-relative path, why it is exempt)`. The reason is data, printed by the tool on the
#: failure path, so an exemption cannot be a bare path nobody can justify later.
EXEMPT_CONFIGS: tuple[tuple[str, str], ...] = (
    (
        "configs/dev_example.yaml",
        "developer template, never minted for a run; DISARMED by design (R59). R346(f) pruned "
        "configs/ to run6 plus one smoke and this file was cut with the rest — it is BACK, and "
        "the ground is LAW-07: ADJ-13 N-3 makes it the mutation corpus's M1 row, the one real "
        "committed config that demonstrates gate 12 going RED on the real `configs/` tree. "
        "With run5, the shakedown and the plain smoke gone it is the only disarmed config "
        "left, so deleting it would leave the gate with no red-capability demonstration on "
        "the tree it audits. Pinned by "
        "`test_naming_a_DISARMED_config_is_AUDITED_and_never_ignored`.",
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
)


class ArmingSurfaceMissingError(AttributeError):
    """A row's `config_path` does not resolve on a real `RunConfig`.

    Subclasses `AttributeError` deliberately, so every existing caller keeps its behaviour;
    what changes is that the failure is NAMED and carries WHICH ROW is broken, WHAT PATH it
    declared and WHICH SEGMENT does not exist. The tool maps it to `PreflightManifestError`,
    rc 31. Written to the CLASS, not to one row.
    """


def _dotted(obj: Any, path: str, *, row: str = "<unnamed row>") -> Any:
    """Walk a dotted path into a validated config object.

    A MISSING attribute raises `ArmingSurfaceMissingError` naming the row, the full path and
    the failing segment — caught per segment, because only that can say WHICH segment failed.
    A `None` met MID-WALK is an EXPLICITLY DISARMED block and short-circuits to `None`, so a
    legitimately disarmed config reports DISARMED instead of failing the gate at rc 31.

    DISCLOSED RESIDUAL: a typo AFTER a legitimately-`None` segment reports "disarmed" rather
    than raising, because the walk short-circuits before reaching it.
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

    Never branches on a row's `name`: `status` selects the list, `mechanism` the predicate.
    `manifest` is a keyword so a test can drive an in-memory copy with a row flipped.
    """
    required = tuple(row for row in manifest if row.status is Status.REQUIRED)
    deferred = tuple(row for row in manifest if row.status is Status.DEFERRED)
    disarmed = tuple(
        row for row in required
        if not row.mechanism.is_armed(
            _dotted(config, row.config_path, row=row.name),
            # Resolved through the SAME walker as the value, so a typo in a `ceiling_path`
            # raises `ArmingSurfaceMissingError` naming the row as a `config_path` typo does.
            ceiling=(None if row.ceiling_path is None
                     else _dotted(config, row.ceiling_path, row=row.name)),
        )
    )
    return AuditResult(required=required, deferred=deferred, disarmed=disarmed)


class ProducerProbeMissingError(KeyError):
    """A row names a `producer_probe` the caller did not supply.

    A NAMED refusal rather than a default in either direction: True would arm a row whose
    producer nobody looked at, False would report a healthy run's abort DISARMED because a
    caller forgot a key. So an unanswerable row raises and names itself.
    """

    def __init__(self, row: str, probe: str, supplied: tuple[str, ...]) -> None:
        super().__init__(
            f"armed-abort row {row!r} declares producer_probe {probe!r}, which is absent "
            f"from the probes supplied ({list(supplied)}): a live arming audit that cannot "
            "reach a row's producer must refuse, never assume — assuming True arms a row "
            "nobody checked and assuming False reds a healthy run on a caller's omission"
        )


def audit_arming_live(
    config: Any,
    *,
    probes: Mapping[str, Callable[[], bool]],
    manifest: tuple[ArmedAbort, ...] = MANIFEST,
) -> AuditResult:
    """Assertion (c) again, with the LIVE producers a running process can supply.

    The same predicate machinery as `audit_arming` and the same absence of any branch on a
    row's name; only WHO answers changes. A SECOND entry point rather than a keyword on the
    first, because `audit_arming` is what CI gate 12 runs and its contract is that it is PURE
    OVER CONFIG — a defaulted `probes=` would make that a property of the call site.

    Args:
        config: a validated `RunConfig` (or anything `_dotted` can walk).
        probes: liveness answers by probe NAME. Every probe a REQUIRED row names must be
            present; extra entries are ignored, because a caller may own more subsystems than
            the manifest asks about.
        manifest: keyword for the same reason `audit_arming` has one — so a test can drive an
            in-memory copy.

    Returns:
        The same `AuditResult` shape. `disarmed` is still the only field that gates.

    Raises:
        ProducerProbeMissingError: a REQUIRED row names a probe absent from `probes`.
        ArmingSurfaceMissingError: a row's `config_path` or `ceiling_path` does not resolve.
    """
    required = tuple(row for row in manifest if row.status is Status.REQUIRED)
    deferred = tuple(row for row in manifest if row.status is Status.DEFERRED)
    supplied = tuple(probes)
    disarmed: list[ArmedAbort] = []
    for row in required:
        live: bool | None = None
        if row.producer_probe is not None:
            if row.producer_probe not in probes:
                raise ProducerProbeMissingError(row.name, row.producer_probe, supplied)
            live = bool(probes[row.producer_probe]())
        if not row.mechanism.is_armed(
            _dotted(config, row.config_path, row=row.name),
            ceiling=(None if row.ceiling_path is None
                     else _dotted(config, row.ceiling_path, row=row.name)),
            producer_live=live,
        ):
            disarmed.append(row)
    return AuditResult(required=required, deferred=deferred, disarmed=tuple(disarmed))


@dataclass(frozen=True)
class CadenceVerdict:
    """One armed row judged on the cadence axis. `within` is the only field that gates.

    `earliest_step` carries the three currencies `Cadence.earliest_fire_step` answers in — a
    step, `math.inf` for "these operands can never fire", `None` for "no step cadence governs
    this row" — because collapsing them destroys the distinction between a DEFECT and a rule
    that is not step-cadenced at all. `earliest_samples`/`bound_samples` are the pair the
    verdict is DECIDED on, in the axis's own ticks; `period_steps` ties them to the step
    currency and is `None` exactly when `clock` is `NO_STEP_CLOCK`.
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
    """Can every ARMED required row still FIRE inside this config's own run?

    The second half of assertion (c): `audit_arming` asks whether the arming surface is set,
    this asks whether the machinery that reads it ever runs. Scope is REQUIRED and ARMED rows
    only — a disarmed row is already a failure from `audit_arming` — and a row whose cadence
    answers `None` is REPORTED, never failed. A REQUIRED row that declares NO cadence is OUT OF
    BOUND by name: an unjudgeable armed row must fail toward visibility.

    Each row is judged in its OWN sample clock: `earliest_fire_samples <= bound / period`, the
    period derived from a live key. A clock whose period cannot be derived RAISES
    `SampleClockNotDerivableError`; there is no step-clock fallback anywhere on this path.
    """
    armed = audit_arming(config, manifest=manifest)
    disarmed = {row.name for row in armed.disarmed}
    total = _dotted(config, RUN_LENGTH_PATH, row="<cadence bound>")
    # An unreadable run length yields a bound of `-inf`, so every judged row reports OUT of
    # bound: the audit must never widen its own bound because the key it reads went missing.
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
            # A degenerate period cannot divide the bound and must not read as a generous one:
            # `-inf` ticks refuses every row whose clock does not advance.
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

    Looks the row up and returns whatever `exit_code` it carries — never a branch on a rule's
    identity — so the manifest stays the ONE authority and no second literal can disagree.

    `None` has two distinct and equally truthful sources, neither an error: a rule with NO
    manifest row at all, and a registered row carrying `exit_code=None`. A caller must read
    `None` as "this abort has no authored exit code", never as "no abort fired" —
    `ShutdownState.abort_rule is None` is the only thing that means the latter.
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
