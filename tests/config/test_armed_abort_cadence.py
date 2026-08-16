"""R251 / ADJ-D22 — the armed-abort audit's CADENCE half, at the module layer.

The defect these pin, measured: `monitor.gate_interval: 1000000000` on a 40-step run
produces ZERO gate boundaries, so an ARMED `train.draw_rate_abort` is never evaluated and
gate 12 audits that config green. `Mechanism.is_armed` reads a threshold; a threshold that
is never READ is armed in the config and absent in effect, and `ge=1` on the interval bans
exactly one spelling of "never gate" while permitting every larger one.

So the manifest now carries a SECOND data axis beside `mechanism` — `cadence`, the code-
derived answer to "at which training step could this row FIRST fire" — and `audit_cadence`
compares it against `EARLIEST_FIRE_FRACTION * train.max_train_steps`.

What each block below is the only witness to (LAW-07):

* the FRACTION is a named module constant with a live consumer, and is NOT a config key —
  a config that could set its own audit fraction could relax its own audit (ADJ-D20's
  gate-3c self-comparison class relocated);
* every REQUIRED row declares a cadence and every declared `cadence_paths` entry resolves
  on a REAL `RunConfig` — the phantom-input class `ceiling_path` already gets;
* `Cadence.earliest_fire_step` is real ARITHMETIC over the row's operands, not a constant;
* R265 / ADJ-D38 — every row is judged in ITS OWN SAMPLE CLOCK, whose period is derived from
  a live key the CLOCK names (never one the row declares), and a step-clocked member handed
  no period RAISES rather than falling back to one-tick-per-training-step;
* the shipped production config passes with the numbers stated, and a vacuous interval
  FAILS by name;
* the comparison is LIVE — driving the same config at `fraction=inf` flips the verdict, so
  neither the computed step nor the bound is decoration;
* the bound is anchored to the RUN-LENGTH authority and not to the LR-scheduler horizon — the
  WPAX S-4 / F-C class, caught here for the second time;
* the boundary is `exceeds` and not `reaches`, pinned from both sides of it.

>300 justify (R8): one audit, one subject. Every test below judges the SAME function over the
SAME row set, and each block varies exactly one of the four things that can make it wrong — the
fraction, the anchor key, the arithmetic, or the boundary. Splitting them by axis would fork the
`run5` fixture and the `_revalidated` helper (the loader's own `dump -> mutate -> model_validate`
step, which is what makes every synthetic config here one a run could actually be launched from)
into copies with their own ways of drifting from the loader, and a reader chasing "what does the
cadence audit promise" would have to find all of them. Roughly half the length is the per-test
"what defect is this the only witness to" rationale LAW-07 requires.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mantis.config.armed_aborts import (
    EARLIEST_FIRE_FRACTION,
    MANIFEST,
    RUN_LENGTH_PATH,
    ArmedAbort,
    Cadence,
    Mechanism,
    SampleClock,
    SampleClockNotDerivableError,
    Status,
    audit_cadence,
)
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN5 = REPO_ROOT / "configs" / "run5.yaml"


@pytest.fixture(scope="module")
def run5() -> RunConfig:
    return load_config(RUN5)


def _revalidated(config: RunConfig, section: str, key: str, value: object) -> RunConfig:
    """`dump -> mutate ONE key -> model_validate` — the loader's own final step, so every
    cross-field validator re-runs. A synthetic config built any other way would prove
    nothing about a config a run could actually be launched from."""
    raw = config.model_dump()
    raw[section][key] = value
    return RunConfig.model_validate(raw)


# ══ the fraction: one authority, and not a config key ══════════════════════════════════
def test_the_fraction_is_a_named_constant_and_a_config_can_never_set_it(run5) -> None:
    """R251's "the fraction is a schema constant with a live consumer (no code-side
    default, R1)", both halves.

    The NEGATIVE half is the load-bearing one and it is driven, not asserted: minting
    `earliest_fire_fraction` into a config must be REFUSED by the schema. If a config could
    carry it, the very disarm this constant exists to refuse would be re-spellable as
    `earliest_fire_fraction: 1.0` — ADJ-D20's gate-3c self-comparison class, one layer down.
    """
    assert EARLIEST_FIRE_FRACTION == 0.25, (
        "the fraction is a PRE-REGISTERED policy number, not a tuning knob: widening it is "
        "how this check quietly stops refusing anything, which is the silent-disarm class one "
        "layer up. Pinned the way `OVERRIDE_KEYS` is — moving it is a ruling, and a ruling "
        f"edits this line too. Got {EARLIEST_FIRE_FRACTION!r}"
    )
    assert isinstance(EARLIEST_FIRE_FRACTION, float)
    assert 0.0 < EARLIEST_FIRE_FRACTION <= 1.0, (
        "a fraction outside (0, 1] is not a fraction of a run: <= 0 fails every armed row "
        f"and > 1 can never fail one; got {EARLIEST_FIRE_FRACTION!r}"
    )
    for section in ("train", "monitor"):
        with pytest.raises(ValidationError, match="earliest_fire_fraction"):
            _revalidated(run5, section, "earliest_fire_fraction", 1.0)


def test_the_run_length_the_bound_is_taken_from_is_a_real_key(run5) -> None:
    """`RUN_LENGTH_PATH` is walked through the SAME `_dotted` every row's paths go through,
    so a rename of the run-length key is one loud `ArmingSurfaceMissingError` rather than a
    silent bound of 0. Pinned against the real schema, never transcribed.

    NOT SUFFICIENT ON ITS OWN, and the test below says why: `train.total_steps` also resolves
    to a positive int, so this arm is satisfied by the one substitution that matters.
    """
    obj: object = run5
    for part in RUN_LENGTH_PATH.split("."):
        obj = getattr(obj, part)
    assert isinstance(obj, int) and obj > 0, (
        f"{RUN_LENGTH_PATH} must resolve to a positive int on a real RunConfig; got {obj!r}"
    )


def test_the_bound_follows_the_RUN_LENGTH_authority_never_the_scheduler_horizon() -> None:
    """WPAX S-4 / F-C, restated on this axis — and this class has now been caught TWICE.

    `config/schema/core.py`'s own cadence bound records the first time: "the bound is anchored
    to `train.max_train_steps`, the RUN-LENGTH authority, not to `train.total_steps` … a
    2000-step run with `total_steps: 1000000` blessed a cadence of 999 999". R251's bound
    re-creates that anchor choice, and re-pointing `RUN_LENGTH_PATH` at the scheduler horizon
    is INVISIBLE on the whole production set: `PRODUCTION_CONFIGS` is run5 alone and run5 is
    the ONE config where the two keys agree (both 1000000). Every short config is where the
    substitution bites — and a short run is exactly the regime a large interval bites hardest
    in, since ADJ-D22's own reproducer was `stop_step=40`.

    Driven rather than asserted by name: this survives a legitimate rename of the key, and it
    fails the moment the bound starts reading a number that is not the run's length.
    """
    short = load_config(REPO_ROOT / "configs" / "smoke_gnn.yaml")
    assert short.train.total_steps != short.train.max_train_steps, (
        "this pin needs a config on which the RUN LENGTH and the LR-scheduler horizon DIFFER, "
        f"or it proves nothing; both read {short.train.max_train_steps}"
    )
    verdicts = audit_cadence(short)
    assert verdicts, "no verdict means this pin has no subject"
    for verdict in verdicts:
        assert verdict.bound == EARLIEST_FIRE_FRACTION * short.train.max_train_steps, (
            f"row {verdict.row.name!r}'s bound must be a fraction of {RUN_LENGTH_PATH}, the "
            f"RUN-LENGTH authority — anchored to train.total_steps it would read "
            f"{EARLIEST_FIRE_FRACTION * short.train.total_steps} on this 2000-step config, "
            f"which is the F-C defect verbatim; got {verdict.bound}"
        )


# ══ every row declares a cadence, and every declared path resolves ═════════════════════
def test_every_required_row_declares_a_cadence_whose_paths_all_resolve(run5) -> None:
    """LAW-07's phantom-input class, applied to the new axis in BOTH directions: a REQUIRED
    row with no cadence cannot be judged, and a `cadence_paths` entry that resolves to
    nothing is a claim the audit does not make."""
    required = [row for row in MANIFEST if row.status is Status.REQUIRED]
    assert required, "no required row means this test has no subject"
    for row in required:
        assert row.cadence is not None, (
            f"required row {row.name!r} declares no cadence — the audit cannot judge whether "
            "it can still fire inside the run"
        )
        assert len(row.cadence_paths) == row.cadence.arity, (
            f"row {row.name!r} declares {len(row.cadence_paths)} cadence path(s) for "
            f"{row.cadence.value}, which consumes {row.cadence.arity}"
        )
        for path in row.cadence_paths:
            obj: object = run5
            for part in path.split("."):
                obj = getattr(obj, part)
            assert obj is not None, (
                f"row {row.name!r} names cadence path {path!r}, which resolves to None on "
                "the shipped production config"
            )
        # R265 / ADJ-D38: the row's CLOCK must resolve too, and a row may NOT declare its own
        # period. Both halves matter — an unresolvable clock is a row that cannot be judged
        # at all, and a period sitting in `cadence_paths` is a row denominating itself, which
        # is how an axis ends up audited in a clock it does not tick in.
        clock = row.cadence.sample_clock
        if clock.period_path is not None:
            assert clock.period_path not in row.cadence_paths, (
                f"row {row.name!r} declares its own sample period {clock.period_path!r} as a "
                "cadence operand: the period belongs to the CLOCK so every row on an axis "
                "reads one key and no row can name another axis's"
            )
            probe: object = run5
            for part in clock.period_path.split("."):
                probe = getattr(probe, part)
            assert isinstance(probe, int) and probe >= 1, (
                f"row {row.name!r} ticks in {clock.value}, whose period is minted at "
                f"{clock.period_path!r}; that must be a positive number of training steps on "
                f"a real RunConfig, got {probe!r}"
            )


def test_the_arity_rule_is_enforced_at_construction_in_both_directions() -> None:
    """A row is unconstructible with the wrong number of operands for its own cadence —
    the shape `ceiling_path` already has. Both directions, so the invariant is not simply
    "always raise"."""
    common = dict(name="probe", config_path="train.terminal_eval_enabled",
                  mechanism=Mechanism.CONFIG_BOOL, status=Status.REQUIRED, exit_code=None,
                  owner=None, source_pin=None, note="synthetic arity probe")
    with pytest.raises(ValueError, match="cadence_paths"):
        ArmedAbort(cadence=Cadence.GATE_INTERVAL_CONSEC,
                   cadence_paths=("monitor.gate_interval",), **common)
    with pytest.raises(ValueError, match="cadence_paths"):
        ArmedAbort(cadence=Cadence.CLOSE_OUT_TERMINAL,
                   cadence_paths=("monitor.gate_interval",), **common)
    with pytest.raises(ValueError, match="cadence_paths"):
        ArmedAbort(cadence=None, cadence_paths=("monitor.gate_interval",), **common)
    # R265 / ADJ-D38: the WR member consumes five operands and the draw-rate member two —
    # the interval that used to be operand 0 is the CLOCK's now, so a row still naming it
    # is one path over and refused by the same rule.
    with pytest.raises(ValueError, match="cadence_paths"):
        ArmedAbort(cadence=Cadence.GATE_INTERVAL_CONSEC,
                   cadence_paths=("monitor.gate_interval", "train.draw_rate_abort.consec",
                                  "train.draw_rate_abort.min_step"), **common)
    with pytest.raises(ValueError, match="cadence_paths"):
        ArmedAbort(cadence=Cadence.EVAL_ROUND_CONSEC,
                   cadence_paths=("monitor.wr_collapse_consecutive_evals",), **common)
    # …and the legal shapes construct.
    ArmedAbort(cadence=Cadence.CLOSE_OUT_TERMINAL, cadence_paths=(), **common)
    ArmedAbort(cadence=Cadence.STEP_LAG_THRESHOLD,
               cadence_paths=("monitor.actor_lag_threshold_steps",), **common)
    ArmedAbort(cadence=Cadence.GATE_INTERVAL_CONSEC,
               cadence_paths=("train.draw_rate_abort.consec",
                              "train.draw_rate_abort.min_step"), **common)


# ══ the arithmetic is arithmetic ═══════════════════════════════════════════════════════
def test_the_earliest_fire_step_is_derived_and_never_a_constant() -> None:
    """A constant here would silently pass or fail every row at once — the warning
    `Mechanism.is_armed`'s docstring already carries, on the sibling axis.

    Each step-cadenced member is driven with two operand sets that must give DIFFERENT
    answers, and the answers are the ones the evaluating code produces: the draw-rate gate
    fires at a `gate_interval` boundary that is both the `consec`-th observation and at or
    past `min_step`; the grad-norm gate counts consecutive TRAINING steps; the actor-lag
    invariant needs the learner one step PAST its threshold.
    """
    gate = Cadence.GATE_INTERVAL_CONSEC
    assert gate.earliest_fire_step((3, 25000), period_steps=1000) == 25000.0
    assert gate.earliest_fire_step((3, 0), period_steps=1000) == 3000.0
    assert gate.earliest_fire_step((3, 10), period_steps=10) == 30.0
    assert gate.earliest_fire_step((3, 25001), period_steps=1000) == 26000.0, (
        "min_step floors the FIRE, and the fire can only land on a boundary — so a min_step "
        "that is not a multiple of the interval rounds UP to the next boundary"
    )
    assert Cadence.CONSEC_TRAIN_STEPS.earliest_fire_step((3,), period_steps=1) == 3.0
    assert Cadence.CONSEC_TRAIN_STEPS.earliest_fire_step((7,), period_steps=1) == 7.0
    assert Cadence.STEP_LAG_THRESHOLD.earliest_fire_step((100,), period_steps=1) == 101.0
    assert Cadence.STEP_LAG_THRESHOLD.earliest_fire_step((14,), period_steps=1) == 15.0
    assert Cadence.WALL_CLOCK_POLL.earliest_fire_step((), period_steps=None) == 0.0
    assert Cadence.CLOSE_OUT_TERMINAL.earliest_fire_step((), period_steps=None) is None, (
        "a row that fires at close-out has no in-run step cadence at all; claiming one "
        "would be a fabricated number"
    )
    # R265 / ADJ-D38 — the WR member, in EVAL ROUNDS. Trigger C at run5's own shape is the
    # first satisfiable of the three (16 rounds, against B's 26 and A's 21), and the answer
    # moves with the eval cadence, which is the whole point of judging it in this clock.
    wr = Cadence.EVAL_ROUND_CONSEC
    assert wr.earliest_fire_samples((3, 15000, 25000, 2, 20000), period_steps=1000) == 16.0
    assert wr.earliest_fire_step((3, 15000, 25000, 2, 20000), period_steps=1000) == 16000.0
    assert wr.earliest_fire_step((3, 15000, 25000, 2, 20000), period_steps=100) == 15100.0, (
        "a shorter eval cadence reaches the strict `current_step > min_step` floor sooner — "
        "round 151 at period 100, not round 16 at period 1000"
    )
    assert wr.earliest_fire_samples((7, 0, 0, 9, 0), period_steps=1000) == 7.0, (
        "with every min_step at 0 the binding constraint is the SMALLEST consec across the "
        "three triggers, since the abort fires on whichever is first satisfiable"
    )
    assert wr.earliest_fire_samples((0, 0, 0, 0, 0), period_steps=1000) == 1.0, (
        "consec 0 does NOT fire before the first eval round: `if not wr_history: return "
        "None` needs one sample however weak the evidence bar is (ADJ-D38's hair-trigger "
        "observation, stated as arithmetic — 0 arms a weaker rule, it disables nothing)"
    )


def test_an_unjudgeable_operand_reads_as_UNREACHABLE_never_as_early() -> None:
    """Fail toward visibility, never toward silence — `is_armed`'s ceiling rule, restated
    on this axis. A gate that never runs must never read as a gate that fires at step 0.

    The DEGENERATE-PERIOD arms are R265's: a clock that does not advance by at least one
    training step per tick is a clock nothing is sampled on, and `math.inf` is the same
    answer R251 gave a sub-1 `gate_interval` when the interval was still an operand."""
    for operands in ((0, 25000), (None, 3)):
        assert Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
            operands, period_steps=1000) == math.inf
    for period in (0, -1):
        assert Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
            (3, 25000), period_steps=period) == math.inf, (
            "a degenerate sample period must read UNREACHABLE, never as a fast clock — and "
            "never as `nan`, which `inf * 0` would produce and every bound would accept"
        )
    assert Cadence.EVAL_ROUND_CONSEC.earliest_fire_step(
        (3, 15000, 25000, 2, 20000), period_steps=0) == math.inf
    assert Cadence.STEP_LAG_THRESHOLD.earliest_fire_step((None,), period_steps=1) == math.inf
    assert Cadence.CONSEC_TRAIN_STEPS.earliest_fire_step((True,), period_steps=1) == math.inf, (
        "`bool` on a threshold path is a type confusion, not a threshold"
    )


# ══ R265 / ADJ-D38: the sample clock ═══════════════════════════════════════════════════
def test_every_axis_names_its_own_clock_and_no_two_clocks_share_a_key() -> None:
    """The structural half of R265, and the one arm A/B-style value drives cannot see.

    The defect is not "the arithmetic is wrong" — it is "the row is judged against a key its
    axis is not sampled on", which reads perfectly healthy in every number the audit prints.
    Two properties make that unreachable: a step-clocked member's period comes from the CLOCK
    (so a row cannot name it), and no two clocks name the SAME key (so collapsing the period
    table onto one key — the mutation that would restore "every axis judged in the step
    clock" — cannot happen silently)."""
    paths = {clock: clock.period_path for clock in SampleClock
             if clock.period_path is not None}
    assert set(paths) == {SampleClock.GATE_BOUNDARY, SampleClock.EVAL_ROUND}, (
        f"exactly the two config-period clocks may name a key; got {paths}"
    )
    assert len(set(paths.values())) == len(paths), (
        f"two sample clocks share a period key {paths}: one of those axes is being judged in "
        "the other's clock, which is ADJ-D38 verbatim"
    )
    assert SampleClock.TRAIN_STEP.period_path is None
    assert SampleClock.TRAIN_STEP.is_step_clocked, (
        "a `None` period path means two different things and they must not collapse: the "
        "train-step clock's tick is DEFINITIONAL (1), a no-step-clock row has no tick at all"
    )
    assert not SampleClock.NO_STEP_CLOCK.is_step_clocked
    for member in Cadence:
        assert isinstance(member.sample_clock, SampleClock), (
            f"cadence {member.value} names no sample clock, so the audit cannot know which "
            "key its evidence arrives on"
        )


def test_an_underivable_clock_RAISES_and_never_falls_back_to_the_step_clock() -> None:
    """R265's fail-loud half, in every direction it can rot.

    A silent fallback would make "one tick is one training step" and "nobody could derive
    this row's tick" the same observable — and the fallback is the FRIENDLY-looking outcome,
    which is why it has to be a raise and not a warning."""
    absent = SimpleNamespace(train=SimpleNamespace(eval_interval=None))
    with pytest.raises(SampleClockNotDerivableError, match="eval_interval"):
        SampleClock.EVAL_ROUND.period_steps(absent, row="probe")
    with pytest.raises(SampleClockNotDerivableError, match="not a training-step clock"):
        SampleClock.NO_STEP_CLOCK.period_steps(SimpleNamespace(), row="probe")
    with pytest.raises(SampleClockNotDerivableError, match="FALLBACK"):
        Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step((3, 25000), period_steps=None)
    with pytest.raises(SampleClockNotDerivableError):
        Cadence.CLOSE_OUT_TERMINAL.earliest_fire_step((), period_steps=1000)
    with pytest.raises(SampleClockNotDerivableError):
        Cadence.WALL_CLOCK_POLL.earliest_fire_samples((), period_steps=1)
    with pytest.raises(SampleClockNotDerivableError):
        Cadence.GATE_INTERVAL_CONSEC.step_floor()
    # …and the derivation itself works, so the invariant is not simply "always raise".
    assert SampleClock.TRAIN_STEP.period_steps(SimpleNamespace(), row="probe") == 1.0
    assert SampleClock.EVAL_ROUND.period_steps(
        SimpleNamespace(train=SimpleNamespace(eval_interval=250)), row="probe") == 250.0


def test_the_verdict_publishes_the_clock_it_judged_each_row_in(run5) -> None:
    """The vacuity half of R265, on the field an operator needs to tell a green row from a
    row that was green in the wrong units. `within` is decided in the row's OWN ticks, so
    the published pair must be self-consistent: samples x period == the published step, and
    bound_samples x period == the published bound."""
    by_name = {v.row.name: v for v in audit_cadence(run5)}
    draw = by_name["draw_rate_collapse"]
    assert draw.clock is SampleClock.GATE_BOUNDARY
    assert draw.period_steps == float(run5.monitor.gate_interval)
    assert draw.earliest_samples is not None and draw.earliest_step is not None
    assert draw.earliest_samples * draw.period_steps == draw.earliest_step
    assert draw.bound_samples is not None
    assert draw.bound_samples * draw.period_steps == draw.bound
    assert draw.within is (draw.earliest_samples <= draw.bound_samples), (
        "the verdict must be the one the OWN-CLOCK comparison gives, not a step-clock "
        "comparison that happens to agree"
    )
    lag = by_name["actor_lag"]
    assert lag.clock is SampleClock.TRAIN_STEP and lag.period_steps == 1.0
    close_out = by_name["terminal_eval_broken"]
    assert close_out.clock is SampleClock.NO_STEP_CLOCK
    assert close_out.period_steps is None and close_out.bound_samples is None, (
        "a row with no step clock must publish NO period and NO tick-bound: a number there "
        "would be one nobody could have derived"
    )


# ══ the shipped config, and the defect ═════════════════════════════════════════════════
def test_the_production_config_can_fire_every_armed_row_with_margin(run5) -> None:
    """R251's sanity anchor, RE-DERIVED here rather than transcribed: the verdicts are read
    off `audit_cadence`, and each is compared to the bound the constant actually implies."""
    verdicts = audit_cadence(run5)
    assert verdicts, "no verdict means the audit judged nothing"
    by_name = {verdict.row.name: verdict for verdict in verdicts}
    assert not [v for v in verdicts if not v.within], (
        "every armed row on the shipped production config must be able to fire inside the "
        f"bound: {[(v.row.name, v.earliest_step, v.bound) for v in verdicts if not v.within]}"
    )
    bound = EARLIEST_FIRE_FRACTION * run5.train.max_train_steps
    assert by_name["draw_rate_collapse"].earliest_step == float(
        run5.train.draw_rate_abort.min_step
    ), "run5 mints min_step as a multiple of gate_interval, so the fire lands on it exactly"
    assert by_name["draw_rate_collapse"].earliest_step < bound / 4.0, (
        "the anchor must clear the bound with real margin, not squeak past it"
    )
    assert by_name["actor_lag"].earliest_step == float(
        run5.monitor.actor_lag_threshold_steps + 1
    )
    assert by_name["terminal_eval_broken"].earliest_step is None


def test_an_interval_that_outruns_the_run_is_CADENCE_DISARMED(run5) -> None:
    """ADJ-D22's own config, at the module layer. `gate_interval` stays schema-legal
    (`ge=1`) and the threshold stays armed — every check that existed before this ruling
    still reads this config as healthy."""
    vacuous = _revalidated(run5, "monitor", "gate_interval", 1_000_000_000)
    assert vacuous.train.draw_rate_abort is not None
    assert Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(
        vacuous.train.draw_rate_abort.threshold
    ), "the arming predicate must still read ARMED, or this test is about something else"

    failed = [v for v in audit_cadence(vacuous) if not v.within]
    assert [v.row.name for v in failed] == ["draw_rate_collapse"], (
        f"exactly the cadence-disarmed row must fail; got {[v.row.name for v in failed]}"
    )
    assert failed[0].earliest_step == 3_000_000_000.0, (
        "three consecutive observations at a 1e9 stride — the consec-th boundary, floored "
        f"by min_step; got {failed[0].earliest_step!r}"
    )
    assert failed[0].bound == EARLIEST_FIRE_FRACTION * vacuous.train.max_train_steps


def test_the_boundary_is_EXCEEDS_and_not_REACHES_in_both_directions(run5) -> None:
    """R251 fails a row whose earliest fire EXCEEDS the bound. Nothing else in the suite ever
    places a row EXACTLY on it, so the one thing that separates "exceeds" from "reaches" —
    `<=` against `<` — was free to move with every test green.

    It is not a hypothetical shape: run5's own `min_step: 25000` sits exactly on the bound of
    a 100000-step run, which is a perfectly ordinary thing for an operator to mint. Both
    directions are driven off ONE construction so the pair cannot drift: the run length that
    puts the row on the bound, and that same length minus one step.
    """
    on_the_bound = int(run5.train.draw_rate_abort.min_step / EARLIEST_FIRE_FRACTION)
    exact = _revalidated(run5, "train", "max_train_steps", on_the_bound)
    draw = {v.row.name: v for v in audit_cadence(exact)}["draw_rate_collapse"]
    assert draw.earliest_step == draw.bound, (
        f"this pin needs the row exactly ON the bound; got {draw.earliest_step} against "
        f"{draw.bound}"
    )
    assert draw.within, (
        "a row whose earliest fire lands exactly on the bound has not EXCEEDED it — the "
        "ruling's word is 'exceeds', and a strict comparison quietly reds a config it permits"
    )

    just_short = _revalidated(run5, "train", "max_train_steps", on_the_bound - 1)
    one_over = {v.row.name: v for v in audit_cadence(just_short)}["draw_rate_collapse"]
    assert one_over.earliest_step > one_over.bound and not one_over.within, (
        "…and one step the other side of it must fail, or 'exceeds' has become 'never' — a "
        f"boundary pinned in one direction is not pinned; got {one_over!r}"
    )


def test_a_required_row_that_declares_no_cadence_fails_toward_visibility(run5) -> None:
    """The undeclared arm. A required row nobody gave a cadence is UNJUDGEABLE, and an
    unjudgeable armed row must gate rather than pass — otherwise "nobody declared it" and
    "it is fine" become the same observable, which is MF-7's defect on a new axis."""
    row = ArmedAbort(
        name="_synthetic_uncadenced", config_path="train.terminal_eval_enabled",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.REQUIRED, exit_code=None,
        owner=None, source_pin=None, note="synthetic subject; not a shipped row.",
    )
    assert row.cadence is None
    failed = [v for v in audit_cadence(run5, manifest=(row,)) if not v.within]
    assert [v.row.name for v in failed] == [row.name]
    assert "cadence" in failed[0].detail


def test_a_DISARMED_row_is_left_to_the_arming_audit_and_never_double_judged(run5) -> None:
    """Scope. `audit_cadence` judges ARMED rows only: a disarmed row is already rc 30 from
    assertion (c), and reporting it twice under two names would make the operator chase a
    cadence question about an abort that is simply off."""
    disarmed = run5.model_dump()
    disarmed["train"]["draw_rate_abort"] = None
    judged = audit_cadence(RunConfig.model_validate(disarmed))
    assert "draw_rate_collapse" not in [verdict.row.name for verdict in judged]
    assert "actor_lag" in [verdict.row.name for verdict in judged], (
        "…and the OTHER armed rows are still judged, or the scope rule is just a skip"
    )


# ══ the comparison is live (the mutation arm) ══════════════════════════════════════════
def test_neutering_the_bound_flips_the_verdict_so_the_comparison_is_not_decoration(
    run5,
) -> None:
    """The audit's own mutation pin. If the fraction were read once and discarded, or the
    computed step ignored, a config the audit refuses at the shipped fraction would still
    be refused at `fraction=inf` — and both directions are driven so a verdict that is
    constant in EITHER direction is caught."""
    vacuous = _revalidated(run5, "monitor", "gate_interval", 1_000_000_000)
    assert [v.row.name for v in audit_cadence(vacuous) if not v.within]
    assert not [v.row.name for v in audit_cadence(vacuous, fraction=math.inf) if not v.within], (
        "at an infinite bound nothing can be cadence-disarmed — a row still failing means "
        "the bound is not read"
    )
    tightened = audit_cadence(run5, fraction=1e-12)
    assert [v.row.name for v in tightened if not v.within], (
        "at a bound below every earliest-fire step the healthy config must fail — a row "
        "still passing means the computed step is not read"
    )
