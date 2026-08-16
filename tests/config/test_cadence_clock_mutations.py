"""R265 / ADJ-D38 — the PER-AXIS mutation battery on the fireability audit itself.

`tests/config/test_armed_abort_cadence.py` pins what the audit computes. This file pins that
the audit BITES, per axis, in the axis's own sample clock — LAW-07's "every gate input cites
a live producer AND a mutation self-test", applied to the thing R265 changed.

The shape of every drive is the same and it is the shape the ruling asks for: take the real
production config, make ONE armed row unfireable IN ITS OWN CLOCK, assert the audit reds for
THAT row and stays green for the others, then put the key back and assert green. A per-axis
battery rather than a per-config one, because the defect R265 closes is precisely that one
axis's verdict was being computed from another axis's key — an all-rows-at-once assertion
cannot see that.

THE LOAD-BEARING DRIVE is `test_the_WR_axis_audits_GREEN_when_judged_in_the_GATE_clock`: it
builds the row R251 would have produced for the sealbot-WR axis — same arithmetic, same
manifest machinery, the GATE-BOUNDARY clock — and measures it GREEN on the very config the
correct row refuses. That is ADJ-D38's "worse than D36 on the audit side" as a number, and it
is the reason the clock had to move onto the axis instead of onto the row.

The WR row ships DEFERRED (operator ruling G-3 mints `wr_hard_abort_enabled` false on every
production config, and a CI gate may not overrule a pre-registered value), so every WR drive
here flips it REQUIRED and arms it IN MEMORY — the §8.5 one-field data edit, the same way
`test_armed_abort_manifest.py::test_flipping_the_deferred_row_to_required_needs_no_code_change`
drives its own subject. Nothing on disk moves; no armed VALUE moves.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from mantis.config.armed_aborts import (
    MANIFEST,
    ArmedAbort,
    Cadence,
    SampleClock,
    Status,
    audit_cadence,
)
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN5 = REPO_ROOT / "configs" / "run5.yaml"

#: Well past the old `WR_HISTORY_DEPTH = 5` — the exact region ADJ-D38 names as armed in the
#: config and permanently unfireable in effect before the ring fix. A test INPUT.
_ABOVE_OLD_WR_DEPTH = 9


@pytest.fixture(scope="module")
def run5() -> RunConfig:
    return load_config(RUN5)


def _revalidated(config: RunConfig, section: str, key: str, value: object) -> RunConfig:
    """`dump -> mutate ONE key -> model_validate` — the loader's own final step, so every
    cross-field validator re-runs and every mutation below is one a run could be launched
    from. A synthetic config built any other way would prove nothing about a real mint."""
    raw = config.model_dump()
    raw[section][key] = value
    return RunConfig.model_validate(raw)


def _armed_wr(config: RunConfig, **monitor_overrides: object) -> RunConfig:
    """`run5` with the sealbot-WR abort ARMED in memory, plus any monitor deltas."""
    raw = config.model_dump()
    raw["monitor"]["wr_hard_abort_enabled"] = True
    raw["monitor"].update(monitor_overrides)
    return RunConfig.model_validate(raw)


def _required_manifest() -> tuple[ArmedAbort, ...]:
    """The shipped manifest with every DEFERRED row flipped REQUIRED — the §8.5 data edit.

    Derived from `MANIFEST`, never a re-typed row set: a transcribed copy would stop being
    the shipped rows the first time one moved, and this battery would then be auditing a
    manifest nobody ships. `owner` is dropped because `__post_init__` forbids it on a
    REQUIRED row; `source_pin` is kept, which a REQUIRED row may carry (N-1 / R73).
    """
    return tuple(
        dataclasses.replace(row, status=Status.REQUIRED, owner=None)
        if row.status is Status.DEFERRED else row
        for row in MANIFEST
    )


def _out_of_bound(config: RunConfig, manifest: tuple[ArmedAbort, ...]) -> list[str]:
    return [v.row.name for v in audit_cadence(config, manifest=manifest) if not v.within]


def _judged(config: RunConfig, manifest: tuple[ArmedAbort, ...]) -> dict:
    return {v.row.name: v for v in audit_cadence(config, manifest=manifest)}


# ══ the baseline: with every row REQUIRED and armed, run5 is GREEN ═════════════════════
def test_the_battery_baseline_is_green_or_every_kill_below_is_meaningless() -> None:
    """Every mutation below claims "this key alone reds this row alone". That claim needs a
    green start, and it needs the WR row to actually BE judged — a row the audit skips
    (disarmed, or still deferred) would make every drive here vacuously green."""
    manifest = _required_manifest()
    armed = _armed_wr(load_config(RUN5))
    judged = _judged(armed, manifest)
    assert "sealbot_wr_abort" in judged, (
        "the flipped WR row must reach the cadence audit, or this file has no subject — a "
        "row that is DEFERRED or DISARMED is skipped and every kill below reads green"
    )
    assert judged["sealbot_wr_abort"].clock is SampleClock.EVAL_ROUND, (
        "…and it must be judged in the EVAL-ROUND clock, which is the whole ruling; got "
        f"{judged['sealbot_wr_abort'].clock}"
    )
    assert _out_of_bound(armed, manifest) == [], (
        f"the baseline must be green; got {[(n, v.detail) for n, v in judged.items()]}"
    )


# ══ per-axis kills: one key, one row, in that row's own clock ══════════════════════════
@pytest.mark.parametrize(
    "label,section,key,value,expected",
    [
        # The draw-rate axis's own clock: gate boundaries. ADJ-D22's measured config.
        ("gate_interval outruns the run", "monitor", "gate_interval", 1_000_000_000,
         "draw_rate_collapse"),
        # The BOUND rather than a cadence key — a run short enough that the draw-rate row's
        # own min_step no longer fits inside it, chosen to sit BETWEEN the two axes'
        # earliest fires so it reds one and not the other.
        ("run too short for the draw-rate min_step", "train", "max_train_steps", 80_000,
         "draw_rate_collapse"),
        # The WR axis's own clock: eval rounds. THE R265 CASE — this key is invisible to a
        # step-clock audit, and `monitor.gate_interval` says nothing whatever about it.
        ("eval_interval outruns the run", "train", "eval_interval", 1_000_000_000,
         "sealbot_wr_abort"),
        # The actor-lag axis: the train-step clock, its threshold past the bound.
        ("actor-lag threshold past the bound", "monitor", "actor_lag_threshold_steps",
         900_000, "actor_lag"),
    ],
)
def test_ONE_key_reds_ONE_axis_in_that_axis_own_clock(
    label: str, section: str, key: str, value: object, expected: str,
) -> None:
    """The kill table, as code. Each row makes exactly one axis unfireable and asserts the
    audit names THAT axis and no other — which is the property a single all-rows assertion
    cannot give, and the property the D38 defect violated in the quietest possible way.

    `train.max_train_steps` is the odd one out and is here deliberately: it moves the BOUND
    rather than a cadence key, so it proves the comparison is live from the other side. Its
    value is chosen to land BETWEEN the two step answers — the draw-rate row's 25000-step
    min_step no longer fits in an 80000-step run's quarter (20000) while the WR row's 16
    eval rounds (16000 steps) still does — so even the bound mutation names ONE row.
    """
    manifest = _required_manifest()
    armed = _armed_wr(load_config(RUN5))
    assert _out_of_bound(armed, manifest) == [], "premise: the unmutated config is green"
    mutated = _revalidated(armed, section, key, value)
    failed = _out_of_bound(mutated, manifest)
    assert failed == [expected], (
        f"{label}: expected exactly {expected!r} to go out of bound in its own sample "
        f"clock; got {failed}. A key that reds the WRONG row means an axis is being judged "
        "against a cadence it does not tick on (R265 / ADJ-D38)"
    )
    # …and putting it back is green again, so the kill is the KEY and not the mutation ritual.
    assert _out_of_bound(_revalidated(mutated, section, key,
                                      getattr(getattr(armed, section), key)), manifest) == []


def test_a_WR_consec_past_the_old_ring_depth_is_VISIBLE_to_the_audit() -> None:
    """The specific ADJ-D38 case, on the audit side.

    Before this ruling the WR axis had NO manifest row, so a consec past the old ring depth
    was invisible to gate 12 in both directions — it could not be reported unfireable and it
    could not be reported fireable either. Now the published number MOVES with the consec
    (so the operand is read), and a consec absurd enough to outrun the run's own eval budget
    is OUT OF BOUND by name.

    The min_steps are zeroed so the CONSEC is what binds; at run5's minted min_steps the
    step floors dominate and this drive would be measuring those instead.
    """
    manifest = _required_manifest()
    base = _armed_wr(load_config(RUN5), wr_early_death_min_step=0, wr_collapse_min_step=0,
                     wr_rolling_min_step=0)
    deeper = _revalidated(
        _revalidated(base, "monitor", "wr_collapse_consecutive_evals", _ABOVE_OLD_WR_DEPTH),
        "monitor", "wr_rolling_consecutive_evals", _ABOVE_OLD_WR_DEPTH + 1)
    verdict = _judged(deeper, manifest)["sealbot_wr_abort"]
    assert verdict.earliest_samples == float(_ABOVE_OLD_WR_DEPTH), (
        "the published earliest fire must be the minted consec in EVAL ROUNDS — a value "
        "the pre-D38 ring could never deliver and the pre-D38 audit could not even name; "
        f"got {verdict.earliest_samples!r}"
    )
    assert verdict.within, (
        "…and at run5's eval budget nine rounds is comfortably inside the bound, so this "
        "consec is armed AND fireable AND audited, which is the state ADJ-D38 asks for"
    )
    absurd = _revalidated(base, "monitor", "wr_collapse_consecutive_evals", 10_000_000)
    absurd = _revalidated(absurd, "monitor", "wr_rolling_consecutive_evals", 10_000_000)
    assert _out_of_bound(absurd, manifest) == ["sealbot_wr_abort"], (
        "a consec beyond the run's whole eval budget must be OUT OF BOUND by name: it is "
        "armed in the config and unfireable in the run, which is what this audit refuses"
    )


# ══ the false affirmative R265 exists to kill ══════════════════════════════════════════
def test_the_WR_axis_audits_GREEN_when_judged_in_the_GATE_clock() -> None:
    """ADJ-D38's "worse than D36 on the audit side", measured rather than argued.

    The row below is the one R251's machinery would have produced for this axis: the same
    `Cadence` arithmetic, the same manifest, the same audit — judged on the GATE-BOUNDARY
    clock, because that is the only step-cadence key a pre-R265 row had to reach for. On a
    config whose eval cadence outruns the run three orders of magnitude it reports the row
    fireable, WITH A CONCRETE NUMBER, while the correct row refuses the same config.

    That difference is the whole reason the period moved onto the CLOCK: an author could not
    have got this wrong on purpose, and nothing in the old machinery would have told them.
    """
    vacuous = _revalidated(_armed_wr(load_config(RUN5)),
                           "train", "eval_interval", 1_000_000_000)
    correct = next(row for row in _required_manifest() if row.name == "sealbot_wr_abort")
    wrong_clock = dataclasses.replace(
        correct,
        cadence=Cadence.GATE_INTERVAL_CONSEC,
        cadence_paths=("monitor.wr_collapse_consecutive_evals",
                       "monitor.wr_early_death_min_step"),
    )
    assert wrong_clock.cadence.sample_clock is SampleClock.GATE_BOUNDARY
    assert correct.cadence is not None
    assert correct.cadence.sample_clock is SampleClock.EVAL_ROUND

    judged_wrong = _judged(vacuous, (wrong_clock,))["sealbot_wr_abort"]
    assert judged_wrong.within and judged_wrong.earliest_step is not None, (
        "premise: the step-clock row must read this config as FIREABLE — if it did not, "
        "there would have been no false affirmative to close"
    )
    judged_right = _judged(vacuous, (correct,))["sealbot_wr_abort"]
    assert not judged_right.within, (
        "the EVAL-ROUND row must REFUSE the same config: the axis ticks on "
        f"train.eval_interval and this one delivers no rounds. Got {judged_right.detail}"
    )
    assert judged_wrong.clock is not judged_right.clock, (
        "the two verdicts must differ by the CLOCK and nothing else — same row, same "
        "arithmetic family, same bound; only the key the period came from moved"
    )


def test_the_audit_itself_RAISES_when_a_rows_clock_cannot_be_derived() -> None:
    """The fail-loud path, driven through `audit_cadence` rather than through the clock in
    isolation — a raise that never reaches the audit closes nothing.

    A duck-typed config is used deliberately (`audit_cadence` takes `Any`): the schema's
    `ge=1` on `train.eval_interval` makes an underivable period unreachable through the
    loader, so the only honest drive is one that supplies the shape directly. What must NOT
    happen is the audit answering anyway with a one-step tick — that answer is friendlier,
    which is exactly why it has to be a raise.
    """
    from types import SimpleNamespace

    from mantis.config.armed_aborts import SampleClockNotDerivableError

    wr_row = next(row for row in _required_manifest() if row.name == "sealbot_wr_abort")
    shaped = SimpleNamespace(
        monitor=SimpleNamespace(
            wr_hard_abort_enabled=True, wr_collapse_consecutive_evals=3,
            wr_early_death_min_step=0, wr_collapse_min_step=0,
            wr_rolling_consecutive_evals=2, wr_rolling_min_step=0),
        train=SimpleNamespace(max_train_steps=1_000_000, eval_interval=1_000),
    )
    judged = audit_cadence(shaped, manifest=(wr_row,))
    assert [v.row.name for v in judged] == ["sealbot_wr_abort"] and judged[0].within, (
        "premise: with a derivable period the audit judges the row normally"
    )
    shaped.train.eval_interval = None
    with pytest.raises(SampleClockNotDerivableError, match="eval_interval"):
        audit_cadence(shaped, manifest=(wr_row,))
