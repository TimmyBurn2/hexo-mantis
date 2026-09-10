"""The PER-AXIS mutation battery on the fireability audit itself.

`tests/config/test_armed_abort_cadence.py` pins what the audit computes; this file pins that it
BITES, per axis, in that axis's own sample clock. Every drive has one shape: take the real
production config, make ONE armed row unfireable IN ITS OWN CLOCK, assert the audit reds for
THAT row and stays green for the others, then put the key back and assert green. Per-axis rather
than per-config, because the defect being closed is one axis's verdict computed from another
axis's key. The WR row ships DEFERRED, so every WR drive flips it REQUIRED and arms it IN
MEMORY: nothing on disk moves, and no armed VALUE moves.
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
RUN5 = REPO_ROOT / "configs" / "run6.yaml"

#: Well past the old `WR_HISTORY_DEPTH = 5`: armed in config, permanently unfireable in effect
#: before the ring fix. A test INPUT.
_ABOVE_OLD_WR_DEPTH = 9


@pytest.fixture(scope="module")
def run5() -> RunConfig:
    return load_config(RUN5)


def _revalidated(config: RunConfig, section: str, key: str, value: object) -> RunConfig:
    """`dump -> mutate ONE key -> model_validate`, the loader's own final step, so every
    cross-field validator re-runs and every mutation below is one a run could be launched from."""
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
    """The shipped manifest with every DEFERRED row flipped REQUIRED, derived from `MANIFEST`
    rather than re-typed: a transcribed copy would stop being the shipped rows the first time one
    moved. `owner` is dropped because `__post_init__` forbids it on a REQUIRED row."""
    return tuple(
        dataclasses.replace(row, status=Status.REQUIRED, owner=None)
        if row.status is Status.DEFERRED else row
        for row in MANIFEST
    )


def _out_of_bound(config: RunConfig, manifest: tuple[ArmedAbort, ...]) -> list[str]:
    return [v.row.name for v in audit_cadence(config, manifest=manifest) if not v.within]


def _judged(config: RunConfig, manifest: tuple[ArmedAbort, ...]) -> dict:
    return {v.row.name: v for v in audit_cadence(config, manifest=manifest)}


def test_the_battery_baseline_is_green_or_every_kill_below_is_meaningless() -> None:
    """Every mutation below claims "this key alone reds this row alone", which needs a green
    start and needs the WR row to actually BE judged — a skipped row makes them vacuous."""
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


@pytest.mark.parametrize(
    "label,section,key,value,expected",
    [
        # The draw-rate axis's own clock: gate boundaries. ADJ-D22's measured config.
        ("gate_interval outruns the run", "monitor", "gate_interval", 1_000_000_000,
         "draw_rate_collapse"),
        # The BOUND rather than a cadence key, sitting BETWEEN the two axes' earliest fires.
        ("run too short for the draw-rate min_step", "train", "max_train_steps", 80_000,
         "draw_rate_collapse"),
        # The WR axis's own clock: eval rounds — invisible to a step-clock audit.
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
    """The kill table, as code: each row makes exactly one axis unfireable and asserts the audit
    names THAT axis and no other — the property an all-rows assertion cannot give.
    `train.max_train_steps` moves the BOUND instead of a cadence key, and its value lands BETWEEN
    the two step answers so even the bound mutation names ONE row.
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
    """With no manifest row the WR axis was invisible to the gate in both directions. Now the
    published number MOVES with the consec, and a consec outrunning the run's eval budget is OUT
    OF BOUND by name. The min_steps are zeroed so the CONSEC binds, not the step floors."""
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


def test_the_WR_axis_audits_GREEN_when_judged_in_the_GATE_clock() -> None:
    """The false affirmative, measured: the row below is what the previous machinery would have
    produced for this axis — same arithmetic, same manifest, judged on the GATE-BOUNDARY clock.
    On a config whose eval cadence outruns the run it reports the row fireable WITH A CONCRETE
    NUMBER, while the correct row refuses the same config. Hence the period lives on the CLOCK."""
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
    """The fail-loud path, driven through `audit_cadence` rather than the clock in isolation. A
    duck-typed config is used because the schema's `ge=1` makes an underivable period
    unreachable through the loader; the audit must raise, not answer with a one-step tick."""
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
