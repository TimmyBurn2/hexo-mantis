"""The PER-AXIS mutation battery on the fireability audit itself.

`tests/config/test_armed_abort_cadence.py` pins what the audit computes; this file pins that it
BITES, per axis, in that axis's own sample clock. Every drive has one shape: take the real
production config, make ONE armed row unfireable IN ITS OWN CLOCK, assert the audit reds for
THAT row and stays green for the others, then put the key back and assert green. Per-axis rather
than per-config, because the defect being closed is one axis's verdict computed from another
axis's key. The WR axis (the EVAL-ROUND clock) left this battery with the sealbot rung;
the draw-rate axis on the GATE-BOUNDARY clock is what remains.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from mantis.config.armed_aborts import (
    MANIFEST,
    ArmedAbort,
    SampleClock,
    Status,
    audit_cadence,
)
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from test_armed_abort_cadence import _revalidated

_PRODUCTION = production_configs(Path(__file__).resolve().parents[2])


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


@pytest.mark.parametrize("production", _PRODUCTION, ids=lambda p: p.name)
def test_the_battery_baseline_is_green_or_every_kill_below_is_meaningless(production: Path) -> None:
    """Every mutation below claims "this key alone reds this row alone", which needs a green
    start and needs the draw-rate row to actually BE judged in its own clock."""
    manifest = _required_manifest()
    armed = load_config(production)
    judged = _judged(armed, manifest)
    assert "draw_rate_collapse" in judged, (
        "the draw-rate row must reach the cadence audit, or this file has no subject"
    )
    assert judged["draw_rate_collapse"].clock is SampleClock.GATE_BOUNDARY, (
        "…and it must be judged in the GATE-BOUNDARY clock (R265 / ADJ-D38); got "
        f"{judged['draw_rate_collapse'].clock}"
    )
    assert _out_of_bound(armed, manifest) == [], (
        f"the baseline must be green; got {[(n, v.detail) for n, v in judged.items()]}"
    )


@pytest.mark.parametrize(
    "label,section,key,value,expected",
    [
        # The draw-rate axis's own clock: gate boundaries, this module's measured config.
        ("gate_interval outruns the run", "monitor", "gate_interval", 1_000_000_000,
         "draw_rate_collapse"),
        # The BOUND rather than a cadence key.
        ("run too short for the draw-rate min_step", "train", "max_train_steps", 80_000,
         "draw_rate_collapse"),
        # The actor-lag axis: the train-step clock, its threshold past the bound.
        ("actor-lag threshold past the bound", "monitor", "actor_lag_threshold_steps",
         900_000, "actor_lag"),
    ],
)
@pytest.mark.parametrize("production", _PRODUCTION, ids=lambda p: p.name)
def test_ONE_key_reds_ONE_axis_in_that_axis_own_clock(
    label: str, section: str, key: str, value: object, expected: str, production: Path,
) -> None:
    """The kill table, as code: each row makes exactly one axis unfireable and asserts the audit
    names THAT axis and no other — the property an all-rows assertion cannot give.
    `train.max_train_steps` moves the BOUND instead of a cadence key.
    """
    manifest = _required_manifest()
    armed = load_config(production)
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


def test_the_audit_itself_RAISES_when_a_rows_clock_cannot_be_derived() -> None:
    """The fail-loud path, driven through `audit_cadence` rather than the clock in isolation. A
    duck-typed config is used because the schema's `ge=1` makes an underivable period
    unreachable through the loader; the audit must raise, not answer with a one-step tick."""
    from types import SimpleNamespace

    from mantis.config.armed_aborts import SampleClockNotDerivableError

    row = next(r for r in _required_manifest() if r.name == "draw_rate_collapse")
    shaped = SimpleNamespace(
        monitor=SimpleNamespace(gate_interval=1_000),
        train=SimpleNamespace(
            max_train_steps=1_000_000,
            draw_rate_abort=SimpleNamespace(threshold=0.25, consec=3, min_step=25_000, N_pool_min=50),
        ),
    )
    judged = audit_cadence(shaped, manifest=(row,))
    assert [v.row.name for v in judged] == ["draw_rate_collapse"] and judged[0].within, (
        "premise: with a derivable period the audit judges the row normally"
    )
    shaped.monitor.gate_interval = None
    with pytest.raises(SampleClockNotDerivableError, match="gate_interval"):
        audit_cadence(shaped, manifest=(row,))
