"""The suite refuses its OWN vacuity: an empty or shrunken parametrisation roster FAILS.

Every tier in `tests/model/conformance/` is parametrised over `mantis.encoding.all_specs()`
(`src/mantis/encoding/registry.py:64`), a `dict.values()` view over a process-global
`_REGISTRY_CACHE` built at `:41-45` that this suite does not own. `pyproject.toml`'s
`[tool.pytest.ini_options]` sets no `empty_parameter_set_mark`, so pytest's default applies and
an empty `argvalues` collects one SKIPPED item — silently. The refusal discipline ("no tier may
skip") is written as a constraint on tier code and never reached the parametrisation machinery,
which is where the skip actually comes from.

THIS GUARD IS STRICTLY STRONGER THAN THE GLOBAL SETTING WOULD BE, and the global setting was
weighed and refused on measured grounds (237 `parametrize` decorators, 145 with computed
argvalues, any of which would become a COLLECTION error — tier-fatal — instead of a skip).
`fail_at_collect` catches only `len == 0`; this catches a roster that has silently SHRUNK,
which is the likelier regression.

RESIDUE, STATED (R297(b)): the guard compares the roster captured at collection against the
same public surface read at run time. A session-mate that patches `_REGISTRY_CACHE` BEFORE
collection shrinks both sides together and is not caught here; the collected-test-count gate is
the backstop for that, and naming a backstop is not the same as having a control.
"""
from __future__ import annotations

import pytest

import mantis.encoding as encoding

from _corpus import RosterCollapsed, check_roster, roster, roster_names

#: THE ROSTER AS THE TIERS WERE PARAMETRISED OVER IT, captured at module import — which is
#: collection time, the same pass in which every `@pytest.mark.parametrize(..., roster(), ...)`
#: in this suite is evaluated. This is the guard's whole mechanism: the comparison below reads
#: the live registry surface AGAIN at run time and compares it against this capture, so the two
#: sides have two different times of observation. Reading `roster()` inside the test instead
#: gives two live calls one line apart, which shrink together under any registry change and can
#: therefore report only `len == 0` — exactly what `fail_at_collect` already reports.
ROSTER_AT_COLLECTION: tuple[str, ...] = roster_names(roster())


def test_the_parametrisation_roster_is_non_empty_and_matches_the_live_registry(derived):
    """The guard itself. Its cardinality is a derived output of the run, never a typed number."""
    observed = ROSTER_AT_COLLECTION
    live = tuple(sorted(s.name for s in encoding.all_specs()))
    cardinality = check_roster(observed, live)
    derived("roster.names", observed)
    derived("roster.cardinality", cardinality)
    assert cardinality > 0


def test_an_EMPTY_roster_is_refused_rather_than_skipped():
    """PC-1, the half that matters: empty must FAIL, and it must fail by name."""
    live = roster_names(roster())
    with pytest.raises(RosterCollapsed, match="EMPTY"):
        check_roster((), live)


def test_a_SHRUNKEN_roster_is_refused():
    """Three encodings where the registry has four is not empty, and is the likelier regression."""
    live = roster_names(roster())
    assert len(live) > 1, "a one-member registry cannot exercise the shrink control"
    with pytest.raises(RosterCollapsed, match="differs from the live registry surface"):
        check_roster(live[:-1], live)


def test_the_COLLECTION_TIME_capture_is_what_the_guard_compares_against():
    """The shrink half is only reachable because the two sides are observed at two times.

    A registry emptied or shrunk AFTER collection leaves the capture intact and moves the live
    read alone; this asserts the capture is the gate's left-hand side rather than a second live
    call, by driving the helper with the capture against a live read that has lost a member.
    """
    assert ROSTER_AT_COLLECTION == roster_names(roster())
    shrunk_live = ROSTER_AT_COLLECTION[:-1]
    with pytest.raises(RosterCollapsed, match="differs from the live registry surface"):
        check_roster(ROSTER_AT_COLLECTION, shrunk_live)


def test_the_guard_does_NOT_fire_on_the_real_roster():
    """Negative control. A guard that fires on the live registry is measuring nothing."""
    live = roster_names(roster())
    assert check_roster(live, live) == len(live)
