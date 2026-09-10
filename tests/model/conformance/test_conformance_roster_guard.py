"""The suite refuses its OWN vacuity: an empty or shrunken parametrisation roster FAILS.

Every conformance tier is parametrised over the encoding registry, and pytest collects an empty
`argvalues` as one SKIPPED item, silently. The global `empty_parameter_set_mark=fail_at_collect`
was weighed and refused — it would turn 145 computed-argvalue parametrisations into tier-fatal
collection errors — and it catches only `len == 0`, where this also catches a roster that has
silently SHRUNK.

RESIDUE: a session-mate that patches the registry cache BEFORE collection shrinks both sides
together and is not caught here.
"""
from __future__ import annotations

import pytest

import mantis.encoding as encoding

from _corpus import RosterCollapsed, check_roster, roster, roster_names

#: The roster as the tiers were parametrised over it, captured at module import — collection
#: time. This is the guard's whole mechanism: the test reads the live surface AGAIN at run time,
#: so the two sides are observed at two different times. Two live calls one line apart shrink
#: together under any registry change and could report only `len == 0`.
ROSTER_AT_COLLECTION: tuple[str, ...] = roster_names(roster())


def test_the_parametrisation_roster_is_non_empty_and_matches_the_live_registry(derived):
    """The cardinality is a derived output of the run, never a typed number."""
    observed = ROSTER_AT_COLLECTION
    live = tuple(sorted(s.name for s in encoding.all_specs()))
    cardinality = check_roster(observed, live)
    derived("roster.names", observed)
    derived("roster.cardinality", cardinality)
    assert cardinality > 0


def test_an_EMPTY_roster_is_refused_rather_than_skipped():
    """Empty must FAIL, and it must fail by name."""
    live = roster_names(roster())
    with pytest.raises(RosterCollapsed, match="EMPTY"):
        check_roster((), live)


def test_a_SHRUNKEN_roster_is_refused():
    """A roster short of the registry is not empty, and is the likelier regression."""
    live = roster_names(roster())
    assert len(live) > 1, "a one-member registry cannot exercise the shrink control"
    with pytest.raises(RosterCollapsed, match="differs from the live registry surface"):
        check_roster(live[:-1], live)


def test_the_COLLECTION_TIME_capture_is_what_the_guard_compares_against():
    """The shrink half is only reachable because the two sides are observed at two times: a
    registry shrunk AFTER collection leaves the capture intact and moves the live read alone."""
    assert ROSTER_AT_COLLECTION == roster_names(roster())
    shrunk_live = ROSTER_AT_COLLECTION[:-1]
    with pytest.raises(RosterCollapsed, match="differs from the live registry surface"):
        check_roster(ROSTER_AT_COLLECTION, shrunk_live)


def test_the_guard_does_NOT_fire_on_the_real_roster():
    """A guard that fires on the live registry is measuring nothing."""
    live = roster_names(roster())
    assert check_roster(live, live) == len(live)
