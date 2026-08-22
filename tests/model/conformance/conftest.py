"""Shared fixtures for the architecture conformance suite.

The suite's own vacuity refusals live in collected modules beside this file, NOT here:
pytest treats `conftest.py` as a plugin module and does not collect test functions from it
(`python_files = test_*.py`), so a guard written here would never run. The roster guard the
design assigns to this file therefore lands in `test_conformance_roster_guard.py`, unchanged
in mechanism; the divergence is recorded in the leaf's IMPL record rather than smoothed over.

`derived` is the pinning surface used across the suite: a counter or a cardinality is recorded
as an OUTPUT OF THE RUN and asserted non-zero, never typed as a constant in the source. The one
thing this suite pins as a source-level literal is T1's census triples, which is a different
kind of pin and is argued at its own site.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def derived(record_property):
    """Record a derived quantity on the test record. Never a threshold, never compared."""

    def _derived(name: str, value: object) -> object:
        record_property(f"conformance.{name}", repr(value))
        return value

    return _derived
