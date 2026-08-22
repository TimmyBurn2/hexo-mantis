"""A wrong-kind payload handed to the wire adapter is refused by name, not by AttributeError.

`wire, targets = buf.sample_graph_batch(...)` returns a two-tuple whose halves transpose
easily, and `graph_wire_from_rust` read its attributes directly off an `Any`-typed argument.
Every wrong kind therefore produced a bare `AttributeError: 'X' object has no attribute
'contract_version'` — a message that names neither the function, nor the argument, nor what a
graph wire is. The refusal is derived from the payload's own field list, so a field added to
the wire extends it without a second list to keep in step.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    WireSurfaceIncomplete,
    _WIRE_SURFACE,
    graph_wire_from_rust,
)


def _payload() -> GraphWirePayload:
    ones = np.zeros(1, dtype=np.int64)
    return GraphWirePayload(
        contract_version=1,
        builder_impl=0,
        n_graphs=1,
        **{name: ones for name in _WIRE_SURFACE[3:]},
    )


@pytest.mark.parametrize(
    "wrong", [None, {}, 7, "wire", object()], ids=["none", "dict", "int", "str", "object"]
)
def test_a_WRONG_KIND_payload_is_refused_by_name(wrong):
    with pytest.raises(WireSurfaceIncomplete, match="does not carry the GraphWire surface"):
        graph_wire_from_rust(wrong)


def test_a_payload_MISSING_ONE_FIELD_names_that_field():
    class Partial:
        pass

    partial = Partial()
    for name in _WIRE_SURFACE[:-1]:
        setattr(partial, name, np.zeros(1, dtype=np.int64))
    with pytest.raises(WireSurfaceIncomplete, match=_WIRE_SURFACE[-1]):
        graph_wire_from_rust(partial)


def test_the_refusal_does_NOT_fire_on_a_COMPLETE_wire():
    """Negative control. The surface is derived from the payload's fields, so a guard that
    rejected the payload itself would be rejecting the thing it was derived from."""
    assert graph_wire_from_rust(_payload()).contract_version == 1


def test_the_surface_is_DERIVED_from_the_payload_fields_not_transcribed():
    """A hand-kept list drifts the moment a field is added; this asserts there is no list."""
    import dataclasses

    assert _WIRE_SURFACE == tuple(f.name for f in dataclasses.fields(GraphWirePayload))
    assert len(_WIRE_SURFACE) > 1
