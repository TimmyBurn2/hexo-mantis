"""RQ-16 — the dead serve-path transfers are retired PER FIELD, and the wire keeps its own.

R290(c) ordered a per-tensor R128 reachability census over production entry points; R297(c) ruled
its verdicts one field at a time. `node_coords` came back **GENUINELY DEAD** — zero reads of
`GraphBatch.node_coords` anywhere in the tree, tests included — so it is retired here.

THE DISTINCTION THIS FILE EXISTS TO PIN, because blurring it would break the engine. `node_coords`
names TWO different things:

  * the **wire array** — `GraphWirePayload.node_coords`, flat `(N, 2)` axial `(q, r)`. It is very
    much ALIVE: `graph_collate._check_semantic` reshapes it for checks 16/17, the compiled
    `verify_edge_geometry` reads the raw flat array zero-copy, and `mantis-bridge`'s assemble path
    reads `graph.node_coords` directly (`inference.rs:597-598`). The wire contract is versioned and
    is NOT touched.
  * the **device tensor** — `GraphBatch.node_coords`, built by `collate_graph_batch` with a
    `.to(device)` transfer and then read by nobody. That is the dead one, and only that one.

A census verdict of "dead" is a licence for exactly the thing measured. Retiring the array because
the tensor was dead would delete a live Rust consumer, which is the failure R290(c)'s halt existed
to prevent — so the negative control below is not decoration, it is the point.
"""
from __future__ import annotations

import dataclasses

from _retired_batch_fields import RETIRED_BATCH_FIELDS
from mantis.selfplay.graph_collate import GraphBatch, GraphWirePayload

#: Retired by R297(c) on the census's GENUINELY-DEAD verdict.
_RETIRED = "node_coords"
#: Retired in its own commit (R298(d)) after the A4 rows were re-expressed against the gather.
_RETIRED_2 = "legal_mask"

#: Still carried, and each still under its own R297(c) disposition (TEST-ONLY LAW-08 findings
#: resolved one commit per field). Listed so retiring one silently does not pass this file.
_STILL_CARRIED = ("x", "edge_index", "edge_attr", "legal_offsets",
                  "legal_node_gather", "node_offsets", "n_stones")


def _batch_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(GraphBatch)}


def test_the_dead_device_tensor_is_gone_from_the_batch():
    """Derived from the dataclass, never from the module text (R296(f))."""
    for name in RETIRED_BATCH_FIELDS:
        assert name not in _batch_fields(), f"GraphBatch still carries {name!r}"
    assert _RETIRED not in _batch_fields(), (
        f"GraphBatch still carries {_RETIRED!r}; the census found zero reads of it anywhere and "
        "the field costs an H2D transfer per part on every serve"
    )


def test_the_WIRE_still_carries_it_because_the_wire_is_not_dead():
    """The negative control, and the reason the retirement is scoped to one of two same-named
    things. `verify_edge_geometry` and the bridge's assemble path both read this array."""
    wire_fields = {f.name for f in dataclasses.fields(GraphWirePayload)}
    assert _RETIRED in wire_fields, (
        "the WIRE payload lost node_coords — the census's 'dead' verdict was about the device "
        "tensor only; the flat array has live Rust consumers and a versioned contract"
    )


def test_the_other_four_are_still_carried_and_still_owed():
    """Each of the four TEST-ONLY findings is resolved in its OWN commit (R297(c): 'one commit per
    field, never a blanket act'). This fails loudly if one is retired without its own act."""
    missing = sorted(f for f in _STILL_CARRIED if f not in _batch_fields())
    assert not missing, (
        f"{missing} left GraphBatch without their own per-field disposition; R297(c) forbids a "
        "blanket retirement and each is a LAW-08 finding with its own remedy"
    )
