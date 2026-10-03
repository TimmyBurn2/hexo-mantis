"""The collate stages every device array of a part in ONE host block that never aliases the wire."""
from __future__ import annotations

from typing import Any

import pytest
import torch
from _wire_geometry import geometry_kwargs

from mantis.selfplay.graph_collate import GraphWirePayload, collate_graph_batch

GEOMETRY: dict[str, int] = geometry_kwargs()
_DEVICE_FIELDS = ("x", "edge_index", "edge_attr", "legal_offsets", "legal_node_gather", "node_offsets",
                  "n_stones")


def _collate(fields: dict[str, Any], device: str):
    return collate_graph_batch(GraphWirePayload(**fields), device=device, semantic="full", **GEOMETRY)


@pytest.mark.parametrize("name", ["b6", "b1"])
def test_every_device_array_is_a_view_of_one_host_block(payload_fields, name):
    """Seven per-array stagings cost the server more than the bytes they move: one block, one pack."""
    batch = _collate(payload_fields(name), "cpu")
    blocks = {getattr(batch, f).untyped_storage().data_ptr() for f in _DEVICE_FIELDS}
    assert len(blocks) == 1, f"{len(blocks)} host blocks for the seven device arrays"


def test_the_block_does_not_alias_the_wire(payload_fields):
    """The pack COPIES: a later write to the wire must not reach a collated tensor."""
    fields = payload_fields("b6")
    batch = _collate(fields, "cpu")
    before = batch.x.clone()
    fields["node_feat"][:] = -5.0
    assert torch.equal(batch.x, before)
