"""The ONE list of retired `GraphBatch` fields, shared by its three consumers.

It is a list rather than a derivation on purpose: derived from the dataclass alone, a mistyped
golden key would satisfy "absent from the batch" and pass, so the list is what makes the rule
"these and only these". The dataclass supplies the other half — each consumer asserts the batch
genuinely does not carry the name.
"""
from __future__ import annotations

#: Each of these was dead as a DEVICE tensor and was re-expressed onto the surface that genuinely
#: consumes it. The same-named WIRE arrays are NOT dead and are untouched: the bridge and
#: `verify_edge_geometry` read them zero-copy.
RETIRED_BATCH_FIELDS: frozenset[str] = frozenset({
    "node_coords", "legal_mask", "policy_dst_slot", "window_center", "current_player",
})
