"""The ONE list of `GraphBatch` fields retired by the RQ-16 per-tensor census (R297(c)).

WHY A MODULE FOR ONE NAME. Three consumers need it — the retirement oracle, the byte-parity
capture check, and the ADV expectations check — and each of them would otherwise carry its own
copy. Two copies of a "these and only these" list is the duplicate-authority class this repo keeps
paying for; the third copy is where it stops being noticed. Follows the sibling-helper convention
already used by `_fused_graph_harness.py`.

WHY A LIST AT ALL, RATHER THAN DERIVING IT. The rule the captures need is *"this golden key is
absent from the batch BECAUSE the field was retired"*. Derived purely from the dataclass, a
mistyped golden key would satisfy the same condition and pass — so the list is what makes it
"these and only these" instead of "anything missing is fine". The dataclass supplies the other
half: each consumer asserts the batch genuinely does NOT carry the name.
"""
from __future__ import annotations

#: RQ-16 / R297(c). `node_coords` was GENUINELY DEAD as a DEVICE tensor — zero reads of
#: `GraphBatch.node_coords` anywhere in the tree, tests included — so it cost an H2D transfer per
#: part that nothing read. The same-named WIRE array is NOT dead and is untouched: the compiled
#: `verify_edge_geometry` reads the raw flat array zero-copy and the bridge's assemble path reads
#: `graph.node_coords` directly. Two same-named things; only one was measured, so only one moved.
#:
#: The other four census findings (`legal_mask`, `policy_dst_slot`, `window_center`,
#: `current_player`) are TEST-ONLY LAW-08 findings, NOT dead, and R297(c) resolves them one commit
#: per field. They do not belong here until their own act lands.
#: RQ-16 / R297(c), own commit. `legal_mask` was a TEST-ONLY LAW-08 finding: a local SCATTER of
#: `legal_node_gather` that no production path read. Two prose claims said otherwise and both were
#: false — `GraphBatch`'s own docstring named it as `forward_batch`'s input (repointed to
#: `legal_index` by R284/P-MASK), and the A4 file said `train_step_from_graph_batch` consumed it
#: (that function never mentioned it). The A4 agreement rows are re-expressed against the gather
#: and assert MORE than before: the scatter's implicit uniqueness check is now explicit.
RETIRED_BATCH_FIELDS: frozenset[str] = frozenset({"node_coords", "legal_mask"})
