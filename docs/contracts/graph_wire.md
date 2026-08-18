# Contract: graph wire (ragged)

- version: v1
- owner: crate mantis-selfplay (wire types) / mantis-bridge
- status: v1 — filled by the self-play queue port (WP6); pyclass + numpy getter face ported in WP7
- WP7: the `GraphWire` pyclass + its 13 per-array numpy COPY getters + the Python single-read `take()` latch (second read → `WireAlreadyConsumed`) landed in mantis-bridge. No wire-format/version change (v1 semantics unchanged).

## Summary

`GraphWire` is the **block-diagonal ragged graph batch tensor**: the fuse-out of a
batch of per-leaf `AxisGraph`s into ONE flat-`Vec` payload the NN forward consumes
as a single disjoint-union graph. It is produced by `from_axis_graphs`
(`queues/wire.rs`, the verbatim port of the frozen `inference_bridge.rs:1207`
block-diagonal fuse), owned pure-Rust in `mantis-selfplay`; the pyo3 `#[pyclass]` +
per-field numpy `#[getter]` face is `mantis-bridge` (WP7).

Load-bearing laws: every index array (`edge_index`, `legal_node_gather`, the three
offset arrays) comes out **already globally offset** (`i64`); the offsets are
**running prefix sums**; `policy_dst_slot` is a **verbatim concat** that carries the
`-1` off-window sentinel through unchanged (**NO fixed-width fallback**, never
re-densified); the arrays are yielded **exactly once** by `take()` (a second read is
the named `WireAlreadyConsumed`); and every graph must carry the native
`builder_impl` handshake before it reaches the wire.

The 18 structural assertions the fused wire guarantees and the ≥15 named errors the
build/queue/wire path can raise are enumerated below.

## Payload shape (block-diagonal; all fields little-endian on the wire)

`GraphWire` wraps `Option<GraphWireArrays>` (the `Option` is the single-read latch);
`take()` moves out the arrays. `B = n_graphs`; `N = Σ nodes`; `E = Σ directed edges`;
`L = Σ legal-node-gather rows`.

| field | type | shape | meaning |
|---|---|---|---|
| `contract_version` | `u32` | scalar | echoed from the caller (`from_axis_graphs` arg) |
| `builder_impl` | `u8` | scalar | `BUILDER_IMPL_NATIVE` (1); the native-builder handshake stamp |
| `n_graphs` | `usize` | scalar | `B` = number of fused leaves |
| `node_feat` | `Vec<f32>` | `N × F` (flat) | verbatim concat of each graph's `node_feat` |
| `node_coords` | `Vec<i32>` | `N × 2` (flat) | verbatim concat of each graph's `node_coords` |
| `edge_index` | `Vec<i64>` | `2 × E` (flat: `[src(E) ‖ dst(E)]`) | globally-offset src then dst; reshape to `(2, E)` |
| `edge_attr` | `Vec<f32>` | `E × A` (flat) | verbatim concat of each graph's `edge_attr` |
| `node_offsets` | `Vec<i64>` | `B + 1` | running prefix sums of per-graph node counts; `[0] == 0`, `[B] == N` |
| `edge_offsets` | `Vec<i64>` | `B + 1` | running prefix sums of per-graph directed-edge counts; `[0] == 0`, `[B] == E` |
| `legal_offsets` | `Vec<i64>` | `B + 1` | running prefix sums of per-graph legal-gather lengths; `[0] == 0`, `[B] == L` |
| `legal_node_gather` | `Vec<i64>` | `L` | each row = local row `+ node_off` (globally offset) |
| `policy_dst_slot` | `Vec<i32>` | `L` | verbatim concat of per-graph `policy_scatter_index`, `-1` off-window sentinel kept |
| `n_nodes_checksum` | `Vec<u32>` | `B` | per-graph node-count checksum, in order |
| `n_stones` | `Vec<u16>` | `B` | per-graph stone count, in order |
| `window_center` | `Vec<i32>` | `B × 2` (flat `[q,r]`) | per-graph window centre, in order |
| `current_player` | `Vec<i8>` | `B` | per-graph side to move (∈ {−1, +1}), in order |

### The 18 structural assertions

Offsets/indexing (1–7): (1) `node_offsets`/`edge_offsets`/`legal_offsets` are running
prefix sums with `[0] == 0` and length `B + 1`; (2) `edge_index.len() == 2·E` laid out
`[src_global (E) ‖ dst_global (E)]`; (3) each edge `src` = local `+ node_off`; (4) each
edge `dst` = local `+ node_off`; (5) `legal_node_gather` global = local `+ node_off`;
(6) `policy_dst_slot` = verbatim concat of each graph's `policy_scatter_index`,
including the `-1` off-window sentinel, with **NO fixed-width fallback**; (7) single
graph ⇒ `node_off` stays `0` ⇒ local == global (the degenerate O-24 case).

Single-read + handshake (8–9): (8) `take()` yields every array exactly once; a second
call returns `WireAlreadyConsumed`; (9) `builder_impl == BUILDER_IMPL_NATIVE(1)` is
asserted per graph before the wire (die-loud handshake).

Per-graph metadata travels in order (10–18): (10) `n_graphs == graphs.len()`;
(11) `n_nodes_checksum`, (12) `n_stones`, (13) `window_center` (2 per graph, `[q,r]`),
(14) `current_player` (each ∈ {−1, +1}) travel per graph in fuse order;
(15) `node_feat`, (16) `node_coords`, (17) `edge_attr` are verbatim per-graph concats;
(18) `contract_version` is echoed unchanged.

## Named errors

The single-read guard is a typed error; the build/queue reasons are `String`s that
**travel to the failed waiter** (D6 — the build-side reason is preserved, not
`.ok()`-swallowed; the dense path stays reason-free by design). Producer-side reasons
(9–15) are the frozen catalogue the WP7 NN producer emits through the same `String`
channel (`submit_graph_results` / `fail_remaining`).

Wire (typed):
1. `WireAlreadyConsumed` — second `take()`; `Display` = "GraphWire arrays already consumed (single-read take() called twice)".

Build-side (`build_leaf_graph`, reasons now travel — D6):
2. `graph request: current_player {n} out of range (expected +1 / -1)`.
3. `graph request: moves_remaining {n} out of range 0..=255 (narrowing-cast guard, WP1 Attack-4)`.
4. `graph request: stone coord ({q},{r}) exceeds |coord| < i32::MAX-radius (WP1 Attack-2 silent-wrap guard)`.
5. `graph request: stone player {p} out of range (expected +1 / -1)`.
6. `graph request: non-native builder_impl (NonNativeSampleBuilder handshake)` — build-side handshake.

Consumer-side structural (`graph_collate.py::_check_structural`, the Python resolver):
0. `GatherNotStrictlyIncreasing` (check 13) — `legal_node_gather` is not strictly ascending.
   Added under R284's P-MASK design: the gather is the contract ORDER of every per-legal-node
   output, while a boolean-mask gather (`emb[legal_mask]`) returns rows in ascending ROW INDEX.
   The two coincide exactly while this holds, so an out-of-order gather silently mispairs priors
   to cells — for the mask formulation as well as the index one. Checks 9 and 11 do not cover
   it (which graph a row points into; slot aliasing) and nothing else did.

Consumer-side (`GraphQueue::submit_graph_and_wait`):
7. `graph batcher is closed` — closed before submit.
8. `submit_graph_and_wait: non-native builder_impl (NonNativeSampleBuilder handshake)` — pre-pass handshake.
9. `graph batcher closed while request was waiting` — closed mid-wait.

Producer-side reasons that travel (the frozen catalogue; WP7 NN face):
10. `legal_offsets segment [{start},{end}] out of range for id {id}` — bad segment range.
11. `submit_graph_inference_results: segment len {a} != n_legal {b} for id {id}` — segment-length desync.
12. assemble error from `assemble_ls_from_gnn_probs` — travels verbatim.
13. `next_graph_batch: non-native builder_impl on a queued graph` — defense-in-depth handshake at pop.
14. `batch_size must be > 0` — producer batch guard.
15. `length mismatch ids/values: {a}/{b}` — producer submit guard.

Plus any caller-supplied `fail_remaining(ids, reason)` string (e.g. whole-batch death),
which wakes every still-pending waiter with that reason so none is orphaned.

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| block-diagonal offsets are running prefix sums (`node/edge/legal_offsets`) | `queues/wire.rs::from_axis_graphs` (frozen `:1257-1260`) | queue_fuse_pin.rs |
| `edge_index` = `[src_global (E) ‖ dst_global (E)]`, each = local + `node_off` | `queues/wire.rs::from_axis_graphs` (frozen `:1238-1243`) | queue_fuse_pin.rs |
| `legal_node_gather` global = local + `node_off` | `queues/wire.rs::from_axis_graphs` (frozen `:1244-1246`) | queue_fuse_pin.rs |
| `legal_node_gather` is STRICTLY ASCENDING over the whole fused wire — the ORDER every per-legal-node output is paired by (`policy_dst_slot[i]`, `segment_softmax` segment `i`, `assemble_ls_from_gnn_probs` position `i`) | producer: `mantis_graph::build` (`n_stones + j`, ascending per graph) × `queues/wire.rs::from_axis_graphs` (non-decreasing `node_off`). CONSUMER-side check 13 `GatherNotStrictlyIncreasing` in `graph_collate.py::_check_structural` | tests/selfplay/test_graph_collate_gather_order.py |
| `policy_dst_slot` verbatim concat incl. `-1` off-window sentinel; NO fixed-width fallback | `queues/wire.rs::from_axis_graphs` (frozen `:1247`) | queue_fuse_pin.rs |
| single graph ⇒ `node_off == 0` ⇒ local == global (O-24 degenerate) | `queues/wire.rs::from_axis_graphs` | queue_fuse_pin.rs |
| single-read: `take()` yields all arrays exactly once; second read = `WireAlreadyConsumed` | `queues/wire.rs::take` | queue_fuse_pin.rs |
| per-graph metadata (`n_stones`/`window_center`/`current_player`/`n_nodes_checksum`) travels in fuse order | `queues/wire.rs::from_axis_graphs` | queue_fuse_pin.rs |
| corrupt one `node_offsets` entry ⇒ reconstruction fires (LAW-07 mutation self-test) | queue_fuse_pin.rs (bite proof) | queue_fuse_pin.rs |
| `builder_impl == BUILDER_IMPL_NATIVE(1)` handshake per graph (die-loud) | `queues/graph.rs::build_leaf_graph` + `submit_graph_and_wait` | queue_roundtrip.rs |
| build-side range/narrowing guards (`current_player`/`moves_remaining`/stone coord/stone player) → named `Err` | `queues/graph.rs::build_leaf_graph` | queue_roundtrip.rs |
| graph-build failure reason TRAVELS to the failed waiter (D6 fix; not `.ok()`-swallowed) | `queues/graph.rs` (build) → `fail_remaining` / `submit_graph_and_wait` | queue_roundtrip.rs |
| inference-failure / segment-range / desync / assemble reasons travel verbatim; `fail_remaining` orphans none | `queues/graph.rs::{submit_graph_results, fail_remaining}` | queue_roundtrip.rs |
| single-read delivery: a second `submit_graph_results` for a consumed id is a no-op | `queues/graph.rs::submit_graph_results` | queue_roundtrip.rs |
| cross-queue: the graph batcher never touches the dense pool (disjoint `Inner`) | `queues/graph.rs` + `queues/dense.rs` (frozen `:1415`) | queue_roundtrip.rs |
| F-19: exactly one native build per leaf (build count == leaves; a redundant build fails) | `queues/graph.rs::build_leaf_graph` | queue_roundtrip.rs, graph_build_bench.rs |

## Pinning tests

The gating tests live under `crates/mantis-selfplay/tests/`:

- `queue_fuse_pin.rs` (P-09 ⊕, written first) — the multi-graph block-diagonal fuse
  round-trip: offsets = frozen prefix sums, `edge_index`/`legal_node_gather` = local +
  `node_off`, `policy_dst_slot` verbatim concat incl. the `-1` sentinel, the
  single-graph degenerate case, the `take()` single-read guard (second read =
  `WireAlreadyConsumed`), and the LAW-07 corrupt-`node_offsets` mutation self-test.
  Inputs are the 3 dispatcher-frozen fuse-INPUT `AxisGraph`s (the old fuse is
  `pub(crate)`-unreachable, so P-09 pins the INPUTS and RECONSTRUCTS).
- `queue_roundtrip.rs` (P-07 / P-08) — the dense + graph queue round-trips: submit →
  mock pop → submit results → single-read take; the D6 graph reason-travels
  (inference-failure, `fail_remaining`, and the build-side reason) with no orphaned
  waiter; the native `builder_impl` handshake rejection; the disjoint-pool invariant;
  and the F-19 one-native-build-per-leaf structural assertion.
- `graph_build_bench.rs` (`crates/mantis-selfplay/benches/`) — carries the F-19
  build-INVOCATION-COUNT structural assertion alongside the Metric (1) build median.
