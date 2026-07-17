# Contract: replay persist

- version: HEXB v9 / HEXG v1
- owner: crate mantis-selfplay (replay)
- status: v1 — filled by the replay subsystem port (WP5)

## Summary

The replay subsystem owns two disjoint on-disk ring formats and the sample-return
field-order contract:

- **grid ring → HEXB v9** (dense CNN encodings). Magic `0x48455842` ("HEXB"),
  write-version 9, versioned header, per-row layout below, wire-signature
  cross-load law.
- **graph ring → HEXG v1** (axis-graph GNN encodings). Magic `0x48455847`
  ("HEXG"), version strict 1, slot-geometry guard, two-pass atomic load.

The two magics are DISJOINT — a file of one format handed to the other loader
LOUD-rejects on the magic check (both directions). Representation is a closed
`grid | graph` enum; a graph ring refuses a grid encoding at construction and vice
versa. f16 tensors are stored as raw u16 bits with NO f16→f32→f16 round-trip on
the data path (weights are decoded to f32 only for sampling-bucket math). Save is
a native-endian pointer dump; load is an explicit little-endian decode (the format
is correct on little-endian hosts — a pre-existing, verbatim-ported property).

The registered encoding set is `{v6, v6w25, v6_live2_ls, gnn_axis_v1}`; the first
three are `grid`, the last is `graph`.

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| HEXB magic `0x48455842`, write-version 9 | `replay/persist/mod.rs` (`HEXB_MAGIC`/`HEXB_VERSION`) | O-1, O-12 |
| HEXB load reads v9/v8/v7 (shared header) + v6 (legacy, no encoding name, deprecation warning); v5 and earlier HARD-REJECT | `replay/persist/load.rs` version dispatch | O-2, O-3, O-4, O-5 |
| HEXB per-row layout: state(n_planes×n_cells u16) · chain(n_chain_planes×n_cells u16) · policy(policy_logit_count f32) · outcome f32 · game_id i64 · weight u16 · ownership(n_cells u8) · winning_line(n_cells u8) · is_full_search u8 · [v8+] position_index u16 · [v9] value_target_valid u8 | `replay/persist/load.rs` entry-byte math + `replay/persist/mod.rs` `save_to_path` | O-1, O-12, O-34 |
| all widths/strides are spec-derived (`state_stride()`/`chain_stride()`/`aux_stride()`/`policy_stride()`/`n_planes`/`n_cells()`) — no code-side v6 constant | `replay/*` (every stride read from `RegistrySpec`) | O-6, O-13 (positive width pins) |
| wire-signature cross-load: file & buffer must share `(n_planes, board_size, policy_logit_count, has_pass_slot, sym_table_id)` — the NAME may differ | `replay/persist/load.rs` (compares `wire_signature()`, not the name string) | O-8 (reject leg), O-9 (accept-on-name-mismatch leg), O-10 (unknown-name reject), O-11 (n_planes header guard) |
| grid ring = HEXB v9; graph ring = HEXG v1; magics DISJOINT → cross-format load LOUD-rejects both ways | `replay/persist/*` + `replay/hexg/persist.rs` magic checks | O-16, O-22 |
| HEXG magic `0x48455847`, version strict 1, slot-geometry `(MAX_STONES=256, MAX_VISITS=128)` guard; two-pass atomic load (parse-then-commit); game_id rebase past loaded max (`saturating_add`) | `replay/hexg/persist.rs` | O-20, O-21, O-22, O-26, O-29 |
| HEXG record round-trips byte-identically (`record_at` inverts `push_record_impl`); over-cap push LOUD; push-time validation (finite/non-negative visit prob, finite outcome, ±1 stone player) | `replay/hexg/{push.rs,mod.rs}` | O-17, O-18, O-19, O-28 |
| HEXG rebuild-at-sample: per sampled record, D6-rotate stones + visit keys, rebuild via `build_axis_graph` (stamps `builder_impl = 1`), align to legal nodes, mass-drop guard | `replay/hexg/sample.rs` | O-24, O-25, O-27, O-30 |
| f16 stored as raw u16 bits; no f16→f32→f16 on the data path (NaN/subnormal/−0/max-normal survive) | `replay/{push.rs, sample.rs, persist/*}` | O-34 |
| 12-fold D6 augmentation tables + weight schedule | `replay/sym.rs`, `replay/schedule.rs` | O-13, O-14, O-15, O-31, O-32, O-33 |
| sample-return positional order (grid) is versioned `SAMPLE_ORDER_V1` (8) / `SAMPLE_WITH_POS_ORDER_V1` (9); the 9-form SPLICES position_indices at index 7 before the trailing value_target_valid | `replay/sample.rs` order consts + carrier-derived type tags | O-35 |

## Pinning tests

The gating oracle bank is O-1..O-35 (WP5 DESIGN §b). New homes:

- `crates/mantis-selfplay/tests/replay_hexb_roundtrip.rs` — O-1..O-8, O-10, O-11, O-12, O-34a.
- `crates/mantis-selfplay/src/replay/persist/load.rs` (`#[cfg(test)]`) — O-9 (the
  accept-on-name-mismatch witness; needs the test-only `with_encoding` ctor).
- `crates/mantis-selfplay/src/replay/sym.rs` (`#[cfg(test)]`) — O-13, O-14, O-15,
  and the positive per-encoding policy-width pins.
- `crates/mantis-selfplay/src/replay/schedule.rs` (`#[cfg(test)]`) — O-31
  (bracket lookup + uniform).
- `crates/mantis-selfplay/tests/replay_hexg.rs` — O-16..O-30.
- `crates/mantis-selfplay/tests/replay_sample_aux.rs` — O-31 (seeded distribution), O-32, O-33, O-34b.
- `crates/mantis-selfplay/tests/replay_tuple_order.rs` — O-35 (⊕, written first).
