# Contract: replay persist

- version: HEXG v2
- owner: crate mantis-selfplay (replay)
- status: v2 — the HEXB dense ring was deleted with the grid path (R346(f)); `archive/grid-path`
  carries it, and the HEXB rows of this contract went with it.

## Summary

The replay subsystem owns ONE on-disk ring format:

- **graph ring → HEXG v2** (axis-graph GNN encodings). Magic `0x48455847`
  ("HEXG"), version strict 2, slot-geometry guard, two-pass atomic load. v2 (R347(a))
  added the per-record tail mass alpha of the SPARSE Gumbel row, written after `weight`;
  a v1 file is REFUSED BY NAME and never re-parsed, because from that offset on its bytes
  are self-consistent under both readings and only the version field can tell them apart.
  There is no in-place upgrade — a v1 ring is regenerated.

f16 tensors are stored as raw u16 bits with NO f16->f32->f16 round-trip on the data path
(weights are decoded to f32 only for sampling-bucket math). Save is a native-endian pointer
dump; load is an explicit little-endian decode (the format is correct on little-endian hosts —
a pre-existing, verbatim-ported property).

The registered encoding set is `{gnn_axis_v1, gnn_axis_r8}`; both are `graph`.

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| HEXG magic `0x48455847`, version strict 2, slot-geometry guard: `MAX_STONES=256` fixed; the header's `max_visits` is the buffer's COMPOSED `visit_capacity` (R255/ADJ-D34 under `search.kind: puct`, derived from the sims regime; R347(a) under `gumbel`, the minted `gumbel_m` — reject on mismatch either way: a file written under a different regime is a different record geometry); two-pass atomic load (parse-then-commit); game_id rebase past loaded max (`saturating_add`) | `replay/hexg/persist.rs` | O-20, O-21, O-22, O-26, O-29, `persist_roundtrips_a_non_default_capacity_and_rejects_mismatch` |
| HEXG v1 is REFUSED by name and leaves the buffer byte-identical (the two-pass load's atomicity, exercised on a real format break) | `replay/hexg/persist.rs` version check | `the_v1_byte_golden_is_refused_by_name_and_leaves_the_buffer_untouched` |
| HEXG carries the R347(a) sparse row: per-record `tail_mass` alpha, refused at insert unless it is a probability, and an over-m explicit-entry count refused at insert before any slot is touched | `replay/hexg/{push.rs,persist.rs}` | `a_sparse_row_over_the_minted_m_is_refused_at_insert`, `a_tail_mass_that_is_not_a_probability_is_refused_at_insert` |
| HEXG record round-trips byte-identically (`record_at` inverts `push_record_impl`); over-cap push LOUD; push-time validation (finite/non-negative visit prob, finite outcome, ±1 stone player) | `replay/hexg/{push.rs,mod.rs}` | O-17, O-18, O-19, O-28 |
| HEXG rebuild-at-sample: per sampled record, D6-rotate stones + visit keys, rebuild via `build_axis_graph` (stamps `builder_impl = 1`), align to legal nodes, mass-drop guard | `replay/hexg/sample.rs` | O-24, O-25, O-27, O-30 |
| f16 stored as raw u16 bits; no f16->f32->f16 on the data path (NaN/subnormal/-0/max-normal survive) | `replay/hexg/*` | O-34 |
| the D6 axial rotation primitive + the weight schedule | `replay/sym.rs`, `replay/schedule.rs` | O-13, O-31 |

## Pinning tests

The gating oracle bank is O-1..O-35 (WP5 DESIGN §b). O-1..O-12, O-32..O-35 were the HEXB rows
and are RETIRED with the format. Live homes:

- `crates/mantis-selfplay/src/replay/sym.rs` (`#[cfg(test)]`) — O-13.
- `crates/mantis-selfplay/src/replay/schedule.rs` (`#[cfg(test)]`) — O-31
  (bracket lookup + uniform).
- `crates/mantis-selfplay/tests/replay_hexg.rs` — O-16..O-30.
