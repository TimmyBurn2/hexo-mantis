# Contract: replay persist

- version: HEXG v3
- owner: crate mantis-selfplay (replay)
- status: v3 (RING-V3, R382(d)) — each record carries the search's root value and its flag; a v2
  ring still loads, every row default-filled to no root value. The HEXB dense ring was deleted
  with the grid path (R346(f)); `archive/grid-path` carries it, and the HEXB rows went with it.

## Summary

The replay subsystem owns ONE on-disk ring format:

- **graph ring → HEXG v3** (axis-graph GNN encodings). Magic `0x48455847`
  ("HEXG"), version 3 written and 2 or 3 read, slot-geometry guard, two-pass atomic load,
  and a payload that does not end exactly at its last record refused. v2 (R347(a))
  added the per-record tail mass alpha of the SPARSE Gumbel row, written after `weight`;
  a v1 file is REFUSED BY NAME and never re-parsed, because from that offset on its bytes
  are self-consistent under both readings and only the version field can tell them apart.
  There is no in-place upgrade — a v1 ring is regenerated. v3 (R382(d)) appends to each
  record's fixed head, after `tail_mass`, `root_value: f32` (the search's backed-up root
  value in the row-mover's frame, in [-1, 1]; a proven root carries the proof's value) and
  `root_value_valid: u8`; an invalid row stores exactly `+0.0`. The version picks the record
  layout: a v2 file LOADS, every row `root_value_valid = 0`, `root_value = +0.0`, so its
  rows are the rows it was — re-saved as v3 and stripped of the two fields, it is the v2
  file byte for byte. A v3 payload stamped 2 (or the reverse) is refused by the over-cap and
  trailing-bytes guards unless its garbage realigns exactly; the version field is the authority.

f16 tensors are stored as raw u16 bits with NO f16->f32->f16 round-trip on the data path
(weights are decoded to f32 only for sampling-bucket math). Save is a native-endian pointer
dump; load is an explicit little-endian decode (the format is correct on little-endian hosts —
a pre-existing, verbatim-ported property).

The registered encoding set is `{gnn_axis_v1, gnn_axis_r8}`; both are `graph`.

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| HEXG magic `0x48455847`, version 3 written and 2 or 3 read (any other refused by name), slot-geometry guard: `MAX_STONES=256` fixed; the header's `max_visits` is the buffer's COMPOSED `visit_capacity` (R255/ADJ-D34 under `search.kind: puct`, derived from the sims regime; R347(a) under `gumbel`, the minted `gumbel_m` — reject on mismatch either way: a file written under a different regime is a different record geometry); two-pass atomic load (parse-then-commit); game_id rebase past loaded max (`saturating_add`) | `replay/hexg/persist.rs` | O-20, O-22, O-26, O-29, `the_v2_byte_golden_loads_with_no_root_value_and_its_v3_resave_carries_its_rows_byte_for_byte` (O-21's successor), `persist_roundtrips_a_non_default_capacity_and_rejects_mismatch` |
| HEXG v1 is REFUSED by name and leaves the buffer byte-identical (the two-pass load's atomicity, exercised on a real format break) | `replay/hexg/persist.rs` version check | `the_v1_byte_golden_is_refused_by_name_and_leaves_the_buffer_untouched` |
| HEXG carries the R347(a) sparse row: per-record `tail_mass` alpha, refused at insert unless it is a probability, and an over-m explicit-entry count refused at insert before any slot is touched | `replay/hexg/{push.rs,persist.rs}` | `a_sparse_row_over_the_minted_m_is_refused_at_insert`, `a_tail_mass_that_is_not_a_probability_is_refused_at_insert` |
| HEXG v3 carries the root value and its flag: refused at insert and at load unless a valid value is finite in [-1, 1] and an invalid one is `+0.0`; a v2 ring loads default-filled and its v3 re-save carries its rows byte for byte; the v3 golden re-saves byte-identical; a payload read through the other version's layout, or with bytes past its last record, is refused and touches nothing; the field rides each sampled row's `GraphTargets` unrotated | `replay/hexg/{mod.rs,push.rs,persist.rs,sample.rs}` | `the_v2_byte_golden_loads_with_no_root_value_and_its_v3_resave_carries_its_rows_byte_for_byte`, `the_v3_byte_golden_loads_exact_and_resaves_byte_identical`, `a_v3_payload_read_through_the_v2_layout_is_refused_and_leaves_the_buffer_untouched`, `a_payload_with_bytes_past_its_last_record_is_refused`, `a_root_value_that_is_not_a_search_value_is_refused_at_insert`, `a_v3_root_field_outside_its_contract_is_refused_at_load`, `the_sampled_targets_carry_each_rows_root_value_beside_its_outcome`; Python `tests/bridge/test_hexg_root_value.py` (the v2 sample golden's arrays byte-equal) and `tests/diagnostics/test_ring_reader.py` (the pure-numpy reader's two layouts) |
| The self-play producer writes the root value on every searched row, quick arm included: its search's W/N in the side-to-move frame (bit-equal to the 1-in-N `search_stats` root value, which stays a cross-check, not a source), or +1 on a decided root (a finish, the owed proof stone, a new proof) and -1 on a root lost on cover; the drain row carries it as one `(root_value, root_value_valid)` pair before the game id, and `push_graph` and the `ReplayFacade` forward it by keyword; a row no search produced (the BC corpus, tools) pushes none | `runner/{search_drive.rs,record.rs}`, `mantis-bridge/src/runner.rs`, `selfplay/{pool_push.py,buffers.py}` | `crates/mantis-selfplay/tests/root_value_rows.rs` (both kinds, and the proof's sign read off the outcome), `tests/selfplay/test_search_stats_end_to_end.py::test_every_row_the_pool_pushes_carries_its_searchs_root_value`, `tests/selfplay/test_drain_row_shape_parity.py`, `tests/selfplay/test_pool_drain_parity.py::test_graph_drain_push_rows` |
| HEXG record round-trips byte-identically (`record_at` inverts `push_record_impl`); over-cap push LOUD; push-time validation (finite/non-negative visit prob, finite outcome, ±1 stone player) | `replay/hexg/{push.rs,mod.rs}` | O-17, O-18, O-19, O-28 |
| HEXG rebuild-at-sample: per sampled record, D6-rotate stones + visit keys, rebuild via `build_axis_graph` (stamps `builder_impl = 1`), align to legal nodes, mass-drop guard | `replay/hexg/sample.rs` | O-24, O-25, O-27, O-30 |
| f16 stored as raw u16 bits; no f16->f32->f16 on the data path (NaN/subnormal/-0/max-normal survive) | `replay/hexg/*` | O-34 |
| the D6 axial rotation primitive; the per-slot weight column is stored at 1.0 and read by no sampler (its schedule left at HYGIENE-1) | `replay/sym.rs`, `replay/hexg/push.rs` | O-13 |

## Pinning tests

The gating oracle bank is O-1..O-35 (WP5 DESIGN §b). O-1..O-12, O-32..O-35 were the HEXB rows
and are RETIRED with the format. Live homes:

- `crates/mantis-selfplay/src/replay/sym.rs` (`#[cfg(test)]`) — O-13.
- `crates/mantis-selfplay/tests/replay_hexg.rs` — O-16..O-30.
