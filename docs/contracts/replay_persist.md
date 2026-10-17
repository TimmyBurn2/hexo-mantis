# Contract: replay persist

- version: HEXG v5
- owner: crate mantis-selfplay (replay)
- status: v5 (DT-1) — each record carries the proven-tail flag, set on a decided tail's row and never
  on a full-search one; v4 (AUX-1) added its game's next placements, the future-occupancy field, and v3
  (RING-V3, R382(d)) the search's root value and its flag. A v2, v3 or v4 ring still loads, every row
  default-filled to no root value (v2), no future field (v2, v3) and no flag (v2, v3, v4). The HEXB dense
  ring was deleted with the grid path (R346(f)); `archive/grid-path` carries it, and the HEXB rows went with it.

## Summary

The replay subsystem owns ONE on-disk ring format:

- **graph ring → HEXG v5** (axis-graph GNN encodings). Magic `0x48455847`
  ("HEXG"), version 5 written and 2, 3, 4 or 5 read, slot-geometry guard, two-pass atomic load,
  and a payload that does not end exactly at its last record refused. v2 (R347(a))
  added the per-record tail mass alpha of the SPARSE Gumbel row, written after `weight`;
  a v1 file is REFUSED BY NAME and never re-parsed, because from that offset on its bytes
  are self-consistent under both readings and only the version field can tell them apart.
  There is no in-place upgrade — a v1 ring is regenerated. v3 (R382(d)) appends to each
  record's fixed head, after `tail_mass`, `root_value: f32` (the search's value of the root
  in the row-mover's frame, in [-1, 1]: Σ π′·completedQ under a completed-Q search since
  RUN11-PRE (R384(a)), W/N under PUCT and on every ring written before it, so a ring's
  semantics are read off its writer's commit, not its version, and a search weight > 0
  trains only on rings written since; a proven root carries the proof's value) and
  `root_value_valid: u8`; an invalid row stores exactly `+0.0`. The version picks the record
  layout: a v2 file LOADS, every row `root_value_valid = 0`, `root_value = +0.0`, so its
  rows are the rows it was — re-saved as v3 and stripped of the two fields, it is the v2
  file byte for byte. A v3 payload stamped 2 (or the reverse) is refused by the over-cap and
  trailing-bytes guards unless its garbage realigns exactly; the version field is the authority.
  v4 (AUX-1) adds the future-occupancy field: `future_valid: u8` and `n_future: u8` after
  `root_value_valid`, and `n_future × (q:i16, r:i16, p:i8)` after the visits — the stones the
  game places from the row's ply on, `FUTURE_HORIZON` (20) of them, fewer only where the game
  ended (then `(ply + n).div_ceil(2) == game_length`; at 20, `≤`), never on a stone the row holds,
  never a cell twice, each stone's player the side whose turn it is (the row's `current_player` for
  its `moves_remaining` stones, then two per turn alternating); a row without the field stores flag 0, count 0. A v2 or v3 file
  loads every row without it, and its v4 re-save stripped of the two head bytes and the stones is
  the v3 file byte for byte. The sampler labels each legal node of the rebuilt graph
  `FUTURE_EMPTY`/`FUTURE_MINE`/`FUTURE_THEIRS` in the row-mover's frame, the stones rotated by the
  row's own D6 element; a stone outside the reader's legal set labels nothing.
  v5 (DT-1) adds the proven-tail flag: `proven_tail: u8` after `n_future`, 1 on a row written at a
  ply of a decided tail (the game ran out from a proven win with no search), a row that never trains
  a policy, so a flag on a row whose `is_full_search` is set is refused at insert and at load, and a
  byte over 1 at load. A v2, v3 or v4 file loads every row unflagged, and its v5 re-save stripped of
  the byte is the v4 file byte for byte. The flag reaches no sampled output: the trainer treats a
  flagged row as the no-target row it is, and a ring reader reads the flag off the saved ring
  (`mantis.diagnostics.ring_reader`). The drain row carries it in its `(root_value, root_value_valid,
  future_stones, proven_tail)` group, forwarded by keyword like the rest; a tool that copies ring rows
  into another ring carries it, and one that builds rows no game produced pushes none.

f16 tensors are stored as raw u16 bits with NO f16->f32->f16 round-trip on the data path
(weights are decoded to f32 only for sampling-bucket math). Save is a native-endian pointer
dump; load is an explicit little-endian decode (the format is correct on little-endian hosts —
a pre-existing, verbatim-ported property).

The registered encoding set is `{gnn_axis_v1, gnn_axis_r8, gnn_axis_r8_pruned}`; all are `graph`. A record is stones and a
visit map over the legal cells, so a ring loads into a buffer whose row differs from its writer's at most in the edges
the builder emits, and its rows rebuild under the READER's edge set (ENCODE-1).

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| HEXG magic `0x48455847`, version 5 written and 2, 3, 4 or 5 read (any other refused by name), slot-geometry guard: `MAX_STONES=256` fixed; the header's `max_visits` is the buffer's COMPOSED `visit_capacity` (R255/ADJ-D34 under `search.kind: puct`, derived from the sims regime; R347(a) under `gumbel`, the minted `gumbel_m` — reject on mismatch either way: a file written under a different regime is a different record geometry); two-pass atomic load (parse-then-commit); game_id rebase past loaded max (`saturating_add`) | `replay/hexg/persist.rs` | O-20, O-22, O-26, O-29, `the_v2_byte_golden_loads_with_no_root_value_and_its_v5_resave_carries_its_rows_byte_for_byte` (O-21's successor), `persist_roundtrips_a_non_default_capacity_and_rejects_mismatch` |
| HEXG v1 is REFUSED by name and leaves the buffer byte-identical (the two-pass load's atomicity, exercised on a real format break) | `replay/hexg/persist.rs` version check | `the_v1_byte_golden_is_refused_by_name_and_leaves_the_buffer_untouched` |
| HEXG carries the R347(a) sparse row: per-record `tail_mass` alpha, refused at insert unless it is a probability, and an over-m explicit-entry count refused at insert before any slot is touched | `replay/hexg/{push.rs,persist.rs}` | `a_sparse_row_over_the_minted_m_is_refused_at_insert`, `a_tail_mass_that_is_not_a_probability_is_refused_at_insert` |
| HEXG v3 carries the root value and its flag: refused at insert and at load unless a valid value is finite in [-1, 1] and an invalid one is `+0.0`; a v2 ring loads default-filled and its v3 re-save carries its rows byte for byte; the v3 golden re-saves byte-identical; a payload read through the other version's layout, or with bytes past its last record, is refused and touches nothing; the field rides each sampled row's `GraphTargets` unrotated | `replay/hexg/{mod.rs,push.rs,persist.rs,sample.rs}` | `the_v2_byte_golden_loads_with_no_root_value_and_its_v5_resave_carries_its_rows_byte_for_byte`, `the_v3_byte_golden_loads_with_no_future_field_and_its_v5_resave_carries_its_rows_byte_for_byte`, `a_payload_read_through_another_versions_layout_is_refused_and_leaves_the_buffer_untouched`, `a_payload_with_bytes_past_its_last_record_is_refused`, `a_root_value_that_is_not_a_search_value_is_refused_at_insert`, `a_v3_root_field_outside_its_contract_is_refused_at_load`, `the_sampled_targets_carry_each_rows_root_value_beside_its_outcome`; Python `tests/bridge/test_hexg_root_value.py` (the v2 sample golden's arrays byte-equal) and `tests/diagnostics/test_ring_reader.py` (the pure-numpy reader's two layouts) |
| HEXG v4 carries the future-occupancy field: refused at insert and at load unless it holds at most 20 stones, each stone's player in turn order from the row's `(current_player, moves_remaining)`, no cell twice, none on a stone the row holds, and a count `game_length` agrees with (fewer than 20 only where the game ended); a v3 ring loads with no field and its v4 re-save carries its rows byte for byte; the v4 golden re-saves byte-identical; the v3 and v4 goldens sample byte-equal to the v2 tree; each legal node's label follows its cell under every D6 element | `replay/hexg/{mod.rs,push.rs,persist.rs,sample.rs}` | `the_v3_byte_golden_loads_with_no_future_field_and_its_v5_resave_carries_its_rows_byte_for_byte`, `the_v4_byte_golden_loads_with_no_proven_tail_and_its_v5_resave_carries_its_rows_byte_for_byte`, `crates/mantis-selfplay/tests/replay_hexg_future.rs` (insert and load refusals, the round trip, the label exact and under every element); Python `tests/bridge/test_hexg_future.py` and `tests/diagnostics/test_ring_reader.py` (the reader's v4 layout) |
| HEXG v5 carries the proven-tail flag: refused at insert and at load on a row whose `is_full_search` is set, and at load as a byte over 1; a v4 ring loads unflagged and its v5 re-save carries its rows byte for byte; the v5 golden re-saves byte-identical and, stripped of its flags, is the v4 golden; the v4 and v5 goldens sample byte-equal to the v2 tree; the flag reaches no sampled output and rides the saved ring, the drain row's `(root_value, root_value_valid, future_stones, proven_tail)` group through `push_graph` and the `ReplayFacade`, and every ring-to-ring copy (`tools/ring_future.py`, `tools/run_monitor`, `tools/value_instrument`, the distill control ring) | `replay/hexg/{mod.rs,push.rs,persist.rs}`, `mantis-bridge/src/{hexg,runner}.rs`, `selfplay/{pool_push.py,buffers.py}` | `the_v4_byte_golden_loads_with_no_proven_tail_and_its_v5_resave_carries_its_rows_byte_for_byte`, `the_v5_byte_golden_loads_exact_and_resaves_byte_identical`, `a_v5_proven_tail_outside_its_contract_is_refused_at_load`, `a_proven_tail_row_claiming_a_full_search_is_refused_at_insert`, `a_proven_tail_row_round_trips_through_save_and_load`, `a_payload_read_through_another_versions_layout_is_refused_and_leaves_the_buffer_untouched`; Python `tests/bridge/test_hexg_proven_tail.py`, `tests/diagnostics/test_ring_reader.py` (the reader's v5 layout), `tests/selfplay/test_drain_row_shape_parity.py` and `tests/selfplay/test_pool_drain_parity.py::test_graph_drain_push_rows` |
| The self-play producer writes the root value on every searched row, quick arm included: its search's value in the side-to-move frame — Σ π′·completedQ under a completed-Q search (π′ the exported improved policy before any audit edit), W/N under PUCT — bit-equal to the 1-in-N `search_stats` entry's `search_value` (a cross-check, not a source) and within 1e-4 of REG-1's 0a rebuilt from that entry alone, or +1 on a decided root (a finish, the owed proof stone, a new proof) and -1 on a root lost on cover; the drain row carries it in its `(root_value, root_value_valid, …)` group before the game id, and `push_graph` and the `ReplayFacade` forward it by keyword; a row no search produced (the BC corpus, tools) pushes none | `runner/{search_drive.rs,record.rs}`, `mantis-bridge/src/runner.rs`, `selfplay/{pool_push.py,buffers.py}` | `crates/mantis-selfplay/tests/root_value_rows.rs` (both kinds, and the proof's sign read off the outcome), `tests/selfplay/test_search_stats_end_to_end.py::test_every_row_the_pool_pushes_carries_its_searchs_root_value`, `tests/selfplay/test_drain_row_shape_parity.py`, `tests/selfplay/test_pool_drain_parity.py::test_graph_drain_push_rows` |
| The self-play producer writes the future-occupancy field on every row of a finalized game, the cap's game included: its game's next ≤ 20 placements from `move_history`, each stone's player read off the final board (`records::future_stones`); the drain row carries it in the `(root_value, root_value_valid, future_stones, proven_tail)` group before the game id, and `push_graph`, the `ReplayFacade` and `tools/run_monitor`'s ring copy forward it by keyword; `tools/ring_future.py` gives an existing ring's rows theirs from the run's game records through the same derivation (`_engine.game_future_stones`), a ring game settled whole by the one record its rows agree on and unsettled (no field) otherwise; `mantis.diagnostics.ring_audit` reads every field against its game's next row in the ring (`future_disagree`, a band reads 0) | `runner/finalize.rs`, `records.rs`, `mantis-bridge/src/{runner,hexg}.rs`, `selfplay/{pool_push.py,buffers.py}`, `tools/ring_future.py`, `diagnostics/ring_audit.py` | `crates/mantis-selfplay/tests/future_occupancy_rows.rs` (a win and the cap, against a replayed board; the planted off-by-one), `tests/selfplay/test_future_occupancy_end_to_end.py` (a real pool against its game records), `tests/diagnostics/test_ring_audit_future.py` (the planted contradictions), `tests/tools/test_ring_future.py` (settled by game, the planted contradicting record), `tests/selfplay/test_drain_row_shape_parity.py` |
| HEXG record round-trips byte-identically (`record_at` inverts `push_record_impl`); over-cap push LOUD; push-time validation (finite/non-negative visit prob, finite outcome, ±1 stone player) | `replay/hexg/{push.rs,mod.rs}` | O-17, O-18, O-19, O-28 |
| HEXG rebuild-at-sample: per sampled record, D6-rotate stones + visit keys, rebuild via `build_axis_graph` (stamps `builder_impl = 1`) under the buffer's edge set, align to legal nodes, mass-drop guard | `replay/hexg/sample.rs` | O-24, O-25, O-27, O-30 |
| a ring's writer encoding is named in its header; it loads into a buffer whose row differs from the writer's at most in `empty_edges` (`RegistrySpec::reads_rings_of`), and any other difference, or an unregistered writer, refuses (`HEXG encoding mismatch`) | `replay/hexg/persist.rs`, `mantis-encoding` `spec/mod.rs` | `a_ring_crosses_the_edge_set_and_nothing_else`, `rings_cross_between_rows_only_across_the_edge_set` |
| f16 stored as raw u16 bits; no f16->f32->f16 on the data path (NaN/subnormal/-0/max-normal survive) | `replay/hexg/*` | O-34 |
| the D6 axial rotation primitive; the per-slot weight column is stored at 1.0 and read by no sampler (its schedule left at HYGIENE-1) | `replay/sym.rs`, `replay/hexg/push.rs` | O-13 |

## Pinning tests

The gating oracle bank is O-1..O-35 (WP5 DESIGN §b). O-1..O-12, O-32..O-35 were the HEXB rows
and are RETIRED with the format. Live homes:

- `crates/mantis-selfplay/src/replay/sym.rs` (`#[cfg(test)]`) — O-13.
- `crates/mantis-selfplay/tests/replay_hexg.rs` — O-16..O-30.
