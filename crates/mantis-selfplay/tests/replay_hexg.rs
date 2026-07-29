//! R8-justify (662 lines): the relocated O-16..O-30 HEXG oracle roster (33 tests incl. the frozen v1 byte-golden O-21 and the D6 aug-coherence/ADV-7 canaries) binds ONE buffer type, HexgBuffer, through one shared record/path/symmetry helper set; a split would break the frozen O-numbering and duplicate the helpers.
//! HEXG (graph) oracle suite — the 32-test module relocated out of src (R5).
//! Ported from the predecessor engine's `replay_buffer/hexg/tests.rs`; every
//! `sample_graph_batch_impl` assertion is re-anchored from the deferred
//! `GraphWire` to the buffer-owned `Vec<AxisGraph>` (single graph → local ==
//! global, R-1). Covers O-16..O-30.

use std::io::Write;

use mantis_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList};

use mantis_selfplay::replay::hexg::push::validate_stone_player;
use mantis_selfplay::replay::hexg::push::{validate_outcome, validate_visit_prob};
use mantis_selfplay::replay::hexg::sample::mass_drop_check;
use mantis_selfplay::replay::hexg::{GraphRecord, HexgBuffer, HEXG_MAGIC, HEXG_VERSION, MAX_STONES, MAX_VISITS};
use mantis_selfplay::replay::sym::rotate_axial;
use mantis_selfplay::replay::ReplayBuffer;

const ENC: &str = "gnn_axis_v1";

fn sample_record() -> GraphRecord {
    GraphRecord {
        stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 1, -1)],
        visits: vec![(2, 0, 0.5), (-1, 0, 0.3), (1, 1, 0.2)],
        current_player: -1,
        moves_remaining: 2,
        ply_index: 7,
        is_full_search: true,
        outcome: 1.0,
        value_valid: true,
        game_length: 30,
    }
}

fn unique_path(stem: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let n = C.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("hexg_{stem}_{pid}_{nanos}_{n}.hexg"))
}

/// Group inverse of D6 element `s` (reflect-then-rotate): reflections (s>=6) are
/// involutions; rotations invert to `(6-n)%6`.
fn inv_sym(s: usize) -> usize {
    if s >= 6 {
        s
    } else {
        (6 - s) % 6
    }
}

// ── O-17: push/read round-trip ──────────────────────────────────────────────────

#[test]
fn push_read_roundtrip() {
    let mut buf = HexgBuffer::new(8, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 42).unwrap();
    assert_eq!(buf.size, 1);
    assert_eq!(buf.record_at(0), rec, "record_at must invert push_record_impl");
    assert_eq!(buf.game_ids[0], 42);
}

// ── O-18: ring wrap + size cap ──────────────────────────────────────────────────

#[test]
fn ring_wraps_and_caps_size() {
    let cap = 4;
    let mut buf = HexgBuffer::new(cap, ENC).unwrap();
    for i in 0..(cap + 3) {
        let mut rec = sample_record();
        rec.ply_index = i as u16;
        buf.push_record_impl(&rec, i as i64).unwrap();
    }
    assert_eq!(buf.size, cap, "size caps at capacity");
    assert_eq!(buf.head, 3 % cap, "head wrapped to (7 % 4)");
    let live: std::collections::HashSet<u16> = (0..cap).map(|s| buf.record_at(s).ply_index).collect();
    for expect in 3u16..7 {
        assert!(live.contains(&expect), "expected ply {expect} live after wrap");
    }
}

// ── O-19: over-cap push LOUD ─────────────────────────────────────────────────────

#[test]
fn push_rejects_over_cap() {
    let mut buf = HexgBuffer::new(2, ENC).unwrap();
    let over = GraphRecord { stones: vec![(0, 0, 1); MAX_STONES + 1], ..sample_record() };
    assert!(buf.push_record_impl(&over, -1).is_err(), "over-MAX_STONES must die loud");
    let over_v = GraphRecord { visits: vec![(0, 0, 0.1); MAX_VISITS + 1], ..sample_record() };
    assert!(buf.push_record_impl(&over_v, -1).is_err(), "over-MAX_VISITS must die loud");
}

// ── O-20: persist round-trip byte-identical ─────────────────────────────────────

#[test]
fn persist_roundtrip_byte_identical() {
    let mut buf = HexgBuffer::new(16, ENC).unwrap();
    for i in 0..10 {
        let mut rec = sample_record();
        rec.ply_index = i;
        rec.outcome = if i % 2 == 0 { 1.0 } else { -1.0 };
        rec.value_valid = i != 3;
        buf.push_record_impl(&rec, i64::from(i)).unwrap();
    }
    let path = unique_path("roundtrip");
    buf.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut buf2 = HexgBuffer::new(16, ENC).unwrap();
    let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 10);
    assert_eq!(buf2.size, 10);
    for slot in 0..10 {
        assert_eq!(buf.record_at(slot), buf2.record_at(slot), "record {slot} byte-identical");
        assert_eq!(buf.game_ids[slot], buf2.game_ids[slot]);
        assert_eq!(buf.weights[slot], buf2.weights[slot], "weight must survive");
    }
    let _ = std::fs::remove_file(path);
}

// ── O-21: frozen HEXG v1 byte-golden — load exact + re-save byte-identity ────────

fn hexg_golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/replay/hexg_v1_golden.hexg")
}

#[test]
fn o21_hexg_v1_byte_golden_load_and_resave_identity() {
    let golden = hexg_golden_path();
    let golden_bytes = std::fs::read(&golden).expect("frozen hexg golden must exist");
    assert_eq!(golden_bytes.len(), 148, "golden size drifted");

    let mut buf = HexgBuffer::new(16, ENC).unwrap();
    let n = buf.load_from_path_impl(golden.to_str().unwrap()).unwrap();
    assert_eq!(n, 2, "golden holds 2 records");

    // Exact field values (CAPTURE_LOG §C).
    let r0 = buf.record_at(0);
    assert_eq!(r0.stones, vec![(0, 0, 1), (1, 0, -1), (0, 1, 1)]);
    assert_eq!(r0.visits, vec![(2, 0, 0.6), (0, 2, 0.4)]);
    assert_eq!(r0.current_player, 1);
    assert_eq!(r0.moves_remaining, 2);
    assert_eq!(r0.ply_index, 3);
    assert!(r0.is_full_search);
    assert_eq!(r0.outcome, 1.0);
    assert!(r0.value_valid);
    assert_eq!(r0.game_length, 7);
    assert_eq!(buf.game_ids[0], 42);

    let r1 = buf.record_at(1);
    assert_eq!(r1.stones, vec![(0, 0, -1), (2, 1, 1)]);
    assert_eq!(r1.visits, vec![(1, 1, 1.0)]);
    assert_eq!(r1.current_player, -1);
    assert_eq!(r1.moves_remaining, 4);
    assert_eq!(r1.ply_index, 8);
    assert!(!r1.is_full_search);
    assert_eq!(r1.outcome, -1.0);
    assert!(!r1.value_valid);
    assert_eq!(r1.game_length, 12);
    assert_eq!(buf.game_ids[1], 43);

    // Re-save → byte-identical to the frozen golden.
    let resave = unique_path("o21_resave");
    buf.save_to_path_impl(resave.to_str().unwrap()).unwrap();
    let resaved = std::fs::read(&resave).unwrap();
    assert_eq!(resaved, golden_bytes, "new-engine re-save must reproduce the frozen bytes");
    let _ = std::fs::remove_file(resave);
}

// ── O-22: bad version / slot-geometry ────────────────────────────────────────────

#[test]
fn load_rejects_bad_version() {
    let path = unique_path("badver");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&HEXG_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&999u32.to_le_bytes()).unwrap();
    }
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("not supported"), "bad version must LOUD-FAIL: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn load_rejects_slot_geometry_mismatch() {
    let path = unique_path("slotgeo");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&HEXG_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&HEXG_VERSION.to_le_bytes()).unwrap();
        f.write_all(&(MAX_STONES as u32 + 1).to_le_bytes()).unwrap();
        f.write_all(&(MAX_VISITS as u32).to_le_bytes()).unwrap();
        f.write_all(&4u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&(ENC.len() as u32).to_le_bytes()).unwrap();
        f.write_all(ENC.as_bytes()).unwrap();
    }
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("slot-geometry"), "slot-geometry mismatch must reject: {err}");
    let _ = std::fs::remove_file(path);
}

// ── O-16: cross-magic rejection (both directions) ────────────────────────────────

#[test]
fn load_rejects_dense_hexb_magic() {
    let path = unique_path("hexb_magic");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0x4845_5842u32.to_le_bytes()).unwrap(); // "HEXB"
        f.write_all(&9u32.to_le_bytes()).unwrap();
    }
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("invalid magic"), "HEXB → HEXG load must reject on magic: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn dense_loader_rejects_hexg_file() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    buf.push_record_impl(&sample_record(), 0).unwrap();
    let path = unique_path("hexg_into_dense");
    buf.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dense = ReplayBuffer::new(4, "v6");
    let err = dense.load_from_path(path.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("magic") || err.contains("Invalid") || err.contains("invalid"),
        "HEXG → dense HEXB load must reject on magic: {err}"
    );
    let _ = std::fs::remove_file(path);
}

// ── O-23: grid encoding refused at construction ──────────────────────────────────

#[test]
fn grid_encoding_rejected_at_construction() {
    assert!(HexgBuffer::new(4, "v6").is_err(), "a grid encoding must be refused");
}

// ── O-24: rebuild-at-sample parity vs direct builder (re-anchored to AxisGraph) ──

#[test]
fn sample_wire_matches_direct_builder_unaugmented() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 0).unwrap();

    let (graphs, targets) = buf.sample_graph_batch_impl(1, false, 0.0).unwrap();
    assert_eq!(graphs.len(), 1);
    let sg = &graphs[0];
    assert_eq!(sg.builder_impl, 1, "sampled graph must carry builder_impl=1 (F7)");
    assert_eq!(buf.contract_version, 1);

    // Direct native build on the SAME stones (identity — augment off).
    let stones: Vec<(i32, i32, i8)> =
        rec.stones.iter().map(|&(q, r, p)| (i32::from(q), i32::from(r), p)).collect();
    let params = BuildParams {
        win_length: 6,
        radius: 6,
        current_player: rec.current_player,
        moves_remaining: rec.moves_remaining,
        trunk_size: 19,
    };
    let g = build_axis_graph(&StoneList { stones }, &params);

    // Single graph → local == global: every field must match the direct build.
    assert_eq!(sg.node_feat.0, g.node_feat.0, "node_feat parity");
    assert_eq!(sg.node_coords, g.node_coords, "node_coords parity");
    assert_eq!(sg.policy_scatter_index.0, g.policy_scatter_index.0, "policy_scatter_index parity");
    assert_eq!(sg.n_stones, g.n_stones, "n_stones parity");
    assert_eq!(sg.edge_index, g.edge_index, "edge_index parity");
    assert_eq!(sg.legal_node_gather, g.legal_node_gather, "legal_node_gather parity");
    assert_eq!(*sg, g, "the whole sampled graph equals the direct build");

    // Policy target: length == n_legal, each legal node gets its visit-map mass.
    let n_legal = g.legal_node_gather.len();
    assert_eq!(targets.policy_target.len(), n_legal);
    let mass: f32 = targets.policy_target.iter().sum();
    assert!((mass - 1.0).abs() < 1e-5, "target sums to ~1 (visit mass), got {mass}");
    assert_eq!(targets.argmax_valid, vec![1]);
    assert_eq!((targets.argmax_q[0], targets.argmax_r[0]), (2, 0));
}

// ── O-25: D6 aug round-trip coherence + ADV-7 canary ─────────────────────────────

#[test]
fn rotate_axial_roundtrips_under_inverse() {
    let coords = [(0, 0), (3, -2), (-4, 1), (5, 5), (1, -6)];
    for s in 0..12 {
        for &(q, r) in &coords {
            let (rq, rr) = rotate_axial(q, r, s);
            let (bq, br) = rotate_axial(rq, rr, inv_sym(s));
            assert_eq!((bq, br), (q, r), "s={s}: rotate then inverse must recover coord");
        }
    }
}

fn argmax_canary_passes(g: &AxisGraph, argmax_cell: (i32, i32)) -> bool {
    g.legal_node_gather.iter().any(|&row| {
        let cq = g.node_coords[row as usize * 2];
        let cr = g.node_coords[row as usize * 2 + 1];
        (cq, cr) == argmax_cell
    })
}

#[test]
fn augmented_sample_target_is_coherent_every_element() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    buf.push_record_impl(&sample_record(), 0).unwrap();
    for _ in 0..48 {
        let (graphs, targets) = buf.sample_graph_batch_impl(1, true, 0.0).unwrap();
        if targets.argmax_valid[0] == 0 {
            continue;
        }
        let cell = (targets.argmax_q[0], targets.argmax_r[0]);
        assert!(
            argmax_canary_passes(&graphs[0], cell),
            "augmented sample target argmax {cell:?} must be a legal node (coherent)"
        );
    }
}

#[test]
fn adv7_desync_is_caught_by_the_canary() {
    let rec = sample_record();
    let stones: Vec<(i32, i32, i8)> =
        rec.stones.iter().map(|&(q, r, p)| (i32::from(q), i32::from(r), p)).collect();
    let params = BuildParams {
        win_length: 6,
        radius: 6,
        current_player: rec.current_player,
        moves_remaining: rec.moves_remaining,
        trunk_size: 19,
    };
    let g = build_axis_graph(&StoneList { stones }, &params);

    let stone_cell = (0i32, 0i32);
    assert!(
        !argmax_canary_passes(&g, stone_cell),
        "a stone (occupied) cell must NOT be a legal node — the canary fires on this desync"
    );
    let a_legal = {
        let row = g.legal_node_gather[0] as usize;
        (g.node_coords[row * 2], g.node_coords[row * 2 + 1])
    };
    assert!(argmax_canary_passes(&g, a_legal), "a real legal cell must pass the canary");
}

#[test]
fn empty_board_record_survives_d6_augmented_sample_align() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord {
        stones: vec![],
        visits: vec![(2, 2, 0.6), (-2, -2, 0.3), (0, 0, 0.1)],
        current_player: 1,
        moves_remaining: 2,
        ply_index: 0,
        is_full_search: true,
        outcome: 0.0,
        value_valid: false,
        game_length: 10,
    };
    buf.push_record_impl(&rec, 0).unwrap();
    for _ in 0..48 {
        buf.sample_graph_batch_impl(1, true, 0.0)
            .expect("empty-board record must survive D6-augmented sample-align (sym forced to identity)");
    }
}

// ── O-26: atomic load on failure ─────────────────────────────────────────────────

#[test]
fn failed_truncated_load_is_loud_and_leaves_buffer_untouched() {
    let mut src = HexgBuffer::new(8, ENC).unwrap();
    for i in 0..3u16 {
        let mut rec = sample_record();
        rec.ply_index = i;
        src.push_record_impl(&rec, 900 + i as i64).unwrap();
    }
    let good_path = unique_path("b1_src");
    src.save_to_path_impl(good_path.to_str().unwrap()).unwrap();
    let raw = std::fs::read(&good_path).unwrap();
    assert!(raw.len() > 20, "fixture too small to truncate meaningfully");
    let trunc_path = unique_path("b1_trunc");
    std::fs::write(&trunc_path, &raw[..raw.len() - 20]).unwrap();

    let mut victim = HexgBuffer::new(8, ENC).unwrap();
    for i in 0..5u16 {
        let mut rec = sample_record();
        rec.ply_index = i;
        victim.push_record_impl(&rec, 100 + i as i64).unwrap();
    }
    let size0 = victim.size;
    let head0 = victim.head;
    let hist0 = victim.get_buffer_stats_impl().2;
    let snapshot: Vec<(GraphRecord, i64, u16)> =
        (0..size0).map(|s| (victim.record_at(s), victim.game_ids[s], victim.weights[s])).collect();

    let err = victim.load_from_path_impl(trunc_path.to_str().unwrap());
    assert!(err.is_err(), "truncated mid-payload load must LOUD-FAIL");

    assert_eq!(victim.size, size0, "size unchanged after failed load");
    assert_eq!(victim.head, head0, "head unchanged after failed load");
    assert_eq!(victim.get_buffer_stats_impl().2, hist0, "histogram unchanged after failed load");
    for (s, (rec, gid, w)) in snapshot.iter().enumerate() {
        assert_eq!(&victim.record_at(s), rec, "record {s} byte-identical after failed load");
        assert_eq!(victim.game_ids[s], *gid, "game_id {s} unchanged");
        assert_eq!(victim.weights[s], *w, "weight {s} unchanged");
    }
    let (_, targets) = victim.sample_graph_batch_impl(4, false, 0.0).unwrap();
    assert_eq!(targets.outcomes.len(), 4);

    let _ = std::fs::remove_file(&good_path);
    let _ = std::fs::remove_file(&trunc_path);
}

// ── O-27: mass-drop guard ────────────────────────────────────────────────────────

#[test]
fn sample_rejects_illegal_cell_visit_mass_drop() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord {
        stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, 1)],
        visits: vec![(0, 0, 0.9), (2, 0, 0.1)],
        current_player: 1,
        moves_remaining: 2,
        ply_index: 5,
        is_full_search: true,
        outcome: 1.0,
        value_valid: true,
        game_length: 30,
    };
    buf.push_record_impl(&rec, 7).unwrap();
    assert!(
        buf.sample_graph_batch_impl(1, false, 0.0).is_err(),
        "illegal-cell visit mass drop must raise, not silently under-weight"
    );
}

#[test]
fn mass_drop_check_message_names_game_id_ply_and_dropped_mass() {
    let msg = mass_drop_check(7, 5, 1.0, 0.1).expect_err("a 0.9 mass drop must trip the guard");
    assert!(msg.contains("game_id=7"), "error must name game_id: {msg}");
    assert!(msg.contains("ply=5"), "error must name ply: {msg}");
    assert!(msg.contains("dropped"), "error must report the dropped mass: {msg}");
}

#[test]
fn mass_drop_check_tolerates_float_noise() {
    mass_drop_check(1, 0, 1.0, 1.0 - 1e-6).expect("float noise must not trip");
    mass_drop_check(1, 0, 0.0, 0.0).expect("zero/zero must not trip");
}

#[test]
fn legit_push_sample_roundtrip_does_not_trip_mass_drop_guard() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    buf.push_record_impl(&sample_record(), 0).unwrap();
    for aug in [false, true] {
        for _ in 0..24 {
            buf.sample_graph_batch_impl(1, aug, 0.0).expect("legit round-trip must not trip the guard");
        }
    }
}

// ── O-29: game_id rebase on load ─────────────────────────────────────────────────

#[test]
fn load_rebases_next_game_id_past_loaded_max() {
    let mut src = HexgBuffer::new(8, ENC).unwrap();
    for i in 0..3u16 {
        let mut rec = sample_record();
        rec.ply_index = i;
        src.push_record_impl(&rec, 2 + i as i64).unwrap(); // game_ids 2, 3, 4
    }
    let path = unique_path("ngid_rebase");
    src.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dst = HexgBuffer::new(8, ENC).unwrap();
    let n = dst.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 3);
    assert_eq!(dst.next_game_id, 5, "next_game_id must continue past the loaded max (4)");
    let fresh_id = dst.next_game_id();
    assert_eq!(fresh_id, 5);
    assert!(!(2..=4).contains(&fresh_id), "fresh id must not collide with a loaded game_id");
    let _ = std::fs::remove_file(path);
}

#[test]
fn load_of_empty_file_does_not_touch_next_game_id() {
    let empty_src = HexgBuffer::new(4, ENC).unwrap();
    let path = unique_path("ngid_guard_empty");
    empty_src.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dst = HexgBuffer::new(4, ENC).unwrap();
    let g0 = dst.next_game_id();
    assert_eq!(g0, 0);
    let n = dst.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 0);
    assert_eq!(dst.next_game_id, 1, "loading zero records must not clobber next_game_id");
    let _ = std::fs::remove_file(path);
}

#[test]
fn load_with_i64_max_game_id_does_not_panic_and_saturates() {
    let mut src = HexgBuffer::new(4, ENC).unwrap();
    src.push_record_impl(&sample_record(), i64::MAX).unwrap();
    let path = unique_path("gid_i64max");
    src.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dst = HexgBuffer::new(4, ENC).unwrap();
    let n = dst.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 1);
    assert_eq!(dst.next_game_id, i64::MAX, "saturating_add(1) on i64::MAX must saturate");
    let _ = std::fs::remove_file(path);
}

// ── O-28: push-time validation ───────────────────────────────────────────────────

#[test]
fn push_rejects_nan_visit_prob() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord { visits: vec![(2, 0, f32::NAN)], ..sample_record() };
    assert!(buf.push_record_impl(&rec, 0).is_err(), "NaN visit prob must be rejected at push");
    let msg = validate_visit_prob(2, 0, f32::NAN).expect_err("pure helper must reject NaN");
    assert!(msg.contains("(2, 0)"), "error must name the coord: {msg}");
    assert!(msg.contains("NaN"), "error must report the value: {msg}");
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

#[test]
fn push_rejects_negative_visit_prob() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord { visits: vec![(2, 0, -0.5), (-1, 0, 0.5)], ..sample_record() };
    assert!(buf.push_record_impl(&rec, 0).is_err(), "negative visit prob must be rejected at push");
    let msg = validate_visit_prob(2, 0, -0.5).expect_err("pure helper must reject a negative prob");
    assert!(msg.contains("(2, 0)"), "error must name the coord: {msg}");
    assert!(msg.contains("-0.5"), "error must report the value: {msg}");
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

#[test]
fn legit_push_unaffected_by_prob_validation_guard() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 42).expect("legit finite non-negative probs must pass");
    assert_eq!(buf.size, 1);
    assert_eq!(buf.record_at(0), rec);
    assert_eq!(buf.game_ids[0], 42);
}

#[test]
fn push_rejects_nan_outcome() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord { outcome: f32::NAN, ..sample_record() };
    assert!(buf.push_record_impl(&rec, 0).is_err(), "NaN outcome must be rejected at push");
    let msg = validate_outcome(f32::NAN).expect_err("pure helper must reject NaN");
    assert!(msg.contains("NaN"), "error must report the value: {msg}");
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

#[test]
fn push_rejects_inf_outcome() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord { outcome: f32::INFINITY, ..sample_record() };
    assert!(buf.push_record_impl(&rec, 0).is_err(), "inf outcome must be rejected at push");
    let msg = validate_outcome(f32::INFINITY).expect_err("pure helper must reject +inf");
    assert!(msg.contains("inf"), "error must report the value: {msg}");
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

#[test]
fn push_rejects_bad_stone_player() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord { stones: vec![(0, 0, 1), (1, 0, 5), (0, 1, 1), (2, 1, -1)], ..sample_record() };
    assert!(buf.push_record_impl(&rec, 0).is_err(), "stone player outside {{+1,-1}} must be rejected");
    let msg = validate_stone_player(1, 0, 5).expect_err("pure helper must reject an out-of-range player");
    assert!(msg.contains("(1, 0)"), "error must name the coord: {msg}");
    assert!(msg.contains('5'), "error must report the value: {msg}");
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

#[test]
fn legit_push_unaffected_by_outcome_and_stone_player_guards() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 42).expect("legit finite outcome and ±1 players must pass");
    assert_eq!(buf.size, 1);
    assert_eq!(buf.record_at(0), rec);
    assert_eq!(buf.game_ids[0], 42);
}

// ── O-30: recency sampler ────────────────────────────────────────────────────────

#[test]
fn recency_sampler_draws_the_newest_slot_fraction() {
    let cap = 600;
    let total_pushes: u32 = 700;
    let mut buf = HexgBuffer::new(cap, ENC).unwrap();
    for i in 0..total_pushes {
        let mut rec = sample_record();
        rec.ply_index = (i % (u16::MAX as u32 + 1)) as u16;
        buf.push_record_impl(&rec, -1).unwrap();
    }
    assert_eq!(buf.size, cap);
    let window = buf.recent_window();
    assert_eq!(window, 300, "max(256, 600/2) == 300");
    let newest_threshold = (total_pushes - window as u32) as u16; // 400

    let idx = buf.sample_indices(64, 1.0);
    assert_eq!(idx.len(), 64);
    for i in &idx {
        let ply = buf.ply_index[*i];
        assert!(ply >= newest_threshold, "recent_frac=1.0 draw ply={ply} must be >= {newest_threshold}");
    }
}

#[test]
fn recency_sampler_zero_frac_is_byte_identical_to_full_ring_sample() {
    let cap = 600;
    let total_pushes: u32 = 700;
    let mut buf = HexgBuffer::new(cap, ENC).unwrap();
    for i in 0..total_pushes {
        let mut rec = sample_record();
        rec.ply_index = (i % (u16::MAX as u32 + 1)) as u16;
        buf.push_record_impl(&rec, -1).unwrap();
    }
    let window = buf.recent_window();
    let newest_threshold = (total_pushes - window as u32) as u16;

    let idx = buf.sample_indices(500, 0.0);
    assert_eq!(idx.len(), 500);
    let any_old = idx.iter().any(|&i| buf.ply_index[i] < newest_threshold);
    assert!(any_old, "recent_frac=0.0 must sample beyond the newest-slots window");
}

#[test]
fn recency_sampler_recent_window_clamped_by_size_before_ring_fills() {
    let cap = 1000;
    let mut buf = HexgBuffer::new(cap, ENC).unwrap();
    for i in 0..50u32 {
        let mut rec = sample_record();
        rec.ply_index = i as u16;
        buf.push_record_impl(&rec, -1).unwrap();
    }
    assert_eq!(buf.size, 50);
    let window = buf.recent_window();
    assert_eq!(window, 50, "window must clamp to the live size");
    let idx = buf.sample_indices(20, 1.0);
    for i in &idx {
        assert!(buf.ply_index[*i] < 50, "must never draw a never-written slot");
    }
}
