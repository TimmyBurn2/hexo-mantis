//! R8-justify: one test per golden × (positive pin + mutation arm) keeps the record-byte pin
//! auditable as a single unit.
//!
//! The record/finalize producers must reproduce the frozen goldens byte-for-byte over the
//! pinned splitmix64 inputs. Each golden carries a mutation self-test: flipping ONE
//! load-bearing input element must DIVERGE the serialized output from the golden.
//!
//! g1–g4 pinned the four dense cluster-aggregation producers, which went with the grid path;
//! their goldens are orphaned and no producer can regenerate them. g5–g7 — graph assemble,
//! graph record, outcome finalizer — are what remains.

use fxhash::FxHashMap;
use mantis_core::{Board, Player};
use mantis_graph::{build_axis_graph, BuildParams, StoneList};
use mantis_search::LegalSetPolicy;
use mantis_selfplay::records::{
    assemble_ls_from_gnn_probs, finalize_graph_outcome, record_position_graph,
};

// Pinned constants
const WORKER_GOLDEN_SEED: u64 = 0xB0A2_D601_D000_0006;
const N_ACTIONS: usize = 362;
const TRUNK: i32 = 19;
const HALF: i32 = 9;

// Mock-NN splitmix64 stream
fn splitmix64_step(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `fill_stream(seed, n)` = `n` unit logits `(step >> 40) as f32 / 2^24 ∈ [0,1)`.
fn fill_stream(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let step = splitmix64_step(&mut s);
        out.push((step >> 40) as f32 / 16_777_216.0_f32);
    }
    out
}

/// Compact board: the frozen `records.rs::small_board()` — 3 stones, P1 to move.
fn compact_board() -> Board {
    let mut b = Board::new();
    b.apply_move(0, 0).unwrap();
    b.apply_move(2, 0).unwrap();
    b.apply_move(0, 2).unwrap();
    b
}

// Serializers — pinned byte layouts, all little-endian.
fn ser_vec_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + v.len() * 4);
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// `LegalSetPolicy`: `[u32 dense_len | dense f32…] [u32 overflow_ct | (i32 q,
/// i32 r, f32 p)… SORTED by (q,r)]`.
fn ser_ls(ls: &LegalSetPolicy) -> Vec<u8> {
    let mut out = ser_vec_f32(&ls.dense);
    let mut ov: Vec<((i32, i32), f32)> = ls.overflow.iter().map(|(&k, &v)| (k, v)).collect();
    ov.sort_by_key(|&((q, r), _)| (q, r));
    out.extend_from_slice(&(ov.len() as u32).to_le_bytes());
    for ((q, r), p) in ov {
        out.extend_from_slice(&q.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
        out.extend_from_slice(&p.to_le_bytes());
    }
    out
}

/// `GraphRecord`: `[u32 stones_ct | (i16 q,i16 r,i8 p)… SORTED] [u32 visits_ct
/// | (i16 q,i16 r,f32 p)… SORTED] [i8 cur][u8 mv_rem][u16 ply][u8 full][f32
/// outcome][u8 valid][u16 game_len]` (stones/visits canonicalised by (q,r) sort).
fn ser_graph_record(rec: &mantis_selfplay::replay::hexg::GraphRecord) -> Vec<u8> {
    let mut out = Vec::new();
    let mut stones = rec.stones.clone();
    stones.sort_by_key(|&(q, r, _)| (q, r));
    out.extend_from_slice(&(stones.len() as u32).to_le_bytes());
    for (q, r, p) in stones {
        out.extend_from_slice(&q.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
        out.push(p as u8);
    }
    let mut visits = rec.visits.clone();
    visits.sort_by_key(|&(q, r, _)| (q, r));
    out.extend_from_slice(&(visits.len() as u32).to_le_bytes());
    for (q, r, p) in visits {
        out.extend_from_slice(&q.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
        out.extend_from_slice(&p.to_le_bytes());
    }
    out.push(rec.current_player as u8);
    out.push(rec.moves_remaining);
    out.extend_from_slice(&rec.ply_index.to_le_bytes());
    out.push(u8::from(rec.is_full_search));
    out.extend_from_slice(&rec.outcome.to_le_bytes());
    out.push(u8::from(rec.value_valid));
    out.extend_from_slice(&rec.game_length.to_le_bytes());
    out
}

/// finalize rows: `[u32 ct | (f32 outcome, u8 valid)…]`.
fn ser_finalize(rows: &[(f32, u8)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + rows.len() * 5);
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for &(outcome, valid) in rows {
        out.extend_from_slice(&outcome.to_le_bytes());
        out.push(valid);
    }
    out
}

fn read_golden(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read golden {}: {e}", path.display()))
}

// Reconstructions: canonical pinned inputs → producer → serialized bytes. `mutate` flips ONE
// load-bearing input element for the mutation self-test.

/// g5 `assemble_ls_from_gnn_probs`: build_axis_graph on two far clusters (q∈[0,5)
/// P1, q∈[30,35) P2; win_length 6/radius 6/trunk 19); legal_probs =
/// fill_stream(SEED^0x20, n_legal) normalised to sum 1.
fn produce_g5(mutate: bool) -> Vec<u8> {
    let mut stones: Vec<(i32, i32, i8)> = Vec::new();
    for q in 0..5i32 {
        stones.push((q, 0, 1));
    }
    for q in 30..35i32 {
        stones.push((q, 0, -1));
    }
    let params = BuildParams {
        win_length: 6,
        radius: 6,
        current_player: 1,
        moves_remaining: 2,
        trunk_size: 19,
    };
    let g = build_axis_graph(&StoneList { stones }, &params);
    let slots = g.policy_scatter_index.0.clone();
    let n_legal = slots.len();
    let mut coords: Vec<(i32, i32)> = g
        .legal_node_gather
        .iter()
        .map(|&row| {
            (
                g.node_coords[row as usize * 2],
                g.node_coords[row as usize * 2 + 1],
            )
        })
        .collect();
    let raw = fill_stream(WORKER_GOLDEN_SEED ^ 0x20, n_legal);
    let s: f32 = raw.iter().sum();
    let probs: Vec<f32> = raw.iter().map(|p| p / s).collect();
    if mutate {
        // Flipping one off-window node's coord changes its overflow KEY while preserving the
        // sum-1 invariant, so assemble still returns Ok.
        let off_idx = slots
            .iter()
            .position(|&sl| sl == mantis_graph::OFF_WINDOW_SLOT)
            .expect("mixed fixture has an off-window node");
        coords[off_idx].0 += 100;
    }
    let ls = assemble_ls_from_gnn_probs(N_ACTIONS, &probs, &slots, &coords).expect("assemble ok");
    ser_ls(&ls)
}

/// g6 `record_position_graph`, two legs over the pinned stream:
///   (a) the RAW planting (Σ ≈ n/2, not a distribution) refuses with `MassNotUnity` carrying
///       the exact pre-filter legal-scan sum, which pins the read-by-coord scan;
///   (b) the NORMALIZED planting records, and its bytes equal a re-derived record over the
///       same coords/masses.
/// Returns (refusal_sum, ok_record_bytes). The byte pin for valid targets lives in
/// `target_export_parity.rs`.
fn produce_g6(mutate: bool) -> (f64, Vec<u8>) {
    let board = compact_board();
    let legal = board.legal_moves();
    let (bcq, bcr) = board.window_center();
    let mut stream = fill_stream(WORKER_GOLDEN_SEED ^ 0x30, legal.len());
    if mutate {
        // Flip the planted mass for legal[0] → the refusal sum diverges.
        stream[0] = if stream[0] > 0.5 { 0.125 } else { 0.875 };
    }
    let mut dense = vec![0.0f32; N_ACTIONS];
    let mut raw_sum = 0.0f64;
    for (i, &(q, r)) in legal.iter().enumerate() {
        let idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, TRUNK, HALF);
        assert!(idx < N_ACTIONS, "compact board must be fully in-window");
        dense[idx] = stream[i];
        raw_sum += f64::from(stream[i]);
    }
    let ls = LegalSetPolicy {
        dense: dense.clone(),
        overflow: FxHashMap::default(),
    };
    let refusal_sum = match record_position_graph(
        &board,
        &ls,
        TRUNK,
        board.current_player as i8,
        board.moves_remaining,
        board.ply.index() as u16,
        true,
        128,
        None,
    ) {
        Err(mantis_selfplay::records::TargetIntegrityError::MassNotUnity { sum, .. }) => {
            assert!(
                (sum - raw_sum).abs() < 1e-9,
                "MassNotUnity must carry the pre-filter legal-scan sum: {sum} vs {raw_sum}"
            );
            sum
        }
        other => panic!("a non-distribution target must refuse with MassNotUnity, got {other:?}"),
    };

    // (b) normalized planting → records; equality vs a re-derived record.
    let total = raw_sum as f32;
    let mut norm_dense = vec![0.0f32; N_ACTIONS];
    let mut expected_visits: Vec<(i16, i16, f32)> = Vec::new();
    for (i, &(q, r)) in legal.iter().enumerate() {
        let idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, TRUNK, HALF);
        let p = stream[i] / total;
        norm_dense[idx] = p;
        if p > 0.0 {
            expected_visits.push((q as i16, r as i16, p));
        }
    }
    let norm_ls = LegalSetPolicy {
        dense: norm_dense,
        overflow: FxHashMap::default(),
    };
    let rec = record_position_graph(
        &board,
        &norm_ls,
        TRUNK,
        board.current_player as i8,
        board.moves_remaining,
        board.ply.index() as u16,
        true,
        128,
        None,
    )
    .expect("a normalized target must record");
    let expected = mantis_selfplay::replay::hexg::GraphRecord {
        stones: rec.stones.clone(), // stones come from the board either way
        visits: expected_visits,
        tail_mass: 0.0,
        current_player: board.current_player as i8,
        moves_remaining: board.moves_remaining,
        ply_index: board.ply.index() as u16,
        is_full_search: true,
        outcome: 0.0,
        value_valid: true,
        game_length: 0,
        game_id: -1,
    };
    assert_eq!(
        ser_graph_record(&rec),
        ser_graph_record(&expected),
        "normalized record must carry every planted coord's mass (layout parity)"
    );
    (refusal_sum, ser_graph_record(&rec))
}

/// g7 `finalize_graph_outcome`: 6 enumerated rows (ply_cap_value=-0.5,
/// draw_reward=-0.1): win / loss / ply-cap(tr=2) / organic-draw(tr=3) /
/// P2-win-as-P2 / P2-win-as-P1.
fn produce_g7(mutate: bool) -> Vec<u8> {
    // (rec_player, winner, terminal_reason)
    let mut rows: [(i8, Option<Player>, u8); 6] = [
        (1, Some(Player::One), 0),  // win
        (1, Some(Player::Two), 0),  // loss
        (1, None, 2),               // ply-cap
        (1, None, 3),               // organic draw
        (-1, Some(Player::Two), 0), // P2-win-as-P2
        (-1, Some(Player::One), 0), // P2-win-as-P1
    ];
    if mutate {
        // Flip the ply-cap row's terminal_reason 2 → 3 (ply-cap → organic
        // draw): outcome -0.5 → -0.1 AND value_valid 0 → 1.
        rows[2].2 = 3;
    }
    let out: Vec<(f32, u8)> = rows
        .iter()
        .map(|&(rp, w, tr)| finalize_graph_outcome(rp, w, tr, -0.5, -0.1))
        .collect();
    ser_finalize(&out)
}

#[test]
fn pin_g5_assemble_ls_from_gnn_probs() {
    assert_eq!(
        produce_g5(false),
        read_golden("assemble_ls_from_gnn_probs.bin")
    );
}
#[test]
fn pin_g6_record_position_graph() {
    // The in-fn asserts ARE the pin: there is no golden file, because the old one encoded the
    // pre-fix arbitrary-mass acceptance and was deleted with its manifest row.
    let (refusal_sum, ok_bytes) = produce_g6(false);
    assert!(
        refusal_sum > 1.0,
        "the raw stream planting must overshoot unity"
    );
    assert!(!ok_bytes.is_empty());
}
#[test]
fn pin_g7_finalize_graph_outcome() {
    assert_eq!(produce_g7(false), read_golden("finalize_graph_outcome.bin"));
}

// Mutation self-tests: flipping ONE input element must diverge from the golden.
#[test]
fn mut_g5_diverges() {
    assert_ne!(
        produce_g5(true),
        read_golden("assemble_ls_from_gnn_probs.bin")
    );
}
#[test]
fn mut_g6_diverges() {
    // Flipping ONE planted mass must diverge BOTH the refusal sum and the record bytes.
    let (sum_a, bytes_a) = produce_g6(false);
    let (sum_b, bytes_b) = produce_g6(true);
    assert_ne!(
        sum_a.to_bits(),
        sum_b.to_bits(),
        "refusal sum must feel the flip"
    );
    assert_ne!(
        bytes_a, bytes_b,
        "normalized record bytes must feel the flip"
    );
}
#[test]
fn mut_g7_diverges() {
    assert_ne!(produce_g7(true), read_golden("finalize_graph_outcome.bin"));
}
