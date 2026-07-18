//! R8-justify: one integration test per golden × (positive pin + LAW-07
//! mutation) for 7 goldens keeps the P-04 pin auditable as a single unit.
//!
//! P-04 record byte pin (WP6, `_v1`) — the 7 pyo3-free record/finalize
//! producers reproduce the dispatcher-frozen goldens byte-for-byte over the
//! pinned splitmix64 inputs (CAPTURE_LOG §B/§C, mtime-before-IMPL). Written
//! FIRST-discipline: this pin asserts the ported producers reproduce the frozen
//! bytes; the dispatcher (not IMPL) produced the goldens.
//!
//! Each golden carries a LAW-07 mutation self-test: flipping ONE load-bearing
//! input element must DIVERGE the serialized output from the golden. A checker
//! that passes a flipped input is a test failure (the mutation tests bite).
//!
//! Deviation (CAPTURE_LOG §A): the full mock-NN worker loop is uncapturable
//! old-side (embedded-numpy wall), so this pin covers the load-bearing
//! record-byte math directly; the multi-graph fuse INPUT (g8) is the P-09
//! wire-stage oracle, not this file.

use fxhash::FxHashMap;
use mantis_core::{Board, Player};
use mantis_graph::{build_axis_graph, BuildParams, StoneList};
use mantis_search::LegalSetPolicy;
use mantis_selfplay::records::{
    aggregate_policy, aggregate_policy_ls, aggregate_policy_to_local,
    aggregate_policy_to_local_ls, assemble_ls_from_gnn_probs, finalize_graph_outcome,
    record_position_graph,
};

// ── Pinned constants (CAPTURE_LOG §B/§C) ────────────────────────────────────
const WORKER_GOLDEN_SEED: u64 = 0xB0A2_D601_D000_0006;
const N_ACTIONS: usize = 362;
const TRUNK: i32 = 19;
const HALF: i32 = 9;

// ── Mock-NN splitmix64 stream (CAPTURE_LOG §B, PREREG C-10) ──────────────────
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

// ── Pinned boards (CAPTURE_LOG §C — PUB `apply_move` API only) ───────────────
/// Compact board: the frozen `records.rs::small_board()` — 3 stones, P1 to move.
fn compact_board() -> Board {
    let mut b = Board::new();
    b.apply_move(0, 0).unwrap();
    b.apply_move(2, 0).unwrap();
    b.apply_move(0, 2).unwrap();
    b
}

/// Spread board: walking-line `apply_move` of q ∈ {0,4,…,32}, r=0 (each hop
/// hex-distance 4 ≤ radius 5). Result: `window_center = (16,0)`, `legal = 434`,
/// 230 off-window cells.
fn spread_board() -> Board {
    let mut b = Board::new();
    for q in [0i32, 4, 8, 12, 16, 20, 24, 28, 32] {
        b.apply_move(q, 0).unwrap();
    }
    b
}

// ── Serializers (CAPTURE_LOG §C byte layouts, all little-endian) ─────────────
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

// ── Reconstructions (canonical pinned inputs → producer → serialized bytes) ──
// The `mut_*` closures flip ONE load-bearing input element for the LAW-07 self-test.

/// g1 `aggregate_policy`: compact board, 1 centre = window_center,
/// cluster_policies = [fill_stream(SEED, 362)].
fn produce_g1(mutate: bool) -> Vec<u8> {
    let board = compact_board();
    let centers = [board.window_center()];
    let mut cp0 = fill_stream(WORKER_GOLDEN_SEED, N_ACTIONS);
    if mutate {
        // Bump the local slot read for legal[0] under the single centre.
        let (cq, cr) = centers[0];
        let (q, r) = board.legal_moves()[0];
        let local_idx = (q - cq + HALF) as usize * TRUNK as usize + (r - cr + HALF) as usize;
        cp0[local_idx] += 1.0;
    }
    let out = aggregate_policy(N_ACTIONS, true, TRUNK, &board, &centers, &[cp0]);
    ser_vec_f32(&out)
}

/// g2 `aggregate_policy_to_local`: compact board, centre = window_center,
/// global = fill_stream(SEED^0x01, 362), legal = board.legal_moves().
fn produce_g2(mutate: bool) -> Vec<u8> {
    let board = compact_board();
    let center = board.window_center();
    let legal = board.legal_moves();
    let mut global = fill_stream(WORKER_GOLDEN_SEED ^ 0x01, N_ACTIONS);
    if mutate {
        let (bcq, bcr) = board.window_center();
        let (q, r) = legal[0];
        let idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, TRUNK, HALF);
        global[idx] += 1.0;
    }
    let out = aggregate_policy_to_local(N_ACTIONS, true, TRUNK, &board, &center, &global, &legal);
    ser_vec_f32(&out)
}

/// g3 `aggregate_policy_ls`: spread board, centres [(4,0),(28,0)],
/// cluster_policies = [fill_stream(SEED^0x10,362), fill_stream(SEED^0x11,362)].
fn g3_ls(mutate: bool) -> (Board, LegalSetPolicy) {
    let board = spread_board();
    let centers = [(4, 0), (28, 0)];
    let mut cp0 = fill_stream(WORKER_GOLDEN_SEED ^ 0x10, N_ACTIONS);
    let cp1 = fill_stream(WORKER_GOLDEN_SEED ^ 0x11, N_ACTIONS);
    if mutate {
        // Bump the slot read for legal[0] under centre 0 (an in-window write).
        let (cq, cr) = centers[0];
        let (q, r) = board.legal_moves()[0];
        let local_idx = (q - cq + HALF) as usize * TRUNK as usize + (r - cr + HALF) as usize;
        if local_idx < cp0.len() {
            cp0[local_idx] += 1.0;
        }
    }
    let ls = aggregate_policy_ls(N_ACTIONS, true, TRUNK, &board, &centers, &[cp0, cp1]);
    (board, ls)
}

fn produce_g3(mutate: bool) -> Vec<u8> {
    ser_ls(&g3_ls(mutate).1)
}

/// g4 `aggregate_policy_to_local_ls`: spread board, project g3's ls into centre
/// (28,0), legal = board.legal_moves().
fn produce_g4(mutate: bool) -> Vec<u8> {
    let (board, mut ls) = g3_ls(false);
    let legal = board.legal_moves();
    if mutate {
        // Flip the ls value the projection into centre (28,0) actually READS —
        // the first legal cell inside centre-(28,0)'s local window (a far-cluster
        // cell near legal[0] is not projected, so bumping it would be inert).
        let (cq, cr) = (28i32, 0i32);
        let (bcq, bcr) = board.window_center();
        for &(q, r) in &legal {
            let wq = q - cq + HALF;
            let wr = r - cr + HALF;
            if (0..TRUNK).contains(&wq) && (0..TRUNK).contains(&wr) {
                let mcts_idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, TRUNK, HALF);
                if mcts_idx < ls.dense.len() {
                    ls.dense[mcts_idx] += 1.0;
                } else {
                    *ls.overflow.entry((q, r)).or_insert(0.0) += 1.0;
                }
                break;
            }
        }
    }
    let out = aggregate_policy_to_local_ls(N_ACTIONS, true, TRUNK, &board, &(28, 0), &ls, &legal);
    ser_vec_f32(&out)
}

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
    let params = BuildParams { win_length: 6, radius: 6, current_player: 1, moves_remaining: 2, trunk_size: 19 };
    let g = build_axis_graph(&StoneList { stones }, &params);
    let slots = g.policy_scatter_index.0.clone();
    let n_legal = slots.len();
    let mut coords: Vec<(i32, i32)> = g
        .legal_node_gather
        .iter()
        .map(|&row| (g.node_coords[row as usize * 2], g.node_coords[row as usize * 2 + 1]))
        .collect();
    let raw = fill_stream(WORKER_GOLDEN_SEED ^ 0x20, n_legal);
    let s: f32 = raw.iter().sum();
    let probs: Vec<f32> = raw.iter().map(|p| p / s).collect();
    if mutate {
        // Flip ONE off-window legal node's coord → its overflow KEY changes
        // (sum-1 segmented-softmax invariant preserved, so assemble still Ok).
        let off_idx = slots
            .iter()
            .position(|&sl| sl == mantis_graph::OFF_WINDOW_SLOT)
            .expect("mixed fixture has an off-window node");
        coords[off_idx].0 += 100;
    }
    let ls = assemble_ls_from_gnn_probs(N_ACTIONS, &probs, &slots, &coords).expect("assemble ok");
    ser_ls(&ls)
}

/// g6 `record_position_graph`: compact board; ls.dense planted with
/// fill_stream(SEED^0x30, legal.len()) at each legal cell's window_flat_idx;
/// args (trunk 19, cur, mv_rem, ply, is_full_search=true, max_visits=128).
fn produce_g6(mutate: bool) -> Vec<u8> {
    let board = compact_board();
    let legal = board.legal_moves();
    let (bcq, bcr) = board.window_center();
    let mut stream = fill_stream(WORKER_GOLDEN_SEED ^ 0x30, legal.len());
    if mutate {
        // Flip the planted mass for legal[0] → that visit's prob diverges.
        stream[0] = if stream[0] > 0.5 { 0.125 } else { 0.875 };
    }
    let mut dense = vec![0.0f32; N_ACTIONS];
    for (i, &(q, r)) in legal.iter().enumerate() {
        let idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, TRUNK, HALF);
        if idx < N_ACTIONS {
            dense[idx] = stream[i];
        }
    }
    let ls = LegalSetPolicy { dense, overflow: FxHashMap::default() };
    let rec = record_position_graph(
        &board,
        &ls,
        TRUNK,
        board.current_player as i8,
        board.moves_remaining,
        board.ply.index() as u16,
        true,
        128,
    );
    ser_graph_record(&rec)
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

// ── The 7 record byte pins (positive: ported producer == frozen golden) ─────
#[test]
fn pin_g1_aggregate_policy() {
    assert_eq!(produce_g1(false), read_golden("aggregate_policy.bin"));
}
#[test]
fn pin_g2_aggregate_policy_to_local() {
    assert_eq!(produce_g2(false), read_golden("aggregate_policy_to_local.bin"));
}
#[test]
fn pin_g3_aggregate_policy_ls() {
    assert_eq!(produce_g3(false), read_golden("aggregate_policy_ls.bin"));
}
#[test]
fn pin_g4_aggregate_policy_to_local_ls() {
    assert_eq!(produce_g4(false), read_golden("aggregate_policy_to_local_ls.bin"));
}
#[test]
fn pin_g5_assemble_ls_from_gnn_probs() {
    assert_eq!(produce_g5(false), read_golden("assemble_ls_from_gnn_probs.bin"));
}
#[test]
fn pin_g6_record_position_graph() {
    assert_eq!(produce_g6(false), read_golden("record_position_graph.bin"));
}
#[test]
fn pin_g7_finalize_graph_outcome() {
    assert_eq!(produce_g7(false), read_golden("finalize_graph_outcome.bin"));
}

// ── LAW-07 mutation self-tests (flip ONE input element ⇒ diverge from golden) ─
#[test]
fn mut_g1_diverges() {
    assert_ne!(produce_g1(true), read_golden("aggregate_policy.bin"));
}
#[test]
fn mut_g2_diverges() {
    assert_ne!(produce_g2(true), read_golden("aggregate_policy_to_local.bin"));
}
#[test]
fn mut_g3_diverges() {
    assert_ne!(produce_g3(true), read_golden("aggregate_policy_ls.bin"));
}
#[test]
fn mut_g4_diverges() {
    assert_ne!(produce_g4(true), read_golden("aggregate_policy_to_local_ls.bin"));
}
#[test]
fn mut_g5_diverges() {
    assert_ne!(produce_g5(true), read_golden("assemble_ls_from_gnn_probs.bin"));
}
#[test]
fn mut_g6_diverges() {
    assert_ne!(produce_g6(true), read_golden("record_position_graph.bin"));
}
#[test]
fn mut_g7_diverges() {
    assert_ne!(produce_g7(true), read_golden("finalize_graph_outcome.bin"));
}
