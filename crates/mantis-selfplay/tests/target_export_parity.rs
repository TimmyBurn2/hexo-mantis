// R8 >300 justify: ONE oracle family (the exported-target parity Rust leg) = flat fixture
// reader + the production search harness + three HEAD-runnable legs + the gated post-fix
// record-chain leg; the reader and harness must sit in the same frozen file as the tests that
// prove they read the committed fixture correctly.
//! Exported-target parity, Rust leg, against the frozen fixture family
//! `tests/fixtures/eval_selfplay_parity/target_parity_v1.json` (+ `_dispersed_`).
//!
//! Three legs run at HEAD: the production expand → `get_policy_ls` export must equal the frozen
//! `(coord, mass)` pairs and sum to 1; the same pairs must survive a `GraphRecord` push and
//! re-sample unchanged. The fourth, `o1r_record_chain_full_mass`, drives the full chain through
//! `record_position_graph`'s `Result` signature and is compiled only under the `phase_t_postfix`
//! feature — a loud gate rather than a silent skip.
//!
//! Killers: M-A (the export legs), M-J (the record-chain and buffer legs). M-D/M-K stay green
//! here by design: every fixture position carries <= 57 nonzero cells and this chain never
//! passes finalize.

use mantis_core::board::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_search::{LegalSetPolicy, MCTSTree};
use mantis_selfplay::replay::hexg::{GraphRecord, HexgBuffer};

const N_SIMS: usize = 50;
const LEAF_BATCH: usize = 8;
const PAIR_TOL: f64 = 1e-6;

// Flat fixture reader, house pattern: no serde dep.
fn fixture_text(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/eval_selfplay_parity")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {} unreadable: {e}", path.display()))
}

fn value_of<'a>(src: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":");
    let at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("fixture key {key:?} absent"));
    let rest = src[at + needle.len()..].trim_start();
    if let Some(inner) = rest.strip_prefix('[') {
        let end = inner
            .find(']')
            .unwrap_or_else(|| panic!("unterminated array for {key:?}"));
        &inner[..end]
    } else if let Some(inner) = rest.strip_prefix('"') {
        let end = inner
            .find('"')
            .unwrap_or_else(|| panic!("unterminated string for {key:?}"));
        &inner[..end]
    } else {
        let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
        rest[..end].trim_end()
    }
}

fn ints(src: &str, key: &str) -> Vec<i64> {
    value_of(src, key)
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<i64>()
                .unwrap_or_else(|e| panic!("{key:?}: {t:?} ({e})"))
        })
        .collect()
}

fn floats(src: &str, key: &str) -> Vec<f64> {
    value_of(src, key)
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("{key:?}: {t:?} ({e})"))
        })
        .collect()
}

fn scalar(src: &str, key: &str) -> i64 {
    let raw = value_of(src, key);
    raw.trim()
        .parse::<i64>()
        .unwrap_or_else(|e| panic!("{key:?}: {raw:?} ({e})"))
}

fn pairs(flat: &[i64]) -> Vec<(i32, i32)> {
    assert_eq!(flat.len() % 2, 0, "coord array must be pairs");
    flat.chunks_exact(2)
        .map(|c| (c[0] as i32, c[1] as i32))
        .collect()
}

/// One fixture position, fully decoded + honesty-checked against a replayed board.
struct Pos {
    id: String,
    board: Board,
    n_actions: usize,
    trunk: i32,
    stones: Vec<(i16, i16, i8)>,
    current_player: i8,
    moves_remaining: u8,
    ply_index: u16,
    expected: Vec<((i32, i32), f64)>,
}

fn load_pos(src: &str, i: usize) -> Pos {
    let spec = lookup_or_panic("gnn_axis_v1");
    let key = |s: &str| format!("p{i}_{s}");
    let id = value_of(src, &key("id")).to_string();
    let geom = BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    };
    let mut board = Board::with_geometry(geom);
    for (q, r) in pairs(&ints(src, &key("moves"))) {
        board
            .apply_move(q, r)
            .unwrap_or_else(|e| panic!("{id}: move ({q},{r}) rejected: {e}"));
    }
    // Fixture honesty: re-derive every recorded claim from the replayed board.
    assert_eq!(
        board.legal_moves().len() as i64,
        scalar(src, &key("n_legal")),
        "{id}: n_legal drift"
    );
    assert_eq!(
        i64::from(board.current_player as i8),
        scalar(src, &key("current_player")),
        "{id}: player drift"
    );
    assert_eq!(
        i64::from(board.moves_remaining),
        scalar(src, &key("moves_remaining")),
        "{id}: mr drift"
    );
    let wc = pairs(&ints(src, &key("window_center")));
    assert_eq!(board.window_center(), wc[0], "{id}: window_center drift");

    let stones_flat = ints(src, &key("stones"));
    assert_eq!(stones_flat.len() % 3, 0);
    let stones: Vec<(i16, i16, i8)> = stones_flat
        .chunks_exact(3)
        .map(|c| (c[0] as i16, c[1] as i16, c[2] as i8))
        .collect();
    assert_eq!(
        stones.len(),
        board.cells_iter().count(),
        "{id}: stone-count drift"
    );

    let coords = pairs(&ints(src, &key("expected_coords")));
    let mass = floats(src, &key("expected_mass"));
    assert_eq!(coords.len(), mass.len(), "{id}: pair arrays misaligned");
    let sum: f64 = mass.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "{id}: fixture pairs must sum to 1, got {sum}"
    );
    assert!(
        mass.len() <= 128,
        "{id}: fixture pairs exceed the suite's 128-slot test geometry"
    );

    Pos {
        id,
        board,
        n_actions: spec.policy_logit_count,
        trunk: spec.trunk_size as i32,
        stones,
        current_player: scalar(src, &key("current_player")) as i8,
        moves_remaining: scalar(src, &key("moves_remaining")) as u8,
        ply_index: scalar(src, &key("ply_index")) as u16,
        expected: coords.into_iter().zip(mass).collect(),
    }
}

fn no_drop_uniform(board: &Board, n_actions: usize) -> LegalSetPolicy {
    let legal = board.legal_moves();
    let p = 1.0_f32 / legal.len().max(1) as f32;
    let mut ls = LegalSetPolicy {
        dense: vec![0.0; n_actions],
        overflow: Default::default(),
    };
    for (q, r) in legal {
        let idx = board.window_flat_idx(q, r);
        if idx < n_actions {
            ls.dense[idx] = p;
        } else {
            ls.overflow.insert((q, r), p);
        }
    }
    ls
}

fn searched_tree(pos: &Pos) -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(pos.board.clone());
    let mut done = 0;
    while done < N_SIMS {
        let take = LEAF_BATCH.min(N_SIMS - done);
        let boards = tree
            .select_leaves(take)
            .expect("select_leaves: no desync in this fixture");
        if boards.is_empty() {
            break;
        }
        let policies: Vec<LegalSetPolicy> = boards
            .iter()
            .map(|b| no_drop_uniform(b, pos.n_actions))
            .collect();
        let values = vec![0.0_f32; boards.len()];
        let centers: Vec<(i32, i32)> = boards.iter().map(|b| b.window_center()).collect();
        tree.expand_and_backup_ls_at(&policies, &values, &centers, pos.trunk);
        done += boards.len();
    }
    tree
}

fn export_map(pos: &Pos, ls: &LegalSetPolicy) -> Vec<((i32, i32), f64)> {
    let mut out = Vec::new();
    for &(q, r) in &pos.board.legal_moves() {
        let flat = pos.board.window_flat_idx(q, r);
        let m = if flat < pos.n_actions {
            f64::from(ls.dense[flat])
        } else {
            f64::from(ls.overflow.get(&(q, r)).copied().unwrap_or(0.0))
        };
        if m > 0.0 {
            out.push(((q, r), m));
        }
    }
    out.sort_unstable_by_key(|&((q, r), _)| {
        (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF)
    });
    out
}

fn assert_pairs_match(id: &str, got: &[((i32, i32), f64)], want: &[((i32, i32), f64)]) {
    let g: Vec<(i32, i32)> = got.iter().map(|&(c, _)| c).collect();
    let w: Vec<(i32, i32)> = want.iter().map(|&(c, _)| c).collect();
    assert_eq!(
        g, w,
        "{id}: exported nonzero coord set != frozen fixture pairs"
    );
    for (&(c, gm), &(_, wm)) in got.iter().zip(want) {
        assert!(
            (gm - wm).abs() <= PAIR_TOL,
            "{id}: mass at {c:?} = {gm} != frozen {wm} (tol {PAIR_TOL})"
        );
    }
    let total: f64 = got.iter().map(|&(_, m)| m).sum();
    assert!(
        (total - 1.0).abs() <= PAIR_TOL,
        "{id}: exported mass {total} != 1"
    );
}

fn check_export(src: &str, i: usize) {
    let pos = load_pos(src, i);
    let tree = searched_tree(&pos);
    let ls = tree.get_policy_ls(1.0, pos.n_actions);
    let got = export_map(&pos, &ls);
    assert_pairs_match(&pos.id, &got, &pos.expected);
}

fn check_roundtrip(src: &str, i: usize) {
    let pos = load_pos(src, i);
    let visits: Vec<(i16, i16, f32)> = pos
        .expected
        .iter()
        .map(|&((q, r), m)| (q as i16, r as i16, m as f32))
        .collect();
    let rec = GraphRecord {
        stones: pos.stones.clone(),
        visits,
        tail_mass: 0.0,
        current_player: pos.current_player,
        moves_remaining: pos.moves_remaining,
        ply_index: pos.ply_index,
        is_full_search: true,
        outcome: 0.0,
        value_valid: true,
        game_length: 0,
        game_id: -1,
    };
    let mut buf = HexgBuffer::new(2, "gnn_axis_v1", 128).expect("graph buffer");
    buf.push_record_impl(&rec, 1)
        .unwrap_or_else(|e| panic!("{}: push refused: {e}", pos.id));
    let (graphs, targets) = buf
        .sample_graph_batch_impl(1, false, 0.0, 1)
        .unwrap_or_else(|e| panic!("{}: sample failed (mass_drop_check?): {e}", pos.id));
    assert_eq!(graphs.len(), 1);
    let g = &graphs[0];
    let mut got: Vec<((i32, i32), f64)> = Vec::new();
    for (j, &row) in g.legal_node_gather.iter().enumerate() {
        let q = g.node_coords[row as usize * 2];
        let r = g.node_coords[row as usize * 2 + 1];
        let m = f64::from(targets.policy_target[j]);
        if m > 0.0 {
            got.push(((q, r), m));
        }
    }
    got.sort_unstable_by_key(|&((q, r), _)| {
        (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF)
    });
    assert_pairs_match(&pos.id, &got, &pos.expected);
}

#[test]
fn o1r_export_matches_fixture() {
    let src = fixture_text("target_parity_v1.json");
    assert_eq!(scalar(&src, "schema"), 1);
    assert_eq!(value_of(&src, "encoding"), "gnn_axis_v1");
    let n = scalar(&src, "n_positions");
    assert!(n >= 3, "v1 fixture is pre-registered at >=3 positions");
    for i in 0..n as usize {
        check_export(&src, i);
    }
}

#[test]
fn o1r_export_matches_dispersed_fixture() {
    let src = fixture_text("target_parity_dispersed_v1.json");
    assert_eq!(scalar(&src, "schema"), 1);
    let n = scalar(&src, "n_positions");
    assert_eq!(n, 3, "dispersed companion is pre-registered at 3 positions");
    // Band preconditions: p0 in the 193-235 n_legal band, p1 the HIGH-magnitude degenerate row,
    // p2 the >=5000-legal row.
    assert!(
        (193..=235).contains(&scalar(&src, "p0_n_legal")),
        "p0 left the 193-235 band"
    );
    assert!(
        scalar(&src, "p2_n_legal") >= 5000,
        "p2 left the >=5000-legal regime"
    );
    for i in 0..n as usize {
        check_export(&src, i);
    }
}

#[test]
fn o1r_buffer_roundtrip_preserves_pairs() {
    for name in ["target_parity_v1.json", "target_parity_dispersed_v1.json"] {
        let src = fixture_text(name);
        for i in 0..scalar(&src, "n_positions") as usize {
            check_roundtrip(&src, i);
        }
    }
}

// Compiles only under the `phase_t_postfix` feature: the pre-fix `record_position_graph`
// signature makes this leg non-compilable, so the gate is loud rather than a silent skip.
#[cfg(feature = "phase_t_postfix")]
#[test]
fn o1r_record_chain_full_mass() {
    use mantis_selfplay::records::record_position_graph;
    const MAX_VISITS: usize = 128; // test slot geometry; derived in production

    for name in ["target_parity_v1.json", "target_parity_dispersed_v1.json"] {
        let src = fixture_text(name);
        for i in 0..scalar(&src, "n_positions") as usize {
            let pos = load_pos(&src, i);
            let tree = searched_tree(&pos);
            let ls = tree.get_policy_ls(1.0, pos.n_actions);
            let rec = record_position_graph(
                &pos.board,
                &ls,
                pos.trunk,
                pos.current_player,
                pos.moves_remaining,
                pos.ply_index,
                true,
                MAX_VISITS,
                None,
            )
            .unwrap_or_else(|e| panic!("{}: a full-mass export must record: {e}", pos.id));
            let mut buf = HexgBuffer::new(2, "gnn_axis_v1", 128).expect("graph buffer");
            buf.push_record_impl(&rec, 1)
                .unwrap_or_else(|e| panic!("{}: push: {e}", pos.id));
            let (graphs, targets) = buf
                .sample_graph_batch_impl(1, false, 0.0, 1)
                .unwrap_or_else(|e| panic!("{}: sample: {e}", pos.id));
            let g = &graphs[0];
            let mut got: Vec<((i32, i32), f64)> = Vec::new();
            for (j, &row) in g.legal_node_gather.iter().enumerate() {
                let q = g.node_coords[row as usize * 2];
                let r = g.node_coords[row as usize * 2 + 1];
                let m = f64::from(targets.policy_target[j]);
                if m > 0.0 {
                    got.push(((q, r), m));
                }
            }
            got.sort_unstable_by_key(|&((q, r), _)| {
                (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF)
            });
            assert_pairs_match(&pos.id, &got, &pos.expected);
        }
    }
}
