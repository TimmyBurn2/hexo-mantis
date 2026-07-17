//! Golden replay oracle: replays the predecessor-engine capture
//! (`tests/fixtures/board/board_replay_golden_v1.json`, schema
//! `board-golden-v1`) against this crate's Board and asserts EVERY recorded
//! field matches — moves, zobrist u128, legal-set sha256 digests, turn
//! structure, check_win, window_center, periodic full legal sets, sorted
//! cluster centers, sorted threat anchors, winning/threat move surfaces,
//! forced-win probes, and terminal outcome.
//!
//! A second test proves the checker BITES: flipping one move of game 0 must
//! produce loud zobrist + legal-digest divergences naming game/ply.
//!
//! Fixture provenance note: the cluster-center field inherits HashMap
//! iteration order in the massive-cluster dedup path; the capture records the
//! toolchain, and a cluster-center-only mismatch after a toolchain change is
//! fixture-regeneration territory, not a port defect.

#![cfg_attr(miri, allow(dead_code))]

use mantis_core::board::{Board, BoardGeometry, Player};
use serde::Deserialize;
use sha2::{Digest, Sha256};

// ── Fixture schema (board-golden-v1) ─────────────────────────────────────────

#[derive(Deserialize)]
struct Golden {
    schema: String,
    n_games: usize,
    #[allow(dead_code)]
    seed: String,
    #[allow(dead_code)]
    seed_formula: String,
    games: Vec<Game>,
}

#[derive(Deserialize)]
struct Game {
    index: usize,
    construction: Construction,
    geometry: Geometry,
    plies: Vec<PlyRec>,
    terminal: Terminal,
}

#[derive(Deserialize, Clone, Copy)]
struct Geometry {
    legal_move_radius: i32,
    cluster_threshold: i32,
    cluster_window_size: usize,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Construction {
    Default,
    Geometry,
    RadiusOverride { radius: i32 },
}

#[derive(Deserialize)]
struct PlyRec {
    mv: (i32, i32),
    zobrist: String,
    legal_sha256: String,
    current_player: i8,
    moves_remaining: u8,
    ply: u32,
    check_win: bool,
    window_center: (i32, i32),
    #[serde(default)]
    legal_moves: Option<Vec<(i32, i32)>>,
    #[serde(default)]
    cluster_centers: Option<Vec<(i32, i32)>>,
    #[serde(default)]
    threat_anchors: Option<Vec<(i32, i32)>>,
    #[serde(default)]
    winning_moves_p1: Option<Vec<(i32, i32)>>,
    #[serde(default)]
    winning_moves_p2: Option<Vec<(i32, i32)>>,
    #[serde(default)]
    threat_moves_p1: Option<Vec<(i32, i32)>>,
    #[serde(default)]
    threat_moves_p2: Option<Vec<(i32, i32)>>,
    /// Some(vec): computed; empty = no forced win, one element = the move.
    #[serde(default)]
    forced_win_move_d2: Option<Vec<(i32, i32)>>,
}

#[derive(Deserialize)]
struct Terminal {
    winner: i8,
    winning_line: Vec<(i32, i32)>,
    terminal_value_to_move: f32,
}

// ── Replay checker ───────────────────────────────────────────────────────────

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/board/board_replay_golden_v1.json")
}

fn load_golden() -> Golden {
    let raw = std::fs::read_to_string(fixture_path())
        .expect("golden fixture must exist at tests/fixtures/board/board_replay_golden_v1.json");
    let g: Golden = serde_json::from_str(&raw).expect("golden fixture must parse");
    assert_eq!(g.schema, "board-golden-v1");
    assert_eq!(g.n_games, g.games.len());
    g
}

fn sha_legal(legal_sorted: &[(i32, i32)]) -> String {
    let mut h = Sha256::new();
    for &(q, r) in legal_sorted {
        h.update(format!("{q},{r}\n").as_bytes());
    }
    format!("{:x}", h.finalize())
}

fn player_i8(p: Player) -> i8 {
    match p {
        Player::One => 1,
        Player::Two => -1,
    }
}

fn build_board(c: Construction, g: Geometry) -> Board {
    match c {
        Construction::Default => Board::new(),
        Construction::Geometry => Board::with_geometry(BoardGeometry {
            legal_move_radius: g.legal_move_radius,
            cluster_threshold: g.cluster_threshold,
            cluster_window_size: g.cluster_window_size,
        }),
        Construction::RadiusOverride { radius } => {
            let mut b = Board::new();
            b.set_legal_move_radius(radius);
            b
        }
    }
}

fn cmp<T: PartialEq + std::fmt::Debug>(
    div: &mut Vec<String>, game: usize, ply: u32, field: &str, got: &T, want: &T,
) {
    if got != want {
        div.push(format!(
            "game {game} ply {ply}: {field} diverged — replay produced {got:?}, golden records {want:?}"
        ));
    }
}

/// Replay every game; return every field divergence (empty = clean).
fn verify(golden: &Golden) -> Vec<String> {
    let mut div = Vec::new();
    for game in &golden.games {
        let g = game.index;
        let mut board = build_board(game.construction, game.geometry);
        for rec in &game.plies {
            let (q, r) = rec.mv;
            if let Err(e) = board.apply_move(q, r) {
                div.push(format!("game {g} ply {}: recorded move ({q},{r}) rejected: {e}", rec.ply));
                break;
            }
            cmp(&mut div, g, rec.ply, "zobrist", &format!("{:032x}", board.zobrist_hash), &rec.zobrist);
            let legal = board.legal_moves();
            cmp(&mut div, g, rec.ply, "legal_sha256", &sha_legal(&legal), &rec.legal_sha256);
            cmp(&mut div, g, rec.ply, "current_player", &player_i8(board.current_player), &rec.current_player);
            cmp(&mut div, g, rec.ply, "moves_remaining", &board.moves_remaining, &rec.moves_remaining);
            cmp(&mut div, g, rec.ply, "ply", &board.ply.index(), &rec.ply);
            cmp(&mut div, g, rec.ply, "check_win", &board.check_win(), &rec.check_win);
            cmp(&mut div, g, rec.ply, "window_center", &board.window_center(), &rec.window_center);
            if let Some(want) = &rec.legal_moves {
                cmp(&mut div, g, rec.ply, "legal_moves", &legal, want);
            }
            if let Some(want) = &rec.cluster_centers {
                let (_views, mut centers) = board.get_cluster_views();
                centers.sort_unstable();
                cmp(&mut div, g, rec.ply, "cluster_centers", &centers, want);
            }
            if let Some(want) = &rec.threat_anchors {
                let mut anchors = board.get_threat_anchors();
                anchors.sort_unstable();
                cmp(&mut div, g, rec.ply, "threat_anchors", &anchors, want);
            }
            if let Some(want) = &rec.winning_moves_p1 {
                cmp(&mut div, g, rec.ply, "winning_moves_p1", &board.winning_moves(Player::One), want);
            }
            if let Some(want) = &rec.winning_moves_p2 {
                cmp(&mut div, g, rec.ply, "winning_moves_p2", &board.winning_moves(Player::Two), want);
            }
            if let Some(want) = &rec.threat_moves_p1 {
                cmp(&mut div, g, rec.ply, "threat_moves_p1", &board.threat_moves(Player::One), want);
            }
            if let Some(want) = &rec.threat_moves_p2 {
                cmp(&mut div, g, rec.ply, "threat_moves_p2", &board.threat_moves(Player::Two), want);
            }
            if let Some(want) = &rec.forced_win_move_d2 {
                let got = board.forced_win_move(2).map(|m| vec![m]).unwrap_or_default();
                cmp(&mut div, g, rec.ply, "forced_win_move_d2", &got, want);
            }
        }
        let ply = board.ply.index();
        cmp(&mut div, g, ply, "terminal.winner", &board.winner().map_or(0, player_i8), &game.terminal.winner);
        cmp(&mut div, g, ply, "terminal.winning_line", &board.find_winning_line(), &game.terminal.winning_line);
        cmp(&mut div, g, ply, "terminal.value_to_move", &board.terminal_value_to_move(), &game.terminal.terminal_value_to_move);
    }
    div
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// ⊕ The golden replay: every recorded field of all 27 games must match.
#[test]
#[cfg_attr(miri, ignore)] // file I/O — excluded set per prereg
fn golden_replay_matches_every_recorded_field() {
    let golden = load_golden();
    assert_eq!(golden.n_games, 27, "board-golden-v1 carries 27 games");
    let div = verify(&golden);
    assert!(
        div.is_empty(),
        "golden replay diverged on {} field(s):\n{}",
        div.len(),
        div.join("\n")
    );
}

/// Mutation self-test: the checker must BITE. Flipping one move of game 0
/// (q+1) must produce loud zobrist + legal-digest divergences naming
/// game/ply. A checker that passes a flipped move fails this test.
#[test]
#[cfg_attr(miri, ignore)] // file I/O — excluded set per prereg
fn golden_replay_mutation_self_test() {
    let mut golden = load_golden();
    golden.games[0].plies[0].mv.0 += 1;
    let div = verify(&golden);
    assert!(
        !div.is_empty(),
        "MUTATION SELF-TEST FAILED: the checker passed a flipped move"
    );
    assert!(
        div.iter().any(|d| d.starts_with("game 0 ply 1:") && d.contains("zobrist")),
        "mutation must surface a zobrist divergence naming game 0 ply 1; got:\n{}",
        div.join("\n")
    );
    assert!(
        div.iter().any(|d| d.contains("legal_sha256")),
        "mutation must surface a legal-digest divergence; got:\n{}",
        div.join("\n")
    );
}
