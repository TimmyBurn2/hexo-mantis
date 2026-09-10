//! PUCT root-Dirichlet golden byte-identity harness (site S4).
//!
//! Captures `f32::to_bits()` of every post-noise root prior under a FIXED seed, and
//! asserts it against the frozen bits in
//! `tests/fixtures/search_golden/completed_q/golden_bits.txt`. BIT-EXACT — an FMA→mul
//! regression is sub-ULP but nonzero, so `==` on bits catches it and `abs()<1e-6` would
//! not.
//!
//! S1/S2/S3 ARE GONE WITH THEIR SUBJECT. They pinned the LEGACY completed-Q exporters
//! and the legacy `GumbelSearchState::score`, all three of which the search-kind
//! collapse deletes; a golden over a deleted arm is not a guard, it is a fixture nobody
//! can re-derive. Their rows stay in the golden FILE, which is manifest-sha-pinned and
//! therefore not editable here — `parse_golden` reads by key, so the unread rows are
//! inert. S4 is untouched and its bits still hold, which is the property that matters:
//! the deletion did not perturb the PUCT path.

// The seeded RNG constant (`0x5_4D_1_5EED`) is FROZEN — the golden was captured with
// this exact seed. Its underscore grouping is preserved verbatim; suppress the cosmetic
// `unusual_byte_groupings` lint.
#![allow(clippy::unusual_byte_groupings)]

use super::*;
use mantis_core::board::Board;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 19-window stride with a pass slot — the shape the golden was captured at.
const N_ACTIONS: usize = 19 * 19 + 1;

/// S4 PUCT Dirichlet root-noise bits: build a seeded tree, expand root, sample
/// symmetric Dirichlet on a FIXED seed, apply to root, dump
/// every root child's post-noise `prior` via `to_bits()`.
///
/// This pins the PUCT root-noise sequence f32::to_bits-identical. Dirichlet is a PUCT
/// mechanism only — the Gumbel kind's root exploration IS its Gumbel draw — so S4
/// holding byte-identical proves a change to the Gumbel side did not reach the PUCT
/// path.
const S4_DIRICHLET_SEED: u64 = 0x5_4D_1_5EED;
const S4_DIRICHLET_ALPHA: f32 = 0.3;
const S4_DIRICHLET_EPSILON: f32 = 0.25;

fn s4_puct_dirichlet_bits() -> Vec<u32> {
    let mut tree = MCTSTree::new(1.5);
    let mut board = Board::new();
    board.apply_move(0, 0).expect("(0,0) legal");
    tree.new_game(board);
    let n_actions = N_ACTIONS;
    let policy = vec![1.0 / n_actions as f32; n_actions];
    let _ = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    tree.expand_and_backup(&[policy], &[0.0]);

    let n_ch = tree.pool[0].n_children as usize;
    let mut rng = StdRng::seed_from_u64(S4_DIRICHLET_SEED);
    let noise = crate::mcts::dirichlet::sample_dirichlet(S4_DIRICHLET_ALPHA, n_ch, &mut rng);
    tree.apply_dirichlet_to_root(&noise, S4_DIRICHLET_EPSILON);

    let first = tree.pool[0].first_child as usize;
    (0..n_ch)
        .map(|j| tree.pool[first + j].prior.to_bits())
        .collect()
}

// Golden roster: (name, live_bits) computed from the live code

/// All goldens as `(line-key, live u32 bits)`. The line-key must match the
/// prefix in `golden_bits.txt`. One source of truth for both capture + assert.
fn all_goldens() -> Vec<(String, Vec<u32>)> {
    vec![("S4_PUCT_DIRICHLET".to_string(), s4_puct_dirichlet_bits())]
}

/// Frozen golden bits (decimal u32 to_bits()). One line per fixture:
/// `KEY v0 v1 v2 ...`. Relocated to the repo-root fixtures tree.
const GOLDEN_BITS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/search_golden/completed_q/golden_bits.txt"
));

fn parse_golden(key: &str) -> Vec<u32> {
    for line in GOLDEN_BITS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let k = it.next().expect("golden line has a key");
        if k == key {
            return it
                .map(|t| t.parse::<u32>().expect("u32 golden token"))
                .collect();
        }
    }
    panic!("golden key {key:?} not found in golden_bits.txt");
}

// Capture (ignored) — regenerates golden_bits.txt from the live code

#[test]
#[ignore]
fn test_capture_goldens_print() {
    use std::fmt::Write as _;
    let mut s = String::new();
    s.push_str("# completed-Q goldens (f32::to_bits, decimal u32).\n");
    s.push_str("# One line per fixture: KEY bit0 bit1 ...\n");
    s.push_str("# Regenerate: cargo test -p mantis-search --lib golden_tests::test_capture_goldens_print -- --ignored\n");
    for (key, bits) in all_goldens() {
        write!(s, "{key}").unwrap();
        for b in &bits {
            write!(s, " {b}").unwrap();
        }
        s.push('\n');
    }
    // Print to stdout — operator pipes into the golden file once at HEAD.
    print!("{s}");
}

// Byte-identity assertions (non-ignored)

#[test]
fn test_golden_s4_puct_dirichlet_unchanged() {
    let live = s4_puct_dirichlet_bits();
    let golden = parse_golden("S4_PUCT_DIRICHLET");
    assert_eq!(
        live, golden,
        "S4_PUCT_DIRICHLET: PUCT root Dirichlet noise bits changed — a change on the \
         Gumbel side must NOT perturb the PUCT path (sample_dirichlet + \
         apply_dirichlet_to_root untouched contract)"
    );
}
