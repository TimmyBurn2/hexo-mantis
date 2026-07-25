//! Soundness-fix proof subset for the legal-move cache (`UnsafeCell` payload,
//! `&mut`-gated invalidation, no `unsafe impl Sync` — Board is `Send + !Sync`).
//! These tests MUST run (and pass, zero UB findings) under
//! `cargo +nightly miri test -p mantis-core` — they exercise every real
//! `unsafe` in the crate: interleaved apply/undo/rebuild (exclusive-rebuild /
//! shared-return cycles), clone cache independence, coexisting shared
//! borrows, radius-override invalidation, and Clone's shared cap read while a
//! `&FxHashSet` from `legal_moves_set()` is live
//! (`miri_clone_while_set_borrowed` — the fifth named proof test, replacing
//! the Candidate-A-specific `miri_dirty_flag_set_while_borrowed_panics`,
//! whose trigger is statically unwritable now: invalidate-while-borrowed is
//! rejected at compile time, pinned by the `compile_fail,E0502` doctest on
//! `legal_moves_set`).

use mantis_core::board::Board;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// 200-step splitmix64-scripted interleaving of apply_move_tracked /
/// undo_move / legal_moves_set / legal_move_count on one board
/// (deterministic, no fs, no proptest).
#[test]
fn miri_apply_undo_legal_interleave() {
    let mut b = Board::new();
    let mut diffs = Vec::new();
    let mut s = 0x0dd5_eed0_0000_0042u64;
    for step in 0..200u32 {
        match splitmix(&mut s) % 4 {
            0 | 3 => {
                // apply a scripted legal move (tracked)
                let legal = b.legal_moves();
                assert!(!legal.is_empty(), "step {step}: no legal moves");
                let (q, r) = legal[(splitmix(&mut s) as usize) % legal.len()];
                diffs.push(b.apply_move_tracked(q, r).expect("legal move"));
            }
            1 => {
                if let Some(d) = diffs.pop() {
                    b.undo_move(d);
                }
            }
            _ => {
                // read through the cache both ways
                let n1 = b.legal_moves_set().len();
                let n2 = b.legal_move_count();
                assert_eq!(n1, n2, "step {step}: set len and count must agree");
                assert!(n1 > 0);
            }
        }
    }
    // Drain: undo everything and confirm the initial 25-cell region returns.
    while let Some(d) = diffs.pop() {
        b.undo_move(d);
    }
    assert_eq!(b.legal_move_count(), 25);
}

/// Clone while the parent cache is clean AND while dirty; mutate both;
/// assert legal sets diverge correctly (no aliased cache).
#[test]
fn miri_clone_shares_no_cache() {
    // Case 1: parent cache CLEAN at clone time.
    let mut parent = Board::new();
    parent.apply_move(0, 0).unwrap();
    let _ = parent.legal_moves_set().len(); // force rebuild → clean
    let mut child = parent.clone();
    parent.apply_move(1, 0).unwrap(); // P2 on parent only
    child.apply_move(-1, 0).unwrap(); // P2 on child only
    let p: Vec<(i32, i32)> = parent.legal_moves();
    let c: Vec<(i32, i32)> = child.legal_moves();
    assert!(!p.contains(&(1, 0)), "parent occupied (1,0)");
    assert!(p.contains(&(-1, 0)), "parent did NOT occupy (-1,0)");
    assert!(!c.contains(&(-1, 0)), "child occupied (-1,0)");
    assert!(c.contains(&(1, 0)), "child did NOT occupy (1,0)");
    assert_ne!(p, c, "diverged boards must have diverged legal sets");

    // Case 2: parent cache DIRTY at clone time.
    let mut parent = Board::new();
    parent.apply_move(0, 0).unwrap(); // cache dirty (no read since)
    let mut child = parent.clone();
    parent.apply_move(2, 0).unwrap();
    child.apply_move(-2, 0).unwrap();
    let p: Vec<(i32, i32)> = parent.legal_moves();
    let c: Vec<(i32, i32)> = child.legal_moves();
    assert!(!p.contains(&(2, 0)) && p.contains(&(-2, 0)));
    assert!(!c.contains(&(-2, 0)) && c.contains(&(2, 0)));
    assert_ne!(p, c);
}

/// Two clean-path `legal_moves_set()` references must coexist (shared borrows).
#[test]
fn miri_nested_shared_borrows() {
    let mut b = Board::new();
    b.apply_move(0, 0).unwrap();
    let g1 = b.legal_moves_set();
    let g2 = b.legal_moves_set();
    assert_eq!(g1.len(), g2.len());
    // Read through both references while both are live.
    for mv in g1.iter() {
        assert!(g2.contains(mv));
    }
    drop(g1);
    assert_eq!(g2.len(), 90);
}

/// Clone's shared cap read must coexist with a live `&FxHashSet` returned by
/// `legal_moves_set()` (both are shared reads through the `UnsafeCell` —
/// INV-1/INV-4 guarantee no exclusive borrow can be live when `clone()`
/// runs), and the clone's own first `legal_moves_set()` must rebuild
/// correctly from its skip-contents, dirty-marked cache.
#[test]
fn miri_clone_while_set_borrowed() {
    let mut b = Board::new();
    b.apply_move(0, 0).unwrap();
    let set = b.legal_moves_set(); // shared borrow, held across clone()
    assert_eq!(set.len(), 90, "radius 5 ball around (0,0) minus the stone");
    let clone = b.clone(); // shared cap read while `set` is live
    // The held reference stays valid and readable after the clone.
    assert_eq!(set.len(), 90);
    assert!(set.contains(&(1, 0)));
    // The clone's first legal_moves_set() rebuilds correctly (same position).
    let cloned_set = clone.legal_moves_set();
    assert_eq!(cloned_set.len(), 90);
    let mut a: Vec<(i32, i32)> = set.iter().copied().collect();
    let mut c: Vec<(i32, i32)> = cloned_set.iter().copied().collect();
    a.sort_unstable();
    c.sort_unstable();
    assert_eq!(a, c, "clone must rebuild the identical legal set");
}

/// set_legal_move_radius invalidation + rebuild.
/// Hex-ball area is 3r² + 3r + 1; minus the occupied stone.
#[test]
fn miri_radius_change_rebuild() {
    let mut b = Board::new();
    b.apply_move(0, 0).unwrap();
    assert_eq!(b.legal_move_count(), 90, "radius 5 default: 91 - 1");
    b.set_legal_move_radius(4);
    assert_eq!(b.legal_move_count(), 60, "radius 4: 61 - 1");
    b.set_legal_move_radius(6);
    assert_eq!(b.legal_move_count(), 126, "radius 6: 127 - 1");
    b.set_legal_move_radius(5);
    assert_eq!(b.legal_move_count(), 90, "back to radius 5");
}
