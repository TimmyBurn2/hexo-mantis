//! D6 axial symmetry primitive + the proof that it is a lattice bijection.
//!
//! Ported from the predecessor engine's `replay_buffer/sym_tables.rs` (the D6 half of the
//! 3-way split). R346(f) deleted the dense ring, and with it everything in this file that
//! existed to scatter a WINDOW-clamped grid record: the `SymTables` LUTs, the per-spec
//! `sym_tables_for` singletons, the dropped-cell derivation and the `WINDOW_PRESERVING_SYMS`
//! draw gate. What survives is the window-free half — the axial transform itself, which the
//! graph (HEXG) sample path uses to rotate coords and visit keys.

// Geometric constants (board-size invariant)

/// D6 group order: 6 rotations × 2 (with/without prior reflection).
pub const N_SYMS: usize = 12;

/// Apply the canonical `(q, r) → (−r, q + r)` 60° rotation, `n_rot` times.
#[inline]
fn rotate_n(mut q: i32, mut r: i32, n_rot: usize) -> (i32, i32) {
    for _ in 0..n_rot {
        let nq = -r;
        let nr = q + r;
        q = nq;
        r = nr;
    }
    (q, r)
}

/// Apply D6 element `sym_idx ∈ 0..N_SYMS` to an axial coordinate `(q, r)`.
///
/// **Single source of the 12 D6 elements.** The element is reflect-then-rotate —
/// `reflect = sym_idx >= 6` swaps axes `(q, r) → (r, q)` FIRST, then `sym_idx % 6`
/// × 60° rotations `(q, r) → (−r, q+r)`. `sym_idx 0` is the identity. Board-size
/// invariant (a pure axial lattice automorphism), so it is the correct primitive
/// for the infinite-board graph coords: there is no window to leave and no clamp,
/// which is why the HEXG sample path draws over the FULL group.
#[inline]
#[must_use]
pub fn rotate_axial(q: i32, r: i32, sym_idx: usize) -> (i32, i32) {
    let (mut q, mut r) = (q, r);
    // Optional reflection first (swap axes).
    if sym_idx >= 6 {
        (q, r) = (r, q);
    }
    rotate_n(q, r, sym_idx % 6)
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Live board sizes, taken from the registry rather than hardcoded, so a newly
    /// registered encoding is covered automatically (LAW-08 / derive-never-transcribe).
    fn registry_board_sizes() -> Vec<usize> {
        let mut sizes: Vec<usize> = mantis_encoding::all_specs().map(|s| s.board_size).collect();
        sizes.sort_unstable();
        sizes.dedup();
        assert!(
            !sizes.is_empty(),
            "the registry must ship at least one encoding"
        );
        sizes
    }

    /// R245: the GRAPH path is structurally unaffected by any window. `rotate_axial` is an
    /// axial lattice automorphism with no window to leave, so it is injective (hence a
    /// bijection of the lattice) for ALL 12 elements. Checked over a box strictly larger
    /// than the widest shipped board, so no element could hide a drop at the boundary.
    #[test]
    fn rotate_axial_is_injective_for_all_twelve_elements() {
        use std::collections::HashSet;
        let max_bs = registry_board_sizes().into_iter().max().expect("non-empty");
        let reach = i32::try_from(max_bs).expect("board_size fits i32");
        for s in 0..N_SYMS {
            let mut seen: HashSet<(i32, i32)> = HashSet::new();
            let mut n = 0usize;
            for q in -reach..=reach {
                for r in -reach..=reach {
                    assert!(
                        seen.insert(rotate_axial(q, r, s)),
                        "rotate_axial: sym {s} is not injective at ({q},{r})"
                    );
                    n += 1;
                }
            }
            assert_eq!(seen.len(), n, "rotate_axial: sym {s} lost points");
        }
    }

    /// The graph sample path must keep the FULL 12-element draw: `rotate_axial` is exact on
    /// every element, so any restriction there would discard augmentation for no correctness
    /// gain. Source-presence, because a narrowed draw compiles cleanly and is silent in a run.
    #[test]
    fn the_graph_sample_site_keeps_the_full_group() {
        const HEXG_SAMPLE: &str = include_str!("hexg/sample.rs");
        assert!(
            HEXG_SAMPLE.contains("self.rng.random_range(0..N_SYMS)"),
            "the graph sample path must draw over the FULL 12-element group"
        );
    }
}
