//! D6 symmetry scatter/axis-perm tables + their derivation tests kept together —
//! the tables and the proofs that they equal the coordinate transform are one
//! indivisible unit (R8 justification: >300 LOC by design; splitting the LUTs
//! from the byte-exact proofs that certify them would break the audit chain).
//!
//! Ported verbatim from the predecessor engine's `replay_buffer/sym_tables.rs`
//! (the D6 half of the 3-way split). The v6-only buffer-stride/action constants
//! (`N_PLANES`/`N_CELLS`/`N_ACTIONS`/`STATE_STRIDE`/`CHAIN_STRIDE`/`POLICY_STRIDE`/
//! `AUX_STRIDE`/`BOARD_H`/`BOARD_W`) are KILLED — all widths come from
//! `RegistrySpec` accessors, the sole authority. The weight schedule moved to
//! `schedule.rs`. Only the board-size-invariant D6/hex-axis geometry constants
//! `N_SYMS` (group order) and `N_CHAIN_PLANES` (table dimension) are retained.

use mantis_encoding::RegistrySpec;

// ── Geometric constants (board-size invariant) ─────────────────────────────────

/// D6 group order: 6 rotations × 2 (with/without prior reflection).
pub const N_SYMS: usize = 12;
/// Number of Q13 chain-length planes: 3 hex axes × 2 players. The compile-time
/// inner dimension of `axis_perm`/`chain_src_lookup`; NOT a buffer stride
/// (`chain_stride()` is spec-derived).
pub const N_CHAIN_PLANES: usize = 6;
/// Offset used when building the chain_src_lookup (chains are their own buffer).
pub const CHAIN_PLANE_OFFSET: usize = 0;

/// Canonical hex-basis directions mirroring the board `HEX_AXES`. Used to derive
/// the axis-plane permutation under each symmetry.
pub(crate) const HEX_BASIS: [(i32, i32); 3] = [
    (1, 0),  // axis 0 — E/W
    (0, 1),  // axis 1 — NE/SW
    (1, -1), // axis 2 — SE/NW
];

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
/// **Single source of the 12 D6 elements**: the CNN cell-scatter (`with_shape`
/// below) and the GNN HEXG sample-path coord/visit-key rotation (`replay::hexg::
/// sample`) BOTH call this, so "D6 element `s`" means the identical geometry on
/// the dense-grid and axis-graph paths. The element is reflect-then-rotate —
/// `reflect = sym_idx >= 6` swaps axes `(q, r) → (r, q)` FIRST, then `sym_idx % 6`
/// × 60° rotations `(q, r) → (−r, q+r)` — byte-exactly matching the CNN
/// scatter-table construction. `sym_idx 0` is the identity. Board-size invariant
/// (pure axial lattice automorphism), so it is the correct primitive for the
/// infinite-board graph coords (no window clamp).
#[inline]
#[must_use]
pub fn rotate_axial(q: i32, r: i32, sym_idx: usize) -> (i32, i32) {
    let (mut q, mut r) = (q, r);
    // Optional reflection first (swap axes) — matches `with_shape` construction.
    if sym_idx >= 6 {
        (q, r) = (r, q);
    }
    rotate_n(q, r, sym_idx % 6)
}

/// Compare two axial vectors in a direction-unsigned sense (axis identity).
#[inline]
fn same_axis(a: (i32, i32), b: (i32, i32)) -> bool {
    a == b || (-a.0, -a.1) == b
}

// ── SymTables ──────────────────────────────────────────────────────────────────

/// Precomputed scatter tables for all 12 hexagonal symmetries.
///
/// `scatter[s]` is the list of `(src_cell, dst_cell)` pairs for symmetry `s`.
/// Cells that fall outside the board window after transformation are omitted —
/// the corresponding output cells remain zero (matching the Python behaviour).
///
/// `axis_perm[s]` is the per-symmetry axis-plane remap for the Q13 chain-length
/// planes; `chain_src_lookup[s]` fuses coordinate + axis-plane remap for the 6
/// chain planes.
pub struct SymTables {
    /// Board side length for which scatter tables were built (odd only). v6: 19.
    pub board_size: usize,
    /// Total cells = `board_size * board_size`. v6: 361. v6w25: 625.
    pub n_cells: usize,
    /// State plane count this table targets. v6: 8.
    pub n_planes: usize,
    pub scatter: [Vec<(u16, u16)>; N_SYMS],
    /// Per-symmetry axis-plane remap for Q13 chain-length planes.
    /// `axis_perm[s][dst_j] = src_i`: destination plane for axis j reads from
    /// source plane for axis i under symmetry s. Board-size invariant.
    pub axis_perm: [[usize; 3]; N_SYMS],
    /// Fused per-symmetry source-plane lookup for the 6 chain-length planes.
    /// `chain_src_lookup[s][dst_p] = src_p`: coordinate + axis-plane remap.
    pub chain_src_lookup: [[usize; N_CHAIN_PLANES]; N_SYMS],
}

impl Default for SymTables {
    /// Equivalent to `SymTables::new()` — v6 default shape (19×19, 8 planes).
    fn default() -> Self {
        Self::new()
    }
}

impl SymTables {
    /// Build sym tables at the v6 default shape (`board_size=19, n_planes=8`).
    pub fn new() -> Self {
        Self::with_shape(19, 8)
    }

    /// Build sym tables at an arbitrary square board shape and plane count.
    ///
    /// The 12-fold scatter LUTs depend on `board_size` (hex window dimensions);
    /// axis_perm is board-size invariant (purely a function of the 3 hex axes);
    /// chain_src_lookup is plane-count invariant (always 6 chain planes
    /// regardless of state plane count).
    ///
    /// Panics if `board_size` is even (sym scatter assumes a centred odd window).
    pub fn with_shape(board_size: usize, n_planes: usize) -> Self {
        assert!(
            board_size % 2 == 1,
            "SymTables: board_size must be odd, got {board_size}"
        );
        let n_cells = board_size * board_size;
        let half = (board_size as i32 - 1) / 2;

        // Axial → flat index.  Returns None if the result is out of the window.
        let to_flat = |q: i32, r: i32| -> Option<u16> {
            let qi = q + half;
            let ri = r + half;
            if qi >= 0 && qi < board_size as i32 && ri >= 0 && ri < board_size as i32 {
                Some((qi as usize * board_size + ri as usize) as u16)
            } else {
                None
            }
        };

        // Flat index → axial coordinates.
        let from_flat = |flat: usize| -> (i32, i32) {
            ((flat / board_size) as i32 - half, (flat % board_size) as i32 - half)
        };

        const EMPTY: Vec<(u16, u16)> = Vec::new();
        let mut scatter = [EMPTY; N_SYMS];
        let mut axis_perm = [[0usize; 3]; N_SYMS];

        for sym_idx in 0..N_SYMS {
            let reflect = sym_idx >= 6;
            let n_rot = sym_idx % 6;
            let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(n_cells);

            for src in 0..n_cells {
                let (sq, sr) = from_flat(src);
                // Single-source D6 transform (reflect-then-rotate).
                let (q, r) = rotate_axial(sq, sr, sym_idx);

                if let Some(dst) = to_flat(q, r) {
                    pairs.push((src as u16, dst));
                }
            }

            scatter[sym_idx] = pairs;

            // Derive the axis-plane permutation for this symmetry by applying the
            // SAME transform to each hex basis vector and matching the result to
            // one of the three canonical axes (direction-unsigned).
            let mut perm = [usize::MAX; 3];
            // src_i indexes HEX_BASIS AND is the derived perm source — kept verbatim.
            #[allow(clippy::needless_range_loop)]
            for src_i in 0..3 {
                let (mut q, mut r) = HEX_BASIS[src_i];
                if reflect {
                    (q, r) = (r, q);
                }
                let (tq, tr) = rotate_n(q, r, n_rot);
                let mut matched = false;
                for dst_j in 0..3 {
                    if same_axis((tq, tr), HEX_BASIS[dst_j]) {
                        perm[dst_j] = src_i;
                        matched = true;
                        break;
                    }
                }
                debug_assert!(
                    matched,
                    "transformed basis axis {:?} did not match any canonical axis \
                     (sym_idx={}, reflect={}, n_rot={})",
                    (tq, tr), sym_idx, reflect, n_rot
                );
            }
            debug_assert!(
                perm.iter().all(|&i| i < 3),
                "axis_perm[{sym_idx}] has an unset slot: {perm:?}"
            );
            // Sanity check: perm must be a bijection on {0,1,2}.
            let mut seen = [false; 3];
            for &i in &perm {
                debug_assert!(!seen[i], "axis_perm[{sym_idx}] is not a bijection: {perm:?}");
                seen[i] = true;
            }
            axis_perm[sym_idx] = perm;
        }

        // Build chain_src_lookup for the 6 chain-length planes.
        let mut chain_src_lookup = [[0usize; N_CHAIN_PLANES]; N_SYMS];
        for s in 0..N_SYMS {
            for dst_axis in 0..3 {
                let src_axis = axis_perm[s][dst_axis];
                for player_off in 0..2 {
                    chain_src_lookup[s][2 * dst_axis + player_off] =
                        CHAIN_PLANE_OFFSET + 2 * src_axis + player_off;
                }
            }
        }

        SymTables {
            board_size,
            n_cells,
            n_planes,
            scatter,
            axis_perm,
            chain_src_lookup,
        }
    }
}

// ── sym_tables_for() — per-spec lazy constructor ───────────────────────────────

use std::sync::LazyLock;

/// Return the pre-built `SymTables` for the given encoding spec, keyed on
/// `spec.sym_table_id` + `spec.n_planes`.
///
/// Lazily initialises one static `SymTables` per distinct combination and
/// returns a `&'static` reference. Panics on an unknown combination — only
/// valid registry entries should ever be passed here.
///
/// Supported (registered set `{v6, v6w25, v6_live2_ls, gnn_axis_v1}`):
///   `("size_19", _)` → 19×19 tables (v6 8-plane AND v6_live2_ls 4-plane —
///     `apply_symmetry_state` deduces plane count from `src.len()/n_cells`, so
///     the 361-cell table is byte-correct for any plane count).
///   `("size_25", 8)` → v6w25 (25×25, 8-plane).
pub fn sym_tables_for(spec: &'static RegistrySpec) -> &'static SymTables {
    // Hardening (R-5): the spec-derived `chain_stride()` and the compile-time
    // `chain_src_lookup` inner dimension can never disagree.
    assert_eq!(
        spec.n_chain_planes, N_CHAIN_PLANES,
        "sym_tables_for: encoding {:?} has n_chain_planes={} but the D6 chain tables \
         are dimensioned for {N_CHAIN_PLANES}",
        spec.name, spec.n_chain_planes
    );

    // "size_19" (v6, v6_live2_ls): board 19×19 — shared singleton (plane count
    // is deduced at scatter time, not baked into the table).
    static SIZE19_8: LazyLock<SymTables> = LazyLock::new(|| SymTables::with_shape(19, 8));
    // "size_25" n_planes=8 (v6w25): board 25×25, 8-plane wire format.
    static SIZE25_8: LazyLock<SymTables> = LazyLock::new(|| SymTables::with_shape(25, 8));

    match (spec.sym_table_id, spec.n_planes) {
        ("size_19", _) => &SIZE19_8,
        ("size_25", 8) => &SIZE25_8,
        (id, np) => panic!(
            "sym_tables_for: no sym table for encoding {:?} (sym_table_id={:?}, n_planes={}). \
             Add a LazyLock<SymTables> entry in replay::sym::sym_tables_for().",
            spec.name, id, np
        ),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(name: &str) -> &'static RegistrySpec {
        mantis_encoding::registry::lookup_or_panic(name)
    }

    /// Apply the sym transform `(reflect, n_rot)` to an axial coordinate.
    fn apply_sym_coord(q: i32, r: i32, reflect: bool, n_rot: usize) -> (i32, i32) {
        let (mut q, mut r) = (q, r);
        if reflect {
            (q, r) = (r, q);
        }
        rotate_n(q, r, n_rot)
    }

    #[test]
    fn identity_axis_perm_is_identity() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[0], [0, 1, 2]);
    }

    #[test]
    fn rot60_axis_perm_cycles_2_0_1() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[1], [2, 0, 1]);
    }

    #[test]
    fn rot120_axis_perm_cycles_1_2_0() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[2], [1, 2, 0]);
    }

    #[test]
    fn rot180_axis_perm_is_identity() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[3], [0, 1, 2]);
    }

    #[test]
    fn rot240_matches_rot60() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[4], tables.axis_perm[1]);
    }

    #[test]
    fn rot300_matches_rot120() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[5], tables.axis_perm[2]);
    }

    #[test]
    fn reflection_only_swaps_axis0_and_axis1() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[6], [1, 0, 2]);
    }

    #[test]
    fn reflection_composed_with_rot60() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[7], [2, 1, 0]);
    }

    #[test]
    fn axis_perm_derived_from_coord_transform() {
        let tables = SymTables::new();
        for sym_idx in 0..N_SYMS {
            let reflect = sym_idx >= 6;
            let n_rot = sym_idx % 6;

            let perm = tables.axis_perm[sym_idx];
            let mut perm_inv = [usize::MAX; 3];
            for (j, &i) in perm.iter().enumerate() {
                perm_inv[i] = j;
            }
            for &v in &perm_inv {
                assert!(v < 3, "axis_perm[{sym_idx}] inverse has unset slot");
            }

            for src_i in 0..3 {
                let (bq, br) = HEX_BASIS[src_i];
                let (tq, tr) = apply_sym_coord(bq, br, reflect, n_rot);
                let expected_dst_j = perm_inv[src_i];
                assert!(
                    same_axis((tq, tr), HEX_BASIS[expected_dst_j]),
                    "sym_idx={} src_i={}: transformed basis {:?} does not match \
                     expected axis {} ({:?})",
                    sym_idx, src_i, (tq, tr), expected_dst_j, HEX_BASIS[expected_dst_j]
                );
            }
        }
    }

    #[test]
    fn axis_perm_is_bijection_for_all_syms() {
        let tables = SymTables::new();
        for sym_idx in 0..N_SYMS {
            let perm = tables.axis_perm[sym_idx];
            let mut seen = [false; 3];
            for &i in &perm {
                assert!(i < 3, "axis_perm[{sym_idx}] has out-of-range index: {perm:?}");
                assert!(!seen[i], "axis_perm[{sym_idx}] is not a bijection: {perm:?}");
                seen[i] = true;
            }
        }
    }

    #[test]
    fn v6_default_byte_exact() {
        let tables = SymTables::new();
        assert_eq!(tables.board_size, 19);
        assert_eq!(tables.n_cells, 361);
        assert_eq!(tables.n_planes, 8);
        assert_eq!(tables.scatter[0].len(), 361);
        for (i, &(src, dst)) in tables.scatter[0].iter().enumerate() {
            assert_eq!(src as usize, i);
            assert_eq!(dst as usize, i);
        }
    }

    #[test]
    fn v6_rot180_preserve_all_cells() {
        let v6 = SymTables::new();
        assert_eq!(v6.scatter[0].len(), 361, "v6 identity must keep all 361 cells");
        assert_eq!(v6.scatter[3].len(), 361, "v6 rot180 must keep all 361 cells");
    }

    // ── O-15: board-size invariance (v6 vs v6w25, re-anchored from v8) ──────────

    #[test]
    fn axis_perm_board_size_invariant_v6_vs_v6w25() {
        let v6 = SymTables::new();
        let w25 = SymTables::with_shape(lookup("v6w25").board_size, lookup("v6w25").n_planes);
        for s in 0..N_SYMS {
            assert_eq!(v6.axis_perm[s], w25.axis_perm[s], "axis_perm[{s}] must be board-size invariant");
        }
    }

    #[test]
    fn chain_src_lookup_board_size_invariant_v6_vs_v6w25() {
        let v6 = SymTables::new();
        let w25 = SymTables::with_shape(lookup("v6w25").board_size, lookup("v6w25").n_planes);
        for s in 0..N_SYMS {
            assert_eq!(
                v6.chain_src_lookup[s], w25.chain_src_lookup[s],
                "chain_src_lookup[{s}] must be board-size invariant"
            );
        }
    }

    // ── O-13 positive width pins (replaces the KILLed N_ACTIONS footgun test) ───

    #[test]
    fn each_encoding_derives_its_own_policy_width() {
        assert_eq!(lookup("v6").policy_stride(), 362, "v6 derives 362");
        assert_eq!(lookup("v6w25").policy_stride(), 626, "v6w25 derives its own 626");
    }

    // ── O-14: sym_tables_for singletons + shape ─────────────────────────────────

    #[test]
    fn sym_tables_for_v6_matches_new() {
        let via_fn = sym_tables_for(lookup("v6"));
        let via_new = SymTables::new();
        assert_eq!(via_fn.board_size, via_new.board_size);
        assert_eq!(via_fn.n_cells, via_new.n_cells);
        assert_eq!(via_fn.n_planes, via_new.n_planes);
        for s in 0..N_SYMS {
            assert_eq!(via_fn.scatter[s], via_new.scatter[s], "scatter[{s}] mismatch");
            assert_eq!(via_fn.axis_perm[s], via_new.axis_perm[s], "axis_perm[{s}] mismatch");
            assert_eq!(
                via_fn.chain_src_lookup[s], via_new.chain_src_lookup[s],
                "chain_src_lookup[{s}] mismatch"
            );
        }
    }

    #[test]
    fn sym_tables_for_v6w25_has_size_25() {
        let tables = sym_tables_for(lookup("v6w25"));
        assert_eq!(tables.board_size, 25);
        assert_eq!(tables.n_cells, 625);
        assert_eq!(tables.n_planes, 8);
        assert_eq!(tables.scatter[0].len(), 625);
    }

    #[test]
    fn sym_tables_for_v6_live2_ls_shares_size_19() {
        // v6_live2_ls is 4-plane, sym_table_id="size_19" — must share the v6
        // 361-cell table (the wildcard arm; plane count is deduced at scatter).
        let v6 = sym_tables_for(lookup("v6"));
        let live2 = sym_tables_for(lookup("v6_live2_ls"));
        assert_eq!(live2.board_size, 19);
        assert_eq!(live2.n_cells, 361);
        assert_eq!(
            v6 as *const SymTables, live2 as *const SymTables,
            "v6 and v6_live2_ls must share the same size_19 static singleton"
        );
    }

    #[test]
    fn sym_tables_for_returns_stable_ref() {
        let t1 = sym_tables_for(lookup("v6")) as *const _;
        let t2 = sym_tables_for(lookup("v6")) as *const _;
        assert_eq!(t1, t2, "sym_tables_for must return the same static singleton");
    }
}
