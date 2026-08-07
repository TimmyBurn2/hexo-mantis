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
use rand::RngExt;

// ── Geometric constants (board-size invariant) ─────────────────────────────────

/// D6 group order: 6 rotations × 2 (with/without prior reflection).
pub const N_SYMS: usize = 12;

/// The D6 elements under which the SQUARE axial window is CLOSED — the only
/// elements whose `with_shape` scatter is a bijection of the window onto itself.
///
/// Derivation (geometric, not measured — the per-size cell counts are DERIVED by
/// `window_preserving_syms_are_derived_not_asserted` below and are deliberately not
/// transcribed here, R8/G-DFIX-4): `to_flat` accepts exactly `max(|q|, |r|) ≤ half`,
/// so the window is the ∞-norm ball in the axial basis. A linear map preserves that
/// ball iff its matrix is a signed permutation matrix — one `±1` per row and per
/// column. Of the 12 D6 matrices in the axial basis only four are: the identity, the
/// 180° rotation `(q, r) → (−q, −r)`, the axis swap `(q, r) → (r, q)`, and their
/// product `(q, r) → (−r, −q)`. That is `{identity, 180°} × {no-reflect, reflect}`,
/// a Klein four-group. Every other element carries the `q + r` term the 60° rotation
/// `(q, r) → (−r, q + r)` introduces, so its matrix has a row `(±1, ±1)`, which sends
/// a window corner to a coordinate of magnitude `2·half` — outside. Under this file's
/// encoding (`reflect = sym_idx >= 6`, then `sym_idx % 6` sixty-degree rotations)
/// those four elements are `0`, `3`, `6`, `9`.
///
/// Why it matters (R245): `with_shape` pushes a scatter pair ONLY when `to_flat`
/// returns `Some` and has no `else` arm, so under one of the other eight elements a
/// dense record is rewritten with a block of its cells silently deleted while the
/// policy/value target rides along unchanged — label noise, not augmentation. But a
/// cell is only deleted if it HELD something: under R245 ruled option (c) a dense
/// site that can certify its subject record gates per record ([`draw_record_sym`] on
/// [`SymTables::dropped_cells`]) and restricts to this set only for the records that
/// would actually be clipped; a site that cannot certify its subject restricts
/// unconditionally ([`draw_window_preserving_sym`]). Uniform-over-12 at a
/// window-clamped DENSE site is never correct.
///
/// This is a property of the WINDOW, never of the transform: [`rotate_axial`] is a
/// lattice automorphism and is exact for all 12 elements. The graph path has no
/// window, so it keeps the full group — restricting it would discard 8/12 of the
/// graph arm's augmentation for no correctness gain.
pub const WINDOW_PRESERVING_SYMS: [usize; 4] = [0, 3, 6, 9];

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

/// Draw one D6 element uniformly from [`WINDOW_PRESERVING_SYMS`].
///
/// The correct draw for every WINDOW-CLAMPED (dense-grid) augmentation or rotation
/// site whose subject cannot be certified lossless: the dense scatter deletes
/// off-window cells, so the other eight elements inject label noise (R245). The
/// graph sample path draws from the full `0..N_SYMS` instead and must keep doing
/// so — `rotate_axial` never drops. A dense site that CAN certify its subject
/// record uses [`draw_record_sym`] and recovers the full group for the records
/// that are provably lossless.
#[inline]
pub fn draw_window_preserving_sym<R: RngExt>(rng: &mut R) -> usize {
    WINDOW_PRESERVING_SYMS[rng.random_range(0..WINDOW_PRESERVING_SYMS.len())]
}

/// Draw one D6 element for a DENSE record, gated on that record's own
/// losslessness (R245 ruled option (c)).
///
/// * `compact == true` — the record is NEUTRAL on every cell of
///   [`SymTables::dropped_cells`], so each of the eight window-dropping elements
///   deletes only neutral content from it and all 12 elements are exact. The full
///   group is drawn.
/// * `compact == false` — some channel carries content on a cell a dropping
///   element would delete, so only the always-lossless subgroup may be applied.
///
/// Either way NO clipped copy is ever produced: the gate is the per-record
/// evaluation of "is this element lossless HERE", and the subgroup is the answer
/// whenever the record cannot be certified. `false` is therefore the safe default
/// for an uncertified slot — it costs augmentation variety, never correctness.
#[inline]
pub fn draw_record_sym<R: RngExt>(rng: &mut R, compact: bool) -> usize {
    if compact {
        rng.random_range(0..N_SYMS)
    } else {
        draw_window_preserving_sym(rng)
    }
}

/// Compare two axial vectors in a direction-unsigned sense (axis identity).
#[inline]
fn same_axis(a: (i32, i32), b: (i32, i32)) -> bool {
    a == b || (-a.0, -a.1) == b
}

/// Derive **D** — the source cells the window-DROPPING D6 elements delete — from a
/// freshly built scatter table (R245(c)).
///
/// DERIVED, never transcribed: an element whose pair list is short of `n_cells`
/// drops every source cell that owns no pair. D is the UNION over those elements,
/// so "neutral on D" implies lossless under EVERY one of them — which is what makes
/// the compact/spread gate a single binary per record.
///
/// The eight dropping elements in fact delete the IDENTICAL set (each of their
/// matrices carries exactly one `(±1, ±1)` row, so the surviving condition is
/// `|q + r| <= half` for all of them, and the other row is signed-unit hence always
/// in range); `dropping_elements_drop_one_shared_cell_set` pins that equivalence,
/// which is what makes the binary gate EXACT rather than merely sound. The union is
/// used regardless, so the gate stays SOUND even if a future window shape broke the
/// equivalence — it would only stop being tight.
fn derive_dropped_cells(scatter: &[Vec<(u16, u16)>; N_SYMS], n_cells: usize) -> Vec<u16> {
    let mut dropped = vec![false; n_cells];
    for pairs in scatter {
        if pairs.len() == n_cells {
            continue; // window-preserving — drops nothing
        }
        let mut present = vec![false; n_cells];
        for &(sc, _) in pairs {
            present[sc as usize] = true;
        }
        for (cell, &seen) in present.iter().enumerate() {
            if !seen {
                dropped[cell] = true;
            }
        }
    }
    dropped
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d)
        .map(|(cell, _)| u16::try_from(cell).expect("cell index fits u16"))
        .collect()
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
    /// **D** — the SOURCE cells that the window-dropping D6 elements delete,
    /// ascending, DERIVED from `scatter` at construction (never a transcribed
    /// index list). Empty is impossible for any shipped window: the eight
    /// non-[`WINDOW_PRESERVING_SYMS`] elements each drop a corner wedge.
    ///
    /// This is the domain of the R245(c) per-record losslessness gate: a dense
    /// record whose every channel equals its NEUTRAL on every cell of this set is
    /// COMPACT — the full 12-element group transforms it exactly — and otherwise
    /// SPREAD, restricted to [`WINDOW_PRESERVING_SYMS`]. See
    /// `ReplayBuffer::slot_is_compact` for the neutrals and [`draw_record_sym`]
    /// for the draw.
    pub dropped_cells: Vec<u16>,
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

        let dropped_cells = derive_dropped_cells(&scatter, n_cells);

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
            dropped_cells,
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

    // ── R245: the window-preserving subgroup, DERIVED ───────────────────────────

    /// Live board sizes, taken from the registry rather than hardcoded, so a newly
    /// registered encoding is covered automatically (LAW-08 / derive-never-transcribe).
    fn registry_board_sizes() -> Vec<usize> {
        let mut sizes: Vec<usize> =
            mantis_encoding::registry::all_specs().map(|s| s.board_size).collect();
        sizes.sort_unstable();
        sizes.dedup();
        assert!(!sizes.is_empty(), "registry must ship at least one encoding");
        sizes
    }

    /// `WINDOW_PRESERVING_SYMS` is COMPUTED from the scatter geometry at every board
    /// size the registry actually ships and then compared to the constant — the
    /// constant is falsifiable by the geometry, never by a transcribed literal.
    ///
    /// A `sym_idx` preserves the window iff its scatter has one pair per cell (no
    /// `to_flat` rejection). Two anti-vacuity arms keep the test honest: every
    /// element OUTSIDE the set must genuinely drop cells (otherwise restricting the
    /// draw would be a no-op and this test would pass for free), and every element
    /// INSIDE it must be a true bijection of the window, not merely a full-length
    /// pair list with a collided destination.
    #[test]
    fn window_preserving_syms_are_derived_not_asserted() {
        for bs in registry_board_sizes() {
            let tables = SymTables::with_shape(bs, 8);
            let derived: Vec<usize> =
                (0..N_SYMS).filter(|&s| tables.scatter[s].len() == tables.n_cells).collect();
            assert_eq!(
                derived.as_slice(),
                WINDOW_PRESERVING_SYMS.as_slice(),
                "board_size={bs}: the DERIVED window-preserving set disagrees with \
                 WINDOW_PRESERVING_SYMS"
            );

            for s in 0..N_SYMS {
                if WINDOW_PRESERVING_SYMS.contains(&s) {
                    continue;
                }
                assert!(
                    tables.scatter[s].len() < tables.n_cells,
                    "board_size={bs}: sym {s} is outside WINDOW_PRESERVING_SYMS yet drops \
                     no cell — the restriction would be vacuous"
                );
            }

            for &s in &WINDOW_PRESERVING_SYMS {
                let mut dsts: Vec<u16> = tables.scatter[s].iter().map(|&(_, d)| d).collect();
                let n_pairs = dsts.len();
                dsts.sort_unstable();
                dsts.dedup();
                assert_eq!(
                    dsts.len(), n_pairs,
                    "board_size={bs}: sym {s} has colliding destinations — not a bijection"
                );
            }
        }
    }

    /// R245: the GRAPH path is structurally unaffected. `rotate_axial` is an axial
    /// lattice automorphism with no window to leave, so it is injective (hence a
    /// bijection of the lattice) for ALL 12 elements — restricting the DENSE draw
    /// removes nothing from the graph arm. Checked over a box strictly larger than
    /// the widest shipped window, so no element could hide a drop at the boundary.
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

    /// R245 boundary pin, both directions: the two window-clamped DENSE SAMPLE sites
    /// use the per-record GATED draw, the per-game rotation site keeps the flat
    /// window-preserving draw, and the GRAPH sample path keeps the FULL group.
    /// Source-presence, because the three are indistinguishable to the type system —
    /// any drift compiles cleanly and is silent in a run.
    #[test]
    fn dense_sites_restricted_and_graph_site_keeps_the_full_group() {
        const DENSE_SAMPLE: &str = include_str!("sample.rs");
        const GAME: &str = include_str!("../runner/game.rs");
        const HEXG_SAMPLE: &str = include_str!("hexg/sample.rs");

        assert_eq!(
            DENSE_SAMPLE
                .matches("draw_record_sym(&mut self.rng, self.compact[idx] != 0)")
                .count(),
            2,
            "both dense replay draw sites must use the R245(c) per-record gated draw"
        );
        assert!(
            !DENSE_SAMPLE.contains("random_range(0..N_SYMS)"),
            "a dense replay draw site still draws from the full group UNGATED"
        );
        assert!(
            !DENSE_SAMPLE.contains("draw_window_preserving_sym"),
            "a dense replay draw site is restricted UNCONDITIONALLY — compact records \
             must recover the full group (R245(c))"
        );
        // The per-game frame sym is drawn BEFORE the first stone is played and rides
        // the search input/inverse scatters and finalize; the compactness of the
        // yet-unplayed game is unknowable at draw time, so the record can never be
        // certified and the gate's answer is permanently "spread". That IS the
        // per-record gate evaluated at the only time it can be — hence the flat
        // window-preserving draw here, not `draw_record_sym`.
        assert!(
            GAME.contains("draw_window_preserving_sym(rng)"),
            "the per-game rotation draw must use the window-preserving draw"
        );
        assert!(
            !GAME.contains("random_range(0..N_SYMS)"),
            "the per-game rotation draw must not draw from the full group"
        );
        assert!(
            HEXG_SAMPLE.contains("self.rng.random_range(0..N_SYMS)"),
            "the graph sample path must keep the FULL 12-element draw (rotate_axial is exact)"
        );
        assert!(
            !HEXG_SAMPLE.contains("draw_window_preserving_sym")
                && !HEXG_SAMPLE.contains("draw_record_sym"),
            "the graph sample path must NOT be gated or restricted — it has no window"
        );
    }

    // ── R245(c): D (the dropped-cell set) + the gated draw ──────────────────────

    /// `SymTables::dropped_cells` is DERIVED — cross-checked against an INDEPENDENT
    /// derivation (the closed form `|q + r| > half` that the geometry argument
    /// predicts) at every board size the registry ships, and asserted non-empty so
    /// the gate can never be vacuously satisfied.
    #[test]
    fn dropped_cell_set_is_derived_not_asserted() {
        for bs in registry_board_sizes() {
            let tables = SymTables::with_shape(bs, 8);
            let half = (i32::try_from(bs).expect("board_size fits i32") - 1) / 2;
            let closed_form: Vec<u16> = (0..tables.n_cells)
                .filter(|&flat| {
                    let q = i32::try_from(flat / bs).expect("fits") - half;
                    let r = i32::try_from(flat % bs).expect("fits") - half;
                    (q + r).abs() > half
                })
                .map(|flat| u16::try_from(flat).expect("fits"))
                .collect();
            assert!(
                !closed_form.is_empty(),
                "board_size={bs}: the dropped set must be non-empty or the gate is vacuous"
            );
            assert_eq!(
                tables.dropped_cells, closed_form,
                "board_size={bs}: scatter-derived dropped set disagrees with the \
                 independent |q + r| > half derivation"
            );
        }
    }

    /// The equivalence the BINARY gate rests on: all eight window-dropping elements
    /// delete the IDENTICAL source-cell set, so "lossless under one dropping element"
    /// and "lossless under all of them" are the same predicate and one flag per record
    /// suffices. If a future window shape broke this, the per-element sets would
    /// diverge and this test REDs (the union `dropped_cells` would stay sound but the
    /// gate would no longer be exact).
    #[test]
    fn dropping_elements_drop_one_shared_cell_set() {
        for bs in registry_board_sizes() {
            let tables = SymTables::with_shape(bs, 8);
            let per_element = |s: usize| -> Vec<u16> {
                let mut present = vec![false; tables.n_cells];
                for &(sc, _) in &tables.scatter[s] {
                    present[sc as usize] = true;
                }
                (0..tables.n_cells)
                    .filter(|&c| !present[c])
                    .map(|c| u16::try_from(c).expect("fits"))
                    .collect()
            };
            let mut n_dropping = 0usize;
            for s in 0..N_SYMS {
                let dropped = per_element(s);
                if WINDOW_PRESERVING_SYMS.contains(&s) {
                    assert!(
                        dropped.is_empty(),
                        "board_size={bs}: preserving sym {s} dropped {} cells",
                        dropped.len()
                    );
                } else {
                    n_dropping += 1;
                    assert_eq!(
                        dropped, tables.dropped_cells,
                        "board_size={bs}: dropping sym {s} deletes a DIFFERENT cell set — \
                         the binary compact/spread gate is no longer exact"
                    );
                }
            }
            assert_eq!(
                n_dropping,
                N_SYMS - WINDOW_PRESERVING_SYMS.len(),
                "board_size={bs}: unexpected number of dropping elements"
            );
        }
    }

    /// `draw_record_sym` support is EXACT on both arms — compact records recover the
    /// whole group, spread records see nothing outside the subgroup. Subset-only
    /// checks would pass for a degenerate always-identity draw, so both directions
    /// are asserted.
    #[test]
    fn draw_record_sym_support_is_exact_on_both_arms() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use std::collections::HashSet;

        let mut rng = StdRng::seed_from_u64(0x0024_500C);
        let compact_seen: HashSet<usize> =
            (0..4096).map(|_| draw_record_sym(&mut rng, true)).collect();
        assert_eq!(
            compact_seen,
            (0..N_SYMS).collect::<HashSet<usize>>(),
            "a COMPACT record must draw over the full D6 group"
        );

        let spread_seen: HashSet<usize> =
            (0..4096).map(|_| draw_record_sym(&mut rng, false)).collect();
        assert_eq!(
            spread_seen,
            WINDOW_PRESERVING_SYMS.iter().copied().collect::<HashSet<usize>>(),
            "a SPREAD record must draw ONLY from the window-preserving subgroup"
        );
    }

    #[test]
    fn sym_tables_for_returns_stable_ref() {
        let t1 = sym_tables_for(lookup("v6")) as *const _;
        let t2 = sym_tables_for(lookup("v6")) as *const _;
        assert_eq!(t1, t2, "sym_tables_for must return the same static singleton");
    }
}
